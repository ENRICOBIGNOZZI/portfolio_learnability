"""Cost/solver/projection pilot; no confirmation loss or rate is inspected."""
import argparse
from dataclasses import asdict
from pathlib import Path
import resource
import time

import numpy as np

from simulations.dgp.balanced import DGPParameters
from simulations.estimator.kernel import NystromBasis
from simulations.estimator.selectors import fit_training
from simulations.experiments.monte_carlo import managed_path
from simulations.experiments.population import population_moments
from simulations.provenance import json_write, source_hashes, utc_now

MASTER = 2026100801


def seed(family, index=0):
    # Economic DGP independently spawns state/factor/idiosyncratic streams.
    families = {'anchor': 1, 'pilot': 2, 'quadrature': 3, 'train': 4,
                'test': 5, 'bootstrap': 6, 'folds': 7, 'spectrum': 8}
    return int(np.random.SeedSequence([MASTER, families[family], index]).generate_state(1, dtype=np.uint32)[0])


def run(destination):
    records = []
    started = utc_now()
    for eta in (0., .35):
        p = DGPParameters(loading_map='rich6d' if eta else 'baseline_original', eta=eta)
        for nu in (1.5, .5, 2.5):
            for rank in (256, 512, 1024):
                start = time.perf_counter()
                basis = NystromBasis(rank=rank, nu=nu, seed=seed('anchor'))
                mean, second = population_moments(p, basis, 8192, seed('pilot', 1))
                eigenvalues = np.linalg.eigvalsh(second)
                floor = p.q_star-mean@np.linalg.solve(second, mean)
                population_seconds = time.perf_counter()-start
                start = time.perf_counter()
                train, _, _ = managed_path(p, basis, 'baseline', seed('pilot', 2), 240)
                data_seconds = time.perf_counter()-start
                start = time.perf_counter()
                fit = fit_training(train, np.geomspace(1e-12, 1e-2, 96), [eigenvalues[-1]*240**(-.6)])
                fit_seconds = time.perf_counter()-start
                coefficients = fit['coefficients']
                residual = train.T@(train@coefficients)/len(train)+coefficients*fit['penalties']-train.mean(axis=0)[:, None]
                records.append({'loading_map': p.loading_map, 'eta': eta, 'nu': nu, 'rank': rank,
                                'parameters': p.to_dict(), 'population_groups': 8192,
                                'population_seed': seed('pilot', 1), 'training_seed': seed('pilot', 2),
                                'anchor_seed': seed('anchor'), 's_ref': float(eigenvalues[-1]),
                                'projection_floor': float(floor), 'floor_fraction_of_q_star': float(floor/p.q_star),
                                'population_seconds': population_seconds, 'seconds_per_date': data_seconds/240,
                                'four_training_fits_seconds_T240': fit_seconds,
                                'max_normal_equation_error': float(np.max(np.abs(residual))),
                                'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
                print(f'Pilot {p.loading_map} nu={nu} rank={rank}: {data_seconds/240:.5f}s/date; floor/q={floor/p.q_star:.5g}', flush=True)
                json_write(Path(destination)/'cost_accuracy_pilot.json', {'schema': 'confirmation-pilot/2.0',
                           'started_utc': started, 'updated_utc': utc_now(), 'records': records,
                           'source_hashes': source_hashes(), 'complete': False,
                           'restriction': 'No confirmation performance, slopes or selector ranking examined.'})
    report = {'schema': 'confirmation-pilot/2.0', 'started_utc': started, 'completed_utc': utc_now(),
              'records': records, 'source_hashes': source_hashes(), 'complete': True,
              'restriction': 'No confirmation performance, slopes or selector ranking examined.'}
    json_write(Path(destination)/'cost_accuracy_pilot.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2/audit'))
    run(parser.parse_args().output)
