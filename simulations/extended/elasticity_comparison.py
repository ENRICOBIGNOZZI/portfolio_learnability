"""Compare N-specific local elasticities on the frozen common history grid.

This is deterministic postprocessing of saved spectra, never a recalibration or
a Monte Carlo selection rule. Ranges across numerical cases are not intervals
for a sampling distribution.
"""
import json
import re

import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT
from simulations.extended.population import spectral_path
from simulations.provenance import file_hash


def run():
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    times = np.asarray(protocol['robustness_T'], dtype=int)
    population = pd.read_csv(OUTPUT / 'population_summary.csv')
    rows = []
    for name in ('N300', 'baseline', 'N1200'):
        for path in sorted((OUTPUT / 'pilot').glob(f'spectra_{name}_*.npz')):
            match = re.fullmatch(r'spectra_\w+_P(\d+)_B(\d+)_Q(\d+)_S(\d+)\.npz', path.name)
            rank, basis, groups, quadrature = map(int, match.groups())
            primary = (rank, basis, groups, quadrature) == (
                protocol['rank'], protocol['basis_seed'],
                protocol['population_groups'], protocol['population_seed'])
            with np.load(path) as saved:
                complexity, _, elasticity = spectral_path(
                    saved['managed'], protocol['a'][name] * times.astype(float)**(-.6))
            if primary:
                reference = population[(population.environment == name) &
                    population['T'].isin(times)].sort_values('T')
                np.testing.assert_array_equal(reference['T'], times)
                np.testing.assert_allclose(complexity, reference.complexity, rtol=1e-12)
                np.testing.assert_allclose(elasticity, reference.local_T_elasticity, rtol=1e-12)
            source_hash = file_hash(path)
            for T, C, E in zip(times, complexity, elasticity):
                rows.append(dict(environment=name, T=int(T), rank=rank,
                    basis_seed=basis, population_groups=groups, population_seed=quadrature,
                    primary=primary, a=protocol['a'][name], complexity=float(C),
                    local_T_elasticity=float(E), source=path.name, source_sha256=source_hash,
                    run_hash=protocol['run_hash']))
    audit = pd.DataFrame(rows)
    keys = ['rank', 'basis_seed', 'population_groups', 'population_seed', 'primary', 'T']
    paired = audit.pivot(index=keys, columns='environment', values='local_T_elasticity')
    assert set(paired.columns) == {'N300', 'baseline', 'N1200'}
    assert paired.notna().all().all(), 'Each numerical case must include all three N values.'
    paired = paired.reset_index()
    paired['N300_minus_baseline'] = paired.N300 - paired.baseline
    paired['N1200_minus_baseline'] = paired.N1200 - paired.baseline
    paired['cross_N_range'] = paired[['N300', 'baseline', 'N1200']].max(axis=1) - paired[
        ['N300', 'baseline', 'N1200']].min(axis=1)
    paired['run_hash'] = protocol['run_hash']
    assert len(paired[paired.primary]) == len(times)
    audit.to_csv(OUTPUT / 'population_elasticity_N_audit.csv', index=False)
    paired.to_csv(OUTPUT / 'population_elasticity_N_comparison.csv', index=False)
    return paired


if __name__ == '__main__':
    run()
