"""Freeze the CV design and compact, immutable reproduction references."""
import json
import os
from pathlib import Path
import numpy as np

from simulations.extended.design import ROOT, ENVIRONMENTS
from simulations.extended.freeze import science_hashes
from simulations.initial_cv.selector import T0, EXPONENT, A_GRID, FOLDS
from simulations.provenance import digest, file_hash, json_write, npz_write, utc_now

REFERENCE = ROOT/'simulations/outputs/rich6d_rough_sr3_extended_v2'
CANONICAL = ROOT/'simulations/outputs/rich6d_initial_cv60_v1'
OUTPUT = Path(os.environ.get('RICH6D_CV_OUTPUT', str(CANONICAL))).resolve()


def freeze():
    old = json.loads((REFERENCE/'protocol.json').read_text())
    sources = science_hashes()
    for name in ('selector', 'run'):
        relative = f'simulations/initial_cv/{name}.py'
        sources[relative] = file_hash(ROOT/relative)
    design = dict(schema='initial-cv60/1', old_protocol_sha256=file_hash(REFERENCE/'protocol.json'),
        rank=old['rank'], baseline_T=old['baseline_T'], robustness_T=old['robustness_T'],
        environments=old['environments'], a_old=old['a'], basis_seed=old['basis_seed'],
        population_groups=old['population_groups'], population_seed=old['population_seed'],
        T0=T0, exponent=EXPONENT, a_grid=A_GRID.tolist(), folds=[list(f) for f in FOLDS],
        validation='Forward expanding training prefix, next disjoint ten observations; observation-weighted response-one squared loss.',
        tie_break='Largest a among exact minimum-score ties.',
        selection_time='Select separately for every economy and path at t=60, before generating observation 61. Never recalibrate.',
        pilot_indices=list(range(12)), production_indices=list(range(300)),
        pilot_promotion='Verified first twelve production-seed paths count toward the final 300; correctness gate only, never performance-based grid revision.',
        pairing='Original production training seeds and extra-group/noise streams; common innovations across all five economies.',
        oracle='For each economy and T, maximize mean population Sharpe on the original predetermined 96-penalty grid; diagnostic only, inaccessible to calibration.',
        numerical_warnings='Preserve original flags; additionally report CV floor/regret checks. No new fitted-policy rank certificate is claimed.',
        slope_windows=old['slope_windows'], primary_bands=old['primary_bands'],
        annualization=old['annualization'], source_hashes=sources)
    path = CANONICAL/'protocol.json'
    if path.exists():
        frozen = json.loads(path.read_text())
        if frozen['design'] != design:
            raise ValueError('Frozen design changed; do not overwrite it.')
        return frozen
    CANONICAL.mkdir(parents=True, exist_ok=True)
    reference_manifest = []
    for index in range(300):
        record = {}
        originals = {}
        for stage in ('production_baseline', 'production_robustness'):
            source = REFERENCE/stage/'P4096'/f'rep_{index:03d}.npz'
            originals[str(source.relative_to(REFERENCE))] = file_hash(source)
            with np.load(source) as z:
                record[stage+'_hashes'] = z['returns_hashes']
                record['training_seed'] = z['training_seed']
                names = ['baseline'] if stage == 'production_baseline' else [n for n in ENVIRONMENTS if n != 'baseline']
                for name in names:
                    record[name+'_T'] = z[name+'_T']
                    for metric in ('sr', 'loss', 'empirical_complexity'):
                        record[name+'_'+metric] = z[name+'_'+metric][:, -1]
        dest = CANONICAL/'references'/f'rep_{index:03d}.npz'
        npz_write(dest, **record)
        reference_manifest.append(dict(index=index, path=str(dest.relative_to(CANONICAL)),
            sha256=file_hash(dest), original_checkpoints=originals))
    json_write(CANONICAL/'reference_manifest.json', reference_manifest)
    frozen = dict(design=design, run_hash=digest(design), frozen_utc=utc_now(),
        references_sha256=file_hash(CANONICAL/'reference_manifest.json'))
    json_write(path, frozen)
    return frozen


def protocol():
    value = json.loads((CANONICAL/'protocol.json').read_text())
    if digest(value['design']) != value['run_hash']:
        raise ValueError('Protocol identity mismatch.')
    for name, expected in value['design']['source_hashes'].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError(f'Frozen scientific source changed: {name}')
    if file_hash(CANONICAL/'reference_manifest.json') != value['references_sha256']:
        raise ValueError('Reference manifest changed.')
    return value


if __name__ == '__main__':
    freeze()
