"""Reproduce canonical fitted policies before admitting a remote production host."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from simulations.extended.design import OUTPUT, ROOT, ENVIRONMENTS
from simulations.extended.remote_compatibility import protocol_checked
from simulations.extended.remote_inputs import download, supporting_paths
from simulations.provenance import file_hash, json_write, utc_now


CANONICAL = ROOT / 'simulations/outputs/rich6d_rough_sr3_extended_v2'


def bootstrap():
    if OUTPUT == CANONICAL or OUTPUT.exists():
        raise ValueError('Use a new isolated RICH6D_EXTENDED_OUTPUT directory for reproduction.')
    OUTPUT.mkdir(parents=True)
    files = ['protocol.json', 'remote_input_manifest.json', *supporting_paths()]
    for relative in files:
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CANONICAL / relative, target)


def compare(path, reference, names, rules, production, protocol):
    checks = []
    with np.load(path) as got, np.load(reference) as old:
        if (int(got['index']) != 0 or int(old['index']) != 0
                or int(got['rank']) != int(old['rank']) or int(got['rank']) != protocol['rank']):
            raise ValueError('Fitted-policy reproduction identity mismatch.')
        if int(got['training_seed']) != int(old['training_seed']):
            raise ValueError('Fitted-policy reproduction training seed mismatch.')
        for name in names:
            times = got[name + '_T']
            expected_times = protocol['baseline_T'] if name == 'baseline' else protocol['robustness_T']
            if not np.array_equal(times, expected_times):
                raise ValueError('Fitted-policy reproduction must cover every frozen horizon.')
            reference_times = old[name + '_T']
            positions = [int(np.flatnonzero(reference_times == t)[0]) for t in times]
            for metric in rules['metrics']:
                actual = got[name + '_' + metric]
                expected = old[name + '_' + metric][positions]
                if actual.shape != (len(times), 97) or expected.shape != actual.shape:
                    raise ValueError('Missing horizons or candidate penalties in fitted-policy comparison.')
                delta = np.abs(actual - expected)
                passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected,
                              rtol=rules['rtol'], atol=rules['atol']))
                checks.append(dict(environment=name, metric=metric, passed=passed,
                    maximum_absolute_difference=float(delta.max()),
                    maximum_tolerance_fraction=float(np.max(delta /
                        (rules['atol'] + rules['rtol'] * np.abs(expected))))))
            if float(got[name + '_floor']) != float(old[name + '_floor']):
                raise ValueError('Frozen population projection floor changed.')
        actual_hashes = json.loads(str(got['returns_hashes']))
        expected_hashes = json.loads(str(old['returns_hashes']))
        keys = ('baseline', 'baseline_T1440', 'baseline_T3240') if production else tuple(
            key for key in actual_hashes if key != 'baseline')
        hash_checks = {key: actual_hashes[key] == expected_hashes[key] for key in keys}
        normal_error = float(got['maximum_normal_equation_error'])
        normal_pass = not production or normal_error <= rules['maximum_production_normal_equation_error']
    return dict(reference=str(reference.relative_to(ROOT)), reference_sha256=file_hash(reference),
        computed_sha256=file_hash(path), metric_checks=checks, history_hash_checks=hash_checks,
        maximum_normal_equation_error=normal_error, normal_equation_check_applicable=production,
        passed=all(row['passed'] for row in checks) and all(hash_checks.values()) and normal_pass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    plan = json.loads((CANONICAL / 'remote_reproduction_plan.json').read_text())
    if platform.system() != 'Darwin' or os.environ.get('RICH6D_RUNNER_LABEL') != plan['runner']:
        raise ValueError('Only the history-qualified macOS runner is eligible for this probe.')
    bootstrap()
    protocol = protocol_checked()
    if plan['run_hash'] != protocol['run_hash']:
        raise ValueError('Reproduction plan belongs to a different protocol.')
    for case in ('baseline_probe', 'paired_probe'):
        if file_hash(CANONICAL / plan[case]['reference']) != plan[case]['reference_sha256']:
            raise ValueError('Canonical fitted-policy reference changed after probe declaration.')
    manifest = download(protocol)
    from simulations.extended.production import execute
    from simulations.extended.compute import run_path
    execute(0, protocol, 'production_baseline')
    baseline = compare(OUTPUT / plan['baseline_probe']['reference'],
        CANONICAL / plan['baseline_probe']['reference'], ['baseline'], plan['comparison'], True, protocol)
    if not baseline['passed']:
        paired = None
    else:
        run_path(0, [protocol['rank']], protocol['baseline_T'], protocol['robustness_T'],
                 protocol['population_groups'], protocol['population_seed'], 'rank_pilot',
                 environment_names=list(ENVIRONMENTS))
        paired = compare(OUTPUT / plan['paired_probe']['reference'],
            CANONICAL / plan['paired_probe']['reference'], list(ENVIRONMENTS), plan['comparison'], False, protocol)
    result = dict(run_hash=protocol['run_hash'], runner=plan['runner'], source_hashes_checked=True,
        plan_sha256=file_hash(CANONICAL / 'remote_reproduction_plan.json'),
        input_manifest_sha256=file_hash(OUTPUT / 'remote_input_manifest.json'),
        input_files_verified=len(manifest['files']), baseline=baseline, paired=paired,
        passed=baseline['passed'] and paired is not None and paired['passed'],
        elapsed_seconds=time.perf_counter() - start, completed_utc=utc_now(),
        scope='Operational fitted-policy equivalence gate only; no new production indices and no change to the scientific 5% approximation thresholds.')
    json_write(args.report, result)
    print('REMOTE_FITTED_REPRODUCTION_RESULT=' + json.dumps(result, sort_keys=True), flush=True)
    if not result['passed']:
        raise SystemExit('Remote fitted-policy reproduction failed; do not admit remote production.')


if __name__ == '__main__':
    main()
