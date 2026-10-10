"""Check host compatibility before moving any frozen production computation.

The reference uses an already audited baseline checkpoint and a short paired
economic-history prefix. Passing this gate does not replace a fitted-policy
reproduction or the per-checkpoint production integrity checks.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from simulations.extended.design import OUTPUT, ROOT
from simulations.extended.histories import PairedHistory
from simulations.provenance import digest, file_hash, json_write, utc_now


REFERENCE = OUTPUT / 'remote_compatibility_reference.json'


def protocol_checked():
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    science = {k: v for k, v in protocol.items()
               if k not in ('run_hash', 'frozen_before_production_utc')}
    if digest(science) != protocol['run_hash']:
        raise ValueError('Invalid frozen protocol identity.')
    actual = {name: file_hash(ROOT / name) for name in protocol['source_hashes']}
    if actual != protocol['source_hashes']:
        raise ValueError('Frozen scientific source mismatch.')
    return protocol


def paired_prefix(training_seed, index, periods):
    sim = PairedHistory(training_seed, index)
    hashes = {name: hashlib.sha256() for name in sim.hashers}
    for _ in range(periods):
        for name, (z, returns) in sim.step(robustness=True).items():
            hashes[name].update(z.tobytes())
            hashes[name].update(returns.tobytes())
    return {name: h.hexdigest() for name, h in hashes.items()}


def create_reference(protocol):
    if REFERENCE.exists():
        raise ValueError('Do not overwrite the canonical compatibility reference.')
    checkpoint = OUTPUT / 'production_baseline' / f'P{protocol["rank"]}' / 'rep_000.npz'
    audit = json.loads((OUTPUT / 'checkpoint_integrity.json').read_text())
    audited = next(row for row in audit['checkpoints']
                   if row['stage'] == 'production_baseline' and row['index'] == 0)
    if audit['run_hash'] != protocol['run_hash'] or file_hash(checkpoint) != audited['sha256']:
        raise ValueError('The canonical baseline checkpoint must have a matching integrity audit.')
    with np.load(checkpoint) as z:
        training_seed = int(z['training_seed'])
        hashes = json.loads(str(z['returns_hashes']))
    reference = dict(run_hash=protocol['run_hash'], index=0, training_seed=training_seed,
        baseline_T=max(protocol['baseline_T']), prefix_T=16,
        checkpoint_sha256=audited['sha256'], created_utc=utc_now(),
        baseline_hashes={key: hashes[key] for key in
                         ('baseline_T1440', 'baseline_T3240', 'baseline')},
        paired_prefix_hashes=paired_prefix(training_seed, 0, 16))
    json_write(REFERENCE, reference)


def run(reference, protocol):
    if reference['run_hash'] != protocol['run_hash']:
        raise ValueError('Reference belongs to a different frozen study.')
    start = time.perf_counter()
    prefix = paired_prefix(reference['training_seed'], reference['index'], reference['prefix_T'])
    sim = PairedHistory(reference['training_seed'], reference['index'])
    hashes = {}
    for t in range(reference['baseline_T']):
        sim.step(robustness=False)
        if t + 1 in (1440, 3240):
            hashes[f'baseline_T{t + 1}'] = sim.hashes()['baseline']
    hashes['baseline'] = sim.hashes()['baseline']
    baseline_checks = {key: hashes[key] == expected
                       for key, expected in reference['baseline_hashes'].items()}
    prefix_checks = {key: prefix[key] == expected
                     for key, expected in reference['paired_prefix_hashes'].items()}
    return dict(run_hash=protocol['run_hash'], source_hashes_checked=True,
        reference_sha256=file_hash(REFERENCE), probe_source_sha256=file_hash(Path(__file__)),
        host=dict(system=platform.system(), machine=platform.machine(), python=sys.version,
                  numpy=version('numpy'), scipy=version('scipy'),
                  runner_label=os.environ.get('RICH6D_RUNNER_LABEL', 'local')),
        index=reference['index'], baseline_T=reference['baseline_T'], prefix_T=reference['prefix_T'],
        baseline_hashes=hashes, paired_prefix_hashes=prefix,
        baseline_hash_checks=baseline_checks, paired_prefix_checks=prefix_checks,
        passed=all(baseline_checks.values()) and all(prefix_checks.values()),
        elapsed_seconds=time.perf_counter() - start, completed_utc=utc_now(),
        scope='One complete baseline history and a 16-period paired prefix in all five economies; no fitted policies or remote production. A full paired fitted-policy reproduction is still required before remote production.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create-reference', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    protocol = protocol_checked()
    if args.create_reference:
        create_reference(protocol)
    result = run(json.loads(REFERENCE.read_text()), protocol)
    json_write(args.report, result)
    print('REMOTE_COMPATIBILITY_RESULT=' + json.dumps(result, sort_keys=True), flush=True)
    if not result['passed']:
        raise SystemExit('Host does not reproduce the canonical economic-history hashes.')


if __name__ == '__main__':
    main()
