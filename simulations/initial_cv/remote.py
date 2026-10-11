"""Bounded remote batches; full production is blocked until the twelve-path gate."""
import argparse
import json
import os
import platform
import time
from importlib.metadata import version

from simulations.extended.remote_reproduction import bootstrap
from simulations.extended.remote_inputs import download
from simulations.extended.remote_compatibility import protocol_checked
from simulations.extended.design import OUTPUT as INPUTS
from simulations.initial_cv.design import OUTPUT, CANONICAL, protocol
from simulations.initial_cv.run import run
from simulations.provenance import json_write, file_hash, utc_now


def indices(stage, batch):
    frozen = protocol()
    if stage == 'pilot':
        if batch not in range(12):
            raise ValueError('Undeclared pilot batch.')
        return [batch]
    gate = json.loads((CANONICAL/'pilot_gate.json').read_text())
    if not gate['passed'] or gate['indices'] != list(range(12)) or gate['run_hash'] != frozen['run_hash']:
        raise ValueError('Verify all twelve pilot paths before production.')
    allocation = json.loads((CANONICAL/'production_allocation.json').read_text())
    if allocation['pilot_gate_sha256'] != file_hash(CANONICAL/'pilot_gate.json'):
        raise ValueError('Production allocation is not tied to the successful pilot gate.')
    flat = [i for b in allocation['batches'] for i in b['indices']]
    if sorted(flat) != list(range(12,300)) or len(flat) != 288:
        raise ValueError('Production must fill exactly the remaining 288 indices.')
    if not 0 <= batch < len(allocation['batches']):
        raise ValueError('Unknown batch.')
    return allocation['batches'][batch]['indices']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pilot','production'], required=True)
    parser.add_argument('--batch', type=int, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or os.environ.get('RICH6D_RUNNER_LABEL') != 'macos-26':
        raise ValueError('Use the original history-qualified macOS runner.')
    if platform.python_version() != '3.13.3' or version('numpy') != '2.3.5' or version('scipy') != '1.16.3':
        raise ValueError('Frozen scientific dependency versions required.')
    work = indices(args.stage, args.batch)
    if INPUTS.exists():
        raise ValueError('Use a new isolated input-cache folder.')
    bootstrap()
    download(protocol_checked())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = dict(stage=args.stage, batch=args.batch, indices=work, completed=[], files=[],
        run_hash=protocol()['run_hash'], source_commit=os.environ.get('GITHUB_SHA'),
        workflow_run_id=os.environ.get('GITHUB_RUN_ID'), started_utc=utc_now())
    start = time.perf_counter()
    try:
        for index in work:
            path = run(index)
            record['completed'].append(index)
            record['files'].append(dict(path=str(path.relative_to(OUTPUT)), sha256=file_hash(path), bytes=path.stat().st_size))
            record.update(updated_utc=utc_now(), elapsed_seconds=time.perf_counter()-start)
            json_write(OUTPUT/'transfer_manifest.json', record)
    finally:
        json_write(OUTPUT/'transfer_manifest.json', record)


if __name__ == '__main__':
    main()
