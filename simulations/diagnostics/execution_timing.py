"""Separate completed checkpoint time from original, resume and render events."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from simulations.provenance import file_hash, json_write, require, utc_now


def report_timing(output):
    output = Path(output)
    manifest = json.loads((output/'run_manifest.json').read_text())
    interruption_path = output/'audit/resource_adjustment.json'
    interruption = json.loads(interruption_path.read_text()) if interruption_path.exists() else {}
    stopped = interruption.get('timestamp_utc')
    events = []
    for event in manifest['execution_events']:
        row = dict(event)
        start = datetime.fromisoformat(row['started_utc'])
        if 'completed_utc' in row:
            row['status'] = row.get('status', 'finished')
            row['wall_clock_seconds'] = (datetime.fromisoformat(row['completed_utc'])-start).total_seconds()
        elif stopped and row['workers'] == interruption.get('previous_production_workers') and start < datetime.fromisoformat(stopped):
            row.update(status='interrupted_by_user_resource_adjustment', stopped_utc=stopped,
                       wall_clock_seconds=(datetime.fromisoformat(stopped)-start).total_seconds(),
                       end_evidence='audit/resource_adjustment.json')
        else:
            row['status'] = 'no_recorded_end'
        events.append(row)
    counts = []
    for env in manifest['configuration']['cases']:
        elapsed = []
        for r in range(env['replications']):
            path = output/'data'/env['name']/'replications'/f'{r:04d}.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as saved:
                seconds = float(saved['runtime_seconds'])
                require(np.isfinite(seconds) and seconds >= 0, 'Invalid checkpoint elapsed time')
                elapsed.append(seconds)
        counts.append({'environment': env['name'], 'completed_evaluations': len(elapsed),
                       'expected_evaluations': env['replications'],
                       'sum_completed_replication_seconds': sum(elapsed)})
    result = {'schema': 'execution-timing/2.0', 'recorded_utc': utc_now(),
              'manifest_sha256': file_hash(output/'run_manifest.json'), 'execution_events': events,
              'environments': counts, 'sum_completed_replication_seconds': sum(r['sum_completed_replication_seconds'] for r in counts),
              'completed_evaluations': sum(r['completed_evaluations'] for r in counts),
              'distinct_economic_seed_indices': max(r['completed_evaluations'] for r in counts),
              'limitations': 'Checkpoint times sum individual completed evaluations, including their fitting/evaluation, not wall-clock or CPU time. They exclude aborted work and separate numerical audits. Paired environments reuse innovation seeds. V1 timing is not replaced by this V2 report.'}
    json_write(output/'audit/execution_timing.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    print(json.dumps(report_timing(parser.parse_args().output), indent=2))
