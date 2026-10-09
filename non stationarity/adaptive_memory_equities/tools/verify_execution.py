"""Verify saved executions, source archives and current-selector decision parity."""
import argparse
import json
from pathlib import Path
import numpy as np
from adaptive_memory_equities.config import PACKAGE, REPO, digest, write_json, provenance
from adaptive_memory_equities.guard import mature_records
from adaptive_memory_equities.prequential import choose_methods


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    assert manifest['status'] == 'REAL_RUN_COMPLETED'
    source = json.loads((directory/'source.json').read_text())
    archived = PACKAGE/'private_runs/code_snapshots'/source['code_hash']
    for relative, expected in source['code'].items():
        original = REPO/relative
        if original.parent == PACKAGE and archived.exists():
            original = archived/original.name
        assert digest(original) == expected, str(original)
    events = [json.loads(p.read_text()) for p in sorted((directory/'events').glob('*.json'))]
    past = []
    for event in events:
        keys = sorted(event['experts'])
        metadata = {k:dict(architecture=k.split('|')[0],family=k.split('|')[1].split(':')[0]) for k in keys}
        selections,gates = choose_methods(keys,metadata,mature_records(past,event['decision_date']),source['config'])
        assert selections == event['selections'], event['decision_date']
        assert gates == event['gates'], event['decision_date']
        past.append(event)
    telemetry = [json.loads(line) for line in (directory/'memory.jsonl').read_text().splitlines()]
    result = dict(status='VERIFIED',actual_months=len(events),source_code_hash=source['code_hash'],
                  current_code_hash=provenance(source['config'])['code_hash'],
                  archived_source_checksums_verified=archived.exists(),
                  exact_current_selector_parity=True,resource_samples=len(telemetry))
    result['current_learning_code_matches_run'] = {
        name:digest(PACKAGE/name)==source['code'][str((PACKAGE/name).relative_to(REPO))]
        for name in ['training.py','networks.py','portfolio.py','memory.py','calendar.py','accounting.py','guard.py']}
    for key in ['rss_mb','available_mb','swap_used_mb','process_cpu_percent','system_cpu_percent',
                'system_read_mib_s','system_write_mib_s','disk_free_gib']:
        values = [r[key] for r in telemetry if key in r]
        if values:
            result[key] = dict(minimum=float(min(values)),median=float(np.median(values)),
                               p95=float(np.quantile(values,.95)),maximum=float(max(values)))
    result['elapsed_minutes'] = (telemetry[-1]['time']-telemetry[0]['time'])/60
    write_json(directory/'execution_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    verify(parser.parse_args().run_dir)
