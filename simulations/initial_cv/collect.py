"""Retrieve pinned remote checkpoints and independently verify every path."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


from simulations.initial_cv.design import OUTPUT, protocol
from simulations.initial_cv.run import verify
from simulations.provenance import file_hash, json_write, utc_now

REPO = 'ENRICOBIGNOZZI/portfolio_learnability'


def gh(*args):
    for attempt in range(4):
        try:
            return subprocess.check_output(['gh',*args],text=True,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(3*(attempt+1))


def collect(stage, run_id, commit, download_root):
    frozen = protocol()
    download_root = Path(download_root)
    download_root.mkdir(parents=True,exist_ok=True)
    audit_path = OUTPUT/f'collection_{stage}.json'
    audit = dict(stage=stage,run_id=str(run_id),commit=commit,run_hash=frozen['run_hash'],
                 batches=[],checkpoints=[],status='collecting')
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit['run_id'] != str(run_id) or audit['commit'] != commit:
            raise ValueError('Collector identity mismatch.')
        for record in audit['checkpoints']:
            if file_hash(OUTPUT/record['path']) != record['sha256']:
                raise ValueError('Previously imported checkpoint changed.')
    processed = set(audit['batches'])
    while True:
        state = json.loads(gh('run','view',str(run_id),'--repo',REPO,'--json','headSha,status,conclusion,jobs'))
        if state['headSha'] != commit:
            raise ValueError('Remote source commit mismatch.')
        for job in state['jobs']:
            if not job['name'].startswith('compute (') or job['status'] != 'completed':
                continue
            batch = int(job['name'].split('(')[1].split(')')[0])
            if batch in processed:
                continue
            folder = download_root/f'batch_{batch}'
            folder.mkdir(exist_ok=True)
            gh('run','download',str(run_id),'--repo',REPO,'--name',f'initial-cv-{stage}-{batch}','--dir',str(folder))
            transfer = json.loads((folder/'transfer_manifest.json').read_text())
            if (transfer['run_hash'] != frozen['run_hash'] or transfer['source_commit'] != commit
                    or transfer['workflow_run_id'] != str(run_id) or transfer['stage'] != stage or transfer['batch'] != batch):
                raise ValueError('Transfer provenance mismatch.')
            if stage == 'pilot':
                expected = [batch]
            else:
                allocation = json.loads((OUTPUT/'production_allocation.json').read_text())
                expected = allocation['batches'][batch]['indices']
            if transfer['indices'] != expected or transfer['completed'] != expected[:len(transfer['completed'])]:
                raise ValueError('Transfer index allocation mismatch.')
            if len(transfer['files']) != len(transfer['completed']):
                raise ValueError('Incomplete transfer file manifest.')
            for index, record in zip(transfer['completed'],transfer['files']):
                if record['path'] != f'replications/rep_{index:03d}.npz':
                    raise ValueError('Unexpected artifact path.')
                source = folder/record['path']
                if source.stat().st_size != record['bytes'] or file_hash(source) != record['sha256']:
                    raise ValueError('Checkpoint transfer failed its checksum.')
                verify(source,index,frozen)
                dest = OUTPUT/record['path']
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists() and file_hash(dest) != record['sha256']:
                    raise ValueError('Conflicting existing checkpoint; do not overwrite it.')
                shutil.copyfile(source,dest)
                audit['checkpoints'].append(dict(index=index,**record,job_id=job['databaseId']))
            processed.add(batch)
            audit['batches'] = sorted(processed)
            audit['updated_utc'] = utc_now()
            json_write(audit_path,audit)
            print(f'{stage}: {len(audit["checkpoints"])} independently verified paths',flush=True)
        if state['status'] == 'completed':
            audit.update(status='completed' if state['conclusion']=='success' else 'failed',
                         conclusion=state['conclusion'],completed_utc=utc_now())
            json_write(audit_path,audit)
            if state['conclusion'] != 'success':
                raise RuntimeError('Remote run failed; completed checkpoints are preserved.')
            break
        time.sleep(30)
    expected_indices = list(range(12)) if stage=='pilot' else list(range(12,300))
    if sorted(r['index'] for r in audit['checkpoints']) != expected_indices:
        raise ValueError('Successful run did not return all assigned indices.')
    if stage == 'pilot':
        gate = dict(passed=True,indices=list(range(12)),run_hash=frozen['run_hash'],
            verified_utc=utc_now(),remote_run_id=str(run_id),source_commit=commit,
            checks=['Original five-economy return hashes','Original fixed-rule metrics at every horizon',
                    'Initial validation minimizer','Same selected a at every T','Normal equations','Population regret floor'],
            checkpoint_sha256={r['path']:r['sha256'] for r in audit['checkpoints']},
            scope='Implementation correctness gate; no performance-dependent change to grid, folds, rank or exponent.')
        json_write(OUTPUT/'pilot_gate.json',gate)
        batches=[dict(id=k,indices=list(range(start,min(start+20,300)))) for k,start in enumerate(range(12,300,20))]
        json_write(OUTPUT/'production_allocation.json',dict(run_hash=frozen['run_hash'],
            pilot_gate_sha256=file_hash(OUTPUT/'pilot_gate.json'),already_completed=list(range(12)),batches=batches))
    else:
        for index in range(300):
            verify(OUTPUT/'replications'/f'rep_{index:03d}.npz',index,frozen)
        json_write(OUTPUT/'production_verification.json',dict(passed=True,replications_per_economy=300,
            economies=5,run_hash=frozen['run_hash'],verified_utc=utc_now(),
            histories_and_old_rule_reproduced=True,initial_calibration_frozen_at_all_horizons=True))
    return audit


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pilot','production'],required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--download-root',type=Path,required=True)
    args=parser.parse_args()
    collect(args.stage,args.run_id,args.commit,args.download_root)
