"""Recover completed GitHub batches promptly and independently audit each import."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import traceback

from simulations.extended.design import OUTPUT
from simulations.extended.remote_batch import STAGES, validate_allocation
from simulations.extended.remote_compatibility import protocol_checked
from simulations.extended.remote_ingest import ingest, validate_transfer
from simulations.extended.remote_inputs import REPOSITORY
from simulations.provenance import file_hash, json_write, output_lock, utc_now


def gh_json(*arguments):
    return json.loads(subprocess.check_output(['gh', *arguments], text=True, timeout=120))


def collect(stage, run_id, commit, download_root, poll_seconds=60):
    protocol = protocol_checked()
    allocation_path = OUTPUT / f'remote_allocation_{stage}.json'
    allocation = json.loads(allocation_path.read_text())
    allocation_hash = file_hash(allocation_path)
    validate_allocation(allocation, protocol, stage)
    seeds = json.loads((OUTPUT / 'seed_manifest.json').read_text())['replications']
    download_root.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT / f'remote_collection_{stage}_{run_id}.json'
    state = dict(run_hash=protocol['run_hash'], stage=stage, workflow_run_id=str(run_id),
        source_commit=commit, allocation_sha256=allocation_hash, started_utc=utc_now(),
        collector_pid=os.getpid(), download_root=str(download_root.resolve()),
        status='polling', downloaded_artifacts=[], verified_counts=None,
        scope='Import only after transfer validation and independent scientific audit. No final-study completion claim.')
    need_ingest = True  # Recover an interrupted import from previously downloaded artifacts.
    terminal_observations = 0
    with output_lock(OUTPUT / f'remote_collector_{stage}'):
        try:
            while True:
                if file_hash(allocation_path) != allocation_hash:
                    raise ValueError('Allocation changed while its collector was running.')
                try:
                    run = gh_json('run', 'view', str(run_id), '--repo', REPOSITORY,
                                  '--json', 'status,conclusion,headSha,jobs')
                    artifacts = gh_json('api',
                        f'repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100')['artifacts']
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    state.update(status='transient_poll_error', last_poll_utc=utc_now(), error=str(error))
                    json_write(status_path, state)
                    time.sleep(poll_seconds)
                    continue
                if run['headSha'] != commit:
                    raise ValueError('Remote workflow source commit does not match the admitted run.')
                state.update(status='polling', last_poll_utc=utc_now(), remote_status=run['status'],
                    remote_conclusion=run['conclusion'], jobs=[dict(name=j['name'], status=j['status'],
                        conclusion=j['conclusion']) for j in run['jobs']])
                state.pop('error', None)
                retry_download = False
                for artifact in artifacts:
                    match = re.fullmatch(r'rich6d-' + re.escape(stage) + r'-batch-(\d+)', artifact['name'])
                    if not match or not 0 <= int(match[1]) < len(allocation['batches']):
                        raise ValueError('Unexpected artifact in the allocated workflow.')
                    destination = download_root / artifact['name']
                    if not destination.exists():
                        if artifact['expired']:
                            raise ValueError('A required checkpoint artifact expired before retrieval.')
                        with tempfile.TemporaryDirectory(prefix='rich6d-artifact-',
                                                         dir=download_root.parent) as scratch:
                            try:
                                subprocess.run(['gh', 'run', 'download', str(run_id), '--repo', REPOSITORY,
                                    '--name', artifact['name'], '--dir', scratch], check=True, timeout=180)
                            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                                state.update(status='transient_download_error', error=str(error))
                                json_write(status_path, state)
                                retry_download = True
                                break
                            validate_transfer(Path(scratch), protocol, allocation, run_id, commit, seeds)
                            json_write(Path(scratch) / 'github_artifact_receipt.json', artifact)
                            os.replace(scratch, destination)
                        need_ingest = True
                    receipt = json.loads((destination / 'github_artifact_receipt.json').read_text())
                    if receipt['id'] != artifact['id']:
                        raise ValueError('Artifact identity changed; inspect a rerun before replacing saved data.')
                    if artifact['id'] not in [row['id'] for row in state['downloaded_artifacts']]:
                        state['downloaded_artifacts'].append(dict(id=artifact['id'], name=artifact['name'],
                            digest=artifact.get('digest'), bytes=artifact['size_in_bytes']))
                if need_ingest and list(download_root.glob('*/transfer_manifest.json')):
                    result = ingest(download_root, stage, run_id, commit)
                    state.update(verified_counts=result['counts'], last_import_utc=utc_now(),
                                 files_published_in_last_import=len(result['published']))
                    need_ingest = False
                    print('INDEPENDENTLY_VERIFIED_COUNTS=' + json.dumps(result['counts']), flush=True)
                if run['status'] == 'completed' and not retry_download:
                    terminal_observations += 1
                    # Allow artifact listing to settle after the last job's upload.
                    if terminal_observations >= 2:
                        counts = state['verified_counts'] or {}
                        names = ['baseline'] if stage == 'production_baseline' else ['N300', 'N1200', 'rho000', 'rho075']
                        complete = all(counts.get(name) == protocol['replications'] for name in names)
                        state.update(status='all_stage_checkpoints_collected' if complete else 'terminal_run_incomplete',
                                     finished_utc=utc_now())
                        json_write(status_path, state)
                        print(json.dumps(state), flush=True)
                        return state
                else:
                    terminal_observations = 0
                json_write(status_path, state)
                time.sleep(poll_seconds)
        except Exception:
            state.update(status='failed', error=traceback.format_exc(), updated_utc=utc_now())
            json_write(status_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--download-root', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.stage, args.run_id, args.commit, args.download_root)
    if result['status'] != 'all_stage_checkpoints_collected':
        raise SystemExit('Remote run ended with missing checkpoints; preserve completed imports and inspect failures.')


if __name__ == '__main__':
    main()
