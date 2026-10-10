"""Run final analysis after a pinned remote collector has audited the whole study."""
import argparse
from contextlib import ExitStack
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

from simulations.extended.design import OUTPUT, ROOT, ENVIRONMENTS
from simulations.extended.remote_compatibility import protocol_checked
from simulations.provenance import file_hash, json_write, output_lock, utc_now


COMMANDS = (
    ('simulations.extended.summarize',),
    ('simulations.extended.verify', '--data-only'),
    ('simulations.extended.report',),
    ('simulations.extended.figures',),
    ('simulations.extended.verify',),
)


def ready(output, protocol, run_id, commit):
    """Require terminal collection and its exact independently audited snapshot."""
    stage = 'production_robustness'
    state = json.loads((output / f'remote_collection_{stage}_{run_id}.json').read_text())
    if (str(state['workflow_run_id']) != str(run_id) or state['source_commit'] != commit
            or state['run_hash'] != protocol['run_hash'] or state['stage'] != stage
            or state['allocation_sha256'] != file_hash(output / f'remote_allocation_{stage}.json')):
        raise ValueError('Collector provenance does not match the admitted remote run.')
    if state['status'] in ('failed', 'terminal_run_incomplete'):
        raise ValueError('Remote collection failed or ended with missing checkpoints.')
    if state['status'] != 'all_stage_checkpoints_collected':
        # A heartbeat file alone is insufficient evidence that work is alive.
        process = subprocess.run(['ps', '-p', str(state['collector_pid']), '-o', 'command='],
                                 capture_output=True, text=True, check=False)
        words = shlex.split(process.stdout)
        run_argument = words.index('--run-id') + 1 if '--run-id' in words else len(words)
        if (process.returncode or 'simulations.extended.remote_collect' not in words
                or run_argument >= len(words) or words[run_argument] != str(run_id)):
            raise RuntimeError('Collector process is missing; inspect before resuming analysis.')
        return False
    expected = {name: protocol['replications'] for name in ENVIRONMENTS}
    audit_path = output / 'checkpoint_integrity.json'
    audit = json.loads(audit_path.read_text())
    ingestion = json.loads((output / f'remote_ingestion_{stage}_{run_id}.json').read_text())
    if (state['verified_counts'] != expected or audit['counts'] != expected
            or ingestion['counts'] != expected or not audit['completed_checkpoint_checks_passed']
            or not ingestion['independent_audit_passed']
            or ingestion['run_hash'] != protocol['run_hash']
            or ingestion['commit'] != commit or str(ingestion['workflow_run_id']) != str(run_id)
            or ingestion['checkpoint_integrity_sha256'] != file_hash(audit_path)
            or audit['run_hash'] != protocol['run_hash']
            or audit['paired_common_history_indices'] != list(range(protocol['replications']))
            or audit['pending_common_history_indices']):
        raise ValueError('Full independent checkpoint audit is missing or inconsistent.')
    return True


def check_audited_bytes(output, protocol):
    """Keep the collector's audit immutable while checking its actual input bytes."""
    audit = json.loads((output / 'checkpoint_integrity.json').read_text())
    expected = {f'{stage}/P{protocol["rank"]}/rep_{index:03d}.npz'
                for stage in ('production_baseline', 'production_robustness')
                for index in range(protocol['replications'])}
    records = audit['checkpoints']
    if len(records) != len(expected) or {row['path'] for row in records} != expected:
        raise ValueError('Audited checkpoint list is incomplete or duplicated.')
    for row in records:
        if file_hash(output / row['path']) != row['sha256']:
            raise ValueError('Checkpoint bytes changed after independent collection audit.')


def finish(run_id, commit, poll_seconds=55):
    protocol = protocol_checked()
    path = OUTPUT / 'remote_final_analysis.json'
    state = dict(run_hash=protocol['run_hash'], workflow_run_id=str(run_id),
                 source_commit=commit, pid=os.getpid(), started_utc=utc_now(),
                 status='waiting_for_complete_collection', completed_commands=[],
                 remaining='Review all ten actual figures, audit the full objective, refresh the final manifest, commit/push deliverables and display the ten figures in chat.')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPYCACHEPREFIX='/tmp/rich6d_no_bytecode_cache',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with output_lock(OUTPUT / 'remote_final_analysis_supervisor'):
        try:
            json_write(path, state)
            while not ready(OUTPUT, protocol, run_id, commit):
                time.sleep(poll_seconds)
            # Wait for the collector to release its lock after writing its terminal record.
            while True:
                locks = ExitStack()
                try:
                    locks.enter_context(output_lock(OUTPUT / 'remote_collector_production_robustness'))
                except RuntimeError:
                    locks.close()
                    time.sleep(poll_seconds)
                    continue
                break
            with locks:
                for name in ('advance_supervisor', 'production_supervisor', 'remote_ingestion'):
                    locks.enter_context(output_lock(OUTPUT / name))
                if not ready(OUTPUT, protocol_checked(), run_id, commit):
                    raise ValueError('Collection state changed before final analysis.')
                check_audited_bytes(OUTPUT, protocol)
                sources = {str(p.relative_to(ROOT)): file_hash(p)
                           for p in (ROOT / 'simulations' / 'extended').glob('*.py')}
                state['analysis_source_hashes'] = sources
                for command in COMMANDS:
                    state.update(status='running', current_command=list(command), updated_utc=utc_now())
                    json_write(path, state)
                    print('RUNNING ' + ' '.join(command), flush=True)
                    subprocess.run([sys.executable, '-m', *command], cwd=ROOT, env=env, check=True)
                    state['completed_commands'].append(list(command))
                if any(file_hash(ROOT / name) != checksum for name, checksum in sources.items()):
                    raise ValueError('Analysis sources changed during postprocessing.')
                state.update(status='ready_for_visual_review', finished_utc=utc_now())
                state.pop('current_command', None)
                json_write(path, state)
        except Exception:
            state.update(status='failed', error=traceback.format_exc(), updated_utc=utc_now())
            json_write(path, state)
            raise
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    finish(args.run_id, args.commit)
