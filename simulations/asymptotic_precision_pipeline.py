"""Resume the frozen precision study, waiting for disk margin when necessary.

Only orchestration changes: every training, evaluation and fit still runs in
the original asymptotic_precision module with the original configuration.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import asymptotic as engine
from . import asymptotic_precision as precision
from .asymptotic_dense import write_json

GIB=1024**3


def disk_failure(text):
    return ('Insufficient free disk for bounded-memory managed-feature scratch' in text
            or '[Errno 28] No space left on device' in text)


def disk_requirement(config,rank,cache):
    files=[cache/f'precision_nystrom{rank}_trial{j:03d}.npz' for j in range(config['trials'])]
    # The unchanged scientific function reuses an existing atomic policy file
    # before its scratch-space check. A fully cached rank needs only exports.
    if all(path.exists() for path in files):
        return 64*1024**2
    return max(config['T_train'])*rank*8+2*GIB


def wait_for_disk(config,rank,cache,state,save,poll_seconds=30):
    stable=0
    while stable<2:
        required=disk_requirement(config,rank,cache)+GIB//4
        free=shutil.disk_usage(cache).free
        state.update(phase='checking_disk_margin',rank=rank,free_disk_bytes=free,required_disk_bytes=required)
        save()
        stable=stable+1 if free>=required else 0
        if stable<2:
            time.sleep(poll_seconds)


def complete_rank(config,out,rank,signature):
    path=out/f'precision_nystrom{rank}_completed.json'
    if not path.exists():
        return False
    record=json.loads(path.read_text())
    assert record['source_hash']==signature and record['replications']==config['trials']
    assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
               for name,sha in record['files_sha256'].items())
    return True


def run(output,config_path=None,wait_main_output=None,compute_only=False,artifact_marker=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    config_path=Path(config_path) if config_path else out/'config.json'
    config=json.loads(config_path.read_text());signature=precision.source_hash(config)
    marker=cache/'precision_source_hash.txt'
    if marker.exists() and marker.read_text()!=signature:
        raise RuntimeError('Frozen scientific code or design changed; use a new output directory')
    with (cache/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=out/'execution.json'
        if old.exists():
            history=cache/'execution_history';history.mkdir(exist_ok=True)
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            (history/f'{stamp}.json').write_bytes(old.read_bytes())
        state=dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_hash=signature,ranks=config['precision_ranks'],workers=1,numerical_thread_cap=1,
            completed_ranks=[],attempts=[],orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scientific_code_or_parameters_changed=False)
        def save():write_json(out/'execution.json',state)
        save()
        env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        awake=None
        if sys.platform=='darwin':
            awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
        try:
            with (out/'run.log').open('ab',buffering=0) as log:
                for rank in config['precision_ranks']:
                    if rank==max(config['precision_ranks']) and wait_main_output and not complete_rank(config,out,rank,signature):
                        main=Path(wait_main_output);state.update(phase='waiting_for_main_confirmation',rank=rank);save()
                        while not (main/'completed.json').exists():time.sleep(30)
                        main_config=json.loads((main/'config.json').read_text())
                        main_done=json.loads((main/'completed.json').read_text())
                        assert main_done['source_hash']==engine.scientific_hash(main_config)
                        assert main_done['completed_replications']==len(main_config['cells'])*main_config['trials']
                    while True:
                        if not complete_rank(config,out,rank,signature):
                            wait_for_disk(config,rank,cache,state,save)
                        state.update(phase='computing_or_reusing_rank',rank=rank);save()
                        print(f'Running frozen precision rank {rank}; complete policies are reused',flush=True)
                        offset=log.tell()
                        result=subprocess.run(['nice','-n','5',sys.executable,'-u','-m','simulations.asymptotic_precision',
                            '--config',str(config_path.resolve()),'--output',str(out.resolve()),'--rank',str(rank),'--workers','1'],
                            cwd=engine.ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                        attempt=dict(rank=rank,returncode=result.returncode)
                        state['attempts'].append(attempt);save()
                        if result.returncode==0:
                            assert complete_rank(config,out,rank,signature)
                            state['completed_ranks'].append(rank);save();break
                        with (out/'run.log').open('rb') as reader:
                            reader.seek(offset);tail=reader.read().decode('utf-8',errors='replace')
                        if not disk_failure(tail):
                            raise RuntimeError(f'Precision rank {rank} failed; inspect run.log')
                        attempt['reason']='insufficient_disk_margin'
                        state['phase']='waiting_for_disk_recovery';save()
                        print('Disk margin interrupted the rank; retained policies will be reused after recovery',flush=True)
                        time.sleep(30)
                state.update(status='computed',phase='scientific_data_complete');save()
                if not compute_only:
                    if artifact_marker:
                        kind='edit' if (out/'precision_plots.pdf').exists() else 'create'
                        subprocess.run(['node',str(artifact_marker),'--operation-kind',kind,
                            '--expected-output-count','1','--output-format','pdf'],check=True)
                    for module in ['simulations.precision_report','simulations.precision_validate']:
                        state.update(phase=module);save()
                        subprocess.run([sys.executable,'-m',module,str(out.resolve())],cwd=engine.ROOT.parent,
                            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    state.update(report_status='generated_and_numerically_validated',
                        remaining='Final visual review, scientific interpretation, commit and push');save()
        except BaseException:
            state['status']='failed';save();raise
        finally:
            if awake is not None:awake.terminate()
    print('Precision pipeline finished; review and publication remain',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--wait-main-output',type=Path)
    p.add_argument('--compute-only',action='store_true');p.add_argument('--artifact-marker',type=Path)
    a=p.parse_args();run(a.output,a.config,a.wait_main_output,a.compute_only,a.artifact_marker)
