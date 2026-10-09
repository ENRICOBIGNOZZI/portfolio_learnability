"""Benchmark two already-declared pilot paths, with automatic supervisor resume."""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from simulations.extended.design import OUTPUT,PILOT_PATHS
from simulations.provenance import json_write,utc_now


def command(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return result.stdout.strip()


def disk_reserve(minimum,timeout=60):
    """Allow exited-child memory/swap to settle without lowering the reserve.

    Three consecutive five-second readings must satisfy the existing threshold.
    Record every reading so this resource decision remains independently auditable.
    """
    started=time.monotonic()
    samples=[]
    consecutive=0
    while True:
        free=shutil.disk_usage(OUTPUT).free
        elapsed=time.monotonic()-started
        samples.append(dict(elapsed_seconds=elapsed,free_bytes=free))
        consecutive=consecutive+1 if free>=minimum else 0
        if consecutive>=3 or elapsed>=timeout:
            return free,consecutive>=3,samples
        time.sleep(min(5,timeout-elapsed))


def run(parent):
    if '-m simulations.extended.preflight' not in command(parent):
        raise ValueError('PID does not identify this study preflight supervisor.')
    children=[]
    os.kill(parent,signal.SIGSTOP)
    try:
        ids=subprocess.check_output(['pgrep','-P',str(parent)],text=True).split()
        if len(ids)!=1:
            raise ValueError('Expected one active sequential pilot child.')
        current=int(ids[0]);description=command(current)
        match=re.search(r'-m simulations.extended.compute --index (\d+)',description)
        if match is None:
            raise ValueError('The current child is not a declared rank pilot.')
        index=int(match.group(1))
        print('Waiting for active pilot',index,'PID',current,flush=True)
        while True:
            result=subprocess.run(['ps','-p',str(current),'-o','stat='],capture_output=True,text=True)
            state=result.stdout.strip()
            if not state or state.startswith('Z'):
                break
            time.sleep(5)
        if not (OUTPUT/'rank_pilot'/'P4096'/f'rep_{index:03d}.npz').exists():
            raise RuntimeError('Current pilot ended without its full-rank checkpoint.')
        selected=[index+1,index+2]
        minimum=int(1.5*1024**3)
        free,stable,samples=disk_reserve(minimum)
        report=dict(previous_pilot=index,selected_pilots=selected,
            disk_free_before_bytes=free,minimum_free_bytes=minimum,
            disk_reserve_stable=stable,disk_reserve_samples=samples,
            started_utc=utc_now())
        if not stable or selected[-1]>=PILOT_PATHS:
            report.update(status='not_launched',reason='Insufficient free disk reserve for simultaneous process memory pressure, or fewer than two remaining predeclared pilots.',recommended_workers=1)
            json_write(OUTPUT/'pilot'/'concurrency_probe.json',report)
            json_write(OUTPUT/'pilot'/f'concurrency_probe_after_pilot_{index:03d}.json',report)
            print(report,flush=True)
            return
        start=time.perf_counter()
        for i in selected:
            children.append(subprocess.Popen([sys.executable,'-m','simulations.extended.compute','--index',str(i)]))
        codes=[child.wait() for child in children]
        elapsed=time.perf_counter()-start
        serial=json.loads((OUTPUT/'rank_pilot'/f'resources_{index:03d}.json').read_text())['elapsed_seconds']
        report.update(status='complete' if all(code==0 for code in codes) else 'failed',
            exit_codes=codes,parallel_wall_seconds=elapsed,preceding_single_path_seconds=serial,
            throughput_ratio_vs_preceding_single=2*serial/elapsed,
            disk_free_after_bytes=shutil.disk_usage(OUTPUT).free,completed_utc=utc_now())
        report['recommended_workers']=2 if all(code==0 for code in codes) and elapsed<1.6*serial and report['disk_free_after_bytes']>=512*1024**2 else 1
        json_write(OUTPUT/'pilot'/'concurrency_probe.json',report)
        json_write(OUTPUT/'pilot'/f'concurrency_probe_after_pilot_{index:03d}.json',report)
        print(report,flush=True)
    finally:
        # The existing supervisor resumes and validates/skips completed indices.
        # On failure it will safely resume incomplete deterministic checkpoints.
        for child in children:
            if child.poll() is None:
                child.wait()
        os.kill(parent,signal.SIGCONT)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervisor-pid',type=int,required=True)
    args=parser.parse_args()
    run(args.supervisor_pid)
