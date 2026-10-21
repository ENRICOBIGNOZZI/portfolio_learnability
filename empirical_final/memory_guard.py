"""Run one command with single-threaded BLAS and sampled process-tree limits."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import psutil


def run(command, destination, limit_mib=600., minimum_available_mib=256., interval=1.):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Avoid opening old bytecode evicted by iCloud in this Desktop checkout.
    env.setdefault('PYTHONPYCACHEPREFIX', str(Path(tempfile.gettempdir())/'paper_codice_pycache'))
    for name in ['VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS']:
        env[name] = '1'
    started = time.monotonic()
    process = subprocess.Popen(command, env=env)
    root = psutil.Process(process.pid)
    peak, low_count, reason, samples = 0., 0, None, 0
    with (destination/'memory_samples.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['elapsed_seconds', 'tree_rss_mib', 'system_available_mib', 'swap_used_mib'])
        while process.poll() is None:
            try:
                children = root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            rss = 0
            for item in [root, *children]:
                try:
                    rss += item.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            rss /= 2**20
            available = psutil.virtual_memory().available/2**20
            swap = psutil.swap_memory().used/2**20
            peak = max(peak, rss)
            samples += 1
            writer.writerow([round(time.monotonic()-started, 2), round(rss, 2), round(available, 2), round(swap, 2)])
            stream.flush()
            low_count = low_count+1 if available < minimum_available_mib else 0
            if rss > limit_mib or low_count >= 5:
                reason = 'process_tree_rss_limit' if rss > limit_mib else 'system_available_memory_floor'
                for item in reversed(children):
                    try:
                        item.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(interval)
    code = process.wait()
    result = dict(command=command, exit_code=code, stopped_reason=reason,
                  sampled_tree_peak_mib=peak, limit_mib=limit_mib,
                  minimum_system_available_mib=minimum_available_mib,
                  interval_seconds=interval, samples=samples,
                  elapsed_seconds=time.monotonic()-started,
                  measurement='Sampled RSS, not an OS high-water mark; includes descendants. Other applications are not terminated.')
    (destination/'memory_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return 75 if reason else code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit-mib', type=float, default=600.)
    parser.add_argument('--minimum-available-mib', type=float, default=256.)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command after -- is required')
    sys.exit(run(command, args.out, args.limit_mib, args.minimum_available_mib))


if __name__ == '__main__':
    main()
