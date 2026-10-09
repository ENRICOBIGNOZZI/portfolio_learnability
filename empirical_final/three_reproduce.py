"""Reproduce the three empirical experiments and their separate Section VI PDF.

The existing confidential stock/cache inputs remain local. The small frozen
French factor table is included among the public results, so reproduction does
not silently download a different factor vintage.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from data_pipeline import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/empirical_three_experiments_20261009'


def main():
    # Verify the previous run's inputs before any output can overwrite its audit.
    audit = json.loads((OUT/'audit/execution.json').read_text())
    for name, expected in audit['inputs'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    commands = [
        [sys.executable, '-m', 'empirical_final.three_experiments'],
        [sys.executable, '-m', 'empirical_final.three_figures'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/test_three_empirical_experiments.py', 'tests/test_local_learnability.py'],
        [sys.executable, '-m', 'empirical_final.three_publication'],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
