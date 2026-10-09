"""Reproduce economic interpretation without retuning existing portfolios."""
import os
import subprocess
import sys


def main():
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    steps=[['empirical_final.economic_holdings','--phase','holdings'],
           ['empirical_final.economic_postprocess'],
           ['empirical_final.economic_holdings','--phase','summaries'],
           ['empirical_final.economic_financing'],
           ['empirical_final.economic_figures'],
           ['empirical_final.economic_publication'],
           ['empirical_final.economic_delivery']]
    for args in steps:
        subprocess.run([sys.executable,'-m',*args],env=env,check=True)
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_economic_holdings.py',
                    'tests/test_three_empirical_experiments.py'],env=env,check=True)


if __name__=='__main__':main()
