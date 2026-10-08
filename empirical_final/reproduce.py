"""Recompute E1--E3 sequentially into a fresh destination using authorized inputs."""
import os
# Set before importing numpy or any repository numerical module.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import shutil
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    from empirical_final import run
    original=run.OUT
    out=args.out.resolve()
    if 'simulations' in out.parts:raise ValueError('Simulation paths are outside scope')
    out.mkdir(parents=True,exist_ok=False)
    for part in ('audit','tables','notes','figures'):(out/part).mkdir()
    shutil.copyfile(original/'audit/protocol_freeze.json',out/'audit/protocol_freeze.json')
    run.OUT=out
    for phase in ('accounting','spectral','missing'):
        run.INPUTS.clear()
        getattr(run,phase)()
    from empirical_final.render import render
    render(out)
    print('Numerical phases and figures complete. Build the manuscript separately; see README.')

if __name__=='__main__':main()
