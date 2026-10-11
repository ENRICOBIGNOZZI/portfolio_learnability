"""Reproduce the complete package in fresh directories without overwriting it.

python3 -m empirical_final.final_theory_reproduce \
  --output outputs/final_theory_regularization_reproduction \
  --private results/final_theory_regularization_reproduction
"""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits

from empirical_final.final_theory_regularization import replicate,estimate,costs
from empirical_final.final_theory_analysis import summarize,rolling,figures
from empirical_final import final_theory_dense as dense
from empirical_final.final_theory_report import deliver


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();private=args.private.resolve()
    if out.exists() or private.exists():raise FileExistsError('Both destinations must be fresh')
    with threadpool_limits(limits=1):
        replicate(out);estimate(out,private);costs(out,private)
        summarize(out);rolling(out);figures(out)
        # The exploratory module reads its primary baseline via this parameter.
        dense.OUT=out
        extra_out=out/'dense_lengthscale';extra_private=private/'dense_lengthscale'
        dense.cache(extra_out,extra_private);dense.estimate(extra_out,extra_private)
        dense.costs(extra_out,extra_private);dense.summary(extra_out)
        deliver(out)


if __name__=='__main__':main()
