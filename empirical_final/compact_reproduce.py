"""Reproduce the entire compact empirical revision, or public figures/documents only."""
import argparse
import json
import os
import subprocess
import sys
from empirical_final.compact_common import ROOT,OUT,PROTOCOL,setup
from data_pipeline import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--public-only',action='store_true');a=p.parse_args()
    setup();env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    def run(module,*args):subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,env=env,check=True)
    if a.public_only:
        manifest=json.loads((OUT/'audit/run_manifest.json').read_text())
        for name,expected in manifest['artifact_sha256'].items():
            if name.startswith('tables/') and digest(OUT/name)!=expected:raise ValueError('Changed aggregate '+name)
    else:
        run('empirical_final.neural_portfolio','--pilot')
        run('empirical_final.compact_study','--phase','kernels')
        run('empirical_final.compact_study','--phase','costs')
        for seed in PROTOCOL['neural']['seeds']:run('empirical_final.neural_portfolio','--seed',str(seed))
        run('empirical_final.compact_study','--phase','bandwidth')
        run('empirical_final.compact_robustness')
        run('empirical_final.compact_study','--phase','inference')
        from empirical_final.compact_delivery import reconciliation,bandwidth_intervals
        reconciliation();bandwidth_intervals()
        run('pytest','-q','tests/test_compact_empirical.py','tests/test_pipeline.py','tests/test_bandwidth_tuning.py',
            'tests/test_final_empirical.py','tests/test_economic_holdings.py')
    run('empirical_final.compact_figures')
    run('empirical_final.compact_publication')
    from empirical_final.compact_delivery import source_map,report
    source_map();report()
    run('empirical_final.compact_verify')


if __name__=='__main__':main()
