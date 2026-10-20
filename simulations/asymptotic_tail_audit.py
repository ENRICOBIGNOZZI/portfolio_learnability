"""Check representation error at the longer reference endpoint, separately.

The rank-selection pilot ends at 12800; this diagnostic extends two of its
paired, independent streams to 25600. It never selects ranks from rate fits
and never changes the frozen confirmation estimator or its samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import asymptotic as engine
from .asymptotic_dense import sha,write_json


def audit(out,workers=2,trials=2):
    out=Path(out)
    config=json.loads((out/'config.json').read_text())
    signature=engine.scientific_hash(config)
    cache=out/'cache'
    assert (cache/'source_hash.txt').read_text()==signature
    selection=json.loads((out/'rank_selection.json').read_text())
    N,rho=config['fixed_N'],config['fixed_rho']
    times=[max(config['T_train_reference'])]
    ranks=sorted({selection['selected_rank'],selection['reference_rank']})
    constant=json.loads((out/'frozen_constants.json').read_text())[str(N)]
    settings=[(N,rho,'nystrom',rank) for rank in ranks]
    design=dict(N=N,rho=rho,T_train=times,trials=trials,ranks=ranks,
        seed=config['fidelity_seed'],source_hash=signature,driver_sha256=sha(Path(__file__)),
        reason='The primary numerical pilot stopped before the extended confirmation endpoint.',
        decision='Diagnostic only; frozen confirmation rank and all rate-fit windows remain unchanged.')
    marker=out/'tail_audit_design.json'
    if marker.exists():
        assert json.loads(marker.read_text())==design
    else:
        write_json(marker,design)
    jobs=[(config,N,rho,j,config['fidelity_seed'],times,kind,rank,constant,str(cache),'tail_fidelity')
          for N,rho,kind,rank in settings for j in range(trials)]
    engine.execute(jobs,engine.training_job,workers,'Extended-horizon numerical audit')
    for _,_,kind,rank in settings:
        assert (cache/f'evaluation_{kind}{rank}_N{N}.npz').exists()
    frame=engine.collect(config,out,'tail_fidelity',settings,{(N,rho):times},trials,config['fidelity_seed'])
    records=[]
    for mult in config['lambda_multipliers']:
        means=frame[frame.lambda_multiplier==mult].groupby('rank')[['complexity','SR_hat']].mean()
        candidate=means.loc[selection['selected_rank']]
        reference=means.loc[selection['reference_rank']]
        c_error=float(abs(candidate.complexity/reference.complexity-1))
        sr_error=float(abs(candidate.SR_hat-reference.SR_hat))
        records.append(dict(lambda_multiplier=mult,T_train=times[0],
            selected_rank=selection['selected_rank'],reference_rank=selection['reference_rank'],
            selected_C=float(candidate.complexity),reference_C=float(reference.complexity),
            relative_C_difference=c_error,absolute_SR_difference=sr_error,
            meets_original_value_tolerances=bool(c_error<=config['rank_C_tolerance'] and sr_error<=config['rank_SR_tolerance'])))
    result=dict(design=design,checks=records,all_finite=bool(np.isfinite(frame.select_dtypes('number')).all().all()),
                note='Two paired pilot trials diagnose numerical representation error; they are excluded from all 100-trial confirmation means, SD, percentiles and rate fits.')
    write_json(out/'tail_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',nargs='?',type=Path,default=engine.ROOT/'outputs/asymptotic_dense_v3')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--trials',type=int,default=2)
    args=parser.parse_args();audit(args.output,args.workers,args.trials)
