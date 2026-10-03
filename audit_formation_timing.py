"""Rebuild formation-only ranks and test whether the frozen snapshot supplies payoffs.

Unknown payoffs remain NaN. This audit never makes an unidentified return equal to zero.
Stock-level rebuilt panels are private; only counts and construction checks are exported.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import formation_mask,rank_months,choose_names,digest,write_json,META


def rebuild(raw, destination, output):
    started=time.perf_counter();raw=Path(raw);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((raw/'manifest.json').read_text())
    selection=json.loads(Path('characteristic_selection.json').read_text())
    names=selection['selected'];candidates=manifest['characteristics']
    files=[raw/x['name'] for x in manifest['files']]
    # Exact adjacent calendar-month match by JKP security identifier, no nearest-date filling.
    current=pd.concat([pd.read_parquet(p,columns=['id','eom','current_excess_return']) for p in files],ignore_index=True)
    current['eom']=pd.to_datetime(current.eom)
    assert not current.duplicated(['id','eom']).any()
    current=current.rename(columns={'eom':'return_date','current_excess_return':'next_current_excess'})
    coverage=pd.Series(0,index=candidates,dtype='int64');coverage_n=0
    counts=[];recovered=0;remaining=0;original_missing=0;rank_difference=0.;independence=0.;saved=[]
    for index,p in enumerate(files):
        frame=pd.read_parquet(p,columns=list(dict.fromkeys(META+candidates)))
        frame['eom']=pd.to_datetime(frame.eom);frame=frame.loc[formation_mask(frame)].copy()
        pre=frame.loc[frame.eom.le('1972-12-31')]
        coverage+=pre[candidates].replace([np.inf,-np.inf],np.nan).notna().sum();coverage_n+=len(pre)
        if frame.empty:continue
        formed=rank_months(frame,names).sort_values(['eom','id'])
        # A counterfactual future-availability vector cannot change the formation panel.
        perturbed=frame.copy();perturbed['ret_exc_lead1m']=np.nan
        reranked=rank_months(perturbed,names).sort_values(['eom','id'])
        assert formed[['id','eom']].equals(reranked[['id','eom']])
        error=float(np.max(np.abs(formed[names].to_numpy()-reranked[names].to_numpy())))
        independence=max(independence,error);assert error==0
        valid=np.isfinite(formed.ret_exc_lead1m)
        old=rank_months(frame.loc[np.isfinite(frame.ret_exc_lead1m)],names).sort_values(['eom','id'])
        known=formed.loc[valid]
        assert old[['id','eom']].reset_index(drop=True).equals(known[['id','eom']].reset_index(drop=True))
        if len(old):rank_difference=max(rank_difference,float(np.max(np.abs(old[names].to_numpy()-known[names].to_numpy()))))
        formed['return_date']=formed.eom+pd.offsets.MonthEnd(1)
        formed=formed.merge(current,on=['id','return_date'],how='left',validate='one_to_one')
        missing=~np.isfinite(formed.ret_exc_lead1m)
        recover=missing&np.isfinite(formed.next_current_excess)
        formed['r']=formed.ret_exc_lead1m.where(~recover,formed.next_current_excess)
        formed['return_known']=np.isfinite(formed.r)
        original_missing+=int(missing.sum());recovered+=int(recover.sum());remaining+=int((~formed.return_known).sum())
        for date,m in formed.groupby('eom'):
            counts.append(dict(formation_date=date,stocks_at_formation=len(m),
                missing_original=int((~np.isfinite(m.ret_exc_lead1m)).sum()),
                missing_unresolved=int((~m.return_known).sum())))
        path=destination/p.name
        formed[['id','eom','return_date','me','r','return_known',*names]].to_parquet(path,index=False)
        saved.append(dict(name=path.name,rows=len(formed),sha256=digest(path)))
        print('Formation-only rebuilt',p.name,flush=True)
    chosen,_=choose_names(coverage,coverage_n)
    assert chosen==names
    counts=pd.DataFrame(counts);counts.to_csv(output/'formation_timing_by_month.csv',index=False)
    audit=dict(source_manifest_sha256=digest(raw/'manifest.json'),formation_months=len(counts),
        formation_stock_months=int(counts.stocks_at_formation.sum()),original_missing_payoffs=original_missing,
        recovered_exact_next_calendar_current_excess=recovered,unresolved_payoffs=remaining,
        months_with_unresolved_payoffs=int((counts.missing_unresolved>0).sum()),
        max_characteristic_rank_change_vs_complete_payoff_sample=rank_difference,
        future_availability_perturbation_rank_error=independence,
        characteristic_freeze='1963-1972 formation metadata and characteristic coverage only',
        characteristic_coverage_observations=coverage_n,selected_names_exactly_match=True,
        dictionary_caveat='153 candidates and characteristic values come from retrospective current JKP snapshot; vintage not certified.',
        resolution=('Formation-only ranks and universe rebuilt. Payoff-complete portfolios cannot be computed from this snapshot without an additional unidentified-return assumption. Existing empirical figures are isolated complete-payoff-sample sensitivities, not pristine real-time evidence.' if remaining else 'All missing payoffs reconstructed from exact next-calendar source observations; managed cache must now be rebuilt.'),
        elapsed_seconds=time.perf_counter()-started)
    write_json(output/'formation_timing_audit.json',audit)
    write_json(destination/'manifest.json',dict(status='formation_only_partial_payoffs',audit=audit,files=saved))
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--output',default='outputs/tables');args=p.parse_args()
    rebuild(args.raw,args.destination,args.output)
