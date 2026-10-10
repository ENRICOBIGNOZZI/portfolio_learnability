"""Past-only group-boundary and Gaussian feature-draw sensitivity."""
import json
import time
import numpy as np
import pandas as pd
from data_pipeline import load_panels,digest
from kernels import FeatureBank,median_distance
from bandwidth_tuning import build_cache
from portfolio import fit_windows
from empirical_final.core import spectral_policy
from empirical_final.run import clean_manifest,load_managed,public,read_json,bind,INPUTS
from empirical_final.compact_common import (ROOT,OUT,PRIVATE,PROTOCOL,SCALE,setup,save,audit,
    source,account_month,summarize_accounts,generated_weight_cache)


def main():
    setup();started=time.monotonic()
    sensitivity=dict(feature_seeds=[0,1],rank_boundaries=PROTOCOL['spectral_boundaries'],
        rule='Initial-training grid per feature draw; annual past-only validation, no seed selection',
        reason='Two predeclared kernel draws limit computation on the shared 8 GB machine; neural sensitivity uses five seeds separately')
    p=OUT/'audit/robustness_protocol.json'
    if p.exists() and json.loads(p.read_text())!=sensitivity:raise ValueError('Sensitivity changed')
    p.write_text(json.dumps(sensitivity,indent=2))
    clean=clean_manifest();names=clean['characteristics']
    ell=read_json(ROOT/'results/final/public/gaussian/seed_0/source.json')['feature_bank']['ell']
    rf=pd.read_csv(source(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    from empirical_final.economic_holdings import FAMILIES
    from empirical_final.economic_financing import EXTRA,EXTRA_MEMBERS
    family_dict=dict(FAMILIES,**{EXTRA:EXTRA_MEMBERS})
    all_accounts=[];exposures=[];selection=[];metadata=[]
    for seed in sensitivity['feature_seeds']:
        bank=FeatureBank('gaussian',130,10000,ell,seed)
        if seed==0:g,dates=load_managed('gaussian',clean)
        else:
            panels,_=load_panels(ROOT/'data/clean')
            cache=PRIVATE/f'gaussian_seed_{seed}_managed'
            g=build_cache(panels,bank,[1.],cache,digest(ROOT/'data/clean/manifest.json'))[0]
            dates=panels.dates
        # Same primary grid for seed 0; initial-training complexity grid for seed 1.
        grid=np.asarray(public('gaussian')[1]['lambda_grid']) if seed==0 else None
        boundaries=sensitivity['rank_boundaries'] if seed==0 else [sensitivity['rank_boundaries'][0]]
        policies=[f'seed{seed}_boundary{b}_nested{j}' for b in range(len(boundaries)) for j in range(1,5)]
        states={(policy,sc[0]):pd.Series(dtype=float) for policy in policies for sc in PROTOCOL['cost_scenarios']}
        previous={policy:pd.Series(dtype=float) for policy in policies}
        accounts=[]
        for result in fit_windows(g,dates,grid):
            year=result['year'];chosen=result['choice'];penalty=result['penalties'][chosen]
            selection.append(dict(seed=seed,decision_year=year,selected_penalty=penalty,C=result['complexity'][chosen],
                candidate=chosen,validation_Q=result['validation_loss'][chosen],boundary=chosen in [0,len(result['penalties'])-1]))
            h=np.r_[result['train'],result['validation']];specs=[];coefficients=[]
            for b,fractions in enumerate(boundaries):
                spec=spectral_policy(np.asarray(g[h]),penalty,fractions)
                components=np.diff(np.column_stack([np.zeros(g.shape[1]),spec['beta']]),axis=1)
                coefficients.append(components);specs.append(spec)
                start=0
                for j,end in enumerate(spec['counts']):
                    metadata.append(dict(seed=seed,boundary=b,decision_year=year,group=j+1,
                        first_rank=start+1,last_rank=end,rank=spec['rank'],
                        second_moment_share=spec['eigenvalues'][start:end].sum()/spec['eigenvalues'].sum()))
                    start=end
            coefficients=np.column_stack(coefficients)
            data=pd.read_parquet(source(ROOT/f'data/clean/jkp_{year}.parquet'))
            year_weights=[]
            weight_path=PRIVATE/f'robustness_seed_{seed}_weights_{year}.parquet'
            cached=generated_weight_cache('robustness',weight_path)
            for date,f in data.groupby('eom',sort=True):
                retdate=date+pd.offsets.MonthEnd(1);r=pd.Series(f.r.to_numpy(),index=f.id)
                if cached is None:weights=SCALE*bank.scores(f[names].to_numpy(),coefficients)/len(f)
                else:
                    cw=cached[cached.formation_date.eq(date)].set_index('id')
                    if set(cw.index)!=set(f.id):raise ValueError('Cached sensitivity universe changed')
                    weights=cw.loc[f.id,[f'component_{j}' for j in range(coefficients.shape[1])]].to_numpy()
                    sample=np.arange(min(16,len(f)))
                    expected_w=SCALE*bank.scores(f[names].to_numpy()[sample],coefficients)/len(f)
                    np.testing.assert_allclose(weights[sample],expected_w,rtol=2e-6,atol=2e-9)
                np.testing.assert_allclose(weights.T@r.to_numpy(),SCALE*np.asarray(g[int(np.flatnonzero(dates==date)[0])])@coefficients,
                                           rtol=2e-6,atol=2e-8)
                month=int(np.flatnonzero(dates==date)[0])
                expected=float(np.asarray(g[month])@result['beta'])*SCALE
                scores=np.column_stack([sum(sign*f[name].to_numpy() for name,sign in members.items())/len(members)
                                       for members in family_dict.values()])
                for b in range(len(boundaries)):
                    w=weights[:,4*b:4*(b+1)]
                    np.testing.assert_allclose(w.sum(axis=1)@r.to_numpy(),expected,rtol=2e-6,atol=2e-8)
                    long=np.maximum(w,0);short=np.maximum(-w,0);l=long.sum(axis=0);s=short.sum(axis=0)
                    lm=long.T@scores/l[:,None];sm=short.T@scores/s[:,None];signed=w.T@scores
                    for j in range(4):
                        for k,family in enumerate(family_dict):exposures.append(dict(seed=seed,boundary=b,decision_year=year,
                            return_date=retdate,group=j+1,family=family,long_minus_short=lm[j,k]-sm[j,k],signed_notional=signed[j,k]))
                        policy=f'seed{seed}_boundary{b}_nested{j+1}'
                        target=pd.Series(w[:,:j+1].sum(axis=1),index=f.id)
                        for sc,tb,bb in PROTOCOL['cost_scenarios']:
                            step,states[policy,sc]=account_month(target,r,float(rf.loc[retdate]),states[policy,sc],previous[policy],tb,bb)
                            accounts.append(dict(policy=policy,seed=seed,boundary=b,group=j+1,scenario=sc,decision_year=year,
                                                 formation_date=date,return_date=retdate,**step))
                        previous[policy]=target
                year_weights.append(pd.DataFrame(weights,columns=[f'component_{j}' for j in range(weights.shape[1])]).assign(id=f.id.to_numpy(),formation_date=date))
            if cached is None:pd.concat(year_weights).to_parquet(weight_path,index=False)
            print(f'Grouping/feature sensitivity seed {seed} {year}, elapsed {(time.monotonic()-started)/60:.1f} min',flush=True)
        save(f'robustness_accounting_seed_{seed}',accounts);all_accounts.append(pd.DataFrame(accounts))
    save('robustness_performance',summarize_accounts(pd.concat(all_accounts),keys=('seed','boundary','group','scenario')))
    save('robustness_economic_profiles',pd.DataFrame(exposures).groupby(['seed','boundary','group','family'])[['long_minus_short','signed_notional']].mean().reset_index())
    save('robustness_selections',selection);save('robustness_spectral_metadata',metadata)
    audit('robustness',dict(passed=True,elapsed_seconds=time.monotonic()-started,inputs=dict(INPUTS),protocol=sensitivity))


if __name__=='__main__':main()
