"""User-requested denser chronological lengthscale search, separate from 2x2.

Four extra shared-draw lengths are declared before reading their OOS results.
The initial 200-point calibration and subsequent polynomial rule are unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bandwidth_tuning import MULTIPLIERS,managed_bandwidths
from complexity_schedule import nested_grid
from data_pipeline import digest
from kernels import FeatureBank,cosine
from portfolio import annual_splits,dual_path,sharpe
from empirical_final.core import performance
from empirical_final.final_theory_regularization import ROOT,OUT,PRIVATE,slope,penalty,frozen,load_inputs,SEED,REPLICATES
from empirical_final.lengthscale_spectral import account
from empirical_final.theory_guided_lambda import save,write_json
from empirical_final.three_experiments import block_counts,boot_metrics

EXTRA=(.75,1.25,1.5,3.)
DENSE=tuple(sorted((*MULTIPLIERS,*EXTRA)))


def scores(x,bank,beta,bands):
    result=np.zeros((len(x),beta.shape[1]))
    for start in range(0,len(x),64):
        projection=x[start:start+64]@bank.frequencies.T
        for band in sorted(set(bands)):
            columns=np.flatnonzero(np.asarray(bands)==band)
            phi=np.sqrt(2/10000)*cosine(projection/DENSE[band]+bank.phases)
            result[start:start+64,columns]=phi@beta[:,columns]
    return result/len(x)


def cache(out,private):
    started=time.monotonic();base=frozen(OUT);clean,_,dates,_=load_inputs(OUT)
    out.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    for part in ('audit','tables'):(out/part).mkdir(exist_ok=True)
    if (out/'audit/cache.json').exists():raise FileExistsError('Dense cache already frozen')
    write_json(out/'audit/protocol.json',dict(version='user-requested-dense-lengthscale-v1',
        justification='User requested more hyperparameter optimization during the controlled experiment',
        multipliers=list(DENSE),extra_multipliers=list(EXTRA),kernel='matern32',seed=0,P=10000,
        initial_lambda_grid=200,annual_lambda_grid=0,
        selection='Minimum response-one squared validation loss; expanding inner training; trailing 60 months; same exact tie convention',
        constants='Independent initial 120 training / 60 validation calibrations, T=120 conversion, then frozen',
        admissibility='Same b>1 exclusions and baseline-control refit fallback as primary experiment',
        status='Follow-up specified after inspecting coarse-grid gross results; exploratory, conditional inference; not confirmatory replication',
        inference='Paired circular blocks 6/12/24 months, 5000 draws; net comparisons against original D and joint CV',
        source_sha256=digest(Path(__file__)),data_manifest_sha256=digest(ROOT/'data/clean/manifest.json')))
    bank=FeatureBank('matern32',130,10000,base['feature_bank']['ell'],0)
    assert bank.metadata()==base['feature_bank']
    path=private/'extra_managed.npy';g=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(4,744,10000))
    lookup={d:i for i,d in enumerate(dates)};inputs={}
    for item in clean['files']:
        data=pd.read_parquet(ROOT/'data/clean'/item['name'])
        assert digest(ROOT/'data/clean'/item['name'])==item['sha256'];inputs[item['name']]=item['sha256']
        for date,group in data.groupby('eom',sort=True):
            group=group.sort_values('id');panel=dict(x=group[clean['characteristics']].to_numpy(float),r=group.r.to_numpy(float))
            assert group.return_date.eq(date+pd.offsets.MonthEnd(1)).all()
            g[:,lookup[date]]=managed_bandwidths(panel,bank,EXTRA)
        g.flush();print('Dense shared-draw cache',item['name'],flush=True)
    assert np.isfinite(g).all()
    write_json(out/'audit/cache.json',dict(passed=True,sha256=digest(path),shape=list(g.shape),
        extra_multipliers=list(EXTRA),feature_bank=bank.metadata(),data_files=inputs,
        elapsed_seconds=time.monotonic()-started))


def estimate(out,private):
    started=time.monotonic();base=frozen(OUT);_,original,dates,_=load_inputs(OUT)
    info=json.loads((out/'audit/cache.json').read_text());path=private/'extra_managed.npy'
    assert digest(path)==info['sha256']
    extra=np.load(path,mmap_mode='r');gs=[original[list(MULTIPLIERS).index(m)] if m in MULTIPLIERS else extra[EXTRA.index(m)] for m in DENSE]
    grams=[g@g.T for g in gs];constants=[];grid_rows=[]
    for j,(g,gram) in enumerate(zip(gs,grams)):
        grid=nested_grid(g[:120]);alpha,mu,_=dual_path(gram[:120,:120],grid);fit=slope(mu)
        losses=np.mean((1-gram[120:180,:120]@alpha)**2,axis=0);choice=int(np.argmin(losses))
        c0=float(grid[choice]*120**(fit['b']/(fit['b']+1)))
        constants.append(dict(candidate=j,multiplier=DENSE[j],**fit,C0=c0,initial_lambda=float(grid[choice]),
            initial_choice=choice,validation_loss=float(losses[choice]),training_months=120,validation_months=60))
        grid_rows.extend(dict(candidate=j,multiplier=DENSE[j],grid_index=i,penalty=float(p),validation_loss=float(losses[i]),selected=i==choice) for i,p in enumerate(grid))
    save(out/'tables/frozen_constants.csv',constants);save(out/'tables/initial_candidates.csv',grid_rows)
    rf=pd.read_csv(ROOT/'data/risk_free.csv',parse_dates=['return_date']).set_index('return_date').rf
    monthly=[];annual=[];candidates=[];hashes={};base_index=DENSE.index(1.)
    order=sorted(range(len(DENSE)),key=lambda j:(abs(np.log2(DENSE[j])),DENSE[j]))
    for year,tr,va,te in annual_splits(dates):
        h=np.r_[tr,va];T=len(h);losses={'C_dense':[],'D_dense':[]};refits=[];fits=[]
        for j,(gram,c) in enumerate(zip(grams,constants)):
            inner=gram[np.ix_(tr,tr)];v,u=np.linalg.eigh((inner+inner.T)/2);v=np.maximum(v,0);fit=slope(v[::-1]/len(tr))
            full=gram[np.ix_(h,h)];vf,uf=np.linalg.eigh((full+full.T)/2)
            if vf.min() < -max(vf.max(),1e-30)*1e-10:raise ValueError('Non-PSD dense full operator')
            vf=np.maximum(vf,0);full_fit=slope(vf[::-1]/T);refits.append((vf,uf));fits.append(full_fit)
            for method,b in (('C_dense',c['b']),('D_dense',fit['b'])):
                eligible=c['admissible'] and (method=='C_dense' or fit['admissible'])
                lam=c['C0']*len(tr)**(-b/(b+1));a=u@((u.T@np.ones(len(tr)))/(v+len(tr)*lam))
                loss=float(np.mean((1-gram[np.ix_(va,tr)]@a)**2))
                losses[method].append(loss if eligible else np.inf)
                candidates.append(dict(year=year,method=method,candidate=j,multiplier=DENSE[j],inner_T=len(tr),
                    full_T=T,inner_b=fit['b'],full_b=full_fit['b'],inner_lambda=lam,C0=c['C0'],eligible=eligible,
                    validation_loss=loss,inner_last_formation=dates[tr[-1]],validation_first_formation=dates[va[0]],decision_date=dates[te[0]]))
        betas=[];bands=[]
        for method in ('C_dense','D_dense'):
            selected=min([j for j in order if np.isfinite(losses[method][j])],key=lambda j:losses[method][j]);j=selected
            if method=='D_dense' and not fits[j]['admissible']:j=base_index
            c=constants[j];b=c['b'] if method=='C_dense' else fits[j]['b'];lam=penalty(c['C0'],T,b)
            v,u=refits[j];a=u@((u.T@np.ones(T))/(v+T*lam));beta=gs[j][h].T@a;pay=grams[j][np.ix_(te,h)]@a
            np.testing.assert_allclose(gs[j][te]@beta,pay,rtol=1e-7,atol=1e-9)
            complexity=float(np.sum((v/T)/(v/T+lam)))
            annual.append(dict(year=year,method=method,candidate=j,multiplier=DENSE[j],T=T,b=b,penalty=lam,
                complexity=complexity,C0=c['C0'],validation_loss=losses[method][selected],fallback=j!=selected))
            for i,r in zip(te,pay):
                rd=dates[i]+pd.offsets.MonthEnd(1)
                monthly.append(dict(method=method,year=year,formation_date=dates[i],return_date=rd,raw_return=float(r),
                    gross_excess_return=float(base['kappa']*r),rf=float(rf.loc[rd]),kappa=base['kappa'],multiplier=DENSE[j]))
            betas.append(beta);bands.append(j)
        path=private/f'dense_coefficients_{year}.npz';np.savez(path,beta=np.column_stack(betas),bands=np.array(bands));hashes[path.name]=digest(path)
        print('Dense chronological selection',year,[DENSE[j] for j in bands],flush=True)
    save(out/'tables/annual_policies.csv',annual);save(out/'tables/monthly_payoffs.csv',monthly);save(out/'tables/candidate_losses.csv',candidates)
    write_json(out/'audit/estimation.json',dict(passed=True,coefficient_hashes=hashes,kappa=base['kappa'],
        feature_bank=base['feature_bank'],source_sha256=digest(Path(__file__)),elapsed_seconds=time.monotonic()-started))


def costs(out,private):
    started=time.monotonic();audit=json.loads((out/'audit/estimation.json').read_text());assert digest(Path(__file__))==audit['source_sha256']
    clean=json.loads((ROOT/'data/clean/manifest.json').read_text());cache=json.loads((out/'audit/cache.json').read_text())
    bank=FeatureBank('matern32',130,10000,audit['feature_bank']['ell'],0);assert bank.metadata()==audit['feature_bank']
    expected=pd.read_csv(out/'tables/monthly_payoffs.csv',parse_dates=['formation_date']).set_index(['method','formation_date'])
    states={};previous={};rows=[];checks=[];holdings_hashes={}
    for year in range(1978,2025):
        name=f'dense_coefficients_{year}.npz';assert digest(private/name)==audit['coefficient_hashes'][name]
        coeff=np.load(private/name);data_path=ROOT/f'data/clean/jkp_{year}.parquet';assert digest(data_path)==cache['data_files'][data_path.name]
        data=pd.read_parquet(data_path);holding_frames=[];max_error=0.
        for date,group in data.groupby('eom',sort=True):
            group=group.sort_values('id').set_index('id');raw=scores(group[clean['characteristics']].to_numpy(float),bank,coeff['beta'],coeff['bands'])
            holding=pd.DataFrame(audit['kappa']*raw,index=group.index,columns=['C_dense','D_dense']).reset_index();holding.insert(0,'formation_date',date);holding_frames.append(holding)
            for col,method in enumerate(('C_dense','D_dense')):
                target=pd.Series(audit['kappa']*raw[:,col],index=group.index);ref=expected.loc[(method,date)]
                err=abs(float(target@group.r)-ref.gross_excess_return);max_error=max(max_error,err)
                np.testing.assert_allclose(target@group.r,ref.gross_excess_return,rtol=2e-6,atol=2e-8)
                values,state=account(target,group.r,ref.rf,states.get(method,pd.Series(dtype=float)),previous.get(method,pd.Series(dtype=float)))
                states[method]=state;previous[method]=target
                for value in values:rows.append(dict(method=method,year=year,formation_date=date,return_date=date+pd.offsets.MonthEnd(1),rf=ref.rf,kappa=audit['kappa'],**value))
        path=private/f'dense_holdings_{year}.parquet';pd.concat(holding_frames,ignore_index=True).to_parquet(path,index=False);holdings_hashes[path.name]=digest(path)
        checks.append(dict(year=year,maximum_payoff_error=max_error));print('Dense actual holdings',year,flush=True)
    save(out/'tables/monthly_accounts.csv',rows);save(out/'tables/holdings_checks.csv',checks)
    write_json(out/'audit/costs.json',dict(passed=True,holdings_hashes=holdings_hashes,maximum_payoff_error=max(c['maximum_payoff_error'] for c in checks),
        actual_weights=True,elapsed_seconds=time.monotonic()-started))


def summary(out):
    assert json.loads((out/'audit/costs.json').read_text())['passed']
    base=pd.read_csv(OUT/'tables/monthly_accounts.csv',parse_dates=['return_date']);dense=pd.read_csv(out/'tables/monthly_accounts.csv',parse_dates=['return_date'])
    combined=pd.concat([base,dense],ignore_index=True);annual=pd.read_csv(out/'tables/annual_policies.csv');rows=[];paired=[]
    methods=['C_dense','D_dense','C','D','B','joint_cv']
    for lo in (1978,1979,1984,2020):
        for scenario in ('gross','trade25_borrow30'):
            q=combined[combined.year.ge(lo)&combined.scenario.eq(scenario)&combined.method.isin(methods)]
            for method,d in q.groupby('method'):
                rows.append(dict(period=f'{lo}-2024',scenario=scenario,method=method,months=len(d),
                    **performance(d.excess_return,d.total_return),mean_leverage=float(d.gross_exposure.mean()),
                    mean_monthly_turnover=float(d.turnover.mean()),raw_unit_quadratic_loss=float(np.mean((1-d.excess_return/d.kappa)**2))))
            wide=q.pivot(index='return_date',columns='method',values='excess_return')[methods];assert not wide.isna().any().any()
            raw=wide.to_numpy()/q.kappa.iloc[0];sr=np.asarray(sharpe(raw));loss=np.mean((1-raw)**2,axis=0)
            for length in (6,12,24):
                draws=boot_metrics(raw,block_counts(len(raw),length,SEED+lo,REPLICATES))
                for right,left in (('D_dense','D'),('D_dense','B'),('D_dense','C_dense'),('D_dense','joint_cv'),('C_dense','C')):
                    i=methods.index(right);j=methods.index(left)
                    for metric,point in (('sharpe',sr),('loss',loss)):
                        low,high=np.quantile(draws[metric][:,i]-draws[metric][:,j],[.025,.975])
                        paired.append(dict(period=f'{lo}-2024',scenario=scenario,comparison=f'{right}-{left}',metric=metric,
                            block_months=length,replicates=REPLICATES,estimate=float(point[i]-point[j]),ci_lower=float(low),ci_upper=float(high),
                            status='Exploratory follow-up; conditional on fitted paths, no correction for sequential exploration'))
    save(out/'tables/performance_summary.csv',rows);save(out/'tables/paired_bootstrap.csv',paired)
    save(out/'tables/selection_frequencies.csv',annual.groupby(['method','multiplier']).size().rename('years').reset_index())
    write_json(out/'audit/summary.json',dict(passed=True,common_dates=True,no_annual_lambda_search=True,
        original_results_preserved=True,exploratory=True,source_sha256=digest(Path(__file__))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('cache','estimate','costs','summary','all'),default='all')
    args=parser.parse_args();out=OUT/'dense_lengthscale';private=PRIVATE/'dense_lengthscale'
    with threadpool_limits(limits=1):
        for phase,fn in (('cache',cache),('estimate',estimate),('costs',costs),('summary',summary)):
            if args.phase in (phase,'all'):
                if phase=='summary':fn(out)
                else:fn(out,private)


if __name__=='__main__':main()
