"""Exact population performance along the fitted boundary-economy ridge paths."""
from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import digest,write_json
from simulation_characteristic_factor import (
    SEED,T_GRID,ECONOMY_CODES,population,factor_path,path_estimates,
)

NAME='R1_b150_J2000_R500'
R=500
J=2000
B=1.5


def population_performance(estimate,mu,theta):
    """Moments of a frozen policy on a fresh stationary draw of the known DGP."""
    expected=(mu*theta)@estimate
    second=np.sum(mu[:,None]*estimate**2,axis=0)
    variance=second-expected**2
    if not np.all(variance>0):
        raise ValueError('Nonpositive population payoff variance.')
    loss=1-2*expected+second
    sharpe=np.sqrt(12)*expected/np.sqrt(variance)
    return expected,second,loss,sharpe


def performance_replication(job):
    r,penalties,reference_risk,reference_complexity=job
    mu,theta,_,_=population(J,B,'R1')
    rng=np.random.default_rng(np.random.SeedSequence([SEED,round(B*100),J,ECONOMY_CODES['R1'],r]))
    f=factor_path(rng,max(T_GRID),mu,theta)
    means=[];seconds=[];max_error=0.
    for k,t in enumerate(T_GRID):
        risk,complexity,estimate=path_estimates(f[:t],penalties,mu,theta,validation=False)
        expected,second,loss,sharpe=population_performance(estimate,mu,theta)
        np.testing.assert_allclose(risk,reference_risk[k],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(complexity,reference_complexity[k],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(loss,.84+risk,rtol=1e-10,atol=1e-12)
        max_error=max(max_error,float(np.max(abs(loss-(.84+reference_risk[k])))))
        means.append(expected);seconds.append(second)
    return r,np.array(means),np.array(seconds),max_error


def compute(output='outputs',cache='results/characteristic_factor',workers=2):
    cache=Path(cache);sim=Path(output)/'simulations';sim.mkdir(parents=True,exist_ok=True)
    reference_file=cache/(NAME+'.npz');meta_file=cache/(NAME+'.json')
    ref=np.load(reference_file);assert int(ref['done'])==R
    penalties=np.array(json.loads(meta_file.read_text())['design']['lambda_values'])
    moments_cache=cache/'performance';moments_cache.mkdir(parents=True,exist_ok=True)
    checkpoint=moments_cache/(NAME+'.npz');meta=moments_cache/(NAME+'.json')
    reference_risk=ref['risks'];reference_complexity=ref['empirical'];choices=ref['choices']
    design=dict(reference_sha256=digest(reference_file),economy='R1',b=B,J=J,R=R,
                T=list(T_GRID),lambda_values=penalties.tolist(),seed=SEED,
                statistic='sqrt(12) times marginal monthly population Sharpe of each fitted policy')
    shape=(R,len(T_GRID),len(penalties));done=0;elapsed=0.
    means=np.full(shape,np.nan);seconds=np.full(shape,np.nan);errors=np.full(R,np.nan)
    if checkpoint.exists() and meta.exists():
        old=json.loads(meta.read_text())
        if old['design']==design:
            loaded=np.load(checkpoint);means=loaded['means'];seconds=loaded['seconds']
            errors=loaded['errors'];done=int(loaded['done']);elapsed=old['elapsed_seconds']
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=((r,penalties,reference_risk[r],reference_complexity[r]) for r in range(done,R))
        for r,mean,second,error in pool.map(performance_replication,jobs):
            means[r]=mean;seconds[r]=second;errors[r]=error
            if (r+1)%25==0 or r+1==R:
                duration=elapsed+time.perf_counter()-start
                with checkpoint.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle,means=means,seconds=seconds,errors=errors,done=r+1)
                checkpoint.with_suffix('.tmp').replace(checkpoint)
                write_json(meta,dict(design=design,elapsed_seconds=duration))
                print('Performance moments',f'{r+1}/{R}',f'{duration:.1f}s',flush=True)
    variance=seconds-means**2
    assert np.isfinite(means).all() and (variance>0).all()
    sharpe=np.sqrt(12)*means/np.sqrt(variance);loss=1-2*means+seconds
    np.testing.assert_allclose(loss,.84+reference_risk,rtol=1e-10,atol=1e-12)
    mu,theta,_,_=population(J,B,'R1')
    c=np.sum(mu[:,None]/(mu[:,None]+penalties),axis=0)
    surfaces=[];selections=[]
    for k,t in enumerate(T_GRID):
        surfaces.append(pd.DataFrame(dict(economy='R1',b=B,J=J,R=R,T=t,
            **{'lambda':penalties},population_C=c,population_C_over_T=c/t,
            mean_population_loss=loss[:,k].mean(0),
            loss_mcse=loss[:,k].std(0,ddof=1)/np.sqrt(R),
            mean_population_sharpe=sharpe[:,k].mean(0),
            sharpe_mcse=sharpe[:,k].std(0,ddof=1)/np.sqrt(R))))
        choice=choices[:,k];selected_c=c[choice];selected_lambda=penalties[choice]
        selected_loss=loss[np.arange(R),k,choice];selected_sr=sharpe[np.arange(R),k,choice]
        row=dict(T=t,R=R,mean_selected_loss=float(selected_loss.mean()),
                 mean_selected_sharpe=float(selected_sr.mean()),
                 selected_loss_mcse=float(selected_loss.std(ddof=1)/np.sqrt(R)),
                 selected_sharpe_mcse=float(selected_sr.std(ddof=1)/np.sqrt(R)))
        for label,values in [('C',selected_c),('lambda',selected_lambda)]:
            row.update({label+'_q25':float(np.quantile(values,.25)),
                        label+'_median':float(np.median(values)),
                        label+'_q75':float(np.quantile(values,.75))})
        selections.append(row)
    pd.concat(surfaces,ignore_index=True).to_parquet(sim/'characteristic_factor_performance_surface.parquet',index=False)
    pd.DataFrame(selections).to_csv(sim/'characteristic_factor_performance_selection.csv',index=False)
    write_json(sim/'characteristic_factor_performance_audit.json',dict(
        design=design,source_sha256=digest(__file__),all_replicates_matched_existing_ridge_risk=True,
        max_loss_identity_error=float(errors.max()),population_variance_positive=True,
        population_optimal_loss=.84,population_max_annualized_monthly_sharpe=float(np.sqrt(12*.16/.84)),
        reference_risk_sha256=digest(sim/'characteristic_factor_full_surface.parquet'),
        elapsed_seconds=json.loads(meta.read_text())['elapsed_seconds'],workers=workers,
        interpretation='Fresh stationary population draw, not a realized next-year path or an AR-conditioned forecast.',
        annualization='sqrt(12) times monthly Sharpe; not the Sharpe of a 12-month sum with serial correlation.'))
    print('Population performance complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--cache',default='results/characteristic_factor')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();compute(args.output,args.cache,args.workers)
