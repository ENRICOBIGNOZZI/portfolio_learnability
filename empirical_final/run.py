"""Sequential E1--E3 execution on frozen real-data inputs only.

Usage: python -m empirical_final.run --phase spectral|accounting|missing
All new artifacts are isolated. Existing empirical outputs and simulations are
never rewritten or executed by this module.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.linalg import svdvals

from data_pipeline import digest
from kernels import FeatureBank, array_hash
from portfolio import fit_windows
from empirical_final.core import accounting_step, incremental_loss, performance, spectral_policy

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/final_empirical_20261008'
KERNELS=('linear','gaussian','matern32')
INPUTS={}


def bind(path,expected=None):
    path=Path(path)
    key=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    actual=digest(path)
    if expected is not None and actual != expected: raise ValueError('Input checksum: '+key)
    if key in INPUTS and INPUTS[key]!=actual: raise ValueError('Input mutated: '+key)
    INPUTS[key]=actual
    return path


def read_json(path):
    return json.loads(bind(path).read_text())


def write_json(path,value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False)


def save_table(name,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    with (OUT/'tables'/name).open('x') as stream: frame.to_csv(stream,index=False)


def finish(phase,started,details):
    write_json(OUT/'audit'/f'{phase}_verification.json',dict(
        passed=True,elapsed_seconds=time.monotonic()-started,python=platform.python_version(),
        inputs=INPUTS,sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'empirical_final').glob('*.py'))},
        **details))


def public(kernel):
    count=131 if kernel=='linear' else 10000
    folder=ROOT/f'results/final/public/{kernel}/seed_0/p_{count}'
    manifest=read_json(folder/'manifest.json')
    for name,expected in manifest['files'].items(): bind(folder/name,expected)
    return folder,manifest


def clean_manifest():
    meta=read_json(ROOT/'data/clean/manifest.json')
    if meta['status']!='complete' or meta['unresolved_returns']: raise ValueError('Canonical sample incomplete')
    return meta


def load_managed(kernel,clean):
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    if kernel=='linear':
        names=clean['characteristics'];totals=np.zeros((744,131));counts=np.zeros(744)
        lookup={d:i for i,d in enumerate(dates)}
        for item in clean['files']:
            path=bind(ROOT/'data/clean'/item['name'],item['sha256'])
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=['eom','r',*names]):
                frame=batch.to_pandas()
                for date,group in frame.groupby('eom',sort=False):
                    i=lookup[pd.Timestamp(date)];x=group[names].to_numpy(float);r=group.r.to_numpy(float)
                    if not np.isfinite(r).all() or not np.isfinite(x).all() or np.max(np.abs(x))>.5+1e-12:
                        raise ValueError('Invalid canonical stock data')
                    totals[i,0]+=r.sum();totals[i,1:]+=x.T@r;counts[i]+=len(r)
        if np.any(counts==0):raise ValueError('Missing managed month')
        return totals/counts[:,None],dates
    if kernel=='matern32':
        meta=read_json(ROOT/'results/schedule_cache/manifest.json')
        path=bind(ROOT/'results/schedule_cache/matern32_managed.npy',meta['managed_file_sha256'])
        if meta['clean_manifest_sha256']!=digest(ROOT/'data/clean/manifest.json'):raise ValueError('Wrong clean cache')
        g=np.load(path,mmap_mode='r',allow_pickle=False)
        if array_hash(g)!=meta['managed_matrix_sha256']:raise ValueError('Array checksum')
    else:
        meta=read_json(ROOT/'results/bandwidth_tuning/gaussian/cache/progress.json')
        identity=meta['identity'];path=bind(ROOT/'results/bandwidth_tuning/gaussian/cache/managed.npy',meta['sha256'])
        if meta['status']!='complete' or identity['clean_manifest_sha256']!=digest(ROOT/'data/clean/manifest.json'):
            raise ValueError('Incomplete or wrong Gaussian cache')
        if identity['multipliers'].count(1.)!=1:raise ValueError('Baseline bandwidth unidentified')
        g=np.load(path,mmap_mode='r',allow_pickle=False)[identity['multipliers'].index(1.)]
        meta=identity
    source=read_json(ROOT/f'results/final/public/{kernel}/seed_0/source.json')
    if meta['feature_bank']!=source['feature_bank'] or not pd.DatetimeIndex(meta['dates']).equals(dates):
        raise ValueError('Feature bank or dates differ from baseline')
    if g.shape!=(744,10000) or not np.isfinite(g).all():raise ValueError('Invalid managed matrix')
    return g,dates


def spectral():
    started=time.monotonic();freeze=read_json(OUT/'audit/protocol_freeze.json');clean=clean_manifest()
    kappa=read_json(ROOT/'results/final/public/linear/seed_0/calibration.json')['scale']
    rf=pd.read_csv(bind(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    annual=[];nested=[];spectra=[];checks=[]
    for kernel in KERNELS:
        folder,manifest=public(kernel)
        reference=pd.read_csv(folder/'monthly.csv',parse_dates=['formation_date','return_date'])
        old_diag=pd.read_csv(folder/'diagnostics.csv'); selected=old_diag[old_diag.selected].set_index('test_year')
        count=131 if kernel=='linear' else 10000
        stored=np.load(bind(ROOT/f'results/final/private/{kernel}/seed_0/p_{count}/selected_coefficients.npz'),allow_pickle=False)['beta']
        g,dates=load_managed(kernel,clean)
        gram=g@g.T
        sv=svdvals(np.asarray(g[:120]))**2/120
        dual=np.maximum(np.linalg.eigvalsh(gram[:120,:120]/120),0)[::-1]
        svd_error=float(np.max(np.abs(sv-dual[:len(sv)]))/sv[0])
        if svd_error>1e-10:raise ValueError('Primal/dual operator mismatch')
        max_payoff=0.;max_beta=0.;max_direct=0.;max_complexity=0.;max_residual=0.
        for yi,result in enumerate(fit_windows(g,dates,np.array(manifest['lambda_grid']))):
            year=result['year'];choice=result['choice'];penalty=float(result['penalties'][choice])
            old=reference[reference.test_year==year].sort_values('formation_date')
            if not np.isclose(penalty,selected.loc[year,'lambda'],rtol=1e-12,atol=0):raise ValueError('Selection differs')
            np.testing.assert_allclose(result['validation_loss'],old_diag[old_diag.test_year==year].validation_loss,rtol=2e-7,atol=1e-9)
            refit=np.r_[result['train'],result['validation']];train_g=np.asarray(g[refit]);t=len(refit)
            spec=spectral_policy(train_g,penalty,freeze['nested_rank_fractions'])
            raw=np.asarray(g[result['test']])@spec['beta']
            error=float(np.max(np.abs(raw[:,-1]-old.raw_excess_return.to_numpy())))
            betaerror=float(np.linalg.norm(spec['beta'][:,-1]-stored[yi])/max(np.linalg.norm(stored[yi]),1e-30))
            cerror=abs(spec['complexity']-selected.loc[year,'effective_complexity'])
            max_payoff=max(max_payoff,error);max_beta=max(max_beta,betaerror);max_complexity=max(max_complexity,cerror)
            max_residual=max(max_residual,spec['feature_operator_relative_residual'])
            if error>2e-7 or betaerror>2e-6 or cerror>2e-6:raise ValueError(f'Baseline reconstruction {kernel} {year}: {error} {betaerror} {cerror}')
            # Direct normalized dual solve, algebra shared with the separately inspected CTF implementation.
            if year in freeze['spectrum_cutoffs']:
                alpha=np.linalg.solve(gram[np.ix_(refit,refit)]/t+penalty*np.eye(t),np.ones(t))/t
                direct=gram[np.ix_(result['test'],refit)]@alpha
                de=float(np.max(np.abs(direct-raw[:,-1])));max_direct=max(max_direct,de)
                if de>2e-7:raise ValueError('Independent normalized dual solve differs')
                for j,mu in enumerate(spec['eigenvalues'][:spec['rank']]):
                    spectra.append(dict(kernel=kernel,decision_year=year,formation_cutoff=str(dates[refit[-1]].date()),
                                        last_known_payoff=str((dates[refit[-1]]+pd.offsets.MonthEnd(1)).date()),T=t,rank=j+1,eigenvalue=mu))
            values=spec['eigenvalues'];positive=values[:spec['rank']];share=positive/positive.sum()
            fit_lo=max(1,int(np.ceil(.1*len(positive))));fit_hi=int(np.floor(.6*len(positive)))
            b=-float(np.polyfit(np.log(np.arange(fit_lo,fit_hi+1)),np.log(positive[fit_lo-1:fit_hi]),1)[0])
            realized=old.return_date.to_numpy();excess=raw[:,-1]*kappa;total=excess+rf.loc[realized].to_numpy()
            annual.append(dict(kernel=kernel,decision_year=year,decision_date=str(dates[result['test'][0]].date()),
                training_first_formation=str(dates[refit[0]].date()),last_training_payoff=str((dates[refit[-1]]+pd.offsets.MonthEnd(1)).date()),
                first_oos_payoff=str(old.return_date.iloc[0].date()),last_oos_payoff=str(old.return_date.iloc[-1].date()),
                T=t,inner_training_months=len(result['train']),validation_months=len(result['validation']),
                selected_lambda=penalty,C=spec['complexity'],C_over_T=spec['complexity']/t,
                positive_rank=spec['rank'],top10_eigenvalue_share=float(share[:10].sum()),
                participation_rank=float(1/np.sum(share**2)),descriptive_b=b,
                b_fit_first_rank=fit_lo,b_fit_last_rank=fit_hi,validation_loss=float(result['validation_loss'][choice]),
                raw_oos_loss=float(np.mean((1-raw[:,-1])**2)),**performance(excess,total)))
            previous=np.zeros(12)
            for stage,fraction in enumerate(freeze['nested_rank_fractions']):
                addition=raw[:,stage]-previous;change,cross,square=incremental_loss(previous,addition)
                for j in range(12):
                    nested.append(dict(kernel=kernel,decision_year=year,formation_date=str(old.formation_date.iloc[j].date()),
                        return_date=str(old.return_date.iloc[j].date()),T=t,rank_fraction=fraction,
                        included_directions=spec['counts'][stage],positive_rank=spec['rank'],selected_lambda=penalty,
                        boundary_relative_gap=spec['boundary_relative_gaps'][stage],raw_payoff=raw[j,stage],
                        previous_payoff=previous[j],added_payoff=addition[j],annual_loss_change=change,
                        annual_loss_cross_term=cross,annual_loss_square_term=square))
                previous=raw[:,stage]
            if (yi+1)%10==0:print(kernel,'spectral refits',yi+1,'/47',flush=True)
        checks.append(dict(kernel=kernel,refits=47,months=564,managed_shape=list(g.shape),
            maximum_payoff_error=max_payoff,maximum_relative_coefficient_error=max_beta,
            maximum_complexity_error=max_complexity,maximum_direct_dual_payoff_error=max_direct,
            primal_svd_dual_eigenvalue_relative_error=svd_error,maximum_feature_operator_relative_residual=max_residual))
        print(kernel,'baseline and nested policies verified',flush=True)
        del g,gram
    save_table('annual_complexity.csv',annual);save_table('nested_spectral_monthly.csv',nested);save_table('managed_spectra.csv',spectra)
    nested=pd.DataFrame(nested);summaries=[]
    for (kernel,fraction),frame in nested.groupby(['kernel','rank_fraction'],sort=False):
        frame=frame.sort_values('return_date');r=frame.raw_payoff.to_numpy();prev=frame.previous_payoff.to_numpy();delta=frame.added_payoff.to_numpy()
        change,cross,square=incremental_loss(prev,delta);e=kappa*r
        perf=performance(e,e+rf.loc[pd.to_datetime(frame.return_date)].to_numpy())
        prev_sd=prev.std(ddof=1)
        summaries.append(dict(kernel=kernel,rank_fraction=fraction,months=len(frame),raw_oos_loss=float(np.mean((1-r)**2)),
            incremental_loss_change=change,cross_term=cross,squared_added_payoff=square,
            previous_sharpe=float(np.sqrt(12)*prev.mean()/prev_sd) if prev_sd else 0.,
            mean_included_directions=float(frame.included_directions.mean()),**perf))
    save_table('spectral_contributions.csv',summaries)
    finish('spectral',started,dict(baseline_checks=checks,partition_fractions=freeze['nested_rank_fractions'],
        nested_cost_scope='Gross incremental diagnostics only: nested stock weights and their net-cost paths are not inferred from returns. E1 costs use actually stored full-policy weights.'))


def accounting():
    started=time.monotonic();freeze=read_json(OUT/'audit/protocol_freeze.json');meta=clean_manifest()
    scale=read_json(ROOT/'results/final/public/linear/seed_0/calibration.json')['scale']
    rf=pd.read_csv(bind(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    refs={k:pd.read_csv(public(k)[0]/'monthly.csv',parse_dates=['formation_date','return_date']).set_index('formation_date') for k in KERNELS}
    states={(k,s['name']):pd.Series(dtype=float) for k in KERNELS for s in freeze['cost_scenarios']}
    previous_target={k:pd.Series(dtype=float) for k in KERNELS};rows=[];max_error=0.;residual=0.
    files={int(Path(x['name']).stem.split('_')[-1]):x for x in meta['files']}
    for year in range(1978,2025):
        item=files[year];path=bind(ROOT/'data/clean'/item['name'],item['sha256'])
        clean=pd.read_parquet(path,columns=['id','eom','r','return_date'])
        for kernel in KERNELS:
            p=131 if kernel=='linear' else 10000
            weights=pd.read_parquet(bind(ROOT/f'results/final/private/{kernel}/seed_0/p_{p}/weights_{year}.parquet'))
            if weights.duplicated(['formation_date','id']).any():raise ValueError('Duplicate stock weights')
            for date,w in weights.groupby('formation_date',sort=True):
                date=pd.Timestamp(date);data=clean[clean.eom==date].set_index('id');w=w.set_index('id')
                if set(data.index)!=set(w.index):raise ValueError('Stock universe mismatch')
                w=w.loc[data.index];raw=w.raw_weight.to_numpy();r=data.r.to_numpy();target=pd.Series(scale*raw,index=data.index)
                retdate=date+pd.offsets.MonthEnd(1)
                if not (data.return_date==retdate).all() or not (w.return_date==retdate).all():raise ValueError('Payoff timing mismatch')
                reference=refs[kernel].loc[date];raw_error=abs(float(raw@r)-reference.raw_excess_return)
                max_error=max(max_error,raw_error)
                np.testing.assert_allclose([raw@r,np.abs(raw).sum(),raw.sum()],reference[['raw_excess_return','raw_gross','raw_net']].to_numpy(float),rtol=1e-8,atol=1e-9)
                target_turnover=float(target.subtract(previous_target[kernel],fill_value=0).abs().sum())
                for scenario in freeze['cost_scenarios']:
                    key=(kernel,scenario['name']);drift=states[key];ids=target.index.union(drift.index)
                    aligned=target.reindex(ids,fill_value=0).to_numpy();drift_values=drift.reindex(ids,fill_value=0).to_numpy()
                    # Exited names retain NaN next returns; no payoff is assigned to a liquidated holding.
                    current_r=pd.Series(r,index=data.index).reindex(ids).to_numpy()
                    step=accounting_step(aligned,current_r,float(rf.loc[retdate]),drift_values,
                                         scenario['trade_bps']/1e4,scenario['borrow_bps_year']/1e4)
                    next_drift=step.pop('next_drift');states[key]=pd.Series(next_drift,index=ids)[aligned!=0]
                    residual=max(residual,step['accounting_residual'])
                    rows.append(dict(kernel=kernel,scenario=scenario['name'],formation_date=str(date.date()),
                        return_date=str(retdate.date()),rf=float(rf.loc[retdate]),raw_payoff=float(raw@r),
                        gross_exposure=float(target.abs().sum()),net_exposure=float(target.sum()),
                        target_turnover=target_turnover,stocks=len(target),**step))
                previous_target[kernel]=target
        if year%5==0:print('Stock accounting through',year,flush=True)
    frame=pd.DataFrame(rows);summaries=[]
    for (kernel,scenario),f in frame.groupby(['kernel','scenario'],sort=False):
        if len(f)!=564:raise ValueError('Incomplete common-period accounting')
        summaries.append(dict(kernel=kernel,scenario=scenario,months=len(f),
            first_payoff=f.return_date.iloc[0],last_payoff=f.return_date.iloc[-1],
            annualized_turnover_two_sided=12*f.turnover.mean(),annualized_target_turnover=12*f.target_turnover.mean(),
            mean_gross_exposure=f.gross_exposure.mean(),mean_net_exposure=f.net_exposure.mean(),
            mean_short_notional=f.short_notional.mean(),annual_trading_fees=12*f.trading_fee.mean(),
            annual_borrowing_fees=12*f.borrowing_fee.mean(),annual_trading_return_drag=12*f.trading_return_drag.mean(),
            **performance(f.excess_return,f.total_return)))
    summary=pd.DataFrame(summaries);old=pd.read_csv(bind(ROOT/'paper/tables/performance.csv'))
    labels={'linear':'Linear','gaussian':'Gaussian','matern32':'Matérn-3/2'}
    for k in KERNELS:
        actual=summary[(summary.kernel==k)&(summary.scenario=='gross')].iloc[0];expected=old[old.Policy==labels[k]].iloc[0]
        np.testing.assert_allclose(actual[['annual_excess_return','annual_volatility','sharpe','maximum_drawdown','mean_gross_exposure','mean_net_exposure']].to_numpy(float),expected.iloc[1:].to_numpy(float),rtol=1e-9,atol=1e-10)
    save_table('monthly_accounting.csv',frame);save_table('performance_scenarios.csv',summary)
    finish('accounting',started,dict(months_per_policy=564,stock_weight_payoff_maximum_error=max_error,
        self_financing_maximum_residual=residual,baseline_performance_reproduced=True,
        scope='Cost scenarios, not observed execution or lending fees; financing at frozen cash rate, no impact or extra funding spread. Initial entry from cash charged; terminal positions marked, not forcibly liquidated.'))


def missing():
    started=time.monotonic();meta=clean_manifest()
    audit=read_json(ROOT/'outputs/tables/formation_timing_audit.json')
    monthly=pd.read_csv(bind(ROOT/'outputs/tables/formation_timing_by_month.csv'))
    formation=read_json(ROOT/'data/formation_only/manifest.json')
    # Audit size/materiality without constructing a return for any unidentified holding.
    missing_rows=[]
    for item in formation['files']:
        path=ROOT/'data/formation_only'/item['name']
        bind(path,item['sha256'])
        frame=pd.read_parquet(path,columns=['eom','me','r','return_known'])
        for date,f in frame.groupby('eom',sort=True):
            unknown=~np.isfinite(f.r.to_numpy());cap=f.me.to_numpy(float)
            valid=np.isfinite(cap)&(cap>=0)
            missing_rows.append(dict(formation_date=str(pd.Timestamp(date).date()),stocks=len(f),
                missing_payoffs=int(unknown.sum()),missing_share=float(unknown.mean()),
                identified_market_cap_share=float(cap[unknown&valid].sum()/cap[valid].sum()) if cap[valid].sum()>0 else np.nan,
                missing_market_cap=int((~valid).sum())))
    table=pd.DataFrame(missing_rows)
    if int(table.missing_payoffs.sum())!=audit['unresolved_payoffs']:raise ValueError('Missing-return audit mismatch')
    save_table('missing_payoff_materiality.csv',table)
    finish('missing',started,dict(stock_months=int(table.stocks.sum()),unidentified_stock_months=int(table.missing_payoffs.sum()),
        unidentified_share=float(table.missing_payoffs.sum()/table.stocks.sum()),
        largest_monthly_missing_share=float(table.missing_share.max()),largest_monthly_market_cap_share=float(table.identified_market_cap_share.max()),
        previous_audit=audit,limitation='Counts, market-cap shares and rank differences do not identify strategy return bias. No finite performance bound is claimed for unidentified outcomes; no returns are imputed.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['spectral','accounting','missing'],required=True)
    args=parser.parse_args()
    if (OUT/'audit'/f'{args.phase}_verification.json').exists():raise FileExistsError('Preserve completed results; choose a new version for reruns')
    globals()[args.phase]()
