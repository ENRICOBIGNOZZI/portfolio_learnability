"""Integrated canonical kernel study, costs, historical panels and sensitivity.

Reuse is allowed only after SHA verification and a numerical policy identity
check. Existing outputs remain legacy snapshots. No simulation is run.
"""
import argparse
import json
import time
import shutil
import numpy as np
import pandas as pd
from data_pipeline import digest
from kernels import FeatureBank
from portfolio import fit_windows, dual_path, sharpe, complexity_grid, annual_splits
from empirical_final.run import clean_manifest, load_managed, public, INPUTS, read_json, bind
from empirical_final.core import spectral_policy
from empirical_final.three_experiments import block_counts, boot_metrics, interval, spectral_groups, FACTORS, hac_regression
from empirical_final.compact_common import (ROOT, OUT, PRIVATE, PROTOCOL, SCALE,
    setup,save,audit,source,account_month,summarize_accounts,verify_audit_inputs,generated_weight_cache)

LEGACY = ROOT/'outputs/empirical_three_experiments_20261009'
ECON = ROOT/'outputs/spectral_economic_content_20261009'
EP = ROOT/'results/spectral_economic_content_20261009'


def kernel_study():
    setup();started=time.monotonic();clean=clean_manifest()
    reference_audit=verify_audit_inputs(LEGACY/'audit/execution.json')
    eigen_rows=[];annual=[];monthly=[];gaussian_results=[];zero=[];private={}
    for kernel in ['linear','gaussian','matern32']:
        g,dates=load_managed(kernel,clean);folder,manifest=public(kernel)
        grid=np.asarray(manifest['lambda_grid'])
        ref=pd.read_csv(folder/'monthly.csv',parse_dates=['return_date'])
        coefficients=[]
        for result in fit_windows(g,dates,grid):
            year=result['year'];j=result['choice'];h=np.r_[result['train'],result['validation']]
            o=result['test'];rdates=dates[o]+pd.offsets.MonthEnd(1)
            np.testing.assert_allclose(result['test_returns'][:,j],ref.set_index('return_date').loc[rdates].raw_excess_return,rtol=2e-6,atol=2e-7)
            coefficients.append(result['beta'])
            for k,penalty in enumerate(grid):
                annual.append(dict(policy=kernel,decision_year=year,candidate=k,penalty=penalty,C=result['complexity'][k],
                    historical_sharpe=result['historical_sharpe'][k],selected=k==j,validation_Q=result['validation_loss'][k],
                    train_months=len(result['train']),validation_months=len(result['validation']),refit_months=len(h),
                    last_known_payoff=dates[h[-1]]+pd.offsets.MonthEnd(1),first_test_payoff=rdates[0]))
            for i,date in enumerate(rdates):
                monthly.append(dict(policy=kernel,decision_year=year,formation_date=dates[o[i]],return_date=date,
                                    raw_return=result['test_returns'][i,j],selected_penalty=grid[j],C=result['complexity'][j]))
            for k,mu in enumerate(result['mu']):
                eigen_rows.append(dict(policy=kernel,decision_year=year,rank=k+1,eigenvalue=mu,selected_penalty=grid[j]))
            if kernel=='gaussian':
                _,c,meta=spectral_groups(np.asarray(g[h]),grid[j])
                raw=np.asarray(g[o])@c
                np.testing.assert_allclose(raw.sum(axis=1),result['test_returns'][:,j],rtol=2e-6,atol=2e-7)
                gaussian_results.append((result,rdates,raw,meta))
                az,mu,cz=dual_path(np.asarray(g[h])@np.asarray(g[h]).T,[0])
                rank=int((mu>mu.max()*1e-12).sum())
                assert abs(cz[0]-rank)<1e-6
                zp=np.asarray(g[o])@np.asarray(g[h]).T@az[:,0]
                for i,date in enumerate(rdates):zero.append(dict(decision_year=year,return_date=date,
                    minimum_norm_return=zp[i],C=cz[0],rank=rank,eligible_for_validation=False))
        np.savez(PRIVATE/f'{kernel}_coefficients.npz',beta=np.stack(coefficients))
        private[kernel]=digest(PRIVATE/f'{kernel}_coefficients.npz')
        print(f'Canonical {kernel}: 47 refits reconstructed',flush=True)
    save('kernel_annual_paths',annual);save('kernel_selected_monthly',monthly);save('managed_spectra',eigen_rows)
    save('gaussian_zero_penalty_diagnostic',zero)
    # Public wide monthly path and group aggregates recomputed from the same managed array.
    paths=[];groups=[];meta=[]
    for r,dates,raw,mm in gaussian_results:
        paths.append(pd.DataFrame(r['test_returns'],columns=[f'lambda_{j:03}' for j in range(len(r['penalties']))]).assign(
            decision_year=r['year'],return_date=dates))
        for j in range(4):
            meta.append(dict(decision_year=r['year'],group=j+1,penalty=r['penalties'][r['choice']],**mm[j]))
            for i,date in enumerate(dates):groups.append(dict(decision_year=r['year'],return_date=date,group=j+1,
                raw_contribution=raw[i,j],scaled_contribution=SCALE*raw[i,j],raw_nested_return=raw[i,:j+1].sum()))
    paths=save('e1_monthly_paths',pd.concat(paths));save('e3_monthly',groups);save('e3_annual_groups',meta)
    for name in ['e1_monthly_paths','e3_monthly']:
        old=pd.read_csv(source(LEGACY/'tables'/f'{name}.csv'))
        new=pd.read_csv(OUT/'tables'/f'{name}.csv')
        columns=[c for c in new if c in old and pd.api.types.is_numeric_dtype(new[c])]
        np.testing.assert_allclose(new[columns],old[columns],rtol=2e-6,atol=2e-7)
    # Factor and history diagnostics are reused only from the verified identical Gaussian run.
    for name in ['french_factors','e2_monthly','e2_annual','e2_summary','e3_training_factor_loadings']:
        path=LEGACY/'tables'/f'{name}.csv'
        if path.exists():
            expected=reference_audit['tables'][path.name]
            bind(path,expected);shutil.copyfile(path,OUT/'tables'/path.name)
    audit('kernels',dict(elapsed_seconds=time.monotonic()-started,passed=True,inputs=dict(INPUTS),
                        private_coefficient_hashes=private,legacy_reuse='SHA-verified inputs and numerical monthly-path reconstruction'))
    historical_panels()


def historical_panels():
    paths=pd.read_csv(OUT/'tables/e1_monthly_paths.csv',parse_dates=['return_date'])
    annual=pd.read_csv(OUT/'tables/kernel_annual_paths.csv').query("policy=='gaussian'")
    selected=pd.read_csv(OUT/'tables/kernel_selected_monthly.csv',parse_dates=['return_date']).query("policy=='gaussian'")
    columns=[c for c in paths if c.startswith('lambda_')]
    curves=[];summaries=[];identifiers=[]
    for lo,hi in PROTOCOL['blocks']:
        label=f'{lo}-{hi}';p=paths[paths.decision_year.between(lo,hi)]
        a=annual[annual.decision_year.between(lo,hi)]
        s=selected[selected.decision_year.between(lo,hi)]
        counts=block_counts(len(p),12,PROTOCOL['bootstrap']['seed'],PROTOCOL['bootstrap']['replicates'])
        r=p[columns].to_numpy();sr=sharpe(r)
        boot=boot_metrics(np.column_stack([r,s.raw_return]),counts)['sharpe'];ci=interval(boot)
        peak=int(np.argmax(sr));month_counts=p.groupby('decision_year').size()
        # Annual complexities weighted by the actual number of realized months.
        complexity=a.assign(n=a.decision_year.map(month_counts)).groupby('candidate').apply(
            lambda f:np.average(f.C,weights=f.n),include_groups=False)
        for j in range(len(columns)):
            curves.append(dict(period=label,candidate=j,penalty=a[a.candidate.eq(j)].penalty.iloc[0],
                mean_C=complexity[j],oos_sharpe=sr[j],sr_low=ci[0,j],sr_high=ci[1,j],
                retrospective_maximum=j==peak,months=len(p),first_return=p.return_date.min(),last_return=p.return_date.max()))
        summaries.append(dict(period=label,months=len(p),first_return=p.return_date.min(),last_return=p.return_date.max(),
            selected_mean_C=s.C.mean(),selected_sharpe=float(sharpe(s.raw_return)),selected_low=ci[0,-1],selected_high=ci[1,-1],
            maximum_candidate=peak,maximum_C=complexity[peak],maximum_sharpe=sr[peak],boundary_maximum=peak in [0,len(columns)-1]))
        identifiers.append(p[['decision_year','return_date']].assign(period=label))
    save('nine_panel_curves',curves);save('nine_panel_summary',summaries)
    save('nine_panel_month_identifiers',pd.concat(identifiers))
    a=annual[annual.decision_year.eq(2024)].sort_values('candidate')
    r=paths[paths.decision_year.eq(2024)][columns].to_numpy()
    save('single_window_complexity',a.assign(oos_sharpe=sharpe(r)).drop(columns='policy'))


def costs_and_economics():
    setup();started=time.monotonic();clean=clean_manifest()
    audit_meta=verify_audit_inputs(ECON/'audit/holdings_execution.json')
    if digest(ROOT/'empirical_final/economic_holdings.py') != audit_meta['source_sha256']:
        raise ValueError('Holdings reconstruction code changed')
    monthly=[];families=[];stock_checks=[]
    policies=['linear','gaussian','matern32']+[f'nested_{j}' for j in range(1,5)]
    states={(p,s[0]):pd.Series(dtype=float) for p in policies for s in PROTOCOL['cost_scenarios']}
    previous={p:pd.Series(dtype=float) for p in policies}
    rf=pd.read_csv(source(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    ref=pd.read_csv(OUT/'tables/kernel_selected_monthly.csv',parse_dates=['formation_date'])
    groups=pd.read_csv(OUT/'tables/e3_monthly.csv',parse_dates=['return_date'])
    from empirical_final.economic_holdings import FAMILIES, family_matrix
    from empirical_final.economic_financing import EXTRA, EXTRA_MEMBERS
    family_dict=dict(FAMILIES,**{EXTRA:EXTRA_MEMBERS})
    audit('economic_dictionary',family_dict)
    for year in range(1978,2025):
        item=next(i for i in clean['files'] if i['name']==f'jkp_{year}.parquet')
        frame=pd.read_parquet(bind(ROOT/'data/clean'/item['name'],item['sha256']))
        cp_path=EP/f'year_{year}/complete.json'
        bind(cp_path,audit_meta['private_checkpoint_hashes'][str(cp_path.relative_to(ROOT))])
        cp=json.loads(cp_path.read_text());hp=EP/f'year_{year}/holdings.parquet'
        bind(hp,cp['files']['holdings.parquet']);holdings=pd.read_parquet(hp)
        weights={}
        for kernel in ['linear','gaussian','matern32']:
            count=131 if kernel=='linear' else 10000
            wp=source(ROOT/f'results/final/private/{kernel}/seed_0/p_{count}/weights_{year}.parquet')
            weights[kernel]=pd.read_parquet(wp)
        max_error=0.
        for date,data in frame.groupby('eom',sort=True):
            data=data.set_index('id');retdate=date+pd.offsets.MonthEnd(1)
            returns=data.r;target={}
            for kernel,w in weights.items():
                ww=w[w.formation_date.eq(date)].set_index('id')
                if set(data.index)!=set(ww.index):raise ValueError('Different canonical universe')
                target[kernel]=SCALE*ww.loc[data.index].raw_weight
                rr=ref[(ref.policy==kernel)&(ref.formation_date==date)].iloc[0].raw_return
                np.testing.assert_allclose(target[kernel]@returns,SCALE*rr,atol=2e-8,rtol=2e-6)
            hh=holdings[holdings.formation_date.eq(date)].set_index('permno').loc[data.index]
            gw=hh[[f'group_{j}_weight' for j in range(1,5)]].to_numpy()
            np.testing.assert_allclose(gw.sum(axis=1),target['gaussian'],rtol=2e-6,atol=2e-9)
            expected=groups[groups.return_date.eq(retdate)].sort_values('group').scaled_contribution.to_numpy()
            np.testing.assert_allclose(gw.T@returns.to_numpy(),expected,rtol=2e-6,atol=2e-9)
            max_error=max(max_error,float(np.max(np.abs(gw.T@returns.to_numpy()-expected))))
            for j in range(4):target[f'nested_{j+1}']=pd.Series(gw[:,:j+1].sum(axis=1),index=data.index)
            x=np.column_stack([sum(sign*data[name].to_numpy() for name,sign in members.items())/len(members)
                               for members in family_dict.values()])
            long=np.maximum(gw,0);short=np.maximum(-gw,0)
            l=long.sum(axis=0);s=short.sum(axis=0)
            lm=long.T@x/l[:,None];sm=short.T@x/s[:,None];signed=gw.T@x
            for j in range(4):
                for k,family in enumerate(family_dict):families.append(dict(decision_year=year,formation_date=date,
                    return_date=retdate,group=j+1,family=family,long_mean=lm[j,k],short_mean=sm[j,k],
                    long_minus_short=lm[j,k]-sm[j,k],signed_notional=signed[j,k],component_gross=l[j]+s[j]))
            for policy,t in target.items():
                for scenario,tb,bb in PROTOCOL['cost_scenarios']:
                    step,states[policy,scenario]=account_month(t,returns,float(rf.loc[retdate]),states[policy,scenario],previous[policy],tb,bb)
                    monthly.append(dict(policy=policy,scenario=scenario,decision_year=year,formation_date=date,return_date=retdate,**step))
                previous[policy]=t
        stock_checks.append(dict(decision_year=year,group_payoff_error=max_error))
        print(f'Canonical stock accounting/netting through {year}',flush=True)
    save('monthly_accounting',monthly);save('performance_scenarios',summarize_accounts(pd.DataFrame(monthly)))
    save('monthly_economic_exposures',families);save('stock_reconstruction_checks',stock_checks)
    # Named characteristics and regressions have identical positions/outcomes; preserve verified aggregates.
    for name in ['dominant_long_short_exposures_named','annual_characteristics','factor_attribution',
                 'financing_factor_attribution','joint_weight_profiles','financing_joint_weight_profiles',
                 'financing_cell_return_attribution']:
        p=ECON/'tables'/f'{name}.csv'
        source(p);shutil.copyfile(p,OUT/'tables'/p.name)
    audit('accounting',dict(passed=True,elapsed_seconds=time.monotonic()-started,inputs=dict(INPUTS),
        nested_accounting='Sum weights at stock level before drift, trading costs or short fees',
        corporate_actions='Upstream JKP total/excess payoffs include documented CRSP treatment. Same identifier links and inferred return drift; no independent share-count, split, or execution-tape reconstruction. Departed stocks liquidated at formation marks, no fabricated next payoff.',
        limitation='Self-financing illustrative friction sensitivity on a conditional sample, not certified realized trading costs'))


def bandwidth_study():
    from bandwidth_tuning import fit_experiment, verify_baseline, MULTIPLIERS
    setup();started=time.monotonic();clean=clean_manifest();dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    selection=[];paths=[];surfaces=[]
    for kernel in ['gaussian','matern32']:
        cache=ROOT/f'results/bandwidth_tuning/{kernel}/cache'
        meta=read_json(cache/'progress.json');identity=meta['identity']
        if identity['clean_manifest_sha256']!=digest(ROOT/'data/clean/manifest.json') or meta['status']!='complete':
            raise ValueError('Wrong bandwidth cache')
        g=np.load(bind(cache/'managed.npy',meta['sha256']),mmap_mode='r')
        baseline=read_json(ROOT/f'results/final/public/{kernel}/seed_0/source.json')['feature_bank']
        if identity['feature_bank']!=baseline or identity['multipliers']!=list(MULTIPLIERS):
            raise ValueError('Bandwidth representation changed')
        m,s,v=fit_experiment(g,dates,MULTIPLIERS,kernel,baseline['ell'])
        check=verify_baseline(m,ROOT/'results/final/public',kernel)
        selection.append(s);paths.append(m);surfaces.append(v.assign(kernel=kernel))
        # Fixed control has already been reconstructed stock by stock. Tuned holdings are newly computed.
        states={sc[0]:pd.Series(dtype=float) for sc in PROTOCOL['cost_scenarios']};previous=pd.Series(dtype=float)
        accounts=[]
        chosen=s[s['mode'].eq('tuned')].set_index('year')
        for year,train,val,test in annual_splits(dates):
            choice=chosen.loc[year];b=list(MULTIPLIERS).index(choice.bandwidth_multiplier);h=np.r_[train,val]
            gg=np.asarray(g[b]);alpha,_,_=dual_path(gg[h]@gg[h].T,[choice['lambda']]);coef=gg[h].T@alpha[:,0]
            bank=FeatureBank(kernel,130,10000,choice.bandwidth,0)
            item=next(i for i in clean['files'] if i['name']==f'jkp_{year}.parquet')
            data=pd.read_parquet(bind(ROOT/'data/clean'/item['name'],item['sha256']))
            weights=[]
            weight_path=PRIVATE/f'bandwidth_{kernel}_weights_{year}.parquet'
            cached=generated_weight_cache('bandwidth',weight_path)
            rf=pd.read_csv(source(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
            for date,f in data.groupby('eom',sort=True):
                retdate=date+pd.offsets.MonthEnd(1)
                if cached is None:w=SCALE*bank.scores(f[clean['characteristics']].to_numpy(),coef)/len(f)
                else:
                    cw=cached[cached.formation_date.eq(date)].set_index('id')
                    if set(cw.index)!=set(f.id):raise ValueError('Cached tuned universe changed')
                    w=cw.loc[f.id].weight.to_numpy()
                    sample=np.arange(min(16,len(f)))
                    expected_w=SCALE*bank.scores(f[clean['characteristics']].to_numpy()[sample],coef)/len(f)
                    np.testing.assert_allclose(w[sample],expected_w,rtol=2e-6,atol=2e-9)
                target=pd.Series(w,index=f.id);r=pd.Series(f.r.to_numpy(),index=f.id)
                expected=m[(m['mode']=='tuned')&(m.formation_date==date)].raw_excess_return.iloc[0]*SCALE
                np.testing.assert_allclose(target@r,expected,rtol=2e-6,atol=2e-8)
                for sc,tb,bb in PROTOCOL['cost_scenarios']:
                    step,states[sc]=account_month(target,r,float(rf.loc[retdate]),states[sc],previous,tb,bb)
                    accounts.append(dict(policy=kernel+'_tuned',scenario=sc,decision_year=year,formation_date=date,return_date=retdate,**step))
                previous=target;weights.append(pd.DataFrame(dict(id=f.id.to_numpy(),formation_date=date,weight=w)))
            if cached is None:pd.concat(weights).to_parquet(weight_path,index=False)
            if year%5==0:print(f'Bandwidth {kernel} stock accounting {year}',flush=True)
        save('bandwidth_'+kernel+'_accounting',accounts)
        audit('bandwidth_'+kernel,dict(baseline_check=check,cache_sha256=meta['sha256'],identity=identity))
    save('bandwidth_selections',pd.concat(selection));save('bandwidth_monthly_raw',pd.concat(paths))
    save('bandwidth_validation_surface',pd.concat(surfaces))
    accounts=pd.concat([pd.read_csv(OUT/'tables'/f'bandwidth_{k}_accounting.csv') for k in ['gaussian','matern32']])
    fixed=pd.read_csv(OUT/'tables/monthly_accounting.csv').query("policy in ['gaussian','matern32']")
    save('bandwidth_performance',summarize_accounts(pd.concat([fixed,accounts])))
    contrasts=[]
    combined=pd.concat([fixed,accounts])
    for kernel in ['gaussian','matern32']:
        for scenario in ['gross','trade25_borrow30']:
            p=combined[combined.scenario.eq(scenario)&combined.policy.isin([kernel,kernel+'_tuned'])].pivot(
                index='return_date',columns='policy',values='excess_return').sort_index()
            for length in PROTOCOL['bootstrap']['blocks']:
                boot=boot_metrics(p[[kernel+'_tuned',kernel]].to_numpy(),block_counts(len(p),length,20261010,5000))['sharpe']
                ci=interval(boot[:,0]-boot[:,1])
                contrasts.append(dict(kernel=kernel,scenario=('Gross' if scenario=='gross' else 'Net'),block_length=length,
                    delta_sharpe=float(sharpe(p[kernel+'_tuned'])-sharpe(p[kernel])),low=ci[0],high=ci[1]))
    save('bandwidth_uncertainty',contrasts)
    audit('bandwidth',dict(elapsed_seconds=time.monotonic()-started,inputs=dict(INPUTS),
        grid='Each bandwidth has its own initial-training payoff spectrum scale; shared random draws. Grid not expanded on test outcomes. Boundary selections retained and reported.'))


def infer():
    m=pd.read_csv(OUT/'tables/monthly_accounting.csv',parse_dates=['return_date'])
    perf=pd.read_csv(OUT/'tables/performance_scenarios.csv')
    contrasts=[];group_rows=[]
    for scenario in ['gross','trade25_borrow30']:
        p=m[m.scenario.eq(scenario)].pivot(index='return_date',columns='policy',values='excess_return').sort_index()
        for length in PROTOCOL['bootstrap']['blocks']:
            counts=block_counts(len(p),length,PROTOCOL['bootstrap']['seed'],5000)
            for a,b,label in [('gaussian','linear','Gaussian minus Linear'),('nested_2','nested_1','Add intermediate group')]:
                boot=boot_metrics(p[[a,b]].to_numpy(),counts)['sharpe'];ci=interval(boot[:,0]-boot[:,1])
                contrasts.append(dict(scenario=scenario,block_length=length,contrast=label,
                    delta_sharpe=float(sharpe(p[a])-sharpe(p[b])),low=ci[0],high=ci[1],months=len(p),
                    inference_scope='conditional paired descriptive interval, not adjusted for multiple contrasts'))
    group=pd.read_csv(OUT/'tables/e3_monthly.csv',parse_dates=['return_date'])
    meta=pd.read_csv(OUT/'tables/e3_annual_groups.csv')
    families=pd.read_csv(OUT/'tables/monthly_economic_exposures.csv')
    exposures=families.groupby(['group','family'])[['long_mean','short_mean','long_minus_short','signed_notional']].mean().reset_index()
    save('economic_exposure_summary',exposures)
    r=group.pivot(index='return_date',columns='group',values='scaled_contribution').sort_index()
    cov=r.cov()*12
    save('group_oos_covariance',cov.reset_index().melt(id_vars='group',var_name='other_group',value_name='annual_covariance'))
    for j in range(1,5):
        gross=perf[(perf.policy==f'nested_{j}')&(perf.scenario=='gross')].iloc[0]
        net=perf[(perf.policy==f'nested_{j}')&(perf.scenario=='trade25_borrow30')].iloc[0]
        standalone=r[j]
        component=families[families.group.eq(j)].groupby('return_date').component_gross.first()
        previous=r.loc[:,:j-1].sum(axis=1) if j>1 else pd.Series(0.,index=r.index)
        group_rows.append(dict(group=j,mean_spectral_share=meta[meta.group.eq(j)].second_moment_share.mean(),
            mean_active_C=meta[meta.group.eq(j)].active_complexity.mean(),component_gross=component.mean(),
            account_gross=gross.mean_gross,annual_mean=12*standalone.mean(),annual_volatility=np.sqrt(12)*standalone.std(ddof=1),
            standalone_sharpe=sharpe(standalone),cumulative_gross_sharpe=gross.sharpe,cumulative_net_sharpe=net.sharpe,
            annual_covariance_with_previous=12*np.cov(previous,standalone,ddof=1)[0,1],
            incremental_annual_variance=12*(standalone.var(ddof=1)+2*np.cov(previous,standalone,ddof=1)[0,1]),
            annual_net_turnover=net.annual_turnover,annual_net_cost=net.annual_trading_fees+net.annual_borrow_fees))
    save('spectral_value',group_rows);save('paired_contrasts',contrasts)
    # Same contrast months in the predeclared nine decision-year blocks.
    sub=[]
    for lo,hi in PROTOCOL['blocks']:
        p=m[m.decision_year.between(lo,hi)&m.scenario.eq('trade25_borrow30')].pivot(index='return_date',columns='policy',values='excess_return')
        for a,b in [('gaussian','linear'),('nested_2','nested_1')]:sub.append(dict(period=f'{lo}-{hi}',contrast=a+' minus '+b,
            delta_sharpe=float(sharpe(p[a])-sharpe(p[b])),months=len(p)))
    save('subperiod_contrasts',sub)
    # Factor regressions: actual contributions and their pretest estimated hedges.
    f=pd.read_csv(OUT/'tables/french_factors.csv',parse_dates=['return_date']).set_index('return_date')
    reg=[]
    hedge=pd.read_csv(OUT/'tables/e3_training_factor_loadings.csv')
    gh=group.copy()
    for j in range(1,5):
        yy=r[j];beta,se,r2=hac_regression(yy,f.loc[r.index,FACTORS])
        for name,b,s in zip(['alpha_monthly']+FACTORS,beta,se):reg.append(dict(group=j,factor=name,coefficient=b,se_hac12=s,r_squared=r2,
            interpretation='full-sample descriptive regression, not a deployed hedge'))
        for year in gh.decision_year.unique():
            mask=gh.group.eq(j)&gh.decision_year.eq(year)
            slopes=hedge[hedge.group.eq(j)&hedge.decision_year.eq(year)&hedge.kind.eq('contribution')].set_index('factor').loc[FACTORS].loading.to_numpy()
            gh.loc[mask,'past_only_hedged_contribution']=gh.loc[mask,'scaled_contribution'].to_numpy()-f.loc[gh.loc[mask,'return_date'],FACTORS].to_numpy()@slopes
    save('group_factor_regressions',reg);save('group_past_only_hedged_monthly',gh)
    if (OUT/'tables/neural_performance_seed_0.csv').exists():
        nn=pd.read_csv(OUT/'tables/neural_performance_seed_0.csv')
        save('table1_performance',pd.concat([perf[perf.policy.isin(['linear','gaussian','matern32'])],nn]))
        seeds=[]
        for seed in PROTOCOL['neural']['seeds']:
            p=OUT/'tables'/f'neural_performance_seed_{seed}.csv'
            if p.exists():seeds.append(pd.read_csv(p).assign(seed=seed))
        save('neural_seed_stability',pd.concat(seeds))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['kernels','costs','bandwidth','inference'],required=True)
    a=p.parse_args()
    {'kernels':kernel_study,'costs':costs_and_economics,'bandwidth':bandwidth_study,'inference':infer}[a.phase]()
