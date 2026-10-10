"""Past-only lengthscale/spectral investigation on the frozen real JKP caches.

python3 -m empirical_final.lengthscale_spectral --phase all
Public exports contain only portfolio/operator aggregates. Holdings stay private.
"""
from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bandwidth_tuning import MULTIPLIERS, select_bandwidth
from data_pipeline import digest
from kernels import FeatureBank, array_hash, cosine
from portfolio import annual_splits, complexity_grid, dual_path, fit_windows, sharpe
from empirical_final.core import accounting_step, performance
from empirical_final.run import ROOT, INPUTS, bind, clean_manifest, public, read_json
from empirical_final.theory_guided_lambda import (anchor_from_history, choose_lambda, complexity,
                                                 sensitivity, spectrum, write_json, save)
from empirical_final.three_experiments import block_counts, boot_metrics

OUT = ROOT/'outputs/lengthscale_spectral_constant_20261010'
KERNELS = ('matern32','gaussian')
RANGES = ((1,10),(11,40),(41,100),(11,60),(61,120),(121,180))
PRIMARY_RANGE = (11,60)
BASE = 2
PRIMARY_METHODS = ('fixed_lambda','annual_cv','one_anchor','joint_cv','spectral_transfer','adaptive_spectral')
SOURCES = ('kernels.py','portfolio.py','bandwidth_tuning.py','empirical_final/theory_guided_lambda.py',
           'empirical_final/core.py','empirical_final/run.py','empirical_final/three_experiments.py')


def log_kappa(b):
    if not np.isfinite(b) or b <= 1:
        raise ValueError('The infinite-spectrum asymptotic requires b>1; do not clip b')
    angle = np.pi*(b-1)/b if b < 2 else np.pi/b
    return float(np.log(np.pi)-np.log(b)-np.log(np.sin(angle)))


def bound_penalty(c, b, T, rho):
    if not np.isfinite([c,b,T,rho]).all() or min(c,T,rho) <= 0:
        raise ValueError('Positive finite amplitude, T and ratio required')
    loglam = b/(b+1)*(np.log(rho)+log_kappa(b)-np.log(b)+np.log(c)/b-np.log(T))
    return float(np.exp(loglam))


def transfer_penalty(anchor, T0, T, log_c, log_c0, b, rho_ratio=1., b_current=None):
    log_kappa(b)
    if b_current is not None and not np.isclose(b_current,b,rtol=0,atol=1e-12):
        raise ValueError('Variable b cannot be inserted into a fixed-b transfer ratio')
    if not np.isfinite([anchor,T0,T,log_c,log_c0,b,rho_ratio]).all() or min(anchor,T0,T,rho_ratio) <= 0:
        raise ValueError('Invalid transfer inputs')
    return float(np.exp(np.log(anchor)+b/(b+1)*np.log(T0/T)+(log_c-log_c0)/(b+1)
                        +b/(b+1)*np.log(rho_ratio)))


def rank_fit(mu, first, last):
    """Descriptive finite-rank regression, never a population tail estimator."""
    mu = np.asarray(mu,float)
    if not 1 <= first < last <= len(mu) or np.any(mu[first-1:last] <= 0):
        raise ValueError('Requested positive rank range is unavailable')
    x = np.log(np.arange(first,last+1)); y = np.log(mu[first-1:last])
    slope, intercept = np.polyfit(x,y,1)
    residual = y-intercept-slope*x
    sse = float(residual@residual)
    sst = float(np.sum((y-y.mean())**2))
    se = np.sqrt(sse/(len(x)-2)/np.sum((x-x.mean())**2)) if len(x)>2 else 0.
    return dict(b=float(-slope), log_c=float(intercept), r_squared=1-sse/sst if sst else 1.,
                max_abs_log_residual=float(np.max(np.abs(residual))),
                descriptive_ols_slope_se=float(se), first_rank=first,last_rank=last,fit_ranks=len(x))


def log_amplitude(mu, b, ranks=PRIMARY_RANGE, estimator='median'):
    first,last = ranks
    if last > len(mu) or np.any(np.asarray(mu)[first-1:last] <= 0):
        raise ValueError('Unavailable amplitude ranks')
    value = np.log(np.asarray(mu)[first-1:last])+b*np.log(np.arange(first,last+1))
    if estimator not in ('median','mean'):
        raise ValueError('Unknown amplitude statistic')
    return float(np.median(value) if estimator=='median' else value.mean())


def choose_from_oos(raw_history, return_dates, cutoff, months=60, decision_history=None):
    """The selector receives only previous realized candidate-policy returns."""
    r = np.asarray(raw_history,float); dates = pd.DatetimeIndex(return_dates)
    if r.ndim != 2 or r.shape[1] != 5 or len(r) != len(dates) or not np.isfinite(r).all():
        raise ValueError('Aligned finite five-candidate history required')
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Sorted unique candidate return dates required')
    eligible = dates <= pd.Timestamp(cutoff)
    if decision_history is not None:
        eligible &= np.asarray(decision_history) >= 1979
    use = np.flatnonzero(eligible)[-months:]
    if len(use) != months:
        raise ValueError('Insufficient genuine post-anchor candidate OOS history')
    losses = np.mean((1-r[use])**2,axis=0)
    order = sorted(range(5),key=lambda j:(abs(np.log2(MULTIPLIERS[j])),MULTIPLIERS[j]))
    choice = min(order,key=lambda j:losses[j])
    return choice, dict(history_months=months,selection_first_return=str(dates[use[0]].date()),
        selection_last_return=str(dates[use[-1]].date()),selection_loss=float(losses[choice]),
        selection_cutoff=str(pd.Timestamp(cutoff).date()))


def private_path(out):
    return ROOT/'results'/out.name


def load_cache(kernel, clean):
    meta = read_json(ROOT/f'results/bandwidth_tuning/{kernel}/cache/progress.json')
    identity = meta['identity']
    path = bind(ROOT/f'results/bandwidth_tuning/{kernel}/cache/managed.npy',meta['sha256'])
    source = read_json(ROOT/f'results/final/public/{kernel}/seed_0/source.json')
    dates = pd.date_range('1963-01-31','2024-12-31',freq='ME')
    if meta['status']!='complete' or identity['clean_manifest_sha256']!=digest(ROOT/'data/clean/manifest.json'):
        raise ValueError('Cache not complete or wrong clean input')
    if identity['feature_bank'] != source['feature_bank'] or identity['multipliers'] != list(MULTIPLIERS):
        raise ValueError('Feature-bank or lengthscale identity mismatch')
    if not pd.DatetimeIndex(identity['dates']).equals(dates):
        raise ValueError('Formation dates differ')
    g = np.load(path,mmap_mode='r',allow_pickle=False)
    if g.shape != (5,744,10000) or g.dtype != np.float64 or not np.isfinite(g).all():
        raise ValueError('Wrong cache shape, units or precision')
    return g,dates,meta


def setup(out):
    protocol_path = out/'audit/protocol.json'
    if not protocol_path.exists():
        protocol = read_json(OUT/'audit/protocol.json')
        for part in ('audit','tables','figures','publication'):
            (out/part).mkdir(parents=True,exist_ok=True)
        write_json(protocol_path,protocol)
    protocol = json.loads(protocol_path.read_text())
    for source in SOURCES:
        bind(ROOT/source)
    bind(ROOT/'empirical_final/compact_protocol.json')  # Inspected state, not a runtime dependency.
    for p in ('paper/theory/direct_port_learning.tex','paper/theory/economomic_spectrum_final.tex',
              'paper/theory/proofs_learning.tex','paper/theory/unbounded_concentration.tex'):
        bind(ROOT/p)
    return protocol


def estimate(out):
    if (out/'audit/estimation.json').exists():
        raise ValueError('Completed estimation already frozen; use a fresh --output')
    started=time.monotonic();protocol=setup(out);clean=clean_manifest()
    kappa = read_json(ROOT/'results/final/public/linear/seed_0/calibration.json')['scale']
    rf = pd.read_csv(bind(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    prior_monthly = pd.read_csv(bind(ROOT/'outputs/bandwidth_tuning/monthly.csv'),parse_dates=['return_date'])
    prior_selection = pd.read_csv(bind(ROOT/'outputs/bandwidth_tuning/selections.csv'))
    prior_spectral = pd.read_csv(bind(ROOT/'outputs/theory_guided_lambda_20261010/tables/monthly_oos.csv'),parse_dates=['return_date'])
    prior_anchor = read_json(ROOT/'outputs/theory_guided_lambda_20261010/audit/anchor.json')
    private=private_path(out);private.mkdir(parents=True,exist_ok=True)
    annual=[];monthly=[];candidate=[];spectra=[];fits=[];ratio_summary=[];hindsight=[];grid_paths=[];anchors={};checks=[];grids_meta={}
    coefficient_hashes={}
    for kernel in KERNELS:
        gs,dates,meta=load_cache(kernel,clean)
        grams=[np.asarray(g)@np.asarray(g).T for g in gs]
        grids=[complexity_grid(np.asarray(g[:120])) for g in gs]
        folder,manifest=public(kernel)
        np.testing.assert_allclose(grids[BASE],manifest['lambda_grid'],rtol=2e-7,atol=0)
        original=pd.read_csv(bind(folder/'monthly.csv'),parse_dates=['return_date']).set_index('return_date')
        independent = {r['year']:r for r in fit_windows(gs[BASE],dates,grids[BASE])}
        anchor=anchor_from_history(gs[BASE],dates,grids[BASE]);rho=anchor['rho0'];lambda0=anchor['lam0']
        mu0,_=spectrum(grams[BASE][:180,:180]);anchor['feature_bank']=meta['identity']['feature_bank']
        if kernel=='gaussian':
            np.testing.assert_allclose([rho,lambda0],[prior_anchor['rho0'],prior_anchor['lam0']],rtol=1e-11,atol=0)
        empirical_b=rank_fit(mu0,*PRIMARY_RANGE)['b']
        if empirical_b is not None:
            initial_fits=[rank_fit(mu0,*r) for r in RANGES if r[0]>=11 and r[1]<=120]
            cross_b=[rank_fit(spectrum(gr[:180,:180])[0],*PRIMARY_RANGE)['b'] for gr in grams]
            primary_fit=rank_fit(mu0,*PRIMARY_RANGE)
            gate=protocol['population_amplitude_gate']
            anchor['finite_range_b']=empirical_b
            anchor['amplitude_gate_passed']=bool(primary_fit['r_squared']>=gate['first_history_minimum_R_squared'] and
                primary_fit['max_abs_log_residual']<=gate['maximum_absolute_log_residual'] and
                np.ptp([f['b'] for f in initial_fits])<=gate['maximum_slope_spread_across_ranges'] and
                np.ptp(cross_b)<=gate['maximum_slope_spread_across_lengths'])
            anchor['b_is_population_exponent']=False
            anchor['amplitude_status']='Finite-rank proxy; no population constant identified from finite history'
        anchors[kernel]=anchor
        grids_meta[kernel]=[dict(multiplier=m,penalties=g.tolist()) for m,g in zip(MULTIPLIERS,grids)]
        rho_endpoints=[]
        _,cv0,_=select_bandwidth(grams,grids,next(annual_splits(dates)),MULTIPLIERS)
        for bi in range(5):
            mm,_=spectrum(grams[bi][:180,:180]);lam=grids[bi][cv0[bi]]
            rho_endpoints.append(180/sensitivity(mm,lam))
            anchor.setdefault('additional_1978_cv_information',[]).append(dict(multiplier=MULTIPLIERS[bi],
                penalty=float(lam),rho_foc_endpoint=float(rho_endpoints[-1]),
                boundary=cv0[bi] in (0,119),interpretation='Extra CV information; boundary FOC values identify only an interval endpoint'))
        history={};history_dates=[];history_years=[]
        max_baseline=max_joint=max_spectral=0.
        for split in annual_splits(dates):
            year,tr,va,te=split;h=np.r_[tr,va];T=len(h);cutoff=dates[te[0]];rdates=dates[te]+pd.offsets.MonthEnd(1)
            joint,cv_choices,losses=select_bandwidth(grams,grids,split,MULTIPLIERS)
            selected_adaptive={}
            if year>=1984:
                for rule in ('spectral','transfer','transfer_nominal','transfer_b15','transfer_b2','transfer_b3','transfer_adjusted'):
                    if rule in history:
                        selected_adaptive[rule]=choose_from_oos(np.concatenate(history[rule]),history_dates,cutoff,60,history_years)
                for rule in ('spectral','transfer'):
                    if rule in history:
                        selected_adaptive[rule+'_36']=choose_from_oos(np.concatenate(history[rule]),history_dates,cutoff,36,history_years)
            current={};rules_by_band={};beta_by_band={};cv_selected=[]
            for bi,(g,gram,grid) in enumerate(zip(gs,grams,grids)):
                mu,psd=spectrum(gram[np.ix_(h,h)])
                spectra.extend(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],T=T,rank=k+1,mu=float(v)) for k,v in enumerate(mu))
                if kernel in KERNELS:
                    for first,last in RANGES:
                        fit=rank_fit(mu,first,last)
                        fits.append(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],T=T,P=10000,
                            threshold=1e-12,active_rank=psd['active_rank'],cutoff=float(mu.max()*1e-12),**fit,
                            common_b=empirical_b,log_c_common_mean=log_amplitude(mu,empirical_b,(first,last),'mean'),
                            log_c_common_median=log_amplitude(mu,empirical_b,(first,last))))
                spec=choose_lambda(mu,T,rho,float(grid.min()),float(grid.max()))
                rules={'spectral':spec['penalty'],'fixed_lambda':lambda0};rule_exponents={}
                if kernel in KERNELS:
                    transfer_specs=[('transfer',empirical_b,PRIMARY_RANGE,'median',1.),
                        ('transfer_nominal',1+3/130,PRIMARY_RANGE,'median',1.),
                        ('transfer_b15',1.5,PRIMARY_RANGE,'median',1.),('transfer_b2',2.,PRIMARY_RANGE,'median',1.),
                        ('transfer_b3',3.,PRIMARY_RANGE,'median',1.),('transfer_ols',empirical_b,PRIMARY_RANGE,'mean',1.),
                        ('transfer_rank_11_40',empirical_b,(11,40),'median',1.),
                        ('transfer_rank_41_100',empirical_b,(41,100),'median',1.),
                        ('transfer_rank_61_120',empirical_b,(61,120),'median',1.),
                        ('transfer_adjusted',empirical_b,PRIMARY_RANGE,'median',rho_endpoints[bi]/rho)]
                    for name,b,ranks,stat,rhoratio in transfer_specs:
                        if kernel=='gaussian' and name=='transfer_nominal':continue
                        if b<=1:continue  # No artificial clipping or forced asymptotic theorem.
                        rule_exponents[name]=b
                        rules[name]=transfer_penalty(lambda0,180,T,log_amplitude(mu,b,ranks,stat),
                            log_amplitude(mu0,b,ranks,stat),b,rhoratio)
                clipped={name:float(np.clip(lam,grid.min(),grid.max())) for name,lam in rules.items()}
                cv=grids[bi][cv_choices[bi]]
                names=list(clipped)
                alpha,_,C=dual_path(gram[np.ix_(h,h)],np.r_[grid,[clipped[n] for n in names]])
                raw=gram[np.ix_(te,h)]@alpha
                gridraw=raw[:,:120]
                betas=np.asarray(g[h]).T@alpha[:,120:]
                beta_by_band[bi]={name:betas[:,i] for i,name in enumerate(names)}
                rules_by_band[bi]={name:dict(penalty=clipped[name],unconstrained_penalty=rules[name],C=float(C[120+i]),
                    boundary='lower' if rules[name]<grid.min() else 'upper' if rules[name]>grid.max() else
                             spec['boundary'] if name=='spectral' else 'interior',
                    kkt_residual=spec['kkt_residual'] if name=='spectral' else None,
                    derivative_residual=spec['derivative_residual'] if name=='spectral' else None,
                    fixed_b=rule_exponents.get(name))
                    for i,name in enumerate(names)}
                for i,name in enumerate(names):current.setdefault(name,np.zeros((12,5)))[:,bi]=raw[:,120+i]
                cvraw=gridraw[:,cv_choices[bi]]
                # Reuse estimator for the selected CV coefficients, not a copied ridge formula.
                beta_cv=np.asarray(g[h]).T@alpha[:,cv_choices[bi]]
                beta_by_band[bi]['cv']=beta_cv
                rules_by_band[bi]['cv']=dict(penalty=float(cv),unconstrained_penalty=float(cv),C=float(C[cv_choices[bi]]),
                    boundary='lower' if cv_choices[bi]==0 else 'upper' if cv_choices[bi]==119 else 'interior',
                    kkt_residual=None,derivative_residual=None,fixed_b=None)
                current.setdefault('cv',np.zeros((12,5)))[:,bi]=cvraw
                cv_selected.append(cv)
                bestq=int(np.argmin(np.mean((1-gridraw)**2,axis=0)))
                bestsr=int(np.nanargmax(sharpe(gridraw)))
                for i,date in enumerate(rdates):
                    grid_paths.append(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],return_date=date,
                        **{f'lambda_{j:03}':float(gridraw[i,j]) for j in range(120)}))
                for name in ['cv',*names]:
                    info=rules_by_band[bi][name]
                    vals=cvraw if name=='cv' else current[name][:,bi]
                    implied=T/sensitivity(mu,info['penalty'])
                    hindsight.append(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],rule=name,
                        penalty=info['penalty'],cv_penalty=float(cv),hindsight_loss_penalty=float(grid[bestq]),
                        hindsight_sharpe_penalty=float(grid[bestsr]),log_error_cv=float(np.log(info['penalty']/cv)),
                        log_error_hindsight_loss=float(np.log(info['penalty']/grid[bestq])),
                        log_error_hindsight_sharpe=float(np.log(info['penalty']/grid[bestsr])),
                        annual_oos_loss=float(np.mean((1-vals)**2)),annual_oos_sharpe=float(sharpe(vals)),
                        best_grid_loss=float(np.mean((1-gridraw[:,bestq])**2)),best_grid_sharpe=float(sharpe(gridraw[:,bestsr])),
                        rho_foc_implied=implied,rho_ratio_to_anchor=implied/rho,
                        status='Hindsight grid diagnostic, not a population oracle'))
                for i,date in enumerate(rdates):
                    for name in ['cv',*names]:
                        candidate.append(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],rule=name,
                            formation_date=dates[te[i]],return_date=date,raw_return=current[name][i,bi],
                            penalty=rules_by_band[bi][name]['penalty'],C=rules_by_band[bi][name]['C']))
            # Every adaptive choice was made above, before current-year candidate payoffs were evaluated.
            methods={'fixed_lambda':('fixed_lambda',BASE),'annual_cv':('cv',BASE),'one_anchor':('spectral',BASE),'joint_cv':('cv',joint)}
            if 'transfer' in current: methods['transfer_fixed_length']=('transfer',BASE)
            if year>=1984:
                for key,(bi,_) in selected_adaptive.items():
                    rule=key.removesuffix('_36')
                    method='adaptive_spectral' if key=='spectral' else 'spectral_transfer' if key=='transfer' else 'adaptive_'+key
                    methods[method]=(rule,bi)
            selected_betas=[];method_names=[];bands=[]
            for method,(rule,bi) in methods.items():
                info=rules_by_band[bi][rule];vals=current[rule][:,bi];beta=beta_by_band[bi][rule]
                # Past score/norm diagnostics are not estimators of the population bound constants.
                gg=np.asarray(gs[bi][h]);residual=1-gg@beta;score=gg*residual[:,None]
                score_mean=score.mean(axis=0)
                selection_key='spectral' if method=='adaptive_spectral' else 'transfer' if method=='spectral_transfer' else method.removeprefix('adaptive_')
                selection_info=selected_adaptive[selection_key][1] if selection_key in selected_adaptive else {}
                annual.append(dict(kernel=kernel,year=year,method=method,T=T,multiplier=MULTIPLIERS[bi],
                    lengthscale=meta['identity']['feature_bank']['ell']*MULTIPLIERS[bi],rho0=rho,**info,
                    first_formation=dates[h[0]],last_formation=dates[h[-1]],last_known_payoff=cutoff,
                    first_test_return=rdates[0],last_test_return=rdates[-1],
                    regularized_policy_norm_squared=float(beta@beta),
                    regularized_score_covariance_trace=float(np.mean(np.sum(score*score,axis=1))-score_mean@score_mean),
                    **selection_info,**performance(kappa*vals,kappa*vals+rf.loc[rdates].to_numpy())))
                for i,date in enumerate(rdates):
                    monthly.append(dict(kernel=kernel,method=method,year=year,formation_date=dates[te[i]],return_date=date,
                        raw_return=float(vals[i]),excess_return=float(kappa*vals[i]),total_return=float(kappa*vals[i]+rf.loc[date]),
                        rf=float(rf.loc[date]),kappa=kappa,penalty=info['penalty'],C=info['C'],multiplier=MULTIPLIERS[bi]))
                selected_betas.append(beta);method_names.append(method);bands.append(bi)
            # Controlled representation-versus-shrinkage cross-comparisons: no causal decomposition.
            for method,bi,penalty in [('joint_length_base_cv_lambda',joint,cv_selected[BASE]),
                                     ('base_length_joint_cv_lambda',BASE,cv_selected[joint])]:
                a,_,c=dual_path(grams[bi][np.ix_(h,h)],[penalty]);vals=grams[bi][np.ix_(te,h)]@a[:,0]
                for i,date in enumerate(rdates):monthly.append(dict(kernel=kernel,method=method,year=year,
                    formation_date=dates[te[i]],return_date=date,raw_return=float(vals[i]),excess_return=float(kappa*vals[i]),
                    total_return=float(kappa*vals[i]+rf.loc[date]),rf=float(rf.loc[date]),kappa=kappa,
                    penalty=float(penalty),C=float(c[0]),multiplier=MULTIPLIERS[bi]))
                annual.append(dict(kernel=kernel,year=year,method=method,T=T,multiplier=MULTIPLIERS[bi],
                    lengthscale=meta['identity']['feature_bank']['ell']*MULTIPLIERS[bi],penalty=float(penalty),C=float(c[0]),
                    rho0=rho,boundary='controlled_cross_comparison',last_known_payoff=cutoff,
                    first_test_return=rdates[0],last_test_return=rdates[-1],**performance(kappa*vals,kappa*vals+rf.loc[rdates].to_numpy())))
            beta_file=private/f'{kernel}_{year}.npz'
            np.savez(beta_file,beta=np.column_stack(selected_betas),methods=np.array(method_names),bands=np.array(bands))
            coefficient_hashes[beta_file.name]=digest(beta_file)
            baseline=current['cv'][:,BASE];joint_raw=current['cv'][:,joint]
            np.testing.assert_allclose(baseline,independent[year]['test_returns'][:,independent[year]['choice']],rtol=1e-7,atol=1e-9)
            np.testing.assert_allclose(baseline,original.loc[rdates].raw_excess_return,rtol=2e-6,atol=2e-7)
            max_baseline=max(max_baseline,float(np.max(np.abs(baseline-original.loc[rdates].raw_excess_return))))
            old=prior_monthly[prior_monthly.kernel.eq(kernel)&prior_monthly['mode'].eq('tuned')&prior_monthly.year.eq(year)].sort_values('return_date')
            np.testing.assert_array_equal(rdates,old.return_date)
            np.testing.assert_allclose(joint_raw,old.raw_excess_return,rtol=1e-7,atol=1e-9)
            oldsel=prior_selection[prior_selection.kernel.eq(kernel)&prior_selection['mode'].eq('tuned')&prior_selection.year.eq(year)].iloc[0]
            assert oldsel.bandwidth_multiplier==MULTIPLIERS[joint]
            np.testing.assert_allclose(cv_selected[joint],oldsel['lambda'],rtol=1e-8,atol=0)
            max_joint=max(max_joint,float(np.max(np.abs(joint_raw-old.raw_excess_return))))
            if kernel=='gaussian':
                old=prior_spectral[prior_spectral.decision_year.eq(year)].sort_values('return_date')
                np.testing.assert_allclose(current['spectral'][:,BASE],old.spectral_raw,rtol=1e-7,atol=1e-9)
                max_spectral=max(max_spectral,float(np.max(np.abs(current['spectral'][:,BASE]-old.spectral_raw))))
            for rule,value in current.items():history.setdefault(rule,[]).append(value)
            history_dates.extend(rdates);history_years.extend([year]*12)
            if year%5==0 or year in (1978,1984,2024):print(f'{kernel} {year}: refit {T}, joint multiplier {MULTIPLIERS[joint]}, adaptive choices { {k:MULTIPLIERS[v[0]] for k,v in selected_adaptive.items()} }',flush=True)
        checks.append(dict(kernel=kernel,years=47,months=564,maximum_original_payoff_error=max_baseline,
            maximum_joint_cv_payoff_error=max_joint,maximum_prior_spectral_payoff_error=max_spectral,
            original_cv_and_joint_selections_reproduced=True))
        diagnostic_sensitivities(gs,dates,kernel,protocol,fits,ratio_summary)
    for name,rows in [('annual_policies',annual),('monthly_policies',monthly),('candidate_policies',candidate),
                      ('empirical_spectra',spectra),('spectral_fits',fits),('spectral_ratio_diagnostics',ratio_summary),
                      ('penalty_prediction_diagnostics',hindsight),('candidate_cv_grid_paths',grid_paths)]:
        save(out/'tables'/f'{name}.csv',rows)
    write_json(out/'audit/anchors.json',anchors);write_json(out/'audit/penalty_grids.json',grids_meta)
    write_json(out/'audit/estimation.json',dict(passed=True,elapsed_seconds=time.monotonic()-started,
        source_hashes={s:digest(ROOT/s) for s in SOURCES},module_sha256=digest(Path(__file__)),
        baseline_checks=checks,coefficient_hashes=coefficient_hashes,kappa=kappa,python=platform.python_version()))
    write_json(out/'audit/input_hashes.json',dict(INPUTS))


def diagnostic_sensitivities(gs,dates,kernel,protocol,fits,ratios):
    for year in protocol['diagnostic_years']:
        cutoff=(year-1963)*12
        histories=[('expanding',np.arange(cutoff))]+[(str(T),np.arange(cutoff-T,cutoff)) for T in protocol['history_sensitivities'] if T<=cutoff]
        for history,h in histories:
            for P in protocol['feature_prefixes']:
                values=[]
                for g in gs:
                    small=np.asarray(g[h,:P])*np.sqrt(10000/P)
                    values.append(spectrum(small@small.T)[0])
                for bi,mu in enumerate(values):
                    for threshold in protocol['rank_thresholds']:
                        n=int((mu>mu.max()*threshold).sum())
                        for first,last in RANGES:
                            if last>n:continue
                            if kernel in KERNELS:
                                fits.append(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],T=len(h),P=P,
                                    history=history,threshold=threshold,active_rank=n,cutoff=float(mu.max()*threshold),
                                    **rank_fit(mu,first,last)))
                    for first,last in [(1,10),(11,60),(61,min(180,len(h)))]:
                        threshold=1e-12;mask=(mu[first-1:last]>mu.max()*threshold)&(values[BASE][first-1:last]>values[BASE].max()*threshold)
                        logratio=np.log(mu[first-1:last][mask]/values[BASE][first-1:last][mask])
                        if not len(logratio):continue
                        x=np.log(np.arange(first,last+1)[mask])
                        slope=float(np.polyfit(x,logratio,1)[0]) if len(x)>1 else 0.
                        center=float(np.median(logratio));dispersion=float(np.sqrt(np.mean((logratio-center)**2)))
                        ratios.append(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],history=history,T=len(h),P=P,
                            first_rank=first,last_rank=last,valid_ranks=len(logratio),log_median_ratio=center,
                            log_ratio_rms_deviation=dispersion,log_ratio_slope=slope,
                            ratio_90_over_10=float(np.exp(np.quantile(logratio,.9)-np.quantile(logratio,.1))),
                            predicted_matern_tail_ratio=MULTIPLIERS[bi]**-3 if kernel=='matern32' else None))


def verify_frozen(out):
    audit=json.loads((out/'audit/estimation.json').read_text())
    for name,expected in json.loads((out/'audit/input_hashes.json').read_text()).items():bind(ROOT/name,expected)
    for name,expected in audit['source_hashes'].items():bind(ROOT/name,expected)
    if digest(Path(__file__))!=audit['module_sha256']:
        raise ValueError('Estimation implementation changed; use a new run')
    return audit


def shared_scores(x, bank, betas, bands):
    """Same shared bank/scaling as managed_bandwidths, with existing float64 cosine."""
    weights=np.zeros((len(x),betas.shape[1]))
    for start in range(0,len(x),64):
        stop=start+64;projection=x[start:stop]@bank.frequencies.T
        for bi in sorted(set(bands)):
            columns=np.flatnonzero(np.asarray(bands)==bi)
            features=np.sqrt(2/10000)*cosine(projection/MULTIPLIERS[bi]+bank.phases)
            weights[start:stop,columns]=features@betas[:,columns]
    return weights/len(x)


def account(target,returns,rf,state,previous):
    ids=target.index.union(state.index);aligned=target.reindex(ids,fill_value=0).to_numpy()
    step=accounting_step(aligned,returns.reindex(ids).to_numpy(),rf,state.reindex(ids,fill_value=0).to_numpy(),.0025,.003)
    drift=pd.Series(step.pop('next_drift'),index=ids)[aligned!=0]
    turnover=float(target.subtract(previous,fill_value=0).abs().sum());short=float(np.maximum(-target,0).sum())
    gross=float(target@returns.reindex(target.index))
    common=dict(gross_exposure=float(target.abs().sum()),net_exposure=float(target.sum()),short_notional=short,
                target_turnover=turnover,gross_excess_return=gross)
    rows=[dict(scenario='gross',**common,excess_return=gross,total_return=gross+rf,turnover=turnover,
               trading_fee=0.,borrowing_fee=0.,trading_return_drag=0.,accounting_residual=0.),
          dict(scenario='target_trade25_borrow30',**common,excess_return=gross-.0025*turnover-.003/12*short,
               total_return=gross+rf-.0025*turnover-.003/12*short,turnover=turnover,trading_fee=.0025*turnover,
               borrowing_fee=.003/12*short,trading_return_drag=.0025*turnover,accounting_residual=0.),
          dict(scenario='trade25_borrow30',**(common|step))]
    return rows,drift


def stock_accounts(out):
    audit=verify_frozen(out);clean=clean_manifest();private=private_path(out);started=time.monotonic()
    monthly=pd.read_csv(out/'tables/monthly_policies.csv',parse_dates=['formation_date']).set_index(['kernel','method','formation_date'])
    anchors=json.loads((out/'audit/anchors.json').read_text());rows=[];checks=[]
    for kernel in KERNELS:
        bmeta=anchors[kernel]['feature_bank'];bank=FeatureBank(kernel,130,10000,bmeta['ell'],0)
        assert bank.metadata()==bmeta
        states={};previous={}
        for year in range(1978,2025):
            coeff=private/f'{kernel}_{year}.npz';beta=np.load(bind(coeff,audit['coefficient_hashes'][coeff.name]),allow_pickle=False)
            methods=list(beta['methods']);bands=beta['bands'];betas=beta['beta']
            leading=set(PRIMARY_METHODS)|{'transfer_fixed_length'}
            indices=[i for i,m in enumerate(methods) if m in leading]
            methods=[methods[i] for i in indices];bands=bands[indices];betas=betas[:,indices]
            item=next(i for i in clean['files'] if i['name']==f'jkp_{year}.parquet')
            frame=pd.read_parquet(bind(ROOT/'data/clean'/item['name'],item['sha256']))
            old_weights=pd.read_parquet(bind(ROOT/f'results/final/private/{kernel}/seed_0/p_10000/weights_{year}.parquet'))
            max_payoff=max_weight=max_account=0.
            for date,data in frame.groupby('eom',sort=True):
                data=data.sort_values('id').set_index('id');retdate=date+pd.offsets.MonthEnd(1)
                if not data.return_date.eq(retdate).all() or not np.isfinite(data.r).all():raise ValueError('Stock timing/payoff mismatch')
                raw_weights=shared_scores(data[clean['characteristics']].to_numpy(float),bank,betas,bands)
                old=old_weights[old_weights.formation_date.eq(date)].set_index('id').reindex(data.index)
                ci=methods.index('annual_cv')
                np.testing.assert_allclose(raw_weights[:,ci],old.raw_weight,rtol=2e-6,atol=1e-9)
                max_weight=max(max_weight,float(np.max(np.abs(raw_weights[:,ci]-old.raw_weight))))
                for mi,method in enumerate(methods):
                    target=pd.Series(audit['kappa']*raw_weights[:,mi],index=data.index)
                    expected=monthly.loc[(kernel,method,date)]
                    error=abs(float(target@data.r)-expected.excess_return);max_payoff=max(max_payoff,error)
                    np.testing.assert_allclose(target@data.r,expected.excess_return,rtol=2e-6,atol=2e-8)
                    windows=['common_1984'] if method in ('spectral_transfer','adaptive_spectral') else ['full_1978','common_1984']
                    for window in windows:
                        if window=='common_1984' and year<1984:continue
                        key=(method,window)
                        values,state=account(target,data.r,float(expected.rf),states.get(key,pd.Series(dtype=float)),previous.get(key,pd.Series(dtype=float)))
                        states[key]=state;previous[key]=target
                        for value in values:
                            max_account=max(max_account,value['accounting_residual'])
                            rows.append(dict(kernel=kernel,method=method,window=window,year=year,formation_date=date,
                                             return_date=retdate,rf=float(expected.rf),multiplier=MULTIPLIERS[int(bands[mi])],**value))
            checks.append(dict(kernel=kernel,year=year,maximum_scaled_payoff_error=max_payoff,
                               maximum_original_cv_weight_error=max_weight,maximum_accounting_residual=max_account))
            save(out/'tables/monthly_accounts.csv',rows);save(out/'tables/stock_checks.csv',checks)
            write_json(out/'audit/cost_checkpoint.json',dict(kernel=kernel,completed_year=year,elapsed_seconds=time.monotonic()-started))
            print(f'Stock accounts {kernel} {year}/2024: {len(methods)} policies, payoff error {max_payoff:.2g}',flush=True)
    frame=pd.DataFrame(rows)
    # The full-history CV account is the identical earlier exact account, with identical entry state.
    old=pd.read_csv(bind(ROOT/'outputs/final_empirical_20261008/tables/monthly_accounting.csv'),parse_dates=['return_date'])
    for kernel in KERNELS:
        actual=frame[frame.kernel.eq(kernel)&frame.method.eq('annual_cv')&frame.window.eq('full_1978')&frame.scenario.eq('trade25_borrow30')].sort_values('return_date')
        expected=old[old.kernel.eq(kernel)&old.scenario.eq('trade25_borrow30')].sort_values('return_date')
        np.testing.assert_array_equal(actual.return_date,expected.return_date)
        np.testing.assert_allclose(actual[['excess_return','turnover','trading_fee','borrowing_fee']],
                                   expected[['excess_return','turnover','trading_fee','borrowing_fee']],rtol=2e-6,atol=1e-9)
    write_json(out/'audit/cost_verification.json',dict(passed=True,actual_weights=True,original_cv_net_reproduced=True,
        primary_entry='Cash at January-close 1984 for every method',elapsed_seconds=time.monotonic()-started,
        maximum_payoff_error=max(c['maximum_scaled_payoff_error'] for c in checks),
        maximum_weight_error=max(c['maximum_original_cv_weight_error'] for c in checks),
        maximum_accounting_residual=max(c['maximum_accounting_residual'] for c in checks)))
    write_json(out/'audit/input_hashes.json',dict(INPUTS))


def summarize(out):
    audit=verify_frozen(out);protocol=json.loads((out/'audit/protocol.json').read_text())
    monthly=pd.read_csv(out/'tables/monthly_policies.csv',parse_dates=['return_date'])
    annual=pd.read_csv(out/'tables/annual_policies.csv')
    accounts=pd.read_csv(out/'tables/monthly_accounts.csv',parse_dates=['return_date'])
    if not (out/'audit/cost_verification.json').exists():raise ValueError('Complete actual-weight accounting required')
    summaries=[];comparisons=[]
    periods=[('1978-2024',1978,2024),('1979-2024',1979,2024),('1984-2024',1984,2024),
             ('1984-1999',1984,1999),('2000-2009',2000,2009),('2010-2019',2010,2019),('2020-2024',2020,2024)]
    for kernel in KERNELS:
        for label,lo,hi in periods:
            m=monthly[monthly.kernel.eq(kernel)&monthly.year.between(lo,hi)]
            required=(hi-lo+1)*12
            available={method:f.sort_values('return_date') for method,f in m.groupby('method') if len(f)==required}
            for method,f in available.items():
                a=annual[annual.kernel.eq(kernel)&annual.method.eq(method)&annual.year.between(lo,hi)]
                common=dict(kernel=kernel,method=method,period=label,months=required,first_return=f.return_date.min(),last_return=f.return_date.max(),
                            mean_C=float(a.C.mean()),mean_multiplier=float(a.multiplier.mean()))
                summaries.append(dict(**common,scenario='gross',window='raw',**performance(f.excess_return,f.total_return)))
                w='common_1984' if lo>=1984 else 'full_1978'
                sub=accounts[accounts.kernel.eq(kernel)&accounts.method.eq(method)&accounts.year.between(lo,hi)&accounts.window.eq(w)]
                for scenario,g in sub.groupby('scenario',sort=False):
                    if len(g)!=required:continue
                    g=g.sort_values('return_date')
                    exposure=dict(mean_gross=float(g.gross_exposure.mean()),mean_net=float(g.net_exposure.mean()),
                        mean_short=float(g.short_notional.mean()),annual_turnover=float(12*g.turnover.mean()),
                        annual_target_turnover=float(12*g.target_turnover.mean()),annual_trade_fees=float(12*g.trading_fee.mean()),
                        annual_borrow_fees=float(12*g.borrowing_fee.mean()))
                    if scenario=='gross':summaries[-1].update(exposure)
                    else:summaries.append(dict(**common,scenario=scenario,window=w,**exposure,**performance(g.excess_return,g.total_return)))
            for scenario in ('gross','trade25_borrow30','target_trade25_borrow30'):
                if scenario=='gross':
                    wide=m[m.method.isin(available)].pivot(index='return_date',columns='method',values='excess_return').sort_index()
                else:
                    w='common_1984' if lo>=1984 else 'full_1978'
                    sub=accounts[accounts.kernel.eq(kernel)&accounts.year.between(lo,hi)&accounts.window.eq(w)&accounts.scenario.eq(scenario)]
                    wide=sub.pivot(index='return_date',columns='method',values='excess_return').sort_index()
                wide=wide.dropna(axis=1)
                if len(wide)!=required:raise ValueError('Unexpected common period')
                observed=sharpe(wide.to_numpy());columns=list(wide.columns)
                for block in protocol['bootstrap']['blocks']:
                    counts=block_counts(required,block,protocol['bootstrap']['seed'],protocol['bootstrap']['replicates'])
                    metrics=boot_metrics(wide.to_numpy(),counts)
                    for reference in ('joint_cv','annual_cv','fixed_lambda'):
                        if reference not in columns:continue
                        ri=columns.index(reference)
                        for mi,method in enumerate(columns):
                            if method==reference:continue
                            delta=metrics['sharpe'][:,mi]-metrics['sharpe'][:,ri];low,high=np.quantile(delta,[.025,.975])
                            comparisons.append(dict(kernel=kernel,period=label,scenario=scenario,method=method,reference=reference,
                                delta_sharpe=float(observed[mi]-observed[ri]),ci_low=float(low),ci_high=float(high),
                                bootstrap_se=float(delta.std(ddof=1)),block=block,replicates=protocol['bootstrap']['replicates'],
                                seed=protocol['bootstrap']['seed'],months=required,
                                delta_annual_mean=float(12*(wide.iloc[:,mi].mean()-wide.iloc[:,ri].mean())),
                                delta_annual_volatility=float(np.sqrt(12)*(wide.iloc[:,mi].std(ddof=1)-wide.iloc[:,ri].std(ddof=1))),
                                scope='Conditional fitted paths; percentile interval; no multiplicity adjustment'))
    save(out/'tables/performance_summary.csv',summaries);save(out/'tables/paired_comparisons.csv',comparisons)
    pred=pd.read_csv(out/'tables/penalty_prediction_diagnostics.csv');rows=[]
    for (kernel,rule),f in pred[pred.year.ge(1984)].groupby(['kernel','rule']):
        rows.append(dict(kernel=kernel,rule=rule,observations=len(f),
            rmse_log_cv=float(np.sqrt(np.mean(f.log_error_cv**2))),mae_log_cv=float(np.abs(f.log_error_cv).mean()),
            rmse_log_hindsight_loss=float(np.sqrt(np.mean(f.log_error_hindsight_loss**2))),
            mean_loss_above_hindsight=float((f.annual_oos_loss-f.best_grid_loss).mean()),
            mean_rho_ratio_to_anchor=float(f.rho_ratio_to_anchor.mean())))
    save(out/'tables/prediction_summary.csv',rows)
    finite_accounts=bool(np.isfinite(accounts.select_dtypes('number')).all().all())
    if not finite_accounts:raise ValueError('Nonfinite account-export field')
    write_json(out/'audit/summary_verification.json',dict(passed=True,common_adaptive_months=492,
        common_first_return='1984-02-29',common_last_return='2025-01-31',bootstrap_conditional=True,
        rho_never_recalibrated_for_primary=True,all_account_numeric_finite=finite_accounts))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--phase',choices=['all','estimate','costs','summary','publication'],default='all')
    args=parser.parse_args(argv);out=args.output.resolve()
    with threadpool_limits(limits=1):
        if args.phase in ('all','estimate'):estimate(out)
        if args.phase in ('all','costs'):stock_accounts(out)
        if args.phase in ('all','summary'):summarize(out)
        if args.phase in ('all','publication'):
            from empirical_final.lengthscale_publication import build
            build(out)
    print(f'Finished {args.phase}: {out}',flush=True)


if __name__=='__main__':main()
