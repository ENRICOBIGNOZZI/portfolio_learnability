"""Controlled Matérn polynomial-spectrum schedules on frozen real JKP data.

Run with python3 -m empirical_final.final_theory_regularization --phase all.
Licensed panels, coefficients and signed stock holdings remain under results/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from complexity_schedule import calibrate, cache_managed, spectral_fit, window
from data_pipeline import digest
from portfolio import annual_splits, complexity_grid, dual_path, sharpe
from empirical_final.run import ROOT, INPUTS, bind, clean_manifest, read_json
from empirical_final.theory_guided_lambda import save, write_json
from bandwidth_tuning import MULTIPLIERS, select_bandwidth
from empirical_final.lengthscale_spectral import load_cache, shared_scores, account
from empirical_final.theory_guided_lambda import choose_lambda
from empirical_final.core import performance
from kernels import FeatureBank, array_hash
from complexity_schedule import nested_grid

OUT = ROOT / 'outputs/final_theory_regularization_20261011'
PRIVATE = ROOT / 'results/final_theory_regularization_20261011'
SEED = 20261011
REPLICATES = 5000
METHODS = ('A','B','C','D','annual_cv','joint_cv','fixed_lambda','one_anchor')


def slope(mu, fractions=(.1,.6)):
    """Original rank convention with explicit, unclipped inadmissibility."""
    mu=np.sort(np.asarray(mu,float))[::-1]
    if not len(mu) or not np.isfinite(mu).all() or mu[0]<=0:
        raise ValueError('Invalid spectrum')
    positive=mu[mu>mu[0]*1e-12]
    lo=max(2,int(np.ceil(fractions[0]*len(positive))))
    hi=int(np.floor(fractions[1]*len(positive)))
    if hi-lo+1<5:raise ValueError('Insufficient positive ranks')
    x=np.log(np.arange(lo,hi+1));y=np.log(positive[lo-1:hi])
    coefficient,intercept=np.polyfit(x,y,1);residual=y-intercept-coefficient*x
    b=float(-coefficient)
    return dict(b=b,r_squared=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
        rank_low=lo,rank_high=hi,positive_rank=len(positive),admissible=bool(np.isfinite(b) and b>1))


def penalty(c0,T,b):
    if not np.isfinite([c0,T,b]).all() or min(c0,T)<=0 or b<=1:
        raise ValueError('Frozen positive constant, positive T and b>1 required; no clipping')
    return float(c0*T**(-b/(b+1)))


def tie_choice(losses, eligible):
    order=sorted(eligible,key=lambda j:(abs(np.log2(MULTIPLIERS[j])),MULTIPLIERS[j]))
    if not order:raise ValueError('No admissible representation')
    return min(order,key=lambda j:losses[j])


def decompose(c0,b0,b,T,base_c0,base_b0,initial_lambda):
    """Exact additive log-penalty accounting, relative to baseline initial CV."""
    a0=base_b0/(base_b0+1);a=b/(b+1);ar=b0/(b0+1)
    sample=-a0*np.log(T/120)
    representation=np.log(c0/base_c0)-(ar-a0)*np.log(T)
    update=-(a-ar)*np.log(T)
    total=np.log(penalty(c0,T,b)/initial_lambda)
    np.testing.assert_allclose(total,sample+representation+update,atol=1e-12)
    return dict(log_sample_size_contribution=float(sample),log_representation_contribution=float(representation),
                log_b_update_contribution=float(update),log_lambda_relative_initial=float(total))


def protocol(out):
    path=out/'audit/protocol.json'
    if path.exists():return read_json(path)
    result=dict(version='final-theory-2x2-v1',kernel='matern32',ell0=4.392107784616423,
        multipliers=list(MULTIPLIERS),features=10000,characteristics=130,feature_seed=0,
        initial_training_formations=['1963-01-31','1972-12-31'],
        initial_validation_formations=['1973-01-31','1977-12-31'],initial_grid=200,
        constant_conversion_T=120,first_refit_T=180,annual_decisions=[1978,2024],
        spectral_fit='Uncentered Gram/T; positive threshold 1e-12; log-rank OLS; ceil(10%)-floor(60%); no clipping',
        rule='C0(ell) * T ** (-b/(b+1)); constants frozen once',
        chronology='Inner training precedes trailing 60 formation months; select length with inner-training lambda only; refit all past history',
        ties='Exact loss ties prefer multiplier 1, then nearest absolute log2 distance, then smaller multiplier',
        admissibility='Initial b<=1: exclude representation permanently. D inner b<=1: exclude for that decision. '
          'D selected full-refit b<=1: fall back to baseline control and record. Invalid baseline: stop. '
          'C uses admissible initial b, subsequent slopes descriptive only. No clipping, grid substitution or OOS-based exclusion.',
        exclusion_sensitivity='Report positive-b extrapolation choices and gross paths separately as outside-theorem diagnostics; never a primary strategy',
        anchor='1978 is context only; per-length calibration shares the initial validation block',
        periods=['1978-2024','1979-2024','1984-2024','2020-2024'],
        subperiods=['1979-1989','1990-1999','2000-2009','2010-2019'],
        primary='1979-2024; principal paired net comparisons D-B, D-C, D-joint_cv',
        bootstrap=dict(seed=SEED,replicates=REPLICATES,blocks=[6,12,24],
            interval='Paired circular moving-block percentile 95%; conditional on fitted paths; no strategy refitting or multiplicity correction'),
        costs=dict(trade_rate=.0025,annual_short_borrow=.003,
            implementation='Existing verified accounting_step via account; actual reconstructed signed holdings',
            state='Cash before February 1978; state carried into every reported subperiod; no terminal liquidation'),
        loss='Response-one quadratic loss in raw payoff units for gross comparisons; net loss divided by same fixed kappa',
        rolling=dict(windows=[180,240],start_formation='1984-01-31',
            pretest_cutoff='1983-12-31 formation, payoff known 1984-01-31',
            translation='Anchor each W to expanding B lambda at T=252; C0(W)=lambda_anchor * W**(b_anchor(W)/(b_anchor(W)+1)); thereafter freeze',
            rank_fractions=[[.1,.6],[.1,.5],[.2,.6],[.1,.7]],
            diagnostic_only=True,
            compelling_gate='For both W, median absolute monthly log-lambda change >= 5%, monthly median cross-band log-lambda spread <= that change, and >=99% primary b admissible; otherwise no monthly backtest'),
        manuscript='Recommend unchanged when incremental net evidence is weak, mixed or block-sensitive; no oracle or causal claim')
    write_json(path,result);return result


def load_inputs(out):
    hashes=json.loads((out/'audit/input_hashes.json').read_text())
    for name,expected in hashes.items():bind(ROOT/name,expected)
    clean=clean_manifest();gs,dates,meta=load_cache('matern32',clean)
    g,original_dates,_=cache_managed(ROOT/'data/clean',ROOT/'results/schedule_cache')
    np.testing.assert_array_equal(dates,original_dates)
    np.testing.assert_allclose(gs[2],g,rtol=1e-9,atol=1e-14)
    # Baseline uses the exact original reduction order for all four strategies.
    arrays=[np.asarray(gs[j]) if j!=2 else g for j in range(5)]
    return clean,arrays,dates,meta


def estimate(out,private):
    started=time.monotonic()
    assert read_json(out/'audit/replication.json')['passed']
    if (out/'audit/estimation.json').exists():raise FileExistsError('Completed estimation is frozen')
    predeclared=protocol(out);clean,gs,dates,meta=load_inputs(out)
    private.mkdir(parents=True,exist_ok=True)
    grams=[g@g.T for g in gs];grids=[complexity_grid(g[:120]) for g in gs]
    kappa=float(read_json(ROOT/'results/final/public/linear/seed_0/calibration.json')['scale'])
    rf=pd.read_csv(bind(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    constants=[];initial_rows=[]
    for j,g in enumerate(gs):
        grid=nested_grid(g[:120]);a,mu,_=dual_path(grams[j][:120,:120],grid)
        fit=slope(mu);loss=np.mean((1-grams[j][120:180,:120]@a)**2,axis=0);choice=int(np.argmin(loss))
        # Record the extrapolation only for the declared exclusion sensitivity.
        c0=float(grid[choice]*120**(fit['b']/(fit['b']+1))) if fit['b']>0 else None
        constants.append(dict(candidate=j,multiplier=MULTIPLIERS[j],lengthscale=predeclared['ell0']*MULTIPLIERS[j],
            **fit,initial_lambda=float(grid[choice]),C0=c0,choice=choice,validation_loss=float(loss[choice]),
            lambda_grid_boundary=choice in (0,199),minimum_eigenvalue=float(mu.min()),
            training_months=120,validation_months=60,training_last_formation=dates[119],validation_last_formation=dates[179]))
        initial_rows.extend(dict(candidate=j,multiplier=MULTIPLIERS[j],grid_index=i,penalty=float(v),
                                 validation_loss=float(loss[i]),selected=i==choice) for i,v in enumerate(grid))
    assert constants[2]['admissible']
    baseline=read_json(out/'audit/replication.json')
    np.testing.assert_allclose([constants[2]['C0'],constants[2]['b']],[baseline['C0'],baseline['b0']],rtol=1e-10)
    save(out/'tables/frozen_constants.csv',constants);save(out/'tables/initial_validation_candidates.csv',initial_rows)
    old_month=pd.read_csv(bind(ROOT/'outputs/bandwidth_tuning/monthly.csv'),parse_dates=['return_date'])
    old_annual=pd.read_csv(bind(ROOT/'outputs/bandwidth_tuning/selections.csv'))
    old_spectral=pd.read_csv(bind(ROOT/'outputs/lengthscale_spectral_constant_20261010/tables/monthly_policies.csv'),parse_dates=['return_date'])
    old_anchor=read_json(ROOT/'outputs/lengthscale_spectral_constant_20261010/audit/anchors.json')['matern32']
    original=pd.read_csv(bind(ROOT/'paper/complexity_schedule/selected_monthly.csv'),parse_dates=['return_date'])
    annual=[];monthly=[];candidates=[];checks=[];exclusions=[];coeff_hashes={};spectra=[]
    for split in annual_splits(dates):
        year,tr,va,te=split;h=np.r_[tr,va];T=len(h);n=len(tr)
        # Both operators and all candidate losses are past-only slices.
        assert max(tr)<min(va) and max(va)<min(te) and len(va)==60
        fits_inner=[];fits_full=[];mus=[];eig=[]
        loss_C=[];loss_D=[];extra_loss_D=[]
        for j,(g,gram,c) in enumerate(zip(gs,grams,constants)):
            inner=gram[np.ix_(tr,tr)];full=gram[np.ix_(h,h)]
            vi,ui=np.linalg.eigh((inner+inner.T)/2);vf,uf=np.linalg.eigh((full+full.T)/2)
            if min(vi.min(),vf.min()) < -max(vf.max(),1e-30)*1e-10:raise ValueError('Non-PSD operator')
            vi=np.maximum(vi,0);vf=np.maximum(vf,0);mi=vi[::-1]/n;mf=vf[::-1]/T
            fi=slope(mi);ff=slope(mf);fits_inner.append(fi);fits_full.append(ff);mus.append(mf);eig.append((vf,uf))
            def inner_loss(b):
                lam=c['C0']*n**(-b/(b+1))
                alpha=ui@((ui.T@np.ones(n))/(vi+n*lam))
                return float(np.mean((1-gram[np.ix_(va,tr)]@alpha)**2)),float(lam)
            lc,pc=inner_loss(c['b']);ld,pd_=inner_loss(fi['b'])
            loss_C.append(lc if c['admissible'] else np.inf)
            loss_D.append(ld if c['admissible'] and fi['admissible'] else np.inf)
            extra_loss_D.append(ld)
            candidates.append(dict(year=year,candidate=j,multiplier=MULTIPLIERS[j],C0=c['C0'],b0=c['b'],
                inner_T=n,full_T=T,inner_first_formation=dates[tr[0]],inner_last_formation=dates[tr[-1]],
                inner_last_known_payoff=dates[tr[-1]]+pd.offsets.MonthEnd(1),
                validation_first_formation=dates[va[0]],validation_last_formation=dates[va[-1]],
                decision_date=dates[te[0]],full_last_formation=dates[h[-1]],
                inner_b=fi['b'],full_b=ff['b'],inner_r_squared=fi['r_squared'],full_r_squared=ff['r_squared'],
                inner_rank_low=fi['rank_low'],inner_rank_high=fi['rank_high'],full_rank_low=ff['rank_low'],
                full_rank_high=ff['rank_high'],inner_positive_rank=fi['positive_rank'],full_positive_rank=ff['positive_rank'],
                initial_admissible=c['admissible'],inner_admissible=fi['admissible'],full_admissible=ff['admissible'],
                C_inner_lambda=pc,D_inner_lambda=pd_,C_validation_loss=lc,D_validation_loss=ld,
                C_eligible=c['admissible'],D_eligible=c['admissible'] and fi['admissible'],
                minimum_full_eigenvalue=float(mf.min())))
            spectra.extend(dict(year=year,candidate=j,rank=i+1,eigenvalue=float(v)) for i,v in enumerate(mf))
        selected_C=tie_choice(loss_C,[j for j in range(5) if np.isfinite(loss_C[j])])
        selected_D=tie_choice(loss_D,[j for j in range(5) if np.isfinite(loss_D[j])])
        original_D=selected_D
        if not fits_full[selected_D]['admissible']:
            selected_D=2
        assert fits_full[2]['admissible']
        joint,cv_choices,cv_losses=select_bandwidth(grams,grids,split,MULTIPLIERS)
        rules={'A':(2,constants[2]['b']),'B':(2,fits_full[2]['b']),
               'C':(selected_C,constants[selected_C]['b']),'D':(selected_D,fits_full[selected_D]['b'])}
        output_rules={m:(j,penalty(constants[j]['C0'],T,b),b) for m,(j,b) in rules.items()}
        output_rules.update(annual_cv=(2,float(grids[2][cv_choices[2]]),None),
            joint_cv=(joint,float(grids[joint][cv_choices[joint]]),None),
            fixed_lambda=(2,constants[2]['initial_lambda'],None),
            one_anchor=(2,choose_lambda(mus[2],T,old_anchor['rho0'],float(grids[2].min()),float(grids[2].max()))['penalty'],None))
        betas=[];bands=[];current={}
        for method in METHODS:
            j,lam,b=output_rules[method];v,u=eig[j]
            alpha=u@((u.T@np.ones(T))/(v+T*lam));beta=gs[j][h].T@alpha
            pay=grams[j][np.ix_(te,h)]@alpha;direct=gs[j][te]@beta
            np.testing.assert_allclose(pay,direct,rtol=1e-7,atol=1e-9)
            residual=np.linalg.norm(grams[j][np.ix_(h,h)]@alpha+T*lam*alpha-1)/np.sqrt(T)
            primal=np.linalg.norm(gs[j][h].T@(gs[j][h]@beta-1)+T*lam*beta)/max(np.linalg.norm(gs[j][h].T@np.ones(T)),1e-30)
            assert max(residual,primal)<1e-6
            comp=float(np.sum(mus[j]/(mus[j]+lam)))
            row=dict(year=year,method=method,T=T,inner_T=n,candidate=j,multiplier=MULTIPLIERS[j],
                     lengthscale=predeclared['ell0']*MULTIPLIERS[j],penalty=lam,complexity=comp,
                     measured_b=fits_full[j]['b'],schedule_b=b,b0=constants[j]['b'],C0=constants[j]['C0'],
                     decision_date=dates[te[0]],last_known_payoff=dates[h[-1]]+pd.offsets.MonthEnd(1),
                     quadratic_loss=float(np.mean((1-pay)**2)),gross_sharpe=float(sharpe(pay)),
                     selected_validation_loss=(loss_C[j] if method=='C' else loss_D[original_D] if method=='D' else np.nan),
                     full_refit_fallback=bool(method=='D' and original_D!=selected_D))
            if method in rules:
                row.update(decompose(constants[j]['C0'],constants[j]['b'],b,T,constants[2]['C0'],constants[2]['b'],constants[2]['initial_lambda']))
            annual.append(row);betas.append(beta);bands.append(j);current[method]=pay
            checks.append(dict(year=year,method=method,dual_relative_residual=float(residual),
                               primal_relative_residual=float(primal),direct_payoff_error=float(np.max(np.abs(pay-direct)))))
            for i,payoff in zip(te,pay):
                rd=dates[i]+pd.offsets.MonthEnd(1)
                monthly.append(dict(year=year,method=method,formation_date=dates[i],return_date=rd,
                    raw_return=float(payoff),gross_excess_return=float(kappa*payoff),rf=float(rf.loc[rd]),kappa=kappa,
                    penalty=lam,complexity=comp,multiplier=MULTIPLIERS[j]))
        # Explicitly exported outside-assumption sensitivity, never primary inference.
        extra_C=tie_choice([r['C_validation_loss'] for r in candidates[-5:]],range(5))
        extra_D=tie_choice(extra_loss_D,range(5))
        for method,j in (('C',extra_C),('D',extra_D)):
            b=constants[j]['b'] if method=='C' else fits_full[j]['b'];lam=constants[j]['C0']*T**(-b/(b+1))
            v,u=eig[j];a=u@((u.T@np.ones(T))/(v+T*lam));pay=grams[j][np.ix_(te,h)]@a
            for i,r in zip(te,pay):exclusions.append(dict(year=year,method=method,candidate=j,
                return_date=dates[i]+pd.offsets.MonthEnd(1),raw_return=float(r),
                changed_selection=bool(j!=output_rules[method][0]),outside_assumptions=bool(not constants[j]['admissible'] or not fits_inner[j]['admissible'] or not fits_full[j]['admissible'])))
        for method,mode in (('annual_cv','fixed'),('joint_cv','tuned')):
            ref=old_month[old_month.kernel.eq('matern32')&old_month['mode'].eq(mode)&old_month.year.eq(year)].sort_values('return_date')
            np.testing.assert_array_equal(ref.return_date,dates[te]+pd.offsets.MonthEnd(1))
            np.testing.assert_allclose(current[method],ref.raw_excess_return,rtol=1e-7,atol=1e-9)
            sel=old_annual[old_annual.kernel.eq('matern32')&old_annual['mode'].eq(mode)&old_annual.year.eq(year)].iloc[0]
            assert output_rules[method][0]==list(MULTIPLIERS).index(sel.bandwidth_multiplier)
            np.testing.assert_allclose(output_rules[method][1],sel['lambda'],rtol=1e-7,atol=1e-14)
        ref=original[original.return_date.isin(dates[te]+pd.offsets.MonthEnd(1))]
        np.testing.assert_allclose(current['B'],ref.raw_excess_return,rtol=1e-7,atol=1e-9)
        for method in ('fixed_lambda','one_anchor'):
            ref=old_spectral[old_spectral.kernel.eq('matern32')&old_spectral.method.eq(method)&old_spectral.year.eq(year)].sort_values('return_date')
            np.testing.assert_allclose(current[method],ref.raw_return,rtol=1e-7,atol=1e-9)
        path=private/f'coefficients_{year}.npz';np.savez(path,beta=np.column_stack(betas),methods=np.array(METHODS),bands=np.array(bands))
        coeff_hashes[path.name]=digest(path)
        print('Controlled annual fit',year,'C/D lengths',MULTIPLIERS[selected_C],MULTIPLIERS[selected_D],flush=True)
    for name,rows in (('annual_policies',annual),('monthly_payoffs',monthly),('candidate_losses',candidates),
                      ('numerical_checks',checks),('exclusion_sensitivity_monthly',exclusions),('annual_spectra',spectra)):
        save(out/'tables'/f'{name}.csv',rows)
    write_json(out/'audit/estimation.json',dict(passed=True,coefficient_hashes=coeff_hashes,kappa=kappa,
        cache_baseline_maximum_difference=float(np.max(np.abs(gs[2]-np.load(ROOT/'results/bandwidth_tuning/matern32/cache/managed.npy',mmap_mode='r')[2]))),
        feature_bank=meta['identity']['feature_bank'],baseline_array_sha256=array_hash(gs[2]),
        all_original_benchmarks_reproduced=True,constants_frozen=True,
        source_hashes={p:digest(ROOT/p) for p in ('empirical_final/final_theory_regularization.py','complexity_schedule.py','portfolio.py','kernels.py','bandwidth_tuning.py','empirical_final/core.py','empirical_final/lengthscale_spectral.py')},
        elapsed_seconds=time.monotonic()-started))
    write_json(out/'audit/input_hashes.json',dict(INPUTS))


def frozen(out):
    audit=read_json(out/'audit/estimation.json')
    for name,h in audit['source_hashes'].items():bind(ROOT/name,h)
    for name,h in json.loads((out/'audit/input_hashes.json').read_text()).items():bind(ROOT/name,h)
    return audit


def costs(out,private):
    started=time.monotonic();audit=frozen(out);clean=clean_manifest()
    if (out/'audit/costs.json').exists():raise FileExistsError('Costs already frozen')
    bank=FeatureBank('matern32',130,10000,audit['feature_bank']['ell'],0)
    assert bank.metadata()==audit['feature_bank']
    monthly=pd.read_csv(out/'tables/monthly_payoffs.csv',parse_dates=['formation_date']).set_index(['method','formation_date'])
    states={};previous={};rows=[];checks=[];holdings_hashes={}
    for year in range(1978,2025):
        name=f'coefficients_{year}.npz';coeff=np.load(bind(private/name,audit['coefficient_hashes'][name]),allow_pickle=False)
        methods=list(coeff['methods']);betas=coeff['beta'];bands=coeff['bands']
        item=next(i for i in clean['files'] if i['name']==f'jkp_{year}.parquet')
        frame=pd.read_parquet(bind(ROOT/'data/clean'/item['name'],item['sha256']))
        old=pd.read_parquet(bind(ROOT/f'results/final/private/matern32/seed_0/p_10000/weights_{year}.parquet'))
        holding_frames=[];max_payoff=max_weight=max_account=0.
        for date,data in frame.groupby('eom',sort=True):
            data=data.sort_values('id').set_index('id');rd=date+pd.offsets.MonthEnd(1)
            assert data.return_date.eq(rd).all()
            raw=shared_scores(data[clean['characteristics']].to_numpy(float),bank,betas,bands)
            reference=old[old.formation_date.eq(date)].set_index('id').reindex(data.index).raw_weight.to_numpy()
            np.testing.assert_allclose(raw[:,methods.index('annual_cv')],reference,rtol=2e-6,atol=1e-9)
            max_weight=max(max_weight,float(np.max(np.abs(raw[:,methods.index('annual_cv')]-reference))))
            holding=pd.DataFrame(audit['kappa']*raw,index=data.index,columns=methods).reset_index()
            holding.insert(0,'formation_date',date);holding_frames.append(holding)
            for mi,method in enumerate(methods):
                target=pd.Series(audit['kappa']*raw[:,mi],index=data.index);expected=monthly.loc[(method,date)]
                got=float(target@data.r);err=abs(got-expected.gross_excess_return);max_payoff=max(max_payoff,err)
                np.testing.assert_allclose(got,expected.gross_excess_return,rtol=2e-6,atol=2e-8)
                accounts,state=account(target,data.r,float(expected.rf),states.get(method,pd.Series(dtype=float)),previous.get(method,pd.Series(dtype=float)))
                states[method]=state;previous[method]=target
                for value in accounts:
                    max_account=max(max_account,value['accounting_residual'])
                    rows.append(dict(method=method,year=year,formation_date=date,return_date=rd,rf=float(expected.rf),
                                     kappa=audit['kappa'],raw_return=float(expected.raw_return),**value))
        path=private/f'holdings_{year}.parquet';pd.concat(holding_frames,ignore_index=True).to_parquet(path,index=False)
        holdings_hashes[path.name]=digest(path)
        checks.append(dict(year=year,maximum_payoff_error=max_payoff,maximum_cv_weight_error=max_weight,maximum_accounting_residual=max_account))
        save(out/'tables/monthly_accounts.csv',rows);save(out/'tables/holdings_checks.csv',checks)
        write_json(out/'audit/cost_checkpoint.json',dict(completed_year=year,elapsed_seconds=time.monotonic()-started))
        print('Actual holdings and accounts',year,flush=True)
    actual=pd.DataFrame(rows)
    ref=pd.read_csv(bind(ROOT/'outputs/final_empirical_20261008/tables/monthly_accounting.csv'),parse_dates=['return_date'])
    a=actual[actual.method.eq('annual_cv')&actual.scenario.eq('trade25_borrow30')].sort_values('return_date')
    b=ref[ref.kernel.eq('matern32')&ref.scenario.eq('trade25_borrow30')].sort_values('return_date')
    np.testing.assert_array_equal(a.return_date,b.return_date)
    np.testing.assert_allclose(a[['excess_return','turnover','trading_fee','borrowing_fee']],b[['excess_return','turnover','trading_fee','borrowing_fee']],rtol=2e-6,atol=1e-9)
    write_json(out/'audit/costs.json',dict(passed=True,actual_signed_holdings=True,holdings_hashes=holdings_hashes,
        original_cv_net_reproduced=True,entry_state='Cash February 1978; carried into subperiods',
        maximum_payoff_error=max(r['maximum_payoff_error'] for r in checks),
        maximum_cv_weight_error=max(r['maximum_cv_weight_error'] for r in checks),
        maximum_accounting_residual=max(r['maximum_accounting_residual'] for r in checks),
        elapsed_seconds=time.monotonic()-started))
    write_json(out/'audit/input_hashes.json',dict(INPUTS))


def replicate(out):
    """Hard gate: reproduce original function calls before any new strategy."""
    started = time.monotonic()
    if (out/'audit/replication.json').exists():
        raise FileExistsError('Replication already frozen; use another output directory')
    for part in ('audit', 'tables', 'figures', 'publication'):
        (out/part).mkdir(parents=True, exist_ok=True)
    write_json(out/'audit/starting_state.json', dict(
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        unrelated_status=subprocess.check_output(['git','status','--short'],text=True),
        baseline_tolerance=dict(rtol=1e-7,atol=1e-9)))
    meta = read_json(ROOT/'paper/complexity_schedule/manifest.json')
    for name in ('complexity_schedule.py','portfolio.py','kernels.py','data_pipeline.py',
                 'bandwidth_tuning.py','empirical_final/theory_guided_lambda.py',
                 'empirical_final/lengthscale_spectral.py','empirical_final/core.py'):
        bind(ROOT/name)
    # The source cache is checksum-bound both to the baseline and clean panels.
    source = meta['source']
    bind(ROOT/'results/schedule_cache/matern32_managed.npy',source['managed_file_sha256'])
    g, dates, actual = cache_managed(ROOT/'data/clean',ROOT/'results/schedule_cache')
    assert actual['managed_matrix_sha256'] == source['managed_matrix_sha256']
    calibration = calibrate(g,dates)
    initial = pd.read_csv(bind(ROOT/'paper/complexity_schedule/initial_validation.csv'))
    for name, col in (('c0','C0'),('lambda','initial_lambda'),('loss','validation_loss')):
        np.testing.assert_allclose(calibration[name],initial[col],rtol=1e-7,atol=1e-12)
    assert calibration['choice'] == meta['initial_choice_index']
    np.testing.assert_allclose(calibration['spectrum']['b'],meta['initial_spectrum']['b'],rtol=1e-10)
    annual_ref = pd.read_csv(bind(ROOT/'paper/complexity_schedule/annual_schedule.csv'))
    monthly_ref = pd.read_csv(bind(ROOT/'paper/complexity_schedule/selected_monthly.csv'),parse_dates=['formation_date','return_date'])
    rows, months = [], []
    for split in annual_splits(dates):
        fit = window(g,dates,split,calibration); j=calibration['choice']
        ref = annual_ref[annual_ref.year.eq(fit['year'])].iloc[0]
        row = dict(year=fit['year'],T=fit['T'],C0=float(calibration['c0'][j]),
                   **fit['spectrum'],lambda_=float(fit['lambda'][j]),
                   complexity=float(fit['complexity'][j]))
        row['lambda']=row.pop('lambda_')
        for key in ('T','C0','b','lambda','complexity','r_squared','rank_low','rank_high','positive_rank'):
            np.testing.assert_allclose(row[key],ref[key],rtol=1e-7,atol=1e-12)
        rows.append(row)
        for i, payoff in zip(fit['test'],fit['returns'][:,j]):
            months.append(dict(formation_date=dates[i],return_date=dates[i]+pd.offsets.MonthEnd(1),raw_excess_return=float(payoff)))
        print('Baseline replication',fit['year'],flush=True)
    frame=pd.DataFrame(months)
    np.testing.assert_array_equal(frame.return_date,monthly_ref.return_date)
    np.testing.assert_array_equal(frame.formation_date,monthly_ref.formation_date)
    np.testing.assert_allclose(frame.raw_excess_return,monthly_ref.raw_excess_return,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(sharpe(frame.raw_excess_return),meta['full_oos_sharpe'],rtol=1e-9)
    assert len(frame)==564
    save(out/'tables/replicated_baseline_annual.csv',rows)
    save(out/'tables/replicated_baseline_monthly.csv',frame)
    write_json(out/'audit/replication.json',dict(passed=True,months=564,annual_refits=47,
        full_gross_sharpe=float(sharpe(frame.raw_excess_return)),
        maximum_payoff_error=float(np.max(np.abs(frame.raw_excess_return-monthly_ref.raw_excess_return))),
        maximum_schedule_relative_error=float(np.max(np.abs(pd.DataFrame(rows)[['b','lambda','complexity']].to_numpy()/annual_ref[['b','lambda','complexity']].to_numpy()-1))),
        C0=float(calibration['c0'][calibration['choice']]),b0=calibration['spectrum']['b'],
        initial_train_months=120,first_refit_months=180,elapsed_seconds=time.monotonic()-started))
    write_json(out/'audit/input_hashes.json',dict(INPUTS))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('replicate','estimate','costs'),default='replicate')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--private',type=Path,default=PRIVATE)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='replicate':replicate(args.output)
        elif args.phase=='estimate':estimate(args.output,args.private)
        elif args.phase=='costs':costs(args.output,args.private)


if __name__ == '__main__':
    main()
