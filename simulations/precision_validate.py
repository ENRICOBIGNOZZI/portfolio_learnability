"""Independent checks of the paired high-precision pilot and its report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .asymptotic_precision import source_hash
from .asymptotic_dense import write_json


def validate_complexity_slope_enclosures(config,frame,cert,spectrum,table):
    """Reconstruct bounds from raw traces and check endpoint regressions."""
    rank=int(frame['rank'].iloc[0]);rows=table[table['rank']==rank]
    assert not rows.duplicated(['metric','T_train_min']).any()
    balanced=frame[frame.lambda_multiplier==1].merge(
        cert[cert.lambda_multiplier==1][['trial','T_train','residual_trace']],
        on=['trial','T_train'],validate='one_to_one')
    penalty=config['precision_frozen_C0']*balanced.T_train**(-2/3)
    balanced['upper']=np.minimum(balanced.complexity+np.maximum(balanced.residual_trace,0)/penalty,balanced.T_train)
    residuals=cert[(cert.lambda_multiplier==1)&(cert.T_train==config['spectrum_T_train'])].set_index('trial').residual_trace
    spectral=spectrum.copy()
    spectral['upper']=np.minimum(spectral.complexity+
        np.maximum(spectral.trial.map(residuals),0)/spectral.lambda_reg,spectral.T_train)
    expected_rows=0
    for cutoff in config['sensitivity_T_min']:
        times=[t for t in config['T_train'] if t>=cutoff]
        if len(times)<3:continue
        low=config['precision_frozen_C0']*max(times)**(-2/3)
        high=config['precision_frozen_C0']*min(times)**(-2/3)
        choices=[('C_T',balanced[balanced.T_train>=cutoff],'T_train',1/3),
            ('C_lambda',spectral[(spectral.lambda_reg>=low*(1-1e-10))&
                (spectral.lambda_reg<=high*(1+1e-10))],'lambda_reg',-.5)]
        for metric,part,index,target in choices:
            curve=part.groupby(index)[['complexity','upper']].mean().sort_index()
            x=curve.index.to_numpy(float);lower=curve.complexity.to_numpy();upper=curve.upper.to_numpy()
            if metric=='C_T':assert list(curve.index)==times
            logx=np.log(x);positive=logx>=logx.mean()
            # Enumerating the minimizing/maximizing ordinate at each coordinate
            # and running polyfit is independent of the producer's weight sum.
            lo=np.polyfit(logx,np.log(np.where(positive,lower,upper)),1)[0]
            hi=np.polyfit(logx,np.log(np.where(positive,upper,lower)),1)[0]
            actual=rows[(rows.metric==metric)&(rows.T_train_min==cutoff)]
            assert len(actual)==1;row=actual.iloc[0]
            assert row.fit_points==len(x) and row.expected_exponent==target
            assert row.T_train_max==max(times)
            np.testing.assert_allclose([row.x_min,row.x_max],[x.min(),x.max()],rtol=1e-11)
            np.testing.assert_allclose(row.approximate_exponent,np.polyfit(logx,np.log(lower),1)[0],atol=1e-11)
            np.testing.assert_allclose([row.full_empirical_exponent_lower,row.full_empirical_exponent_upper],[lo,hi],rtol=1e-10,atol=1e-11)
            assert row.theoretical_exponent_in_enclosure==bool(lo-1e-12<=target<=hi+1e-12)
            np.testing.assert_allclose(row.maximum_mean_C_relative_numerical_uncertainty,np.max(upper/lower-1),rtol=1e-10,atol=1e-12)
            expected_rows+=1
    assert len(rows)==expected_rows


def validate(output, rank=None):
    out=Path(output);config=json.loads((out/'config.json').read_text())
    report=rank is None
    ranks=config['precision_ranks'] if report else [rank]
    assert all(r in config['precision_ranks'] for r in ranks)
    if report:
        audit=json.loads((out/'precision_plot_audit.json').read_text())
        selection=json.loads((out/'precision_selection.json').read_text())
        summary=pd.read_csv(out/'precision_summary.csv')
        windows=pd.read_csv(out/'precision_window_sensitivity.csv')
        enclosures=pd.read_csv(out/'precision_complexity_slope_enclosures.csv')
        assert audit['standard_error_plotted'] is False and audit['SD_ddof']==1
        assert audit['percentile_plotted']==95 and audit['percentile_interpolation']=='linear'
        assert audit['full_training_grid']==config['T_train']
        assert audit['all_sensitivity_cutoffs']==config['sensitivity_T_min']
        assert audit['main_fit_T_train_min']==config['asymptotic_T_min']
        assert set(summary['rank'])==set(config['precision_ranks'])
        assert set(windows['rank'])==set(config['precision_ranks'])
        assert selection['selection_uses_slopes_or_R_squared'] is False
        assert set(enclosures['rank'])==set(config['precision_ranks'])
        assert audit['complexity_slope_enclosures']['sampling_interval'] is False
        assert audit['complexity_slope_enclosures']['used_for_rank_selection'] is False
        assert audit['complexity_slope_enclosures']['pointwise_upper_bound_capped_by_T_train'] is True
    checks=[];observations=0;gaps_below_zero=0
    for rank in ranks:
        prefix=f'precision_nystrom{rank}_'
        complete=json.loads((out/(prefix+'completed.json')).read_text())
        assert complete['source_hash']==source_hash(config)
        assert complete['replications']==config['trials']
        assert complete['evaluation_states']==config['evaluation_states']
        for name,sha in complete['files_sha256'].items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
        f=pd.read_csv(out/(prefix+'trials.csv'))
        cert=pd.read_csv(out/(prefix+'certificates.csv'))
        spectrum=pd.read_csv(out/(prefix+'spectrum.csv'))
        fits=json.loads((out/(prefix+'fits.json')).read_text())
        for data in [f,cert,spectrum]:
            assert np.isfinite(data.select_dtypes('number')).all().all()
            assert set(data['rank'])=={rank}
            assert set(data.trial)==set(range(config['trials']))
        keys=['rank','trial','T_train','lambda_multiplier']
        assert not f.duplicated(keys).any() and not cert.duplicated(keys).any()
        expected=config['trials']*len(config['T_train'])*len(config['lambda_multipliers'])
        assert len(f)==expected and len(cert)==expected
        assert set(f.N)=={config['fixed_N']} and set(f.rho)=={config['fixed_rho']}
        for _,g in f.groupby(['trial','lambda_multiplier']):
            assert set(g.T_train)==set(config['T_train'])
        assert set(f.lambda_multiplier)==set(config['lambda_multipliers'])
        np.testing.assert_allclose(f.lambda_reg,
            config['precision_frozen_C0']*f.T_train**(-2/3)*f.lambda_multiplier,rtol=1e-11)
        np.testing.assert_allclose(f.SR_gap,3-f.SR_hat,atol=1e-13)
        variance=f.payoff_second-f.payoff_mean**2
        assert (variance>0).all()
        np.testing.assert_allclose(f.SR_hat,np.sqrt(12)*f.payoff_mean/np.sqrt(variance),rtol=1e-11)
        np.testing.assert_allclose(f.Q_hat,1-2*f.payoff_mean+f.payoff_second,atol=1e-12)
        np.testing.assert_allclose(f.regret,f.Q_hat-4/7,atol=1e-12)
        assert ((f.complexity>0)&(f.complexity<=np.minimum(rank,f.T_train)+1e-8)).all()
        paired=f.merge(cert,on=keys,validate='one_to_one')
        absolute=np.maximum(paired.residual_trace,0)/paired.lambda_reg
        np.testing.assert_allclose(paired.absolute_C_error_bound,absolute,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(paired.relative_C_error_bound,absolute/paired.complexity,rtol=1e-10,atol=1e-12)
        for (mult,t),g in (f.groupby(['lambda_multiplier','T_train']) if report else []):
            row=summary[(summary['rank']==rank)&(summary.lambda_multiplier==mult)&(summary.T_train==t)]
            assert len(row)==1;row=row.iloc[0]
            assert row.replications==config['trials']
            for name,column in [('SR','SR_hat'),('gap','SR_gap'),('C','complexity')]:
                np.testing.assert_allclose(row[name+'_mean'],g[column].mean(),rtol=1e-11,atol=1e-13)
                np.testing.assert_allclose(row[name+'_std'],g[column].std(ddof=1),rtol=1e-10,atol=1e-13)
                np.testing.assert_allclose(row[name+'_p95'],g[column].quantile(.95,interpolation='linear'),rtol=1e-11,atol=1e-13)
        for trial,g in spectrum.groupby('trial'):
            assert len(g)==65 and not g.lambda_reg.duplicated().any()
            assert set(g.T_train)=={config['spectrum_T_train']}
            assert (np.diff(g.sort_values('lambda_reg').complexity)<0).all()
            assert (g.absolute_C_error_bound>=0).all()
            np.testing.assert_allclose(g.relative_C_error_bound,g.absolute_C_error_bound/g.complexity,rtol=1e-11)
            residual=cert[(cert.trial==trial)&(cert.T_train==config['spectrum_T_train'])].residual_trace
            np.testing.assert_allclose(g.absolute_C_error_bound,np.maximum(residual.iloc[0],0)/g.lambda_reg,rtol=1e-10,atol=1e-12)
        for cutoff in config['sensitivity_T_min']:
            times=[t for t in config['T_train'] if t>=cutoff]
            if len(times)<3:
                continue
            part=f[(f.lambda_multiplier==1)&(f.T_train>=cutoff)]
            means=part.groupby('T_train')[['complexity','SR_gap']].mean().sort_index()
            lower=config['precision_frozen_C0']*max(times)**(-2/3)
            upper=config['precision_frozen_C0']*min(times)**(-2/3)
            spectral=spectrum[(spectrum.lambda_reg>=lower*(1-1e-10))&
                              (spectrum.lambda_reg<=upper*(1+1e-10))]
            sm=spectral.groupby('lambda_reg').complexity.mean().sort_index()
            gaps=part.pivot(index='trial',columns='T_train',values='SR_gap').sort_index(axis=1).to_numpy()
            complexities=part.pivot(index='trial',columns='T_train',values='complexity').sort_index(axis=1).to_numpy()
            spectral_values=spectral.pivot(index='trial',columns='lambda_reg',values='complexity').sort_index(axis=1).to_numpy()
            for metric,x,y,target,values,xvalues in [
                ('SR_T',np.array(times,float),means.SR_gap.to_numpy(),-2/3,gaps,None),
                ('C_T',np.array(times,float),means.complexity.to_numpy(),1/3,complexities,None),
                ('SR_C',means.complexity.to_numpy(),means.SR_gap.to_numpy(),-2,gaps,complexities),
                ('C_lambda',sm.index.to_numpy(float),sm.to_numpy(),-.5,spectral_values,None)]:
                fit=fits[f'{metric}_Tmin{cutoff}']
                np.testing.assert_allclose(fit['x'],x,rtol=1e-11)
                np.testing.assert_allclose(fit['observed'],y,rtol=1e-11,atol=1e-13)
                assert fit['fit_points']==len(x) and fit['expected_exponent']==target
                if metric!='C_lambda':
                    assert fit['T_train_fit_grid']==times
                if np.all(y>0):
                    slope=np.polyfit(np.log(x),np.log(y),1)[0]
                    np.testing.assert_allclose(fit['exponent'],slope,atol=1e-10)
                else:
                    assert fit['exponent'] is None
                basis=x**target;scale=max(0,float(basis@y/(basis@basis)))
                np.testing.assert_allclose(fit['prediction'],scale*basis,rtol=1e-11,atol=1e-13)
                variance=np.sum((y-y.mean())**2)
                r2=1-np.sum((y-scale*basis)**2)/variance
                np.testing.assert_allclose(fit['r_squared'],r2,rtol=1e-10,atol=1e-12)
                # Resample whole raw paths; SR(C) resamples both coordinates.
                # Closed-form centered regression is independent of rate_fit's polyfit.
                interval=None
                if np.all(y>0):
                    indices=np.random.default_rng(540007).integers(0,len(values),
                        size=(config['bootstrap_resamples'],len(values)))
                    by=values[indices].mean(axis=1)
                    bx=np.broadcast_to(x,by.shape) if xvalues is None else xvalues[indices].mean(axis=1)
                    positive=np.all(by>0,axis=1)
                    if np.any(positive):
                        lx=np.log(bx[positive]);ly=np.log(by[positive])
                        lx-=lx.mean(axis=1,keepdims=True);ly-=ly.mean(axis=1,keepdims=True)
                        slopes=np.sum(lx*ly,axis=1)/np.sum(lx**2,axis=1)
                        interval=np.quantile(slopes,[.025,.975])
                if interval is None:assert fit['exponent_ci95']==[None,None]
                else:np.testing.assert_allclose(fit['exponent_ci95'],interval,rtol=1e-9,atol=1e-11)
                good=bool(r2>=.9 and interval is not None and interval[0]<=target<=interval[1])
                assert fit['good_rate_fit']==good
                if report:
                    row=windows[(windows['rank']==rank)&(windows.metric==metric)&(windows.T_train_min==cutoff)]
                    assert len(row)==1
                    if fit['exponent'] is None:
                        assert pd.isna(row.iloc[0].exponent)
                    else:
                        np.testing.assert_allclose(row.iloc[0].exponent,fit['exponent'],atol=1e-10)
        low=config['precision_frozen_C0']*config['spectrum_fit_T_max']**(-2/3)
        high=config['precision_frozen_C0']*config['spectrum_fit_T_min']**(-2/3)
        a=float(cert[cert.lambda_multiplier==1].relative_C_error_bound.max())
        b=float(spectrum[(spectrum.lambda_reg>=low*(1-1e-10))&
                         (spectrum.lambda_reg<=high*(1+1e-10))].relative_C_error_bound.max())
        passed=max(a,b)<=config['precision_relative_C_error_tolerance']
        assert bool(passed)==complete['passes_certified_tolerance']
        if report:
            actual=next(c for c in selection['checks'] if c['rank']==rank)
            assert actual['passes_certified_tolerance']==bool(passed)
        for field,value in [('maximum_balanced_relative_C_error_bound',a),('maximum_fit_window_relative_C_error_bound',b)]:
            if report:np.testing.assert_allclose(actual[field],value,rtol=1e-11)
            np.testing.assert_allclose(complete[field],value,rtol=1e-11)
        checks.append((rank,bool(passed)))
        if report:validate_complexity_slope_enclosures(config,f,cert,spectrum,enclosures)
        observations+=len(f);gaps_below_zero+=int((f.SR_gap<0).sum())
    if not report:
        # A complete rank can be audited while the other planned rank is running.
        # This does not select a rank or certify the final report's plots/statistics.
        files=['config.json',f'precision_nystrom{rank}_completed.json',*complete['files_sha256']]
        result=dict(passed=True,rank=rank,rank_replications=config['trials'],
            observations=observations,all_sensitivity_windows_retained=True,
            bootstrap_exponent_intervals_independently_verified=True,
            source_hash=source_hash(config),report_verified=False,full_precision_study_complete=False,
            passes_certified_tolerance=checks[0][1],negative_estimated_gaps=gaps_below_zero,
            note='Single-rank raw data, moments, complexity certificate and fits verified independently. Final cross-rank selection, summary statistics and plots require the full study.',
            files_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files})
        write_json(out/f'precision_nystrom{rank}_validation.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
        return result
    eligible=[rank for rank,passed in checks if passed]
    expected=min(eligible) if eligible else None
    assert selection['selected_rank']==expected
    assert audit['selection']==selection
    assert selection['benchmark_curve_rank']==(expected if expected is not None else max(config['precision_ranks']))
    # Record only durable report/data artifacts; a live execution log is mutable.
    files=[p for p in out.rglob('*') if p.is_file() and 'cache' not in p.parts
           and p.suffix in ['.json','.csv','.pdf','.png','.html']
           and p.name not in ['precision_validation.json','visual_review.json','execution.json']]
    result=dict(passed=True,rank_replications=config['trials'],ranks=config['precision_ranks'],
        observations=observations,selected_rank=expected,all_sensitivity_windows_retained=True,
        bootstrap_exponent_intervals_independently_verified=True,
        deterministic_full_empirical_complexity_slope_enclosures_verified=True,
        x_axis='T_train',SD_ddof=1,standard_error_plotted=False,percentile_plotted=95,
        negative_estimated_gaps=gaps_below_zero,
        note='Numerical and bookkeeping checks do not assert agreement with a theoretical rate or a passing complexity certificate.',
        files_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    write_json(out/'precision_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--rank',type=int,help='Audit one completed rank, without declaring the full study or report complete')
    a=p.parse_args();validate(a.output,a.rank)
