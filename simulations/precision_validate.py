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


def validate(output):
    out=Path(output);config=json.loads((out/'config.json').read_text())
    audit=json.loads((out/'precision_plot_audit.json').read_text())
    selection=json.loads((out/'precision_selection.json').read_text())
    summary=pd.read_csv(out/'precision_summary.csv')
    windows=pd.read_csv(out/'precision_window_sensitivity.csv')
    assert audit['standard_error_plotted'] is False and audit['SD_ddof']==1
    assert audit['percentile_plotted']==95 and audit['percentile_interpolation']=='linear'
    assert audit['full_training_grid']==config['T_train']
    assert audit['all_sensitivity_cutoffs']==config['sensitivity_T_min']
    assert audit['main_fit_T_train_min']==config['asymptotic_T_min']
    assert set(summary['rank'])==set(config['precision_ranks'])
    assert set(windows['rank'])==set(config['precision_ranks'])
    assert selection['selection_uses_slopes_or_R_squared'] is False
    checks=[];observations=0;gaps_below_zero=0
    for rank in config['precision_ranks']:
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
        for (mult,t),g in f.groupby(['lambda_multiplier','T_train']):
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
            for metric,x,y,target in [
                ('SR_T',np.array(times,float),means.SR_gap.to_numpy(),-2/3),
                ('C_T',np.array(times,float),means.complexity.to_numpy(),1/3),
                ('SR_C',means.complexity.to_numpy(),means.SR_gap.to_numpy(),-2),
                ('C_lambda',sm.index.to_numpy(float),sm.to_numpy(),-.5)]:
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
        actual=next(c for c in selection['checks'] if c['rank']==rank)
        assert actual['passes_certified_tolerance']==bool(passed)
        for field,value in [('maximum_balanced_relative_C_error_bound',a),('maximum_fit_window_relative_C_error_bound',b)]:
            np.testing.assert_allclose(actual[field],value,rtol=1e-11)
            np.testing.assert_allclose(complete[field],value,rtol=1e-11)
        checks.append((rank,bool(passed)))
        observations+=len(f);gaps_below_zero+=int((f.SR_gap<0).sum())
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
        x_axis='T_train',SD_ddof=1,standard_error_plotted=False,percentile_plotted=95,
        negative_estimated_gaps=gaps_below_zero,
        note='Numerical and bookkeeping checks do not assert agreement with a theoretical rate or a passing complexity certificate.',
        files_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    write_json(out/'precision_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    validate(p.parse_args().output)
