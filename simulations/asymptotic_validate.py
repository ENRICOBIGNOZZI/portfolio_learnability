"""Independent completion audit for the long-training experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .asymptotic import scientific_hash
from .model import calibrate, moments


def validate(out):
    out=Path(out)
    config=json.loads((out/'config.json').read_text())
    completion=json.loads((out/'completed.json').read_text())
    assert completion['source_hash']==scientific_hash(config)
    assert completion['completed_replications']==len(config['cells'])*config['trials']
    frame=pd.read_csv(out/'confirmation_trials.csv')
    spectrum=pd.read_csv(out/'complexity_lambda_trials.csv')
    fidelity=pd.read_csv(out/'fidelity_trials.csv')
    assert not frame.duplicated(['N','rho','trial','lambda_multiplier','T_train']).any()
    assert set(map(tuple,frame[['N','rho']].drop_duplicates().to_numpy()))==set(map(tuple,config['cells']))
    assert set(frame.lambda_multiplier)==set(config['lambda_multipliers'])
    assert set(frame.trial)==set(range(config['trials']))
    expected_rows=0
    for (N,rho),group in frame.groupby(['N','rho']):
        times=config['T_train_reference'] if N==config['fixed_N'] and rho==config['fixed_rho'] else config['T_train']
        assert set(group.T_train)==set(times)
        expected=config['trials']*len(times)*len(config['lambda_multipliers'])
        assert len(group)==expected
        expected_rows+=expected
        for key,subset in group.groupby(['trial','lambda_multiplier']):
            assert set(subset.T_train)==set(times)
    assert len(frame)==expected_rows
    assert set(frame.T_pilot_train)=={config['pilot_train']}
    assert set(frame.T_pilot_validation)=={config['pilot_validation']}
    assert set(frame.T_evaluation)=={config['evaluation_states']}
    assert np.isfinite(frame.select_dtypes('number')).all().all()
    np.testing.assert_allclose(frame.lambda_reg,frame.lambda_multiplier*frame.C_lambda*frame.T_train**(-2/3),rtol=1e-11)
    np.testing.assert_allclose(frame.SR_gap,3-frame.SR_hat,atol=1e-13)
    variance=frame.payoff_second-frame.payoff_mean**2
    assert (variance>0).all()
    np.testing.assert_allclose(frame.SR_hat,np.sqrt(12)*frame.payoff_mean/np.sqrt(variance),rtol=1e-11)
    np.testing.assert_allclose(frame.Q_hat,1-2*frame.payoff_mean+frame.payoff_second,atol=1e-12)
    np.testing.assert_allclose(frame.regret,frame.Q_hat-4/7,atol=1e-12)
    assert (frame.complexity>0).all()
    assert (frame.complexity<=np.minimum(frame['rank'],frame.T_train)+1e-8).all()
    assert frame['rank'].nunique()==1
    assert frame['rank'].iloc[0]==completion['rank']
    pilot=pd.read_csv(out/'pilot.csv')
    constants=json.loads((out/'frozen_constants.json').read_text())
    interior=[]
    for N,g in pilot.groupby('N'):
        avg=g.groupby('C_lambda').validation_loss.mean()
        selected=float(avg.idxmin())
        np.testing.assert_allclose(constants[str(N)],selected,rtol=1e-12)
        interior.append(avg.index.min()<selected<avg.index.max())
        np.testing.assert_allclose(frame.loc[frame.N==N,'C_lambda'],selected,rtol=1e-12)
    selection=json.loads((out/'rank_selection.json').read_text())
    f=fidelity[fidelity.lambda_multiplier==1].groupby(['representation','rank','T_train'])[['SR_gap','complexity']].mean()
    ref=f.loc[('nystrom',selection['reference_rank'])]
    recomputed=[]
    for record in selection['checks']:
        rank=record['rank'];candidate=f.loc[('nystrom',rank)]
        keep=ref.index>=config['asymptotic_T_min']
        c_error=float(np.max(np.abs(candidate.complexity[keep]/ref.complexity[keep]-1)))
        sr_error=float(np.max(np.abs(candidate.SR_gap[keep]-ref.SR_gap[keep])))
        np.testing.assert_allclose(c_error,record['max_relative_C_difference'],atol=1e-11)
        np.testing.assert_allclose(sr_error,record['max_absolute_SR_difference'],atol=1e-11)
        passes=c_error<=config['rank_C_tolerance'] and sr_error<=config['rank_SR_tolerance']
        assert passes==record['passes']
        recomputed.append((rank,passes))
    assert min(rank for rank,passes in recomputed if passes)==selection['selected_rank']
    assert selection['selected_rank']==completion['rank']
    for key,g in spectrum.groupby(['N','rho','trial']):
        assert len(g)==65
        assert set(g.T_train)=={config['spectrum_T_train']}
        ordered=g.sort_values('lambda_reg')
        assert (np.diff(ordered.complexity)<0).all()
    chosen=frame[frame.lambda_multiplier==1]
    summary=pd.read_csv(out/'summary.csv')
    for (N,rho,mult,n),g in frame.groupby(['N','rho','lambda_multiplier','T_train']):
        row=summary[(summary.N==N)&(summary.rho==rho)&(summary.lambda_multiplier==mult)&(summary.T_train==n)].iloc[0]
        np.testing.assert_allclose(row.SR_std,g.SR_hat.std(ddof=1),rtol=1e-11)
        np.testing.assert_allclose(row.C_std,g.complexity.std(ddof=1),rtol=1e-11)
        for column,metric in [('SR_p95','SR_hat'),('gap_p95','SR_gap'),('C_p95','complexity')]:
            np.testing.assert_allclose(row[column],g[metric].quantile(.95,interpolation='linear'),rtol=1e-11,atol=1e-13)
    plot=json.loads((out/'plot_audit.json').read_text())
    assert plot['standard_error_plotted'] is False
    assert plot['main_fit_T_train_min']==config['asymptotic_T_min']
    assert plot['full_grid_shown']
    assert plot['percentile_plotted']==95 and plot['percentile_interpolation']=='linear'
    assert plot['temporal_points_per_comparison']==len(config['T_train'])
    assert plot['temporal_points_reference']==len(config['T_train_reference'])
    fits=json.loads((out/'rate_fits.json').read_text())
    windows=pd.read_csv(out/'window_sensitivity.csv')
    for (N,rho),g in chosen.groupby(['N','rho']):
        ts=sorted(g.T_train.unique())
        for cutoff in config['sensitivity_T_min']:
            count=sum(t>=cutoff for t in ts)
            if count<3:
                continue
            for metric in ['SR_T','SR_C','C_T']:
                key=f'N{int(N)}_rho{rho:g}_{metric}_Tmin{cutoff}'
                assert key in fits
                assert fits[key]['fit_points']==count
                assert fits[key]['T_train_fit_grid']==[t for t in ts if t>=cutoff]
                assert fits[key]['all_points_in_declared_window_used']
    # Independent economic audit, including calibration from Fourier orthogonality.
    economic=[]
    rng=np.random.default_rng(93719)
    orders=np.arange(2,1001,dtype=float)
    vpsi=np.sum(1/(orders**7*np.log(orders+1)**2))/2
    for N in config['N_values']:
        cal=calibrate(N)
        p=N*(cal['A_alpha']**2+cal['A_beta']**2*vpsi*np.dot(cal['lambda_factor'],cal['lambda_factor']))/cal['kappa_0']
        np.testing.assert_allclose(p,3/7,rtol=1e-13)
        z=rng.uniform(-1,1,(16,N,3))
        _,B,u,h=moments(z,cal)
        bu=np.einsum('tnk,tn->tk',B,u)
        residual=((cal['sigma_eps']**2+np.sum(u*u,axis=1))[:,None]*u
            +np.einsum('tnk,tk->tn',B,bu@cal['Sigma_factor'])
            +h*np.sum(h*u,axis=1)[:,None]-cal['kappa_0']*u)
        error=float(np.max(np.abs(residual))/np.max(np.abs(cal['kappa_0']*u)))
        assert error<1e-10
        economic.append(dict(N=N,p_star=p,E4_relative_error=error))
    old=Path(__file__).parent/'outputs/new_montecarlo/rate_fits.json'
    old_fits=json.loads(old.read_text())
    reference_key=f"N{config['fixed_N']}_rho{config['fixed_rho']:g}_SR_T_Tmin{config['asymptotic_T_min']}"
    before=old_fits['SR_T_N']['exponent'];after=fits[reference_key]['exponent']
    improvement=dict(previous_full_grid_exponent=before,new_prespecified_tail_exponent=after,
        benchmark=-2/3,old_absolute_exponent_difference=abs(before+2/3),
        new_absolute_exponent_difference=abs(after+2/3),
        exponent_closer_to_benchmark=abs(after+2/3)<abs(before+2/3))
    report=dict(passed=True,completed_replications=completion['completed_replications'],observations=len(frame),
                source_hash_matches=True,all_C0_selections_interior=all(interior),
                all_sensitivity_windows_retained=True,T_train_max=int(frame.T_train.max()),
                SD_ddof=1,percentile_plotted=95,percentile_interpolation='linear',
                temporal_points_per_comparison=len(config['T_train']),temporal_points_reference=len(config['T_train_reference']),
                standard_error_plotted=False,economic_audit=economic,
                minimum_estimated_regret=float(frame.regret.min()),
                negative_estimated_gaps=int((frame.SR_gap<0).sum()),
                coherence_comparison=improvement,
                note='Numerical validity and rate coherence are reported separately; failing to match a benchmark is not relabeled a successful fit.')
    files=[p for p in out.rglob('*') if p.is_file() and 'cache' not in p.parts
           and p.name not in ['validation.json','visual_review.json']]
    report['files_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',nargs='?',type=Path,default=Path(__file__).parent/'outputs/asymptotic_v2')
    validate(parser.parse_args().output)
