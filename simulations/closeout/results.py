"""Verified four-specification summaries, paired outcomes, rates and provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from simulations.initial_cv.design import CANONICAL, REFERENCE, protocol
from simulations.initial_cv.run import verify
from simulations.extended.design import ENVIRONMENTS, PENALTIES, parameters
from simulations.extended.statistics import distribution, gap_slopes
from simulations.provenance import file_hash, json_write, npz_write, utc_now

DEST=CANONICAL.parent/'rich6d_initial_cv60_closeout_v1'
SPECS=('initial_cv','old_fixed','oracle_Q','oracle_SR')


def oracle_indices(regret, annual_sr):
    """One penalty per economy/horizon; never individual replication maxima."""
    if regret.shape != annual_sr.shape or regret.ndim != 3:
        raise ValueError('Need matching replication-by-horizon-by-grid arrays.')
    return np.argmin(regret.mean(axis=0),axis=1), np.argmax(annual_sr.mean(axis=0),axis=1)


def verified_records():
    frozen=protocol()
    gate=json.loads((CANONICAL/'production_verification.json').read_text())
    manifest=json.loads((CANONICAL/'summary_verification.json').read_text())
    if not gate['passed'] or gate['run_hash']!=frozen['run_hash'] or gate['replications_per_economy']!=300:
        raise ValueError('Complete verified original production required.')
    records=[]
    for i in range(300):
        path=CANONICAL/'replications'/f'rep_{i:03d}.npz'
        if file_hash(path)!=manifest['checkpoint_sha256'][str(path.relative_to(CANONICAL))]:
            raise ValueError('Original checkpoint SHA256 mismatch.')
        verify(path,i,frozen)
        with np.load(path) as z: records.append({k:z[k] for k in z.files})
    return frozen,records


def build_results():
    DEST.mkdir(parents=True,exist_ok=True)
    frozen,records=verified_records()
    old_manifest=json.loads((CANONICAL/'summary_verification.json').read_text())
    summaries=[]; slopes=[]; paths=[]; oracle=[]; paired=[]; crossfit=[]; inputs={}; ridge_rows=[]
    for name in ENVIRONMENTS:
        path=REFERENCE/'distributions'/f'{name}.npz'
        expected=old_manifest['original_oracle_input_sha256'][str(path.relative_to(REFERENCE))]
        if file_hash(path)!=expected: raise ValueError('Original oracle grid input changed.')
        inputs[str(path)]=expected
        with np.load(path) as z: old={k:z[k] for k in z.files}
        times=records[0][name+'_T']; nt=len(times); q=parameters(name).q_star
        np.testing.assert_array_equal(times,old['T'])
        choiceQ,choiceSR=oracle_indices(old['regret'][:,:,:96],old['annual_SR'][:,:,:96])
        pop_path=REFERENCE/'population'/f'{name}_theory.npz'
        with np.load(pop_path) as z: eigenvalues=z['eigenvalues']; floor=float(z['projection_floor'])
        inputs[str(pop_path)]=file_hash(pop_path)
        np.testing.assert_allclose(old['regret'][:,:,-1],old['theory_population_bias']+old['theory_estimation_norm']+old['theory_cross_term'],atol=1e-9,rtol=1e-8)
        for i in range(300):
            for t,T in enumerate(times):
                ridge_rows.append(dict(environment=name,index=i,T=int(T),specification='old_fixed',regret=old['regret'][i,t,-1],
                    projection_floor=floor,regularization_bias_within_P=old['theory_population_bias'][i,t]-floor,
                    total_population_bias=old['theory_population_bias'][i,t],estimation_displacement=old['theory_estimation_norm'][i,t],signed_cross_term=old['theory_cross_term'][i,t]))

        data={}
        for metric,source,scale in [('annual_SR','sr',np.sqrt(12)),('regret','loss',1.),
              ('empirical_complexity','empirical_complexity',1.),('population_complexity','population_complexity',1.)]:
            two=np.stack([r[name+'_'+source] for r in records])*scale
            if metric=='regret': two-=1-q
            thirds=[]
            for choice in (choiceQ,choiceSR):
                third=np.broadcast_to(np.sum(eigenvalues[:,None]/(eigenvalues[:,None]+PENALTIES[choice]),axis=0),(300,nt)) if metric=='population_complexity' else old[metric][:,np.arange(nt),choice]
                thirds.append(third[:,:,None])
            data[metric]=np.concatenate([two,*thirds],axis=2)
        data.update(monthly_SR=data['annual_SR']/np.sqrt(12))
        data['annual_gap']=np.sqrt(12)*parameters(name).sr_star-data['annual_SR']
        data['monthly_gap']=data['annual_gap']/np.sqrt(12)
        data['relative_complexity']=data['empirical_complexity']/times[None,:,None]
        data['population_relative_complexity']=data['population_complexity']/times[None,:,None]
        data['normalized_regret']=data['regret']*times[None,:,None]**.6
        data['normalized_monthly_gap']=data['monthly_gap']*times[None,:,None]**.6
        data['normalized_annual_gap']=data['annual_gap']*times[None,:,None]**.6
        data['normalized_empirical_complexity']=data['empirical_complexity']/times[None,:,None]**.4
        data['normalized_population_complexity']=data['population_complexity']/times[None,:,None]**.4
        for metric in ('annual_SR','regret','empirical_complexity'):
            np.testing.assert_allclose(data[metric][:,:,1],old[metric][:,:,-1],atol=1e-8,rtol=1e-7)
        lam=np.concatenate([np.stack([r[name+'_penalties'] for r in records]),
            np.broadcast_to(np.column_stack([PENALTIES[choiceQ],PENALTIES[choiceSR]]),(300,nt,2))],axis=2)
        npz_write(DEST/'distributions'/f'{name}.npz',T=times,penalties=lam,specifications=SPECS,**data)
        for j,spec in enumerate(SPECS):
            for metric,values in data.items():
                stats=distribution(values[:,:,j])
                for t,T in enumerate(times):
                    summaries.append(dict(environment=name,T=int(T),specification=spec,metric=metric,**{k:float(v[t]) for k,v in stats.items()}))
                if metric in ('regret','annual_gap','monthly_gap','empirical_complexity','population_complexity','relative_complexity','population_relative_complexity'):
                    for grid,mask in [('full',np.ones(nt,bool)),('robustness_common',times<=3240)]:
                        for row in gap_slopes(times[mask],values[:,mask,j]):
                            if spec.startswith('oracle'):
                                for k in ('MCSE_delta','MCSE_jackknife','CI_low','CI_high'): row.pop(k,None)
                                row['uncertainty']='Descriptive; no adjustment for grid selection.'
                            slopes.append(dict(environment=name,specification=spec,quantity=metric,grid=grid,**row))
            for i in range(300):
                for t,T in enumerate(times):
                    paths.append(dict(environment=name,index=i,T=int(T),specification=spec,penalty=lam[i,t,j],
                        selected_a=float(records[i][name+'_selected_a']) if j==0 else (frozen['design']['a_old'][name] if j==1 else np.nan),
                        projection_floor=floor,floor_fraction=floor/data['regret'][i,t,j],
                        floor_pass=bool(floor<=.05*data['regret'][i,t,j]),cv_rank_certified=False,
                        **{metric:values[i,t,j] for metric,values in data.items()}))
        for t,T in enumerate(times):
            for spec,choice,objective in [('oracle_Q',choiceQ,'min_mean_regret'),('oracle_SR',choiceSR,'max_mean_annual_SR')]:
                oracle.append(dict(environment=name,T=int(T),specification=spec,penalty_index=int(choice[t]),penalty=PENALTIES[choice[t]],objective=objective,selection_adjusted_CI=False))
        for metric in ('annual_SR','regret','annual_gap','empirical_complexity'):
            stats=distribution(data[metric][:,:,0]-data[metric][:,:,1])
            for t,T in enumerate(times):
                paired.append(dict(environment=name,T=int(T),metric=metric,**{k:float(v[t]) for k,v in stats.items()},CI_low=stats['mean'][t]-1.96*stats['mcse'][t],CI_high=stats['mean'][t]+1.96*stats['mcse'][t]))
        # Even/odd common-grid selection on one half, assessment on the other.
        # Preserves whole histories and never uses a per-path oracle.
        for parity in (0,1):
            train=np.arange(300)%2==parity; test=~train
            cq,cs=oracle_indices(old['regret'][train,:,:96],old['annual_SR'][train,:,:96])
            for spec,choices,metric in [('oracle_Q',cq,'regret'),('oracle_SR',cs,'annual_SR')]:
                stats=distribution(old[metric][test][:,np.arange(nt),choices])
                for t,T in enumerate(times): crossfit.append(dict(environment=name,T=int(T),specification=spec,selection_parity=parity,evaluation_parity=1-parity,penalty=PENALTIES[choices[t]],metric=metric,**{k:float(v[t]) for k,v in stats.items()}))
    pd.DataFrame(ridge_rows).to_csv(DEST/'historical_ridge_decomposition.csv',index=False)
    for filename,rows in [('summary.csv',summaries),('slopes_all_windows.csv',slopes),('policy_metrics.csv',paths),('oracle_choices.csv',oracle),('paired_cv_vs_fixed.csv',paired),('crossfit_oracles.csv',crossfit)]:
        pd.DataFrame(rows).to_csv(DEST/filename,index=False)
    s=pd.DataFrame(slopes)
    baseline=s[(s.environment=='baseline')&(s.specification=='initial_cv')&(s.grid=='full')&(s.window=='full')].set_index('quantity')
    np.testing.assert_allclose(baseline.loc['regret','slope'],-.4378196,atol=1e-6)
    np.testing.assert_allclose(baseline.loc['annual_gap','slope'],-.3986480,atol=1e-6)
    for filename in ('selected_constants.csv','selected_constant_frequencies.csv','selected_constant_summary.csv','numerical_warnings.csv','paired_robustness.csv'):
        (DEST/('historical_'+filename if filename=='numerical_warnings.csv' else filename)).write_bytes((CANONICAL/filename).read_bytes())
        inputs[str(CANONICAL/filename)]=file_hash(CANONICAL/filename)
    warning_rows=[]
    for name in ENVIRONMENTS:
        with np.load(REFERENCE/'population'/f'{name}_theory.npz') as pop:
            floor=float(pop['projection_floor'])
        selected=pd.DataFrame(summaries)
        selected=selected[(selected.environment==name)&(selected.metric=='regret')]
        for _,row in selected.iterrows():
            warning_rows.append(dict(environment=name,T=int(row['T']),specification=row.specification,
                projection_floor=floor,mean_regret=row['mean'],floor_fraction=floor/row['mean'],
                floor_pass=bool(floor<=.05*row['mean']),rank_forward_certificate=False,
                numerically_certified=False,mask_reason='Projection floor threshold and incomplete strategy-specific forward-rank/quadrature audit.'))
    pd.DataFrame(warning_rows).to_csv(DEST/'numerical_warnings.csv',index=False)
    pd.DataFrame(warning_rows).to_csv(DEST/'numerical_reliability_mask.csv',index=False)
    inputs.update({str(CANONICAL/p):h for p,h in old_manifest['checkpoint_sha256'].items()})
    json_write(DEST/'input_verification.json',dict(passed=True,run_hash=frozen['run_hash'],replications=300,old_metrics_reproduced=True,
        regret_slope=float(baseline.loc['regret','slope']),sharpe_gap_slope=float(baseline.loc['annual_gap','slope']),
        sources=inputs,verified_utc=utc_now(),analysis_timing='Post hoc; windows and economic/trading specification unchanged.'))

if __name__=='__main__': build_results()
