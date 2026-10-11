"""Independent checks of delivered CSV arithmetic, provenance and uncertainty."""
import json
import numpy as np
import pandas as pd
from simulations.closeout.results import DEST,SPECS,REFERENCE
from simulations.extended.design import ENVIRONMENTS,parameters,PENALTIES
from simulations.extended.statistics import distribution
from simulations.provenance import file_hash,json_write,utc_now


def verify():
    provenance=json.loads((DEST/'input_verification.json').read_text())
    for source,expected in provenance['sources'].items():
        if file_hash(source)!=expected:raise ValueError(f'Original input altered: {source}')
    summary=pd.read_csv(DEST/'summary.csv');slopes=pd.read_csv(DEST/'slopes_all_windows.csv');choices=pd.read_csv(DEST/'oracle_choices.csv')
    original_summary=pd.read_csv(REFERENCE/'full_penalty_distributions.csv')
    grid=original_summary[original_summary.penalty_index<96]
    for _,row in choices.iterrows():
        metric='regret' if row.specification=='oracle_Q' else 'annual_SR'
        candidates=grid[(grid.environment==row.environment)&(grid['T']==row['T'])&(grid.metric==metric)]
        independent=candidates.loc[candidates['mean'].idxmin() if metric=='regret' else candidates['mean'].idxmax()]
        if int(row.penalty_index)!=int(independent.penalty_index): raise ValueError('Independent original-summary oracle choice mismatch.')
        np.testing.assert_allclose(row.penalty,independent.penalty,rtol=1e-12)
    rows=0;ratechecks=0
    for name in ENVIRONMENTS:
        with np.load(DEST/'distributions'/f'{name}.npz') as z:
            times=z['T']
            np.testing.assert_array_equal(z['specifications'],SPECS)
            for j,spec in enumerate(SPECS):
                for metric in [k for k in z.files if k not in ('T','penalties','specifications')]:
                    values=z[metric][:,:,j]
                    selected=summary[(summary.environment==name)&(summary.specification==spec)&(summary.metric==metric)].sort_values('T')
                    np.testing.assert_array_equal(selected['T'],times)
                    for stat,expected in distribution(values).items():np.testing.assert_allclose(selected[stat],expected,rtol=1e-10,atol=1e-11)
                    rows+=len(selected)
                np.testing.assert_allclose(z['normalized_regret'][:,:,j],z['regret'][:,:,j]*times[None,:]**.6)
                np.testing.assert_allclose(z['monthly_SR'][:,:,j]*np.sqrt(12),z['annual_SR'][:,:,j])
                rate=slopes[(slopes.environment==name)&(slopes.specification==spec)]
                for _,r in rate.iterrows():
                    mask=times>=r.T_min
                    if r.grid=='robustness_common': mask &=times<=3240
                    if mask.sum()<3:continue
                    observed=z[r.quantity][:,mask,j];mean=observed.mean(axis=0)
                    x=np.log(times[mask]);independent=np.polyfit(x,np.log(mean),1)[0]
                    np.testing.assert_allclose(r.slope,independent,atol=1e-12)
                    if not spec.startswith('oracle'):
                        weights=(x-x.mean())/((x-x.mean())**2).sum()
                        gradient=weights/mean
                        cov=np.cov(observed,rowvar=False)/300
                        np.testing.assert_allclose(r.MCSE_delta,np.sqrt(gradient@cov@gradient),atol=1e-12)
                        leave=(observed.sum(axis=0)-observed)/299
                        estimates=np.log(leave)@weights
                        np.testing.assert_allclose(r.MCSE_jackknife,np.sqrt(299*np.mean((estimates-estimates.mean())**2)),atol=1e-12)
                    ratechecks+=1
    policy=pd.read_csv(DEST/'policy_metrics.csv')
    if len(policy)!=300*4*(13+4*12):raise ValueError('Missing or extra per-policy rows.')
    if policy.duplicated(['environment','index','T','specification']).any():raise ValueError('Duplicate policy rows.')
    for name in ENVIRONMENTS:
        rows_subset=policy[(policy.environment==name)&(policy.specification=='initial_cv')]
        if (rows_subset.groupby('index').selected_a.nunique()!=1).any():raise ValueError('Frozen initial constant changed.')
        np.testing.assert_allclose(rows_subset.penalty,rows_subset.selected_a*rows_subset['T'].astype(float)**(-.6),rtol=1e-12)
        for metric in ('annual_SR','regret'):
            paired=pd.read_csv(DEST/'paired_cv_vs_fixed.csv')
            with np.load(DEST/'distributions'/f'{name}.npz') as z:
                expected=distribution(z[metric][:,:,0]-z[metric][:,:,1])
            selected=paired[(paired.environment==name)&(paired.metric==metric)].sort_values('T')
            for stat,v in expected.items():np.testing.assert_allclose(selected[stat],v,atol=1e-11)
    # Re-evaluate every policy identity from saved moments without the helper.
    moment_path=DEST/'audit_policy_moments.csv'
    moment_checks=0
    if moment_path.exists():
        evaluation=json.loads((DEST/'audit_evaluation.json').read_text())
        if file_hash(moment_path)!=evaluation['moments_sha256']:raise ValueError('Evaluated moments CSV changed.')
        m=pd.read_csv(moment_path);q=m.q.to_numpy();u=m.u.to_numpy();v=m.v.to_numpy()
        np.testing.assert_allclose(m.regret,q-2*u+v,atol=1e-10)
        np.testing.assert_allclose(m.regret,q*(u/q-1)**2+v-u*u/q,atol=1e-10)
        active=(v-u*u)>0
        np.testing.assert_allclose(m.monthly_SR[active],u[active]/np.sqrt((v-u*u)[active]),atol=1e-10)
        positive=(u>0)&(v>0)
        np.testing.assert_allclose(m.regret[positive],(m.ray_gap+m.scale_gap)[positive],atol=1e-10)
        moment_checks=len(m)
    for source in sorted((DEST/'audit_fits').glob('*.csv')):
        receipt=json.loads(source.with_suffix('.json').read_text())
        if file_hash(source)!=receipt['moments_sha256'] or file_hash(source.with_suffix('.npz'))!=receipt['sha256']: raise ValueError('Audit sidecar or moments changed.')
        m=pd.read_csv(source)
        np.testing.assert_allclose(m.regret,m.projection_floor+m.regularization_bias_within_P+m.estimation_displacement+m.signed_cross_term,atol=1e-9)
    component_path=DEST/'identified_financial_components.csv'
    if component_path.exists():
        components=pd.read_csv(component_path)
        for name in ENVIRONMENTS:
            c=components[components.environment==name]
            q=parameters(name).q_star
            signed=c.monthly_SR.to_numpy()
            positive=signed>0
            np.testing.assert_allclose(c.ray_gap[positive],q-signed[positive]**2/(1+signed[positive]**2),atol=1e-10)
            np.testing.assert_allclose(c.regret[positive],(c.ray_gap+c.scale_gap)[positive],atol=1e-10)
            np.testing.assert_allclose(c.regret,c.projection_floor+c.regularization_bias_within_P+c.estimation_displacement_plus_cross,atol=1e-10)
        if len(components)!=len(policy):raise ValueError('Incomplete identified component rows.')
    result=dict(passed=True,summary_rows=rows,slope_checks=ratechecks,policy_rows=len(policy),moment_identity_checks=moment_checks,
        original_inputs_hash_checked=True,uncertainty_preserves_whole_histories=True,verified_utc=utc_now())
    json_write(DEST/'closeout_verification.json',result)
    return result

if __name__=='__main__':print(verify())
