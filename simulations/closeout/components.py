"""Identified full-300 financial components and numerical population-ridge bias.

Moments u/v are deliberately not reconstructed from ambiguous quadratic roots.
The normalized directional and ray/scale quantities ARE uniquely identified by
signed Sharpe and response-one regret. Population-ridge bias is evaluated at
each actual lambda in the same checksummed numerical subspace. Its residual is
exported as displacement PLUS cross, never variance or a zero-cross assumption.
"""
import json
import numpy as np
import pandas as pd
from simulations.closeout.results import DEST,SPECS,REFERENCE
from simulations.closeout.metrics import identified_components
from simulations.extended.design import ENVIRONMENTS,parameters
from simulations.extended.statistics import distribution,gap_slopes
from simulations.provenance import json_write,file_hash


def run():
    rows=[];summaries=[];slopes=[];sources={}
    for name in ENVIRONMENTS:
        spectral=DEST/'population_spectral'/f'{"baseline" if name.startswith("rho") else name}.npz'
        with np.load(spectral) as z:values=z['values'];projection=z['mean_projection'];floor=float(z['floor'])
        with np.load(REFERENCE/'population'/f'{name}_theory.npz') as z:
            np.testing.assert_allclose(values,z['eigenvalues'],rtol=1e-10,atol=1e-18)
            np.testing.assert_allclose(floor,z['projection_floor'],atol=1e-10)
        sources[str(spectral)]=file_hash(spectral)
        q=parameters(name).q_star
        np.testing.assert_allclose(floor,q-np.sum(projection**2/values),atol=1e-10)
        energy=projection**2/values
        with np.load(DEST/'distributions'/f'{name}.npz') as z:data={k:z[k] for k in z.files}
        times=data['T']
        for j,spec in enumerate(SPECS):
            regret=data['regret'][:,:,j];sr=data['monthly_SR'][:,:,j];lam=data['penalties'][:,:,j]
            identified=identified_components(regret,sr,q)
            unique,inverse=np.unique(lam,return_inverse=True)
            pop_bias=np.sum(energy[:,None]*(unique[None,:]/(values[:,None]+unique[None,:]))**2,axis=0)[inverse].reshape(lam.shape)
            metrics={k:v for k,v in identified.items() if k!='component_applicable'}
            metrics.update(projection_floor=np.full_like(regret,floor),regularization_bias_within_P=pop_bias,
                estimation_displacement_plus_cross=regret-floor-pop_bias,
                representation_adjusted_regret=regret-floor,
                representation_adjusted_ray_gap=identified['ray_gap']-floor)
            for metric,v in metrics.items():
                stats=distribution(v)
                for t,T in enumerate(times):summaries.append(dict(environment=name,T=int(T),specification=spec,metric=metric,**{k:float(vv[t]) for k,vv in stats.items()}))
                if metric in ('ray_gap','scale_gap','representation_adjusted_regret','representation_adjusted_ray_gap','regularization_bias_within_P') and np.all(v.mean(axis=0)>0):
                    for grid,mask in [('full',np.ones(len(times),bool)),('robustness_common',times<=3240)]:
                        for row in gap_slopes(times[mask],v[:,mask]):
                            if spec.startswith('oracle'):
                                for key in ('MCSE_delta','MCSE_jackknife','CI_low','CI_high'):row.pop(key,None)
                                row['uncertainty']='Descriptive; no adjustment for oracle grid selection.'
                            slopes.append(dict(environment=name,specification=spec,quantity=metric,grid=grid,**row))
            for i in range(300):
                for t,T in enumerate(times):rows.append(dict(environment=name,index=i,T=int(T),specification=spec,penalty=lam[i,t],
                    regret=regret[i,t],monthly_SR=sr[i,t],component_applicable=bool(identified['component_applicable'][i,t]),
                    identification='Unique scale-invariant quantities from signed SR and Q; u/v/c/d unavailable without sidecar.',
                    **{k:v[i,t] for k,v in metrics.items()}))
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'identified_financial_components.csv',index=False)
    pd.DataFrame(summaries).to_csv(DEST/'component_summary.csv',index=False)
    pd.DataFrame(slopes).to_csv(DEST/'component_slopes.csv',index=False)
    # Validate the identified quantities and spectral bias against actual
    # independently regenerated moments and coefficient-level decomposition.
    checks=0
    for path in sorted((DEST/'audit_fits').glob('*.csv')):
        recovered=pd.read_csv(path)
        for _,r in recovered[recovered.audit.isin(['A','C'])].iterrows():
            spec='initial_cv' if r.audit=='A' else 'old_fixed'
            original=frame[(frame.environment==r.environment)&(frame['index']==r['index'])&(frame['T']==r['T'])&(frame.specification==spec)].iloc[0]
            columns=['ray_gap','scale_gap','normalized_directional_error','regularization_bias_within_P']
            np.testing.assert_allclose(original[columns].astype(float),r[columns].astype(float),atol=1e-9,rtol=1e-7)
            np.testing.assert_allclose(original.estimation_displacement_plus_cross,r.estimation_displacement+r.signed_cross_term,atol=1e-9,rtol=1e-7)
            checks+=1
    json_write(DEST/'component_verification.json',dict(passed=True,policy_rows=len(frame),recovered_policy_checks=checks,
        ambiguous_moments_reconstructed=False,population_bias_evaluated_at_each_actual_penalty=True,
        displacement_and_cross_separately_identified_only_on_recovered_subset=True,spectral_sources=sources))

if __name__=='__main__':run()
