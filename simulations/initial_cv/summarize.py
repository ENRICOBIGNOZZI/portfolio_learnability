"""Three-specification comparisons from complete, verified paired production."""
import json
import numpy as np
import pandas as pd

from simulations.extended.design import ENVIRONMENTS, parameters, PENALTIES
from simulations.extended.statistics import distribution, gap_slopes
from simulations.initial_cv.design import OUTPUT, REFERENCE, protocol
from simulations.initial_cv.run import verify
from simulations.initial_cv.selector import A_GRID
from simulations.provenance import npz_write, json_write, file_hash, utc_now

SPECIFICATIONS = ('initial_cv', 'old_fixed', 'population_oracle')


def summarize():
    frozen = protocol()
    gate = json.loads((OUTPUT/'production_verification.json').read_text())
    if not gate['passed'] or gate['replications_per_economy'] != 300 or gate['run_hash'] != frozen['run_hash']:
        raise ValueError('Complete and independently verify all 300 paths first.')
    records=[]
    hashes={}
    for index in range(300):
        path=OUTPUT/'replications'/f'rep_{index:03d}.npz'
        verify(path,index,frozen)
        hashes[str(path.relative_to(OUTPUT))]=file_hash(path)
        with np.load(path) as z:
            records.append({k:z[k] for k in z.files})
    summaries=[];slopes=[];paired=[];alpha_rows=[];frequency=[];warnings=[];oracle_rows=[]
    old_warnings=pd.read_csv(REFERENCE/'production_resolution_final.csv')
    saved={}
    for name in ENVIRONMENTS:
        times=records[0][name+'_T']
        nT=len(times)
        a=np.array([float(r[name+'_selected_a']) for r in records])
        choices=np.array([int(r[name+'_selected_index']) for r in records])
        for index in range(300):
            alpha_rows.append(dict(environment=name,index=index,selected_a=a[index],selected_index=choices[index],
                ratio_to_old_a=a[index]/frozen['design']['a_old'][name],
                at_lower_boundary=bool(choices[index]==0),at_upper_boundary=bool(choices[index]==len(A_GRID)-1)))
        for j,candidate in enumerate(A_GRID):
            frequency.append(dict(environment=name,candidate_a=candidate,index=j,
                                  count=int(np.sum(choices==j)),fraction=float(np.mean(choices==j))))
        with np.load(REFERENCE/'distributions'/f'{name}.npz') as z:
            old={k:z[k] for k in z.files}
        np.testing.assert_array_equal(times,old['T'])
        oracle_index=np.argmax(old['annual_SR'][:,:,:96].mean(axis=0),axis=1)
        with np.load(REFERENCE/'population'/f'{name}_theory.npz') as z:
            spectrum=z['eigenvalues'];floor=float(z['projection_floor'])
        pop_oracle=np.sum(spectrum[:,None]/(spectrum[:,None]+PENALTIES[oracle_index]),axis=0)
        data={}
        for metric,source,scale in [('annual_SR','sr',np.sqrt(12)),('regret','loss',1.),
                                     ('empirical_complexity','empirical_complexity',1.),
                                     ('population_complexity','population_complexity',1.)]:
            two=np.stack([r[name+'_'+source] for r in records])*scale
            if metric=='regret':
                two-=1-parameters(name).q_star
            if metric=='population_complexity':
                third=np.broadcast_to(pop_oracle,(300,nT))
            else:
                third=old[metric][:,np.arange(nT),oracle_index]
            data[metric]=np.concatenate([two,third[:,:,None]],axis=2)
        data['annual_gap']=np.sqrt(12)*parameters(name).sr_star-data['annual_SR']
        data['relative_complexity']=data['empirical_complexity']/times[None,:,None]
        data['population_relative_complexity']=data['population_complexity']/times[None,:,None]
        for metric in ('annual_SR','regret','empirical_complexity'):
            np.testing.assert_allclose(data[metric][:,:,1],old[metric][:,:,-1],atol=1e-8,rtol=1e-7)
        penalties=np.stack([r[name+'_penalties'] for r in records])
        all_penalties=np.concatenate([penalties,np.broadcast_to(PENALTIES[oracle_index],(300,nT))[:,:,None]],axis=2)
        npz_write(OUTPUT/'distributions'/f'{name}.npz',T=times,**data,selected_a=a,
            penalties=all_penalties,oracle_penalty_index=oracle_index,specifications=SPECIFICATIONS,
            run_hash=frozen['run_hash'],replications=300)
        saved[name]=data
        for j,spec in enumerate(SPECIFICATIONS):
            for metric,values in data.items():
                stats=distribution(values[:,:,j])
                for t,T in enumerate(times):
                    summaries.append(dict(environment=name,T=int(T),specification=spec,metric=metric,
                        **{k:float(v[t]) for k,v in stats.items()}))
                if metric in ('annual_gap','empirical_complexity','relative_complexity','population_complexity','population_relative_complexity'):
                    for grid,mask in [('full',np.ones(nT,dtype=bool)),('robustness_common',times<=3240)]:
                        for row in gap_slopes(times[mask],values[:,mask,j]):
                            if spec=='population_oracle':
                                for key in ('MCSE_delta','MCSE_jackknife','CI_low','CI_high'):
                                    row.pop(key,None)
                                row['uncertainty']='Descriptive oracle slope; selection uses final population outcomes.'
                            slopes.append(dict(environment=name,specification=spec,quantity=metric,grid=grid,**row))
            for t,T in enumerate(times):
                original=old_warnings[(old_warnings.environment==name)&(old_warnings['T']==T)].iloc[0]
                regret=data['regret'][:,t,j]
                warnings.append(dict(environment=name,T=int(T),specification=spec,projection_floor=floor,
                    mean_regret=float(regret.mean()),floor_fraction=float(floor/regret.mean()),
                    floor_pass=bool(floor<=.05*regret.mean()),
                    fraction_paths_failing_floor=float(np.mean(floor>.05*regret)),
                    original_floor_pass=bool(original.floor_pass),original_paired_pass=bool(original.paired_pass),
                    original_forward_reference_available=bool(original.forward_reference_available),
                    original_certified=bool(original.certified),
                    strategy_specific_forward_rank_certificate=False,
                    warning='Original numerical limitations preserved; CV-selected policies have no new higher-rank certificate.'))
        for metric in ('annual_SR','annual_gap','regret','empirical_complexity'):
            stats=distribution(data[metric][:,:,0]-data[metric][:,:,1])
            for t,T in enumerate(times):
                paired.append(dict(environment=name,T=int(T),metric=metric,comparison='initial_cv_minus_old_fixed',
                    **{k:float(v[t]) for k,v in stats.items()},
                    mean_CI_low=float(stats['mean'][t]-1.96*stats['mcse'][t]),
                    mean_CI_high=float(stats['mean'][t]+1.96*stats['mcse'][t])))
        for t,T in enumerate(times):
            oracle_rows.append(dict(environment=name,T=int(T),penalty_index=int(oracle_index[t]),
                penalty=float(PENALTIES[oracle_index[t]]),scope='Common maximum of mean population Sharpe on original fixed 96-point grid; diagnostic only.'))
    for filename,rows in [('summary.csv',summaries),('slopes_all_windows.csv',slopes),
        ('paired_cv_vs_fixed.csv',paired),('selected_constants.csv',alpha_rows),
        ('selected_constant_frequencies.csv',frequency),('numerical_warnings.csv',warnings),('oracle_choices.csv',oracle_rows)]:
        pd.DataFrame(rows).to_csv(OUTPUT/filename,index=False)
    robustness=[]
    for name in ENVIRONMENTS:
        if name=='baseline':
            continue
        for j,spec in enumerate(SPECIFICATIONS[:2]):
            for metric in ('annual_SR','annual_gap','empirical_complexity'):
                nT=saved[name][metric].shape[1]
                stats=distribution(saved[name][metric][:,:,j]-saved['baseline'][metric][:,:nT,j])
                for t,T in enumerate(frozen['design']['robustness_T']):
                    robustness.append(dict(environment=name,reference='baseline',specification=spec,
                        metric=metric,T=T,**{k:float(v[t]) for k,v in stats.items()},
                        mean_CI_low=float(stats['mean'][t]-1.96*stats['mcse'][t]),
                        mean_CI_high=float(stats['mean'][t]+1.96*stats['mcse'][t])))
    pd.DataFrame(robustness).to_csv(OUTPUT/'paired_robustness.csv',index=False)
    alpha=pd.DataFrame(alpha_rows)
    statistics=[]
    for name in ENVIRONMENTS:
        values=alpha[alpha.environment==name]
        statistics.append(dict(environment=name,**distribution(values.selected_a.to_numpy()),
            geometric_mean=float(np.exp(np.log(values.selected_a).mean())),
            minimum=float(values.selected_a.min()),maximum=float(values.selected_a.max()),
            lower_boundary_count=int(values.at_lower_boundary.sum()),upper_boundary_count=int(values.at_upper_boundary.sum())))
    pd.DataFrame(statistics).to_csv(OUTPUT/'selected_constant_summary.csv',index=False)
    json_write(OUTPUT/'summary_verification.json',dict(passed=True,replications_per_economy=300,
        run_hash=frozen['run_hash'],checkpoint_sha256=hashes,verified_utc=utc_now(),
        specifications=list(SPECIFICATIONS),old_metrics_reproduced=True,
        original_oracle_input_sha256={str((REFERENCE/'distributions'/f'{name}.npz').relative_to(REFERENCE)):file_hash(REFERENCE/'distributions'/f'{name}.npz') for name in ENVIRONMENTS},
        original_warning_sha256=file_hash(REFERENCE/'production_resolution_final.csv'),
        bands='Central 95% replication outcomes; paired mean confidence intervals are separate.',
        oracle_uncertainty='Selected-grid distributions are descriptive, not selection-adjusted confidence intervals.'))


if __name__=='__main__':
    summarize()
