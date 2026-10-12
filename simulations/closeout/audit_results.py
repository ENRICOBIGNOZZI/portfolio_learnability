"""Paired numerical diagnostics with no unsupported single-history MCSE."""
import json
import numpy as np
import pandas as pd
from simulations.closeout.results import DEST
from simulations.provenance import json_write


def paired_bound(values):
    values=np.asarray(values,float)
    if not np.isfinite(values).all(): return np.nan,np.nan,np.nan
    mean=float(values.mean())
    if len(values)<2:return mean,np.nan,np.nan
    mcse=float(values.std(ddof=1)/np.sqrt(len(values)))
    return mean,mcse,abs(mean)+1.96*mcse


def summarize_audit():
    path=DEST/'audit_policy_moments.csv'
    if not path.exists():
        json_write(DEST/'numerical_status.json',dict(status='not_run',latex_scope='The new rank and quadrature audit has not run; CV-specific certification remains unresolved.'))
        return
    data=pd.read_csv(path)
    # A short and full replay of one index are not independent histories.
    data=data.sort_values('sample_scope').drop_duplicates(['environment','index','T','rank','audit','quadrature'],keep='first')
    comparisons=[]
    keys=['environment','T','audit','cohort']
    metrics=['regret','u','v','annual_gap','normalized_directional_error']
    for values,part in data.groupby(keys):
        for metric in metrics:
            low=part[(part['rank']==4096)&(part.quadrature=='primary')].set_index('index')[metric]
            high=part[(part['rank']==8192)&(part.quadrature=='primary')].set_index('index')[metric]
            ids=low.index.intersection(high.index)
            delta=high.loc[ids].to_numpy()-low.loc[ids].to_numpy()
            mean,mcse,bound=paired_bound(delta)
            if values[-1]=='additional_stress': mcse,bound=np.nan,np.nan
            denominator=abs(float(low.loc[ids].mean()))
            comparisons.append(dict(zip(keys,values),comparison='rank8192_minus4096',metric=metric,
                histories=len(ids),paired_mean=mean,paired_mcse=mcse,paired_bound=bound,denominator=denominator,
                point_fraction=abs(mean)/denominator if denominator>0 else np.nan,
                fraction=bound/denominator if denominator>0 else np.nan,
                passes=bool(len(ids)>=2 and denominator>0 and bound<=.05*denominator),
                status='descriptive_stress' if values[-1]=='additional_stress' else ('not_applicable' if not np.isfinite(mean) else ('not_estimable' if len(ids)<2 else ('passed' if bound<=.05*denominator else 'failed')))))
            for rank in (4096,8192):
                primary=part[(part['rank']==rank)&(part.quadrature=='primary')].set_index('index')[metric]
                independent=part[(part['rank']==rank)&(part.quadrature=='independent')].set_index('index')[metric]
                ids=primary.index.intersection(independent.index)
                mean,mcse,bound=paired_bound(independent.loc[ids].to_numpy()-primary.loc[ids].to_numpy())
                if values[-1]=='additional_stress': mcse,bound=np.nan,np.nan
                denominator=abs(float(primary.loc[ids].mean()))
                comparisons.append(dict(zip(keys,values),comparison=f'independent_minus_primary_P{rank}',metric=metric,
                    histories=len(ids),paired_mean=mean,paired_mcse=mcse,paired_bound=bound,denominator=denominator,
                    point_fraction=abs(mean)/denominator if denominator>0 else np.nan,
                    fraction=bound/denominator if denominator>0 else np.nan,
                    passes=bool(len(ids)>=2 and denominator>0 and bound<=.05*denominator),
                    status='descriptive_stress' if values[-1]=='additional_stress' else ('not_applicable' if not np.isfinite(mean) else ('not_estimable' if len(ids)<2 else ('passed' if bound<=.05*denominator else 'failed')))))
    result=pd.DataFrame(comparisons);result.to_csv(DEST/'audit_paired_comparisons.csv',index=False)
    plan=json.loads((DEST/'audit_protocol.json').read_text())['design']
    stress_rows=[]
    for name,index in plan['stress_indices_by_economy'].items():
        selected=data[(data.environment==name)&(data['index']==index)]
        for (T,audit),part in selected.groupby(['T','audit']):
            for metric in metrics:
                low=part[(part['rank']==4096)&(part.quadrature=='primary')][metric].iloc[0]
                high=part[(part['rank']==8192)&(part.quadrature=='primary')][metric].iloc[0]
                ratio=abs(high-low)/abs(low) if np.isfinite(low) and low!=0 else np.nan
                stress_rows.append(dict(environment=name,index=index,T=int(T),audit=audit,metric=metric,comparison='rank8192_minus4096',
                    relative_change=ratio,passes=bool(np.isfinite(ratio) and ratio<=.05),status='not_applicable' if not np.isfinite(ratio) else ('passed' if ratio<=.05 else 'failed'),
                    scope='Individual predeclared minimum-a stress diagnostic; no IID MCSE or global certificate.'))
                for rank in [4096,8192]:
                    primary=part[(part['rank']==rank)&(part.quadrature=='primary')][metric].iloc[0]
                    independent=part[(part['rank']==rank)&(part.quadrature=='independent')][metric].iloc[0]
                    ratio=abs(independent-primary)/abs(primary) if np.isfinite(primary) and primary!=0 else np.nan
                    stress_rows.append(dict(environment=name,index=index,T=int(T),audit=audit,metric=metric,comparison=f'independent_minus_primary_P{rank}',
                        relative_change=ratio,passes=bool(np.isfinite(ratio) and ratio<=.05),status='not_applicable' if not np.isfinite(ratio) else ('passed' if ratio<=.05 else 'failed'),
                        scope='Individual predeclared minimum-a stress diagnostic; no IID MCSE or global certificate.'))
    pd.DataFrame(stress_rows).to_csv(DEST/'audit_individual_stress.csv',index=False)

    choices=[]
    for path in sorted((DEST/'audit_fits').glob('*.npz')):
        with np.load(path) as z:
            for name in ('baseline','N300','N1200','rho000','rho075'):
                choices.append(dict(index=int(z['index']),environment=name,sample_scope=str(z['sample_scope']),cohort='representative' if int(z['index'])<12 else 'additional_stress',
                    a4096=z[name+'_selected_a'][0],a8192=z[name+'_selected_a'][1],same_choice=bool(z[name+'_selected_a'][0]==z[name+'_selected_a'][1])))
    choices=pd.DataFrame(choices).drop_duplicates(['index','environment'])
    choices.to_csv(DEST/'audit_CV_choices.csv',index=False)
    scope=json.loads((DEST/'audit_evaluation.json').read_text())
    full=scope['full_representative_indices'];short=scope['short_representative_indices']
    plan=json.loads((DEST/'audit_protocol.json').read_text())['design']
    missing_representative_full=sorted(set(plan['representative_indices'])-set(full))
    missing_stress=sorted(set(plan['stress_indices_by_economy'].values())-set(full))
    failed=result[result.status=='failed'];not_estimable=result[result.status=='not_estimable']
    text=(f'The CV audit re-estimates full declared horizons for {len(full)} original production histories; any remaining short-only histories number {len(set(short)-set(full))}. '
          f'There are {len(failed)} failed numerical metric comparisons and {len(not_estimable)} comparisons with insufficient independent histories for MCSE. '
          f'Rank-specific initial CV changes {int((~choices.same_choice).sum())} of {len(choices)} checked selections. '
          'Representative-history MCSE is computed separately from minimum-a stress diagnostics; stress cases are never pooled into an IID MCSE. '
          ('The declared twelve-history and minimum-a stress coverage is complete. ' if not missing_representative_full and not missing_stress else 'Declared full-horizon representative/stress coverage remains incomplete. ') +
          'Rank 8192 has no forward-rank certificate. Population-ridge decomposition at rank 8192 has not run. '
          'Quadrature differences are systematic evaluation diagnostics; replication MCSE does not quantify uncertainty over quadrature seeds.')
    json_write(DEST/'numerical_status.json',dict(status='completed_with_resolution_limits' if not missing_representative_full and not missing_stress else 'unresolved',audit_completed=not missing_representative_full and not missing_stress,full_indices=full,short_indices=short,
        representative_full_indices_not_run=missing_representative_full,stress_full_indices_not_run=missing_stress,
        failed_metric_comparisons=len(failed),not_estimable_comparisons=len(not_estimable),
        individual_stress_passed=sum(r['status']=='passed' for r in stress_rows),individual_stress_failed=sum(r['status']=='failed' for r in stress_rows),
        changed_CV_choices=int((~choices.same_choice).sum()),checked_CV_choices=len(choices),
        high_rank_population_ridge='not_run',forward_rank_certificate=False,latex_scope=text))

if __name__=='__main__':summarize_audit()
