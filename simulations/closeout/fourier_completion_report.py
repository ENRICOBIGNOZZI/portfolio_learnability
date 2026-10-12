"""Full declared Fourier audit: fixed-a and rank/M-specific initial CV separate.

Reuse M128 rank/quadrature results; evaluate fixed M512/M2048 policies under
both their own economy and the common M128 economy, using both fixed seeds.
The latter separates policy changes from changes in evaluation moments.
Stress cases remain descriptive; only fixed representatives supply IID MCSE.
"""
from dataclasses import replace
import json
import time
import numpy as np
import pandas as pd
from simulations.closeout.results import DEST
from simulations.closeout.completion import FOURIER,specification
from simulations.closeout.metrics import financial_decomposition
from simulations.extended.design import ENVIRONMENTS,parameters
from simulations.extended.rank_followup import RawBasis,evaluate
from simulations.provenance import file_hash,json_write,utc_now


def coverage():
    plan=specification();records=[];sources={}
    for i in plan['indices']:
        path=FOURIER/f'rep_{i:03d}_full.npz'
        if not path.exists():raise ValueError(f'Missing full Fourier audit index {i}.')
        receipt=json.loads(path.with_suffix('.json').read_text())
        if file_hash(path)!=receipt['sha256'] or file_hash(path.with_suffix('.csv'))!=receipt['solver_sha256']:raise ValueError('Changed Fourier checkpoint or solver receipt.')
        base=DEST/'audit_fits'/path.name
        if file_hash(base)!=receipt['base_fit_sha256']:raise ValueError('Base fitted policies changed.')
        with np.load(path) as z:record={k:z[k] for k in z.files}
        for name in ENVIRONMENTS:
            expected=plan['baseline_times'] if name=='baseline' else plan['robustness_times']
            np.testing.assert_array_equal(record[name+'_T'],expected)
            for M in [512,2048]:
                if record[f'{name}_M{M}_raw_coefficients'].shape!=(3,2,8192,3):raise ValueError('Incomplete Fourier policy coverage.')
        records.append(record);sources[str(path)]=file_hash(path)
    return plan,records,sources


def run_evaluations():
    plan,records,sources=coverage();base=pd.read_csv(DEST/'audit_policy_moments.csv')
    base=base[base.sample_scope=='full'].copy()
    base['fit_M']=128;base['evaluation_M']=128
    rows=base.to_dict('records');basis=RawBasis(8192);started=time.perf_counter()
    dest=DEST/'fourier_full_evaluation_cache';dest.mkdir(exist_ok=True)
    for names in [('baseline','rho000','rho075'),('N300',),('N1200',)]:
        for fit_M in [512,2048]:
            columns=[];labels=[]
            for record in records:
                i=int(record['index'])
                for name in names:
                    for t,T in enumerate(record[name+'_T']):
                        for r,rank in enumerate([4096,8192]):
                            for j,audit in enumerate(['A','B','C']):
                                columns.append(record[f'{name}_M{fit_M}_raw_coefficients'][t,r,:,j])
                                labels.append(dict(environment=name,index=i,T=int(T),rank=rank,audit=audit,fit_M=fit_M,penalty=record[f'{name}_M{fit_M}_penalties'][t,r,j],
                                    cohort='representative' if i<12 else 'additional_stress',sample_scope='full'))
            coefficients=np.column_stack(columns)
            for evaluation_M in [128,fit_M]:
                for quad,count,qseed in [('primary',plan['primary_groups'],plan['primary_seed']),('independent',plan['independent_groups'],plan['independent_seed'])]:
                    key=f'{names[0]}_fit{fit_M}_eval{evaluation_M}_{quad}'
                    cache=dest/(key+'.csv');receipt=dest/(key+'.json')
                    if cache.exists():
                        saved=json.loads(receipt.read_text())
                        if saved['sources']!=sources or saved['sha256']!=file_hash(cache):raise ValueError('Changed Fourier evaluation cache.')
                        part=pd.read_csv(cache);rows.extend(part.to_dict('records'));continue
                    measured=evaluate(basis,coefficients,count,qseed,replace(parameters(names[0]),fourier_terms=evaluation_M))
                    part=[]
                    for j,label in enumerate(labels):
                        moments=financial_decomposition(measured['mean'][j],measured['second'][j],parameters(label['environment']).q_star)
                        part.append(dict(**label,evaluation_M=evaluation_M,quadrature=quad,groups=count,seed=qseed,**{k:np.asarray(v).item() for k,v in moments.items()}))
                    frame=pd.DataFrame(part);frame.to_csv(cache,index=False)
                    json_write(receipt,dict(sha256=file_hash(cache),sources=sources,completed_utc=utc_now()))
                    rows.extend(part)
                    print(f'Full Fourier evaluation {key}: {time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'fourier_full_policy_moments.csv',index=False)
    choices=[]
    for record in records:
        for name in ENVIRONMENTS:
            with np.load(DEST/'audit_fits'/f'rep_{int(record["index"]):03d}_full.npz') as original:
                a128=original[name+'_selected_a']
            for M in [512,2048]:
                for r,rank in enumerate([4096,8192]):choices.append(dict(environment=name,index=int(record['index']),rank=rank,M=M,a128=a128[r],a_M=record[f'{name}_M{M}_selected_a'][r],choice_unchanged=bool(a128[r]==record[f'{name}_M{M}_selected_a'][r])))
    pd.DataFrame(choices).to_csv(DEST/'fourier_full_CV_choices.csv',index=False)
    json_write(DEST/'fourier_full_evaluation.json',dict(passed=True,rows=len(frame),source_sha256=sources,
        moments_sha256=file_hash(DEST/'fourier_full_policy_moments.csv'),indices=plan['indices'],completed_utc=utc_now(),
        own_economy_and_common_M128_evaluation=True,primary_and_independent_quadrature=True))


def summarize():
    frame=pd.read_csv(DEST/'fourier_full_policy_moments.csv');rows=[]
    metrics=['regret','u','v','annual_gap','normalized_directional_error']
    keys=['environment','T','rank','audit','cohort','quadrature']
    for group,part in frame.groupby(keys):
        base=part[(part.fit_M==128)&(part.evaluation_M==128)].set_index('index')
        for M in [512,2048]:
            for evaluation_M in [128,M]:
                changed=part[(part.fit_M==M)&(part.evaluation_M==evaluation_M)].set_index('index')
                indices=base.index.intersection(changed.index)
                for metric in metrics:
                    delta=changed.loc[indices,metric]-base.loc[indices,metric]
                    denominator=abs(float(base.loc[indices,metric].mean()))
                    average=float(delta.mean());n=len(indices)
                    if group[4]=='representative' and n>=2:
                        mcse=float(delta.std(ddof=1)/np.sqrt(n));bound=abs(average)+1.96*mcse
                        status='passed' if denominator>0 and bound<=.05*denominator else 'failed'
                    else:mcse=bound=np.nan;status='descriptive_stress'
                    rows.append(dict(zip(keys,group),fit_M=M,evaluation_M=evaluation_M,metric=metric,histories=n,
                        paired_mean=average,paired_mcse=mcse,paired_bound=bound,denominator=denominator,
                        fraction=bound/denominator if denominator>0 else np.nan,
                        maximum_individual_relative_change=float((np.abs(delta)/np.abs(base.loc[indices,metric])).max()),status=status,
                        changes_policy_only=evaluation_M==128))
    result=pd.DataFrame(rows);result.to_csv(DEST/'fourier_full_sensitivity.csv',index=False)
    choices=pd.read_csv(DEST/'fourier_full_CV_choices.csv')
    complete_representative=result[(result.cohort=='representative')&(result.histories==12)]
    if len(complete_representative)!=len(result[result.cohort=='representative']):raise ValueError('Missing representative Fourier cells.')
    json_write(DEST/'fourier_completion_status.json',dict(audit_completed=True,representative_histories=12,additional_stress_histories=4,
        horizons=[60,720,3240,4860],ranks=[4096,8192],terms=[128,512,2048],
        representative_sensitivity_failed=int((complete_representative.status=='failed').sum()),
        representative_sensitivity_passed=int((complete_representative.status=='passed').sum()),
        CV_choices_changed=int((~choices.choice_unchanged).sum()),CV_choices_checked=len(choices),
        maximum_rank_forward_certificate=False,remaining_declared_Fourier_cells=[],thresholds_changed=False))

if __name__=='__main__':
    run_evaluations();summarize()
