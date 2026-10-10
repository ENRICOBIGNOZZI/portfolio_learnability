"""Complete-distribution summaries, paired robustness and legacy decomposition."""
import json
import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT,REFERENCE,ENVIRONMENTS,parameters,PENALTIES,T_ORIGINAL
from simulations.extended.statistics import distribution,gap_slopes,deterministic_slopes
from simulations.extended.compute import load_operator
from simulations.estimator.ridge import moment_metrics
from simulations.provenance import npz_write,json_write,utc_now


def read_environment(name,protocol):
    stage='production_baseline' if name=='baseline' else 'production_robustness'
    records=[]
    for index in range(300):
        path=OUTPUT/stage/f'P{protocol["rank"]}'/f'rep_{index:03d}.npz'
        with np.load(path) as z:
            description=json.loads(str(z['description']))
            if description['run_hash']!=protocol['run_hash'] or int(z['index'])!=index:
                raise ValueError(f'Invalid production identity: {path}')
            row={key[len(name)+1:]:z[key] for key in z.files if key.startswith(name+'_')}
            row['returns_hashes']=json.loads(str(z['returns_hashes']))
            records.append(row)
    T=records[0]['T']
    for row in records:
        np.testing.assert_array_equal(row['T'],T)
    return records,T


def make_slopes(name,T,data,population,selected_times=None,grid='environment_full'):
    """Keep full-baseline slopes distinct from matched-grid robustness comparisons."""
    T=np.asarray(T)
    times=T if selected_times is None else np.asarray(selected_times)
    keep=np.isin(T,times)
    np.testing.assert_array_equal(T[keep],times)
    pop=population[population['T'].isin(times)].sort_values('T')
    np.testing.assert_array_equal(pop['T'],times)
    rows=[]
    metadata=dict(environment=name,grid=grid,T_max=int(times[-1]))
    for metric in ('annual_gap','empirical_complexity','relative_complexity'):
        rows.extend(dict(**metadata,quantity=metric,**row)
                    for row in gap_slopes(times,data[metric][:,keep,-1]))
    for metric in ('complexity','relative_complexity'):
        rows.extend(dict(**metadata,quantity='population_'+metric,**row)
                    for row in deterministic_slopes(times,pop[metric].values))
    return rows


def summarize():
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    summary=[];theory=[];slopes=[];common_slopes=[];paired=[];datasets={};hashes={}
    population=pd.read_csv(OUTPUT/'population_summary.csv')
    for name in ENVIRONMENTS:
        records,T=read_environment(name,protocol)
        p=parameters(name)
        sr=np.stack([r['sr'] for r in records])*np.sqrt(12)
        loss=np.stack([r['loss'] for r in records])
        complexity=np.stack([r['empirical_complexity'] for r in records])
        data=dict(annual_SR=sr,annual_gap=p.sr_star*np.sqrt(12)-sr,
                  regret=loss-(1-p.q_star),empirical_complexity=complexity,
                  relative_complexity=complexity/T[None,:,None],
                  coefficient_norm_squared=np.stack([r['coefficient_norm_squared'] for r in records]))
        decomp=np.stack([r['theory_decomposition'] for r in records])
        np.testing.assert_allclose(data['regret'][:,:,-1],decomp.sum(axis=2),atol=1e-10,rtol=1e-8)
        datasets[name]=data
        hashes[name]=[r['returns_hashes'] for r in records]
        npz_write(OUTPUT/'distributions'/f'{name}.npz',T=T,**data,
            theory_population_bias=decomp[:,:,0],theory_estimation_norm=decomp[:,:,1],
            theory_cross_term=decomp[:,:,2],run_hash=protocol['run_hash'],replications=300)
        for metric,values in data.items():
            stats=distribution(values)
            for t,horizon in enumerate(T):
                for j in range(values.shape[2]):
                    row=dict(environment=name,T=int(horizon),metric=metric,penalty_index=j,
                        penalty=float(PENALTIES[j] if j<len(PENALTIES) else protocol['a'][name]*float(horizon)**(-.6)),
                        exact_theory_choice=j==len(PENALTIES),replications=300,
                        **{key:float(value[t,j]) for key,value in stats.items()})
                    summary.append(row)
                    if row['exact_theory_choice']:
                        theory.append(row)
        for key,k in (('population_bias',0),('estimation_norm',1),('cross_term',2)):
            stats=distribution(decomp[:,:,k])
            for t,horizon in enumerate(T):
                theory.append(dict(environment=name,T=int(horizon),metric=key,penalty_index=96,
                    penalty=protocol['a'][name]*float(horizon)**(-.6),exact_theory_choice=True,
                    replications=300,**{field:float(value[t]) for field,value in stats.items()}))
        pop=population[population.environment==name].sort_values('T')
        slopes.extend(make_slopes(name,T,data,pop))
        common_slopes.extend(make_slopes(name,T,data,pop,protocol['robustness_T'],
                                        grid='robustness_common'))
    for name in ENVIRONMENTS:
        if name=='baseline':
            continue
        nt=datasets[name]['annual_SR'].shape[1]
        for index in range(300):
            if hashes[name][index]['baseline_T1440']!=hashes['baseline'][index]['baseline_T1440']:
                raise ValueError('Production stages failed economic-history pairing.')
            if hashes[name][index]['baseline']!=hashes['baseline'][index]['baseline_T3240']:
                raise ValueError('Production stages failed pairing over the full common horizon.')
        for metric in ('annual_SR','annual_gap','empirical_complexity'):
            diff=datasets[name][metric][:,:,-1]-datasets['baseline'][metric][:,:nt,-1]
            stats=distribution(diff)
            for t,T in enumerate(protocol['robustness_T']):
                paired.append(dict(environment=name,reference='baseline',metric=metric,T=T,
                    comparison='Paired innovation difference across independent replication indices',
                    **{key:float(value[t]) for key,value in stats.items()}))
    pd.DataFrame(summary).to_csv(OUTPUT/'full_penalty_distributions.csv',index=False)
    pd.DataFrame(theory).to_csv(OUTPUT/'theory_distributions.csv',index=False)
    pd.DataFrame(slopes).to_csv(OUTPUT/'slopes_all_windows.csv',index=False)
    pd.DataFrame(common_slopes).to_csv(OUTPUT/'slopes_robustness_common_grid.csv',index=False)
    pd.DataFrame(paired).to_csv(OUTPUT/'paired_robustness.csv',index=False)
    legacy_comparison(protocol,datasets['baseline'],hashes['baseline'])
    json_write(OUTPUT/'summary_verification.json',dict(replications_per_environment=300,
        environments=5,all_paired_histories_verified=True,all_regret_decompositions_verified=True,
        primary_bands='Central 95% replication percentiles',completed_utc=utc_now(),run_hash=protocol['run_hash']))


def legacy_comparison(protocol,current,hashes):
    old=[];reevaluated=[]
    m,S,_=load_operator('baseline',512,protocol['population_groups'],protocol['population_seed'])
    for index in range(100):
        with np.load(REFERENCE/'replications'/f'{index:04d}.npz') as z:
            if str(z['stock_return_sha256'])!=hashes[index]['baseline_T1440']:
                raise ValueError('The new baseline is not paired with the historical reference.')
            old.append(z['population_sharpe'][:,-1]*np.sqrt(12))
            _,sr=moment_metrics(z['theory_coefficients'].T,m,S)
            reevaluated.append(sr*np.sqrt(12))
    old=np.asarray(old);reevaluated=np.asarray(reevaluated)
    first100=current['annual_SR'][:100,:len(T_ORIGINAL),-1]
    all300=current['annual_SR'][:,:len(T_ORIGINAL),-1]
    rows=[]
    for t,T in enumerate(T_ORIGINAL):
        rows.append(dict(T=T,old100_rank512_old_quadrature=old[:,t].mean(),
            old100_rank512_new_quadrature=reevaluated[:,t].mean(),
            new100_selected_rank=first100[:,t].mean(),new300_selected_rank=all300[:,t].mean(),
            quadrature_contribution=float((reevaluated-old)[:,t].mean()),
            rank_contribution=float((first100-reevaluated)[:,t].mean()),
            additional_replications_contribution=float(all300[:,t].mean()-first100[:,t].mean()),
            first100_paired_rank_MCSE=float((first100-reevaluated)[:,t].std(ddof=1)/10)))
    pd.DataFrame(rows).to_csv(OUTPUT/'comparison_previous_100.csv',index=False)
    slope_comparison=[]
    fullT=np.asarray(protocol['baseline_T'])
    for label,T,sr in (('old100_rank512_old_Q',T_ORIGINAL,old),
        ('old100_rank512_new_Q',T_ORIGINAL,reevaluated),
        ('new100_selected_rank_original_grid',T_ORIGINAL,first100),
        ('new300_selected_rank_original_grid',T_ORIGINAL,all300),
        ('new300_selected_rank_extended_grid',fullT,current['annual_SR'][:,:,-1])):
        for row in gap_slopes(T,3-np.asarray(sr)):
            slope_comparison.append(dict(comparison=label,**row))
    pd.DataFrame(slope_comparison).to_csv(OUTPUT/'comparison_slopes_previous_100.csv',index=False)

if __name__=='__main__':
    summarize()
