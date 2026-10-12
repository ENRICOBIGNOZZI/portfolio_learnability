"""Explicitly requested, bounded audit; never launches a production campaign.

freeze -> fit --index i [--short] -> evaluate. Representative indices are
original production seeds 0..11, not historical rank-pilot seeds. Low-rank
population decompositions use the archived, checksummed numerical operator.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular, eigh
from simulations.closeout.results import DEST, CANONICAL, REFERENCE
from simulations.closeout.metrics import payoff_moments, financial_decomposition, ridge_decomposition, normal_residual
from simulations.initial_cv.design import protocol
from simulations.initial_cv.selector import calibrate_initial, EXPONENT
from simulations.extended.design import ENVIRONMENTS, BASIS_SEED, seed, parameters
from simulations.extended.histories import raw_managed_history
from simulations.extended.population import basis_cached
from simulations.extended.compute import load_operator
from simulations.extended.rank_followup import RawBasis, evaluate
from simulations.run_rough import seed as original_seed
from simulations.provenance import json_write, npz_write, file_hash, digest, utc_now

CACHE=DEST/'cache'


def freeze():
    frozen=protocol()
    alpha_path=CANONICAL/'selected_constants.csv'
    verification=json.loads((DEST/'input_verification.json').read_text())
    if file_hash(alpha_path)!=verification['sources'][str(alpha_path)]:
        raise ValueError('Archived initial constants changed.')
    alpha=pd.read_csv(alpha_path)
    stress={name:int(alpha[alpha.environment==name].sort_values(['selected_a','index']).iloc[0]['index']) for name in ENVIRONMENTS}
    plan=dict(schema='cv-closeout-audit/1',original_run_hash=frozen['run_hash'],ranks=[4096,8192],
        representative_indices=list(range(12)),stress_indices_by_economy=stress,
        stress_selection='Smallest archived initial constant, index tie-break; no performance or slope selection.',
        baseline_times=[60,720,4860],robustness_times=[60,720,3240],
        primary_groups=32768,primary_seed=seed('population'),independent_groups=131072,independent_seed=seed('quadrature',1),
        A='Hold archived rank4096 a fixed at both ranks.',B='Recalibrate only the identical first60 observations at each rank.',
        C='Historical old-fixed comparison; diagnostic only.',
        tolerances={'projection_floor_over_mean_regret':.05,'paired_absolute_mean_plus_1_96_MCSE_over_low_rank_mean_regret':.05,
          'quadrature_absolute_mean_plus_1_96_MCSE_over_primary_mean_regret':.05,
          'relative_u':.05,'relative_v':.05,'relative_annual_gap':.05,'relative_normalized_directional_error':.05,
          'normal_equation_absolute':1e-8,'normal_equation_relative':1e-8,'rounding_negative_error':1e-10},
        inference='Subset-specific only. Rank8192 has no forward-rank certificate. Short audits never certify long horizons.',
        resource_estimate={'machine_RAM_bytes':8589934592,'worker_limit':1,'dense_8192_matrix_bytes':8192**2*8,
          'estimated_peak_bytes':4_500_000_000,'full_representative_histories':12,'full_stress_histories_upper_bound':len(set(stress.values())-set(range(12))),
          'historical_remote_baseline_seconds_P4096':148.6,'historical_remote_population_seconds_P8192':406.2,
          'local_full_fit_seconds_range_per_path':[300,3600],'representative_fit_hours_range':[1,12],
          'shared_evaluation_seconds_range':[300,7200],
          'estimate_basis':'Historical 32-core remote timings; local 8-core timings uncertain. Measure one worker first.',
          'cloud_spending_authorized':False,'full_production_authorized':False},
        created_before_fits=True,analysis_timing='Post hoc numerical diagnostics; unchanged production protocol.')
    path=DEST/'audit_protocol.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['design']!=plan or saved['protocol_hash']!=digest(plan): raise ValueError('Audit protocol changed.')
    else: json_write(path,dict(design=plan,protocol_hash=digest(plan),frozen_utc=utc_now()))
    CACHE.mkdir(exist_ok=True)
    return plan


def basis_pair():
    low=basis_cached(CACHE,4096,BASIS_SEED)
    high=RawBasis(8192)
    np.testing.assert_array_equal(low.anchors,high.anchors[:4096])
    chol_path=CACHE/'chol8192.npy'
    if not chol_path.exists():
        start=time.perf_counter()
        np.save(chol_path,cholesky(high.raw(high.anchors),lower=True,check_finite=False))
        print(f'High basis Cholesky {time.perf_counter()-start:.1f}s',flush=True)
    return low,high,np.load(chol_path)


def fit(index,short=False,prepared=None,fourier_deltas=None,generation_receipt=None):
    plan=freeze()
    if index not in set(plan['representative_indices'])|set(plan['stress_indices_by_economy'].values()):
        raise ValueError('Index outside fixed representative/stress protocol.')
    suffix='short' if short else 'full'
    path=DEST/'audit_fits'/f'rep_{index:03d}_{suffix}.npz'
    receipt=path.with_suffix('.json')
    base_exists=path.exists()
    fourier_path=DEST/'fourier_full_fits'/f'rep_{index:03d}_full.npz'
    if base_exists:
        saved=json.loads(receipt.read_text())
        if file_hash(path)!=saved['sha256'] or saved['protocol_hash']!=digest(plan) or file_hash(path.with_suffix('.csv'))!=saved['moments_sha256']: raise ValueError('Altered fit or manifest.')
        if fourier_deltas is None or fourier_path.exists(): return
    started=time.perf_counter()
    low,high,L=basis_pair()
    maxT=60 if short else 4860; robustT=60 if short else 3240
    if prepared is None:
        raw,hashes=raw_managed_history(high,original_seed('train',index),index,maxT,robustT,
            progress=lambda t:print(f'CV-audit path {index}: T={t}/{maxT}',flush=True))
        generation_seconds=time.perf_counter()-started
    else: raw,hashes,generation_seconds=prepared
    results={};moment_rows=[];fourier_results={};fourier_solver_rows=[]
    reference_path=CANONICAL/'replications'/f'rep_{index:03d}.npz'
    input_manifest=json.loads((DEST/'input_verification.json').read_text())
    if file_hash(reference_path)!=input_manifest['sources'][str(reference_path)]:
        raise ValueError('Original checkpoint changed after input verification.')
    with np.load(reference_path) as z: reference={k:z[k] for k in z.files}
    if not short:
        expected=json.loads(str(reference['returns_hashes']))
        for key in expected:
            if hashes[key]!=expected[key]: raise ValueError('Original production history changed.')
    for name in ENVIRONMENTS:
        times=np.array([60] if short else plan['baseline_times'] if name=='baseline' else plan['robustness_times'])
        raw_name=raw.pop(name)
        Xlow=raw_name[:,:4096]@low.inverse_root
        Xhigh=solve_triangular(L,raw_name.T,lower=True,check_finite=False).T
        del raw_name
        original_a=float(reference[name+'_selected_a'])
        selections=[calibrate_initial(Xlow[:60]),calibrate_initial(Xhigh[:60])]
        np.testing.assert_allclose(selections[0]['fold_losses'],reference[name+'_fold_losses'],rtol=1e-7,atol=1e-9)
        if selections[0]['selected_a']!=original_a: raise ValueError('Original low-rank CV choice not reproduced.')
        results[name+'_T']=times
        results[name+'_selected_a']=np.array([original_a,selections[1]['selected_a']])
        results[name+'_fold_losses']=np.stack([s['fold_losses'] for s in selections])
        results[name+'_validation_scores']=np.stack([s['mean_validation_loss'] for s in selections])
        if fourier_deltas is not None:
            def whiten_delta(M,rank,T):
                raw_delta=fourier_deltas[M][name][:T]
                return raw_delta[:,:4096]@low.inverse_root if rank==4096 else solve_triangular(L,raw_delta.T,lower=True,check_finite=False).T
            selected_F={M:{} for M in (512,2048)}
            for M in selected_F:
                for rank,X in [(4096,Xlow),(8192,Xhigh)]:
                    selected_F[M][rank]=calibrate_initial(X[:60]+whiten_delta(M,rank,60))
                fourier_results[f'{name}_M{M}_selected_a']=np.array([selected_F[M][rank]['selected_a'] for rank in (4096,8192)])
                fourier_results[f'{name}_M{M}_fold_losses']=np.stack([selected_F[M][rank]['fold_losses'] for rank in (4096,8192)])
            fourier_results[name+'_T']=times
            f_policies={M:[] for M in selected_F};f_penalties={M:[] for M in selected_F}

        m,S,floor=load_operator(name,4096,plan['primary_groups'],plan['primary_seed'])
        eig_name='baseline' if name.startswith('rho') else name
        eig_path=CACHE/f'eigensystem_{eig_name}.npz'
        if eig_path.exists():
            with np.load(eig_path) as cached:
                values,vectors=cached['values'],cached['vectors']
        else:
            values,vectors=eigh(S)
            npz_write(eig_path,values=values,vectors=vectors)
        results[name+'_managed_prefix_low']=Xlow[:60].copy()
        results[name+'_managed_prefix_high']=Xhigh[:60].copy()
        if name=='baseline':
            energy=(vectors.T@m)**2/values
            npz_write(DEST/'spectrum_energy.npz',eigenvalues=values[::-1],target_energy=energy[::-1],
                projection_floor=floor,q=parameters(name).q_star)
        policies=[];complexities=[];residuals=[];penalties=[]
        for T in times:
            per_rank=[];per_complexity=[];per_residual=[];per_lambda=[]
            if fourier_deltas is not None: f_per_rank={M:[] for M in selected_F};f_per_lambda={M:[] for M in selected_F}
            position=int(np.flatnonzero(reference[name+'_T']==T)[0])
            for rank,X,sel in [(4096,Xlow,selections[0]),(8192,Xhigh,selections[1])]:
                # Three fitted policies: A=frozen production a, B=rank-specific initial CV, C=old-fixed.
                lam=np.array([original_a,sel['selected_a'],protocol()['design']['a_old'][name]])*float(T)**EXPONENT
                if fourier_deltas is None: coef,C=ridge_path_local(X[:T],lam)
                else:
                    from simulations.closeout.perturb_ridge import RidgeFactor
                    factor=RidgeFactor(X[:T]);coef,C=factor.path(lam)
                error,relative=normal_residual(X[:T],coef,lam)
                if error>1e-8 or relative>1e-8: raise ValueError('Normal residual threshold failed.')
                raw_coef=low.inverse_root@coef if rank==4096 else solve_triangular(L.T,coef,lower=False,check_finite=False)
                per_rank.append(np.pad(raw_coef,((0,8192-rank),(0,0))))
                per_complexity.append(C);per_residual.append([error,relative]);per_lambda.append(lam)
                if fourier_deltas is not None:
                    for M in selected_F:
                        penalties_F=np.array([original_a,selected_F[M][rank]['selected_a'],protocol()['design']['a_old'][name]])*float(T)**EXPONENT
                        delta=whiten_delta(M,rank,T)
                        coefficient_F,solver=factor.perturb(delta,penalties_F)
                        raw_F=low.inverse_root@coefficient_F if rank==4096 else solve_triangular(L.T,coefficient_F,lower=False,check_finite=False)
                        f_per_rank[M].append(np.pad(raw_F,((0,8192-rank),(0,0))))
                        f_per_lambda[M].append(penalties_F)
                        fourier_solver_rows.append(dict(environment=name,index=index,T=int(T),rank=rank,M=M,**solver))
                        del delta,coefficient_F,raw_F
                    del factor

                if rank==4096:
                    u,v=payoff_moments(coef,m,S)
                    np.testing.assert_allclose(1-2*u[[0,2]]+v[[0,2]],reference[name+'_loss'][position],rtol=1e-7,atol=1e-9)
                    finance=financial_decomposition(u,v,parameters(name).q_star)
                    np.testing.assert_allclose(finance['monthly_SR'][[0,2]],reference[name+'_sr'][position],rtol=1e-7,atol=1e-9)
                    ridge,population,optimum=ridge_decomposition(coef,m,S,parameters(name).q_star,lam,(values,vectors))
                    for j,check in enumerate(('A','B','C')):
                        moment_rows.append(dict(environment=name,index=index,T=int(T),rank=rank,audit=check,
                            penalty=lam[j],sample_scope=suffix,**{k:np.asarray(vv)[j].item() for k,vv in finance.items()},
                            **{k:vv[j] for k,vv in ridge.items()}))
                    results[name+f'_low_coefficients_T{T}']=coef
                    results[name+f'_population_ridge_T{T}']=population
            policies.append(np.stack(per_rank));complexities.append(per_complexity);residuals.append(per_residual);penalties.append(per_lambda)
            if fourier_deltas is not None:
                for M in selected_F:
                    f_policies[M].append(np.stack(f_per_rank[M]));f_penalties[M].append(np.array(f_per_lambda[M]))

        results[name+'_raw_coefficients']=np.stack(policies)
        results[name+'_empirical_complexity']=np.array(complexities)
        results[name+'_normal_residuals']=np.array(residuals)
        results[name+'_penalties']=np.array(penalties)
        if fourier_deltas is not None:
            for M in selected_F:
                fourier_results[f'{name}_M{M}_raw_coefficients']=np.stack(f_policies[M])
                fourier_results[f'{name}_M{M}_penalties']=np.stack(f_penalties[M])

        del Xlow,Xhigh,S,vectors
        print(f'Audit {index} {name}: low metrics reproduced; both rank fits saved',flush=True)
    if base_exists:
        with np.load(path) as old:
            for name in ENVIRONMENTS:
                np.testing.assert_allclose(results[name+'_raw_coefficients'],old[name+'_raw_coefficients'],rtol=1e-7,atol=1e-7)
    else:
        prefixes={k:results.pop(k) for k in list(results) if 'managed_prefix' in k}
        if prefixes: npz_write(CACHE/'training_prefixes'/path.name,**prefixes,index=index,protocol_hash=digest(plan))
        npz_write(path,**results,index=index,protocol_hash=digest(plan),training_seed=original_seed('train',index),
            returns_hashes=json.dumps(hashes,sort_keys=True),sample_scope=suffix)
        pd.DataFrame(moment_rows).to_csv(path.with_suffix('.csv'),index=False)
        json_write(receipt,dict(sha256=file_hash(path),moments_sha256=file_hash(path.with_suffix('.csv')),
            protocol_hash=digest(plan),index=index,sample_scope=suffix,source_checkpoint_sha256=file_hash(CANONICAL/'replications'/f'rep_{index:03d}.npz'),
            generation_seconds=generation_seconds,elapsed_seconds=time.perf_counter()-started,completed_utc=utc_now(),
            numpy_version=np.__version__,
            cache_sha256={p.name:file_hash(p) for p in CACHE.glob('*') if p.is_file()}))
    if fourier_deltas is not None:
        from simulations.closeout.completion import specification
        description=specification()
        npz_write(fourier_path,**fourier_results,index=index,training_seed=original_seed('train',index),protocol_hash=digest(description),
            original_a=np.array([reference[n+'_selected_a'] for n in ENVIRONMENTS]))
        pd.DataFrame(fourier_solver_rows).to_csv(fourier_path.with_suffix('.csv'),index=False)
        json_write(fourier_path.with_suffix('.json'),dict(index=index,sha256=file_hash(fourier_path),solver_sha256=file_hash(fourier_path.with_suffix('.csv')),
            protocol_hash=digest(description),base_fit_sha256=file_hash(path),generation=generation_receipt,
            elapsed_fit_seconds=time.perf_counter()-started,maximum_resident_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
            completed_utc=utc_now()))



def compact_fits():
    """Move regenerable training prefixes to cache; keep policy sidecars public."""
    plan=freeze()
    for path in sorted((DEST/'audit_fits').glob('*.npz')):
        receipt=json.loads(path.with_suffix('.json').read_text())
        if file_hash(path)!=receipt['sha256']: raise ValueError('Fit changed before lossless compaction.')
        with np.load(path) as z: payload={k:z[k] for k in z.files}
        prefixes={k:payload.pop(k) for k in list(payload) if 'managed_prefix' in k}
        if not prefixes: continue
        cache_path=CACHE/'training_prefixes'/path.name
        npz_write(cache_path,**prefixes,index=payload['index'],protocol_hash=digest(plan))
        npz_write(path,**payload)
        with np.load(path) as restored:
            for key,expected in payload.items(): np.testing.assert_array_equal(restored[key],expected)
        evaluation_path=DEST/'audit_evaluation.json'
        if evaluation_path.exists():
            evaluation=json.loads(evaluation_path.read_text())
            if evaluation['fit_sources'].get(str(path))!=receipt['sha256']: raise ValueError('Evaluation source changed before compaction.')
            evaluation['fit_sources'][str(path)]=file_hash(path)
            evaluation.setdefault('lossless_compaction_verified',[]).append(path.name)
            json_write(evaluation_path,evaluation)
        receipt.update(sha256=file_hash(path),training_prefix_cache_sha256=file_hash(cache_path),
            compaction='Lossless move of regenerable initial60 managed arrays to local cache; all policy coefficients, scores, moments and identity fields unchanged.')
        json_write(path.with_suffix('.json'),receipt)
        print('Compacted public policy sidecar',path.name,flush=True)


def ridge_path_local(X,lam):
    from simulations.estimator.ridge import ridge_path
    return ridge_path(X,lam)


def evaluate_fits():
    plan=freeze();files=sorted((DEST/'audit_fits').glob('*.npz'))
    if not files: raise ValueError('No audit fits.')
    matrices={n:[] for n in ENVIRONMENTS};labels={n:[] for n in ENVIRONMENTS};sources={}
    for path in files:
        receipt=json.loads(path.with_suffix('.json').read_text())
        if file_hash(path)!=receipt['sha256'] or file_hash(path.with_suffix('.csv'))!=receipt['moments_sha256'] or receipt['protocol_hash']!=digest(plan): raise ValueError('Altered fit or moments manifest.')
        sources[str(path)]=receipt['sha256']
        with np.load(path) as z:
            if str(z['protocol_hash'])!=digest(plan): raise ValueError('Fit protocol mismatch.')
            for name in ENVIRONMENTS:
                for t,T in enumerate(z[name+'_T']):
                    for r,rank in enumerate((4096,8192)):
                        for j,check in enumerate(('A','B','C')):
                            matrices[name].append(z[name+'_raw_coefficients'][t,r,:,j])
                            labels[name].append(dict(environment=name,index=int(z['index']),T=int(T),rank=rank,audit=check,sample_scope=str(z['sample_scope']),penalty=z[name+'_penalties'][t,r,j]))
    basis=RawBasis(8192);rows=[];start=time.perf_counter()
    # rho economies share the SAME population operator: evaluate all three
    # jointly, without equating their selected penalties or complexities.
    groups=[('baseline','rho000','rho075'),('N300',),('N1200',)]
    for names in groups:
        coefficients=np.column_stack([column for n in names for column in matrices[n]])
        flat_labels=[lab for n in names for lab in labels[n]]
        for quad,count,qseed in [('primary',plan['primary_groups'],plan['primary_seed']),('independent',plan['independent_groups'],plan['independent_seed'])]:
            measured=evaluate(basis,coefficients,count,qseed,parameters(names[0]))
            for j,lab in enumerate(flat_labels):
                finance=financial_decomposition(measured['mean'][j],measured['second'][j],parameters(lab['environment']).q_star)
                rows.append(dict(**lab,quadrature=quad,groups=count,seed=qseed,**{k:np.asarray(v).item() for k,v in finance.items()}))
            print(f'Paired audit evaluation {names}, {quad}: {time.perf_counter()-start:.1f}s',flush=True)
    frame=pd.DataFrame(rows)
    frame['cohort']=np.where(frame['index'].isin(plan['representative_indices']),'representative','additional_stress')
    frame['economy_minimum_a_stress']=frame.apply(lambda r:int(r['index'])==plan['stress_indices_by_economy'][r.environment],axis=1)
    frame.to_csv(DEST/'audit_policy_moments.csv',index=False)
    # Compare low primary metrics with the independently recovered archived moments.
    for path in files:
        recovered=pd.read_csv(path.with_suffix('.csv'))
        for _,row in recovered.iterrows():
            got=frame[(frame.environment==row.environment)&(frame['index']==row['index'])&(frame['T']==row['T'])&(frame['rank']==4096)&(frame.audit==row.audit)&(frame.sample_scope==row.sample_scope)&(frame.quadrature=='primary')].iloc[0]
            np.testing.assert_allclose([got.u,got.v],[row.u,row.v],rtol=1e-7,atol=1e-9)
    json_write(DEST/'audit_evaluation.json',dict(passed=True,fit_sources=sources,moments_sha256=file_hash(DEST/'audit_policy_moments.csv'),protocol_hash=digest(plan),elapsed_seconds=time.perf_counter()-start,
        full_representative_indices=sorted(set(frame[frame.sample_scope=='full']['index'])),short_representative_indices=sorted(set(frame[frame.sample_scope=='short']['index'])),
        high_rank_population_ridge_status='not_run',forward_rank_certificate=False,completed_utc=utc_now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','fit','evaluate','compact'])
    parser.add_argument('--index',type=int,default=0);parser.add_argument('--short',action='store_true')
    args=parser.parse_args()
    if args.stage=='freeze': freeze()
    elif args.stage=='fit': fit(args.index,args.short)
    elif args.stage=='compact': compact_fits()
    else: evaluate_fits()

if __name__=='__main__':main()
