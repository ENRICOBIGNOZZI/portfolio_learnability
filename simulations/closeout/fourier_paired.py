"""Paired initial60 Fourier sensitivity using actual innovations and CV lambdas.

This explicit diagnostic never modifies production a or M. Refit and evaluate
policies in each Fourier economy. Low-rank population operator is only used to
verify reproduction of M128; the M512/M2048 payoff moments are streamed from
those economies themselves.
"""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from simulations.closeout.results import DEST,CANONICAL
from simulations.closeout.numerical import CACHE,freeze
from simulations.closeout.metrics import financial_decomposition,normal_residual
from simulations.initial_cv.selector import calibrate_initial
from simulations.extended.design import ENVIRONMENTS,parameters,BASIS_SEED,seed
from simulations.extended.histories import PairedHistory
from simulations.extended.population import basis_cached
from simulations.extended.rank_followup import RawBasis,evaluate
from simulations.estimator.ridge import ridge_path
from simulations.run_rough import seed as original_seed
from simulations.provenance import npz_write,json_write,file_hash,utc_now


def run(index=0):
    plan=freeze()
    path=DEST/'fourier_paired_protocol.json'
    design=dict(indices=[index],times=[60],terms=[128,512,2048],rank=4096,groups=32768,
        quadrature_seed=seed('population'),tolerance=.05,
        scope='Post hoc initial60 diagnostic; actual CV and frozen-constant fits; no endpoint sensitivity certification.')
    if path.exists():
        if json.loads(path.read_text())['design']!=design: raise ValueError('Fourier paired plan changed.')
    else:json_write(path,dict(design=design,frozen_utc=utc_now()))
    low=basis_cached(CACHE,4096,BASIS_SEED);basis=RawBasis(4096)
    reference_path=CANONICAL/'replications'/f'rep_{index:03d}.npz'
    inputs=json.loads((DEST/'input_verification.json').read_text())
    if file_hash(reference_path)!=inputs['sources'][str(reference_path)]: raise ValueError('Original checkpoint changed.')
    with np.load(reference_path) as z:reference={k:z[k] for k in z.files}
    outputs={};choices=[]
    for M in design['terms']:
        history=PairedHistory(original_seed('train',index),index)
        history.base.parameters=replace(history.base.parameters,fourier_terms=M)
        arrays={n:[] for n in ENVIRONMENTS}
        for t in range(60):
            observations=history.step(True)
            z,r=observations['N1200'];K=basis.raw(z)
            for n,N in [('baseline',600),('N300',300),('N1200',1200)]:arrays[n].append(r[:N]@K[:N]/N)
            for n in ('rho000','rho075'):
                z,r=observations[n];arrays[n].append(r@basis.raw(z)/len(r))
        for n in ENVIRONMENTS:
            X=np.asarray(arrays[n])@low.inverse_root
            selection=calibrate_initial(X)
            olda=float(reference[n+'_selected_a'])
            lam=np.array([olda,selection['selected_a']])*60.**(-.6)
            coef,C=ridge_path(X,lam)
            raw_coef=low.inverse_root@coef
            absres,relres=normal_residual(X,coef,lam)
            if absres>1e-8 or relres>1e-8: raise ValueError('Fourier fit residual failed.')
            if M==128:
                np.testing.assert_allclose(selection['fold_losses'],reference[n+'_fold_losses'],rtol=1e-7,atol=1e-9)
                if selection['selected_a']!=olda: raise ValueError('Initial CV reproduction failed.')
            choices.append(dict(environment=n,index=index,M=M,T=60,production_a=olda,selected_a=selection['selected_a'],choice_unchanged=selection['selected_a']==olda))
            outputs[f'{n}_M{M}_raw_coefficients']=raw_coef
            outputs[f'{n}_M{M}_managed_prefix']=X
            outputs[f'{n}_M{M}_fold_losses']=selection['fold_losses']
        print(f'Fourier paired index {index}: M{M} fits complete',flush=True)
    prefixes={k:outputs[k] for k in outputs if k.endswith('_managed_prefix')}
    npz_write(CACHE/'fourier_training_prefixes.npz',**prefixes,index=index,training_seed=original_seed('train',index))
    npz_write(DEST/'fourier_paired_sidecar.npz',**{k:v for k,v in outputs.items() if not k.endswith('_managed_prefix')},index=index,training_seed=original_seed('train',index))
    rows=[]
    for names in [('baseline','rho000','rho075'),('N300',),('N1200',)]:
        for M in design['terms']:
            coeff=np.column_stack([outputs[f'{n}_M{M}_raw_coefficients'] for n in names])
            p=replace(parameters(names[0]),fourier_terms=M)
            measured=evaluate(basis,coeff,design['groups'],design['quadrature_seed'],p)
            for k,n in enumerate(names):
                for j,spec in enumerate(('fixed_archived_a','reselected_initial_a')):
                    loc=2*k+j
                    metrics=financial_decomposition(measured['mean'][loc],measured['second'][loc],parameters(n).q_star)
                    if M==128 and j==0:
                        np.testing.assert_allclose([metrics['regret'],metrics['monthly_SR']],
                            [reference[n+'_loss'][0,0]-(1-parameters(n).q_star),reference[n+'_sr'][0,0]],atol=1e-9,rtol=1e-7)
                    rows.append(dict(environment=n,index=index,T=60,M=M,specification=spec,
                        **{k:np.asarray(v).item() for k,v in metrics.items()}))
            print(f'Fourier moments {names}, M{M} evaluated',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'fourier_paired_moments.csv',index=False)
    pd.DataFrame(choices).to_csv(DEST/'fourier_paired_choices.csv',index=False)
    json_write(DEST/'fourier_paired_verification.json',dict(passed=True,scope=design,
        unchanged_choices=bool(pd.DataFrame(choices).choice_unchanged.all()),
        sidecar_sha256=file_hash(DEST/'fourier_paired_sidecar.npz'),source_checkpoint_sha256=file_hash(CANONICAL/'replications'/f'rep_{index:03d}.npz'),
        endpoint_policy_sensitivity='not_run',independent_quadrature='not_run'))



def summarize_sensitivity():
    """Measure changes to actual refitted policies, not just to W_star."""
    frame=pd.read_csv(DEST/'fourier_paired_moments.csv')
    with np.load(DEST/'fourier_paired_sidecar.npz') as z:coefficients={k:z[k] for k in z.files if k.endswith('_raw_coefficients')}
    with np.load(CACHE/f'basis_P4096_seed{BASIS_SEED}.npz') as z:gram=z['gram']
    rows=[]
    for name in ENVIRONMENTS:
        for M in (512,2048):
            for j,spec in enumerate(('fixed_archived_a','reselected_initial_a')):
                original=frame[(frame.environment==name)&(frame.M==128)&(frame.specification==spec)].iloc[0]
                changed=frame[(frame.environment==name)&(frame.M==M)&(frame.specification==spec)].iloc[0]
                base=coefficients[f'{name}_M128_raw_coefficients'][:,j]
                delta=coefficients[f'{name}_M{M}_raw_coefficients'][:,j]-base
                numerator=float(delta@gram@delta);denominator=float(base@gram@base)
                if numerator < -1e-10:raise ValueError('Negative RKHS norm beyond rounding tolerance.')
                ratios={metric:abs(changed[metric]-original[metric])/abs(original[metric]) for metric in ['u','v','regret','annual_gap','normalized_directional_error']}
                relative_policy=np.sqrt(max(numerator,0)/denominator)
                rows.append(dict(environment=name,index=int(original['index']),T=60,M=M,specification=spec,
                    relative_RKHS_policy_change=relative_policy,**{'relative_'+k:v for k,v in ratios.items()},
                    passes=bool(relative_policy<=.05 and all(v<=.05 for v in ratios.values())),scope='Single paired initial60 history only; no final-horizon certificate.'))
    result=pd.DataFrame(rows);result.to_csv(DEST/'fourier_paired_sensitivity.csv',index=False)
    metadata=json.loads((DEST/'fourier_paired_verification.json').read_text())
    metadata.update(sensitivity_passed=bool(result.passes.all()),max_relative_RKHS_policy_change=float(result.relative_RKHS_policy_change.max()),
        max_relative_moment_or_gap_change=float(result.filter(regex='^relative_').drop(columns='relative_RKHS_policy_change').max().max()))
    json_write(DEST/'fourier_paired_verification.json',metadata)

if __name__=='__main__':
    run()
    summarize_sensitivity()
