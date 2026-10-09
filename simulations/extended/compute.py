"""Bounded-storage full-penalty simulation checkpoints for pilots and production."""
from pathlib import Path
import json
import time
import resource
import hashlib

import numpy as np

from simulations.extended.design import (OUTPUT, BASIS_SEED, PENALTIES, ENVIRONMENTS,
    T_CANDIDATE,T_REQUIRED,T_ROBUSTNESS,parameters,seed)
from simulations.extended.population import basis_cached
from simulations.extended.histories import raw_managed_history
from simulations.extended.calibration import calibrate
from simulations.estimator.ridge import ridge_path,moment_metrics
from simulations.provenance import npz_write, json_write, utc_now, digest
from simulations.run_rough import seed as legacy_seed


def operator_path(name,rank,groups,quadrature_seed,basis_seed=BASIS_SEED):
    economic_name='baseline' if name.startswith('rho') else name
    return OUTPUT/'pilot'/f'operator_{economic_name}_P{rank}_B{basis_seed}_Q{groups}_S{quadrature_seed}.npz'


def load_operator(name,rank,groups,quadrature_seed):
    path=operator_path(name,rank,groups,quadrature_seed)
    if not path.exists():
        from simulations.extended.archive import read_archived_operator
        return read_archived_operator(name,rank,groups,quadrature_seed)
    with np.load(path) as z:
        return z['mean'],z['second'],float(z['floor'])


def run_path(index,ranks,baseline_times,robustness_times,groups=32768,
             quadrature_seed=None,stage='rank_pilot',run_hash=None,environment_names=None):
    start=time.perf_counter()
    is_production=stage.startswith('production')
    environment_names=list(ENVIRONMENTS) if environment_names is None else list(environment_names)
    quadrature_seed=seed('population') if quadrature_seed is None else quadrature_seed
    calibration=calibrate()['a']
    description=dict(index=index,ranks=list(ranks),baseline_times=list(baseline_times),
        robustness_times=list(robustness_times),groups=groups,quadrature_seed=quadrature_seed,
        stage=stage,run_hash=run_hash)
    if is_production:
        description['environment_names']=environment_names
    identity=digest(description)
    destinations={rank:OUTPUT/stage/f'P{rank}'/f'rep_{index:03d}.npz' for rank in ranks}
    pending=[]
    for rank,path in destinations.items():
        if path.exists():
            with np.load(path) as z:
                if str(z['identity'])!=identity:
                    raise ValueError(f'Checkpoint identity mismatch: {path}')
        else:
            pending.append(rank)
    if not pending:
        return {'index':index,'cached':True}
    maxrank=max(ranks)
    high=basis_cached(OUTPUT/'pilot',maxrank,BASIS_SEED)
    training_seed=legacy_seed('train',index) if is_production else seed('rank_training',index)
    # Separate additional-group streams for pilot and production replications.
    pairing_index=index if is_production else 100000+index
    generation_start=time.perf_counter()
    raw,hashes=raw_managed_history(high,training_seed,pairing_index,
        max(baseline_times),max(robustness_times) if robustness_times else 0,
        progress=lambda t:print(f'{stage} path {index}: generated {t}/{max(baseline_times)}',flush=True))
    generation_seconds=time.perf_counter()-generation_start
    for rank in pending:
        basis=high if rank==maxrank else basis_cached(OUTPUT/'pilot',rank,BASIS_SEED)
        outcomes={}
        maximum_normal_error=0.
        for name in environment_names:
            times=baseline_times if name=='baseline' else robustness_times
            p=parameters(name)
            m,S,floor=load_operator(name,rank,groups,quadrature_seed)
            X=raw[name][:,:rank]@basis.inverse_root
            loss,sr,complexity,coefnorm=[],[],[],[]
            theory_coefficients=[]
            decomposition=[]
            pop_theory=None
            if is_production:
                with np.load(OUTPUT/'population'/f'{name}_theory.npz') as pop:
                    np.testing.assert_array_equal(pop['T'],times)
                    pop_theory=pop['coefficients']
                    pop_bias=pop['bias']
            diagnostic_coefficients=[]
            diagnostic_times=[]
            for t_index,T in enumerate(times):
                penalties=np.r_[PENALTIES,calibration[name]*float(T)**(-.6)]
                coefficients,C=ridge_path(X[:T],penalties)
                L,SR=moment_metrics(coefficients,m,S)
                if not np.isfinite(L).all() or not np.isfinite(SR).all():
                    raise ValueError('Non-finite fitted-policy population metrics.')
                # Independent population quadrature can perturb the analytical
                # ceiling slightly; retain and quantify any excess in audits.
                if pop_theory is not None:
                    checked=np.array([0,32,64,95,96])
                    fitted=coefficients[:,checked]
                    residual=X[:T].T@(X[:T]@fitted)/T+fitted*penalties[checked]-X[:T].mean(axis=0)[:,None]
                    maximum_normal_error=max(maximum_normal_error,float(np.max(np.abs(residual))))
                    if maximum_normal_error>1e-8:
                        raise ValueError('Ridge normal-equation residual exceeds the existing numerical tolerance.')
                    delta=coefficients[:,-1]-pop_theory[t_index]
                    norm=float(delta@S@delta)
                    cross=float(2*delta@(S@pop_theory[t_index]-m))
                    regret=L[-1]-(1-p.q_star)
                    np.testing.assert_allclose(regret,pop_bias[t_index]+norm+cross,atol=1e-10,rtol=1e-8)
                    decomposition.append((pop_bias[t_index],norm,cross))
                loss.append(L)
                sr.append(SR)
                complexity.append(C)
                coefnorm.append(np.sum(coefficients**2,axis=0))
                if stage=='rank_pilot':
                    theory_coefficients.append(coefficients[:,-1])
                    if index==0 and rank==maxrank and T in (60,1440,max(times)):
                        diagnostic_coefficients.append(coefficients)
                        diagnostic_times.append(T)
            outcomes.update({f'{name}_{key}':np.asarray(value) for key,value in
                dict(T=times,loss=loss,sr=sr,empirical_complexity=complexity,
                     coefficient_norm_squared=coefnorm,floor=floor).items()})
            if decomposition:
                outcomes[name+'_theory_decomposition']=np.asarray(decomposition)
            if stage=='rank_pilot':
                outcomes[name+'_theory_coefficients']=np.asarray(theory_coefficients)
                if diagnostic_times:
                    outcomes[name+'_diagnostic_coefficients']=np.asarray(diagnostic_coefficients)
                    outcomes[name+'_diagnostic_times']=np.asarray(diagnostic_times)
            print(f'{stage} path {index} P={rank} {name}: fits complete',flush=True)
            del X,S
        npz_write(destinations[rank],**outcomes,index=index,training_seed=training_seed,
                  pairing_index=pairing_index,rank=rank,identity=identity,
                  description=json.dumps(description,sort_keys=True),
                  returns_hashes=json.dumps(hashes,sort_keys=True),
                  maximum_normal_equation_error=maximum_normal_error,
                  generation_seconds=generation_seconds,elapsed_seconds=time.perf_counter()-start,
                  maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report=dict(description,elapsed_seconds=time.perf_counter()-start,
        generation_seconds=generation_seconds,maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        completed_utc=utc_now())
    json_write(OUTPUT/stage/f'resources_{index:03d}.json',report)
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--ranks',type=int,nargs='+',default=[512,1024,2048,4096])
    parser.add_argument('--groups',type=int,default=32768)
    args=parser.parse_args()
    print(run_path(args.index,args.ranks,T_CANDIDATE,T_ROBUSTNESS,args.groups),flush=True)

if __name__=='__main__':
    main()
