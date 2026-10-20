"""Higher-rank, paired pilot with exact finite-sample complexity error bounds.

The economy, nonperiodic kernel, seeds and frozen penalty are unchanged.
Only managed feature rows are retained, never stock panels. Primal/dual ridge
and direct policy evaluation avoid an unnecessary rank-by-rank evaluation
operator. All fit windows are reported; slopes never select the rank.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import linalg
from threadpoolctl import threadpool_limits

from . import asymptotic as engine
from .asymptotic_dense import write_json
from .model import calibrate,fit_path,moments
from .report import matrix,rate_fit


def source_hash(config):
    return hashlib.sha256(engine.scientific_hash(config).encode()+Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def basis_for(rank,encoded_config):
    return engine.Basis('nystrom',rank,json.loads(encoded_config))


def trace_certificate(full_trace,projected_trace,penalties,complexities):
    """For A >= B >= 0: 0 <= C(A)-C(B) <= tr(A-B)/lambda.

    Eigenvalues are monotone under Loewner order and u/(u+lambda) has
    derivative at most 1/lambda. Sum this scalar inequality over eigenvalues.
    The normalized full managed Gram trace needs only its diagonal entries.
    """
    residual=float(full_trace-projected_trace)
    if residual < -1e-10*max(abs(full_trace),1e-30):
        raise ArithmeticError('Nyström trace exceeds the full-kernel trace')
    residual=max(residual,0.)  # Only tolerate floating-point roundoff at zero.
    absolute=residual/np.asarray(penalties)
    return absolute,absolute/np.asarray(complexities)


def training_job(args):
    config,rank,trial,cache=args
    path=Path(cache)/f'precision_nystrom{rank}_trial{trial:03d}.npz'
    scratch=Path(cache)/f'managed_nystrom{rank}_trial{trial:03d}.dat'
    if path.exists():
        scratch.unlink(missing_ok=True)
        return str(path)
    N,rho=config['fixed_N'],config['fixed_rho']
    times=config['T_train']
    constant=config['precision_frozen_C0']
    if shutil.disk_usage(cache).free < max(times)*rank*8+2*1024**3:
        raise RuntimeError('Insufficient free disk for bounded-memory managed-feature scratch')
    with threadpool_limits(1):
        basis=basis_for(rank,json.dumps(config,sort_keys=True))
        white=np.memmap(scratch,dtype=np.float64,mode='w+',shape=(max(times),rank))
        full_diagonal=np.empty(max(times))
        for start,z,r in engine.draw_blocks(N,rho,trial,config['fidelity_seed'],max(times),config['stream_block']):
            white[start:start+len(z)]=engine.managed_block(basis,z,r)
            for j in range(len(z)):
                full_diagonal[start+j]=r[j]@engine.matern_kernel(z[j],z[j],config['ell'])@r[j]/N**2
            if start%2048==0:
                white.flush()
        # L depends only on fixed centers, not on any training observation.
        # Applying it to all rows together does not leak future training data.
        for start in range(0,max(times),128):
            white[start:start+128]=basis.whiten_rows(white[start:start+128])
            if start%2048==0:
                white.flush()
        white.flush()
        cumulative_full=np.cumsum(full_diagonal)
        cumulative_projected=np.cumsum(np.einsum('ij,ij->i',white,white))
        policies=[];cs=[];full=[];projected=[]
        spectrum=None
        for n in times:
            lambdas=constant*n**(-2/3)*np.asarray(config['lambda_multipliers'])
            theta,complexity,eigenvalues=fit_path(white[:n],lambdas)
            policies.append(basis.raw_coefficients(theta));cs.append(complexity)
            full.append(cumulative_full[n-1]/n);projected.append(cumulative_projected[n-1]/n)
            if n==config['spectrum_T_train']:
                spectrum=eigenvalues.copy()
        assert spectrum is not None
        engine.atomic_npz(path,theta=np.column_stack(policies),complexity=cs,
            full_trace=full,projected_trace=projected,T_train=times,eigenvalues=spectrum)
        white._mmap.close()
        scratch.unlink()
    return str(path)


def direct_evaluation(config,N,basis,coefficients):
    """Same conditional moments and cross-fit control, contracted per policy."""
    rng=np.random.default_rng(np.random.SeedSequence([config['evaluation_seed'],N]))
    cal=calibrate(N)
    count=config['evaluation_states']//2
    block=config.get('precision_evaluation_block',config['evaluation_block'])
    ms=[];ss=[];ps=[]
    with threadpool_limits(1):
        for half in range(2):
            m_sum=np.zeros(coefficients.shape[1]);s_sum=m_sum.copy();p_sum=0.
            for start in range(0,count,block):
                size=min(block,count-start)
                z=rng.uniform(-1,1,(size,N,3))
                _,B,mu,h=moments(z,cal)
                weights=(basis.raw(z.reshape(-1,3))@coefficients/N).reshape(size,N,-1)
                mean=np.einsum('bnp,bn->bp',weights,mu)
                common=np.einsum('bnp,bn->bp',weights,h)
                exposures=np.einsum('bnp,bnk->bpk',weights,B)
                second=(mean**2+common**2+cal['sigma_eps']**2*np.sum(weights**2,axis=1)
                    +np.einsum('bpk,km,bpm->bp',exposures,cal['Sigma_factor'],exposures))
                m_sum+=mean.sum(axis=0);s_sum+=second.sum(axis=0)
                p_sum+=np.sum(mu**2)/cal['kappa_0']
            ms.append(m_sum/count);ss.append(s_sum/count);ps.append(p_sum/count)
    ms,ss,ps=np.asarray(ms),np.asarray(ss),np.asarray(ps)
    controls=ms[::-1]/ps[::-1,None]
    corrected_mean=(ms-controls*(ps[:,None]-3/7)).mean(axis=0)
    corrected_second=(ss-controls**2*(ps[:,None]-3/7)).mean(axis=0)
    raw_mean=ms.mean(axis=0);raw_second=ss.mean(axis=0)
    variance=corrected_second-corrected_mean**2
    if np.any(variance<=0):
        raise ArithmeticError('Nonpositive direct evaluation variance')
    sr=np.sqrt(12)*corrected_mean/np.sqrt(variance)
    return dict(SR_hat=sr,SR_gap=3-sr,SR_raw=np.sqrt(12)*raw_mean/np.sqrt(raw_second-raw_mean**2),
        payoff_mean=corrected_mean,payoff_second=corrected_second,
        Q_hat=1-2*corrected_mean+corrected_second,regret=corrected_second-2*corrected_mean+3/7)


def collect(config,rank,out):
    cache=out/'cache'
    arrays=[]
    for trial in range(config['trials']):
        with np.load(cache/f'precision_nystrom{rank}_trial{trial:03d}.npz') as a:
            arrays.append({k:a[k] for k in a.files})
    coefficients=np.column_stack([a['theta'] for a in arrays])
    with threadpool_limits(1):
        basis=basis_for(rank,json.dumps(config,sort_keys=True))
        metrics=direct_evaluation(config,config['fixed_N'],basis,coefficients)
    rows=[];certificates=[];spectral=[]
    index=0
    constant=config['precision_frozen_C0']
    grid=constant*config['spectrum_T_train']**(-2/3)*np.geomspace(.01,100,65)
    for trial,a in enumerate(arrays):
        for i,n in enumerate(config['T_train']):
            lambdas=constant*n**(-2/3)*np.asarray(config['lambda_multipliers'])
            absolute,relative=trace_certificate(a['full_trace'][i],a['projected_trace'][i],lambdas,a['complexity'][i])
            for j,mult in enumerate(config['lambda_multipliers']):
                row=dict(N=config['fixed_N'],rho=config['fixed_rho'],trial=trial,rank=rank,T_train=n,
                    lambda_multiplier=mult,lambda_reg=lambdas[j],complexity=a['complexity'][i,j])
                row.update({key:float(value[index]) for key,value in metrics.items()})
                rows.append(row);index+=1
                certificates.append(dict(rank=rank,trial=trial,T_train=n,lambda_multiplier=mult,
                    residual_trace=a['full_trace'][i]-a['projected_trace'][i],
                    absolute_C_error_bound=absolute[j],relative_C_error_bound=relative[j]))
        eigenvalues=a['eigenvalues']
        cs=np.sum(eigenvalues[:,None]/(eigenvalues[:,None]+grid),axis=0)
        i=config['T_train'].index(config['spectrum_T_train'])
        absolute,relative=trace_certificate(a['full_trace'][i],a['projected_trace'][i],grid,cs)
        spectral.extend(dict(rank=rank,trial=trial,T_train=config['spectrum_T_train'],lambda_reg=lam,
            complexity=C,absolute_C_error_bound=error,relative_C_error_bound=rel)
            for lam,C,error,rel in zip(grid,cs,absolute,relative))
    frame=pd.DataFrame(rows)
    assert len(frame)==config['trials']*len(config['T_train'])*len(config['lambda_multipliers'])
    assert np.isfinite(frame.select_dtypes('number')).all().all()
    frame.to_csv(out/f'precision_nystrom{rank}_trials.csv',index=False)
    pd.DataFrame(certificates).to_csv(out/f'precision_nystrom{rank}_certificates.csv',index=False)
    spectrum=pd.DataFrame(spectral)
    spectrum.to_csv(out/f'precision_nystrom{rank}_spectrum.csv',index=False)
    fits={}
    chosen=frame[frame.lambda_multiplier==1]
    for cutoff in config['sensitivity_T_min']:
        part=chosen[chosen.T_train>=cutoff]
        gap=matrix(part,'T_train','SR_gap');C=matrix(part,'T_train','complexity')
        if len(gap.columns)<3:
            continue
        T=gap.columns.to_numpy(float)
        for name,x,values,power,xtrials in [
            ('SR_T',T,gap.to_numpy(),-2/3,None),('C_T',T,C.to_numpy(),1/3,None),
            ('SR_C',C.mean(axis=0).to_numpy(),gap.to_numpy(),-2,C.to_numpy())]:
            fit=rate_fit(x,values,power,config['bootstrap_resamples'],xtrials)
            fit['T_train_fit_grid']=T.tolist()
            fits[f'{name}_Tmin{cutoff}']=fit
        low=constant*max(T)**(-2/3);high=constant*min(T)**(-2/3)
        tab=matrix(spectrum[(spectrum.lambda_reg>=low*(1-1e-10))&(spectrum.lambda_reg<=high*(1+1e-10))],
                   'lambda_reg','complexity')
        fits[f'C_lambda_Tmin{cutoff}']=rate_fit(tab.columns.to_numpy(float),tab.to_numpy(),-.5,config['bootstrap_resamples'])
    write_json(out/f'precision_nystrom{rank}_fits.json',fits)
    cert=pd.DataFrame(certificates)
    main_error=float(cert.query('lambda_multiplier==1').relative_C_error_bound.max())
    low=constant*config['spectrum_fit_T_max']**(-2/3)
    high=constant*config['spectrum_fit_T_min']**(-2/3)
    window=spectrum[(spectrum.lambda_reg>=low*(1-1e-10))&(spectrum.lambda_reg<=high*(1+1e-10))]
    spectrum_error=float(window.relative_C_error_bound.max())
    files=[f'precision_nystrom{rank}_{suffix}' for suffix in ['trials.csv','certificates.csv','spectrum.csv','fits.json']]
    write_json(out/f'precision_nystrom{rank}_completed.json',dict(rank=rank,replications=config['trials'],
        source_hash=source_hash(config),evaluation_states=config['evaluation_states'],
        maximum_balanced_relative_C_error_bound=main_error,maximum_fit_window_relative_C_error_bound=spectrum_error,
        passes_certified_tolerance=bool(max(main_error,spectrum_error)<=config['precision_relative_C_error_tolerance']),
        files_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files}))
    print(f'High-precision rank {rank}: evaluated, certified, all fit windows retained',flush=True)


def run(config,out,rank,workers):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    signature=source_hash(config)
    marker=cache/'precision_source_hash.txt'
    if marker.exists() and marker.read_text()!=signature:
        raise RuntimeError('Precision code or design changed; use a new output directory')
    marker.write_text(signature)
    write_json(out/'config.json',config)
    assert rank in config['precision_ranks']
    complete=out/f'precision_nystrom{rank}_completed.json'
    if complete.exists():
        previous=json.loads(complete.read_text())
        assert previous['source_hash']==signature
        assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==checksum
                   for name,checksum in previous['files_sha256'].items())
        print(f'Reusing complete evaluated and certified rank {rank}',flush=True)
        return
    engine.execute([(config,rank,j,str(cache)) for j in range(config['trials'])],training_job,workers,
                   f'Paired high-precision pilot, rank {rank}')
    collect(config,rank,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=engine.ROOT/'config/asymptotic_precision.json')
    parser.add_argument('--output',type=Path,default=engine.ROOT/'outputs/asymptotic_precision_v4')
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();run(json.loads(args.config.read_text()),args.output,args.rank,args.workers)
