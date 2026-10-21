"""Exact single-penalty ridge and complexity for the frozen denser paper grid.

For S=G'G/T (or GG'/T), C=dim(S)-lambda*tr((S+lambda I)^-1).
If L L'=S+lambda I, tr((S+lambda I)^-1)=||L^-1||_F^2.
No stochastic trace or spectral truncation is used. The spectrum at the
unchanged final training sample is reused from the completed paired study.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import linalg
from threadpoolctl import threadpool_limits

from . import asymptotic as engine
from . import asymptotic_precision as precision
from .asymptotic_dense import write_json

INPUTS=['prior_config.json','prior_eigenvalues.csv','prior_trace_anchors.csv','prior_provenance.json']


def exact_ridge_complexity(G,penalty):
    if not np.isfinite(penalty) or penalty<=0:raise ValueError('Positive finite ridge penalty required')
    T,M=G.shape;dual=T<=M;d=min(T,M)
    A=np.array((G@G.T if dual else G.T@G)/T,order='F')
    A.flat[::d+1]+=penalty
    factor=linalg.cholesky(A,lower=True,overwrite_a=True,check_finite=False)
    rhs=np.ones(T)/T if dual else G.mean(axis=0)
    solved=linalg.cho_solve((factor,True),rhs,check_finite=False)
    theta=G.T@solved if dual else solved
    inv,info=linalg.lapack.dtrtri(factor,lower=1,overwrite_c=1)
    if info!=0:raise ArithmeticError(f'Triangular inverse failed: {info}')
    for j in range(d):inv[:j,j]=0
    C=float(d-penalty*np.einsum('ij,ij->',inv,inv))
    if not (0<C<=d+1e-8):raise ArithmeticError('Invalid exact trace complexity')
    return theta[:,None],np.array([C])


def signature(config,out):
    digest=hashlib.sha256(precision.source_hash(config).encode()+Path(__file__).read_bytes())
    for name in INPUTS:digest.update((Path(out)/name).read_bytes())
    return digest.hexdigest()


def validate_inputs(config,out):
    out=Path(out);old=json.loads((out/'prior_config.json').read_text())
    provenance=json.loads((out/'prior_provenance.json').read_text())
    assert precision.source_hash(old)==provenance['completed_source_hash']
    for key in ['fixed_N','fixed_rho','fidelity_seed','stream_block','ell','kernel_jitter',
                'center_seed','feature_seed','precision_frozen_C0','spectrum_T_train',
                'evaluation_seed','evaluation_states','precision_evaluation_block']:
        assert config[key]==old[key],key
    assert max(config['T_train'])==max(old['T_train'])==config['spectrum_T_train']
    assert config['lambda_multipliers']==[1.0] and config['precision_ranks']==[8192]
    for name,sha in provenance['inputs_sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha,name
    return provenance


def training_job(args):
    config,rank,trial,cache=args
    cache=Path(cache);out=cache.parent;path=cache/f'precision_nystrom{rank}_trial{trial:03d}.npz'
    scratch=cache/f'managed_nystrom{rank}_trial{trial:03d}.dat'
    if path.exists():scratch.unlink(missing_ok=True);return str(path)
    N,rho=config['fixed_N'],config['fixed_rho'];times=config['T_train'];constant=config['precision_frozen_C0']
    if shutil.disk_usage(cache).free<max(times)*rank*8+2*1024**3:
        raise RuntimeError('Insufficient free disk for bounded-memory managed-feature scratch')
    old=pd.read_csv(out/'prior_trace_anchors.csv',float_precision='round_trip').query('trial==@trial').set_index('T_train')
    spectrum=pd.read_csv(out/'prior_eigenvalues.csv',float_precision='round_trip')[f'trial_{trial}'].to_numpy()
    assert len(spectrum)==rank and (spectrum>=0).all()
    with threadpool_limits(1):
        basis=precision.basis_for(rank,json.dumps(config,sort_keys=True))
        white=np.memmap(scratch,dtype=np.float64,mode='w+',shape=(max(times),rank))
        full_diagonal=np.empty(max(times))
        for start,z,r in engine.draw_blocks(N,rho,trial,config['fidelity_seed'],max(times),config['stream_block']):
            white[start:start+len(z)]=engine.managed_block(basis,z,r)
            for j in range(len(z)):
                full_diagonal[start+j]=r[j]@engine.matern_kernel(z[j],z[j],config['ell'])@r[j]/N**2
            if start%2048==0:white.flush()
        for start in range(0,max(times),128):
            white[start:start+128]=basis.whiten_rows(white[start:start+128])
            if start%2048==0:white.flush()
        white.flush();cumulative_full=np.cumsum(full_diagonal)
        cumulative_projected=np.cumsum(np.einsum('ij,ij->i',white,white))
        policies=[];cs=[];full=[];projected=[]
        for n in times:
            penalty=constant*n**(-2/3)
            theta,C=exact_ridge_complexity(white[:n],penalty)
            f=float(cumulative_full[n-1]/n);p=float(cumulative_projected[n-1]/n)
            if n in old.index:
                np.testing.assert_allclose([f,p],[old.loc[n,'full_trace'],old.loc[n,'projected_trace']],rtol=1e-11,atol=1e-15)
                np.testing.assert_allclose(C.item(),old.loc[n,'complexity'],rtol=1e-8,atol=1e-9)
            if n==config['spectrum_T_train']:
                np.testing.assert_allclose(C.item(),np.sum(spectrum/(spectrum+penalty)),rtol=1e-8,atol=1e-9)
            policies.append(basis.raw_coefficients(theta));cs.append(C);full.append(f);projected.append(p)
        engine.atomic_npz(path,theta=np.column_stack(policies),complexity=cs,full_trace=full,
            projected_trace=projected,T_train=times,eigenvalues=spectrum)
        white._mmap.close();scratch.unlink()
    return str(path)


def validate_backend(output,require_complete=True):
    out=Path(output);config=json.loads((out/'config.json').read_text());validate_inputs(config,out)
    backend=json.loads((out/'training_backend.json').read_text())
    assert backend['source_signature']==signature(config,out)
    assert backend['exact_complexity_trace'] and not backend['stochastic_trace']
    if require_complete:
        completed=out/'precision_nystrom8192_completed.json'
        assert backend['completed_marker_sha256']==hashlib.sha256(completed.read_bytes()).hexdigest()
        frame=pd.read_csv(out/'precision_nystrom8192_trials.csv')
        prior=pd.read_csv(out/'prior_trace_anchors.csv',float_precision='round_trip')
        merged=prior.merge(frame,on=['trial','T_train'],validate='one_to_one',suffixes=('_prior','_new'))
        assert len(merged)==len(prior)
        np.testing.assert_allclose(merged.complexity_prior,merged.complexity_new,rtol=1e-8,atol=1e-9)
        eig=pd.read_csv(out/'prior_eigenvalues.csv',float_precision='round_trip')
        spectrum=pd.read_csv(out/'precision_nystrom8192_spectrum.csv')
        for trial,g in spectrum.groupby('trial'):
            e=eig[f'trial_{trial}'].to_numpy();lam=g.lambda_reg.to_numpy()
            np.testing.assert_allclose(g.complexity,np.sum(e[:,None]/(e[:,None]+lam),axis=0),rtol=1e-11)
    return dict(passed=True,source_signature=backend['source_signature'],exact_trace=True,
                prior_spectrum_from_identical_final_training_sample=True)


def run(config,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    validate_inputs(config,out);sig=signature(config,out);marker=cache/'fast_source_signature.txt'
    if marker.exists() and marker.read_text()!=sig:raise RuntimeError('Backend or frozen inputs changed; use a new output directory')
    marker.write_text(sig);write_json(out/'config.json',config)
    backend=dict(source_signature=sig,scientific_base_hash=precision.source_hash(config),
                 exact_complexity_trace=True,stochastic_trace=False,
                 solver='Exact ridge Cholesky solve; C=d-lambda*||L^-1||_F^2',
                 spectral_curve='Reuse frozen full spectrum of identical final training input; no spectral truncation')
    write_json(out/'training_backend.json',backend)
    state=dict(status='running',phase='training',source_signature=sig,rank=8192,workers=1,
               numerical_threads=1,backend='exact_cholesky_trace_inverse')
    def save():write_json(out/'execution.json',state)
    save();awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]) if sys.platform=='darwin' else None
    try:
        while True:
            try:
                engine.execute([(config,8192,j,str(cache)) for j in range(config['trials'])],training_job,1,
                               'Dense exact ridge, fixed rank8192')
                break
            except (RuntimeError,OSError) as exc:
                if 'Insufficient free disk' not in str(exc) and getattr(exc,'errno',None)!=28:raise
                state['phase']='waiting_for_disk_margin';save()
                required=max(config['T_train'])*8192*8+2*1024**3+256*1024**2
                while shutil.disk_usage(cache).free<required:time.sleep(30)
                state['phase']='training';save()
        state['phase']='evaluating';save();precision.collect(config,8192,out)
        backend['completed_marker_sha256']=hashlib.sha256((out/'precision_nystrom8192_completed.json').read_bytes()).hexdigest()
        write_json(out/'training_backend.json',backend);validate_backend(out)
        state.update(status='computed',phase='data_and_backend_validated');save()
    except BaseException:
        state['status']='failed';save();raise
    finally:
        if awake is not None:awake.terminate()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();lock_dir=a.output/'cache';lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/'compute.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(json.loads(a.config.read_text()),a.output)
