"""Reusable raw population integrals, with factor/idiosyncratic separation.

Prefix anchors make all rank comparisons exactly nested. The same population
integrals describe every N and rho; N enters only the stated economic weights,
and rho does not enter the stationary marginal operator.
"""
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

from simulations.dgp.balanced import balanced_triplets, beta
from simulations.estimator.kernel import NystromBasis
from simulations.provenance import npz_write, file_hash, json_write


def basis_cached(folder,rank,seed):
    path=Path(folder)/f'basis_P{rank}_seed{seed}.npz'
    if path.exists():
        with np.load(path) as z:
            assert int(z['rank'])==rank and int(z['seed'])==seed
            # Cache the unchanged canonical Matérn basis; no alternative RKHS.
            obj=object.__new__(NystromBasis)
            obj.rank,obj.dimension,obj.nu,obj.ell,obj.seed=rank,6,1.5,1.,seed
            obj.anchors,obj.inverse_root,obj.gram=z['anchors'],z['inverse_root'],z['gram']
            return obj
    start=time.perf_counter()
    obj=NystromBasis(rank=rank,seed=seed)
    npz_write(path,rank=rank,seed=seed,anchors=obj.anchors,inverse_root=obj.inverse_root,gram=obj.gram)
    print(f'Basis P={rank}: {time.perf_counter()-start:.1f}s',flush=True)
    return obj


def raw_integrals(p,basis,groups,seed,batch_groups=256):
    rng=np.random.default_rng(seed)
    h_sum=np.zeros((basis.rank,3))
    hh_sum=np.zeros((basis.rank,basis.rank))
    kk_sum=np.zeros_like(hh_sum)
    for first in range(0,groups,batch_groups):
        count=min(batch_groups,groups-first)
        ranks=np.clip(rng.random((count,6)),np.finfo(float).eps,1-np.finfo(float).eps)
        z=balanced_triplets(ranks)
        raw=basis.raw(z)
        b=beta(z,p).reshape(count,3,3)
        h=np.einsum('grm,grk->gmk',raw.reshape(count,3,basis.rank),b)/3
        flat=h.transpose(1,0,2).reshape(basis.rank,-1)
        h_sum+=h.sum(axis=0)
        hh_sum+=flat@flat.T
        kk_sum+=raw.T@raw
    return {'h_mean':h_sum/groups,'hh':hh_sum/groups,'kk':kk_sum/(3*groups)}


def integrals_cached(folder,p,basis,groups,seed,persist=True):
    path=Path(folder)/f'integrals_P{basis.rank}_basis{basis.seed}_Q{groups}_seed{seed}.npz'
    if path.exists():
        with np.load(path) as z:
            assert int(z['groups'])==groups and int(z['seed'])==seed
            return {k:z[k] for k in ('h_mean','hh','kk')}
    start=time.perf_counter()
    stats=raw_integrals(p,basis,groups,seed)
    if persist:
        npz_write(path,**stats,groups=groups,seed=seed,basis_seed=basis.seed,
                  loading_map=p.loading_map,eta=p.eta,fourier_terms=p.fourier_terms)
    print(f'Integrals P={basis.rank}, Q={groups}: {time.perf_counter()-start:.1f}s',flush=True)
    return stats


def operators(p,basis,stats,components=False):
    P=basis.rank
    h=stats['h_mean'][:P]
    hh=stats['hh'][:P,:P]
    kk=stats['kk'][:P,:P]
    root=basis.inverse_root
    factor_raw=hh/p.G+(1-1/p.G)*(h@h.T)
    total=root@(factor_raw+(p.sigma_eps**2/p.N)*kk)@root
    total=(total+total.T)/2
    mean=root@h@np.asarray(p.mu_F)
    if not components:
        return mean,total
    kernel=root@kk@root
    kernel=(kernel+kernel.T)/2
    noise=(p.sigma_eps**2/p.N)*kernel
    factor=total-noise
    return mean,total,{'kernel':kernel,'factor':factor,'idiosyncratic':noise}


def spectral_path(values,penalties):
    mu=np.asarray(values)[:,None]
    lam=np.atleast_1d(penalties)
    weights=mu/(mu+lam)
    C=weights.sum(axis=0)
    elasticity=(weights*(1-weights)).sum(axis=0)/C
    # Independent finite difference of the finite-spectrum trace.
    step=1e-4
    logplus=np.log(np.sum(mu/(mu+lam*np.exp(step)),axis=0))
    logminus=np.log(np.sum(mu/(mu+lam*np.exp(-step)),axis=0))
    numerical=-(logplus-logminus)/(2*step)
    np.testing.assert_allclose(elasticity,numerical,rtol=1e-7,atol=1e-9)
    return C,elasticity,.6*elasticity


def population_reference(p,basis,stats,penalties):
    m,S=operators(p,basis,stats)
    values,vectors=eigh(S)
    assert values[0]>0
    projection=vectors.T@m
    floor=float(p.q_star-np.sum(projection**2/values))
    coefficient=vectors@(projection[:,None]/(values[:,None]+penalties))
    bias=p.q_star-2*(m@coefficient)+np.sum(coefficient*(S@coefficient),axis=0)
    C,E,local=spectral_path(values,penalties)
    return dict(mean=m,second=S,eigenvalues=values,coefficients=coefficient,
                floor=floor,bias=bias,complexity=C,elasticity=E,local_T_elasticity=local)
