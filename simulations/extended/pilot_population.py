"""Checkpointed preproduction rank, quadrature and operator spectrum audit."""
import argparse
import time
import resource
import shutil

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from simulations.extended.design import (OUTPUT, BASIS_SEED, BASE_A, T_CANDIDATE,
    parameters, pilot_design, seed)
from simulations.extended.population import basis_cached, integrals_cached, operators, spectral_path
from simulations.provenance import json_write, npz_write, digest, utc_now, output_lock


def run(max_rank=2048,groups=32768,basis_seed=BASIS_SEED,quadrature_seed=None,save_operators=True):
    quadrature_seed=seed('population') if quadrature_seed is None else quadrature_seed
    folder=OUTPUT/'pilot'
    folder.mkdir(parents=True,exist_ok=True)
    design=pilot_design()
    frozen=OUTPUT/'pilot_design.json'
    if frozen.exists():
        import json
        assert json.loads(frozen.read_text())['design_hash']==digest(design)
    else:
        json_write(frozen,dict(design,design_hash=digest(design),frozen_utc=utc_now()))
    start=time.perf_counter()
    high=basis_cached(folder,max_rank,basis_seed)
    stats=integrals_cached(folder,parameters(),high,groups,quadrature_seed,persist=False)
    rows=[]
    for rank in (512,1024,2048,4096):
        if rank>max_rank:
            continue
        basis=high if rank==max_rank else basis_cached(folder,rank,basis_seed)
        kernel_spectrum=None
        for name in ('baseline','N300','N1200'):
            p=parameters(name)
            m,S,components=operators(p,basis,stats,components=True)
            values,vectors=eigh(S)
            projection=vectors.T@m
            floor=float(p.q_star-np.sum(projection**2/values))
            lam=BASE_A*np.asarray(T_CANDIDATE,dtype=float)**(-.6)
            C,E,local=spectral_path(values,lam)
            coeff=vectors@(projection[:,None]/(values[:,None]+lam))
            bias=p.q_star-2*m@coeff+np.sum(coeff*(S@coeff),axis=0)
            spectra={'managed':values[::-1]}
            # The kernel spectrum is common across N; component spectra are
            # reported individually, never added as if eigenvectors agreed.
            if kernel_spectrum is None:
                kernel_spectrum=eigh(components['kernel'],eigvals_only=True)[::-1]
            spectra['kernel']=kernel_spectrum
            spectra['idiosyncratic']=p.sigma_eps**2/p.N*kernel_spectrum
            spectra['factor']=eigh(components['factor'],eigvals_only=True)[::-1]
            tag=f'{name}_P{rank}_B{basis_seed}_Q{groups}_S{quadrature_seed}'
            npz_write(folder/f'spectra_{tag}.npz',**spectra)
            if save_operators:
                npz_write(folder/f'operator_{tag}.npz',mean=m,second=S,eigenvalues=values,
                          floor=floor,groups=groups,basis_seed=basis_seed,quadrature_seed=quadrature_seed)
            for k,T in enumerate(T_CANDIDATE):
                rows.append(dict(environment=name,rank=rank,groups=groups,basis_seed=basis_seed,
                    quadrature_seed=quadrature_seed,T=T,lambda_diagnostic=lam[k],
                    a_diagnostic=BASE_A,projection_floor=floor,population_bias=bias[k],
                    floor_over_population_bias=floor/bias[k],complexity=C[k],
                    lambda_elasticity=E[k],local_T_elasticity=local[k],
                    complexity_fraction_rank=C[k]/rank))
            print(f'{name} P={rank} Q={groups}: floor={floor:.8g}; '
                  f'local elasticity {local[0]:.4f} to {local[-1]:.4f}',flush=True)
            del S,components,vectors,coeff
        del basis
    tag=f'P{max_rank}_B{basis_seed}_Q{groups}_S{quadrature_seed}'
    pd.DataFrame(rows).to_csv(folder/f'population_audit_{tag}.csv',index=False)
    json_write(folder/f'resources_{tag}.json',dict(elapsed_seconds=time.perf_counter()-start,
        maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        disk_free_bytes=shutil.disk_usage(OUTPUT).free,completed_utc=utc_now(),
        note='N robustness a is calibrated separately before production; these preliminary curves use baseline a for diagnostic comparability.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-rank',type=int,default=2048,choices=[512,1024,2048,4096])
    parser.add_argument('--groups',type=int,default=32768)
    parser.add_argument('--basis-seed',type=int,default=BASIS_SEED)
    parser.add_argument('--quadrature-seed',type=int)
    parser.add_argument('--spectra-only',action='store_true')
    args=parser.parse_args()
    with output_lock(OUTPUT/'pilot'):
        run(args.max_rank,args.groups,args.basis_seed,args.quadrature_seed,not args.spectra_only)

if __name__=='__main__':
    main()
