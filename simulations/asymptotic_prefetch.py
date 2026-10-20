"""Precompute frozen future-cell jobs with atomic, non-overwriting handoff.

Two low-priority workers can use spare cores while the dense collector runs.
The scientific engine, design, seeds and penalties are unchanged. Staging
directories isolate the engine's temporary files from concurrent main jobs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import asymptotic as engine
from .asymptotic_dense import cell_times,write_json


def handoff(staged,target,cell_marker):
    """Publish a complete inode atomically; preserve every existing cache."""
    staged,target,cell_marker=map(Path,[staged,target,cell_marker])
    if cell_marker.exists():
        staged.unlink()
        return 'cell_already_exported'
    try:
        os.link(staged,target)
        state='published'
    except FileExistsError:
        state='main_cache_already_present'
    staged.unlink()
    return state


def prefetch_job(args):
    config,N,rho,trial,rank,constant,cache,signature=args
    if engine.scientific_hash(config)!=signature:
        raise RuntimeError('Frozen scientific source changed during precomputation')
    cache=Path(cache)
    name=f'confirmation_nystrom{rank}_N{N}_rho{rho:g}_trial{trial:03d}.npz'
    marker=cache.parent/'cells'/f'N{N}_rho{rho:g}'/'completed.json'
    target=cache/name
    row=dict(N=N,rho=rho,trial=trial)
    if marker.exists() or target.exists():
        return dict(row,state='already_available')
    staging=cache/'prefetch';staging.mkdir(exist_ok=True)
    times=cell_times(config,N,rho)
    staged=Path(engine.training_job((config,N,rho,trial,config['confirmation_seed'],times,
        'nystrom',rank,constant,str(staging),'confirmation')))
    with np.load(staged) as saved:
        assert saved['T_train'].tolist()==times
        assert saved['theta'].shape==(rank,len(times)*len(config['lambda_multipliers']))
        assert saved['complexity'].shape==(len(times),len(config['lambda_multipliers']))
        assert all(np.isfinite(saved[key]).all() for key in saved.files)
    checksum=hashlib.sha256(staged.read_bytes()).hexdigest()
    return dict(row,state=handoff(staged,target,marker),policy_sha256=checksum)


def run(output,N=1200,rho=.5,workers=2):
    if not 1<=workers<=2:
        raise ValueError('Precomputation is limited to one or two workers')
    out=Path(output);cache=out/'cache'
    config=json.loads((out/'config.json').read_text())
    signature=engine.scientific_hash(config)
    if (cache/'source_hash.txt').read_text()!=signature:
        raise RuntimeError('Main cache and frozen scientific source differ')
    if [N,rho] not in config['cells']:
        raise ValueError('Only a cell already in the frozen experiment may be precomputed')
    rank=json.loads((out/'rank_selection.json').read_text())['selected_rank']
    constant=json.loads((out/'frozen_constants.json').read_text())[str(N)]
    # flock releases automatically if the process exits or the host reboots.
    with (cache/'prefetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[(config,N,rho,j,rank,constant,str(cache),signature)
              for j in reversed(range(config['trials']))]
        state=dict(source_hash=signature,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cell=[N,rho],workers=workers,trial_order='descending; main collector uses ascending',
            publication='Atomic hard link from separate staging; existing cache entries preserved',
            scientific_engine='Unmodified simulations.asymptotic.training_job',
            status='running',results=[])
        progress=cache/'prefetch_progress.json';write_json(progress,state)
        start=time.time()
        with ProcessPoolExecutor(workers) as pool:
            futures=[pool.submit(prefetch_job,job) for job in jobs]
            for i,future in enumerate(as_completed(futures),1):
                state['results'].append(future.result())
                write_json(progress,state)
                if i%5==0 or i==len(jobs):
                    print(f'Frozen future cell N={N}, rho={rho:g}: {i}/{len(jobs)}; elapsed {(time.time()-start)/60:.1f} min',flush=True)
        state.update(status='complete',elapsed_seconds=time.time()-start)
        write_json(progress,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--N',type=int,default=1200);p.add_argument('--rho',type=float,default=.5)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.output,a.N,a.rho,a.workers)
