"""Complete all declared audit histories; resumable, one local worker.

M128 economic streams remain canonical. M512/M2048 use identical states,
factors and idiosyncratic shocks; only loading differences are applied.
All three managed histories share one unchanged Matern kernel calculation.
Temporary memory maps are removed only after verified coefficient receipts.
"""
import argparse
import copy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from simulations.closeout.results import DEST,CANONICAL
from simulations.closeout.numerical import CACHE,freeze
from simulations.closeout.managed_native import ManagedFeatures
from simulations.closeout.fourier_evaluation import beta_fast,table
from simulations.extended.rank_followup import RawBasis
from simulations.extended.histories import PairedHistory
from simulations.extended.design import ENVIRONMENTS,parameters
from simulations.dgp.balanced import beta
from simulations.run_rough import seed as original_seed
from simulations.provenance import json_write,file_hash,digest,utc_now

FOURIER=DEST/'fourier_full_fits'


def specification():
    plan=freeze()
    description=dict(schema='cv-completion-Fourier/1',original_protocol_hash=digest(plan),
        indices=sorted(set(plan['representative_indices'])|set(plan['stress_indices_by_economy'].values())),
        Fourier_terms=[128,512,2048],ranks=plan['ranks'],baseline_times=plan['baseline_times'],robustness_times=plan['robustness_times'],
        primary_groups=plan['primary_groups'],primary_seed=plan['primary_seed'],independent_groups=plan['independent_groups'],independent_seed=plan['independent_seed'],
        comparisons=['A_frozen_archived_a','B_initial_CV_per_rank_and_M','C_historical_fixed'],
        relative_sensitivity_tolerance=.05,normal_equation_absolute=1e-8,normal_equation_relative=1e-8,
        representative_and_stress_separate=True,source_approximation='FFT/Hermite evaluates the same finite Fourier polynomial with a uniform interpolation error below float64 roundoff; canonical M128 and canonical independent-quadrature loadings retained.',
        interpolation_error_bound={str(M):table(M)[2] for M in (512,2048)},
        future_data_in_selection=False,coefficient_roots_reconstructed=False,
        maximum_rank_forward_certificate=False,full_production_launched=False)
    path=DEST/'fourier_completion_protocol.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['design']!=description or saved['protocol_hash']!=digest(description):raise ValueError('Fourier completion protocol changed.')
    else:json_write(path,dict(design=description,protocol_hash=digest(description),created_utc=utc_now()))
    FOURIER.mkdir(exist_ok=True)
    return description


def reference(index):
    path=CANONICAL/'replications'/f'rep_{index:03d}.npz'
    verified=json.loads((DEST/'input_verification.json').read_text())
    if file_hash(path)!=verified['sources'][str(path)]:raise ValueError('Original training checkpoint changed.')
    with np.load(path) as z:return {k:z[k] for k in z.files}


def prepare(index):
    specification();folder=CACHE/'completion_history'/f'rep_{index:03d}'
    receipt_path=folder/'history.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        for name,sha in receipt['files'].items():
            if file_hash(folder/name)!=sha:raise ValueError('Changed prepared history cache.')
        arrays={M:{n:np.load(folder/f'{n}_M{M}.npy',mmap_mode='r') for n in ENVIRONMENTS} for M in (128,512,2048)}
        return arrays,receipt
    folder.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(folder).free<4_000_000_000:raise ValueError('Insufficient temporary disk for one history; no cache deletion without ownership.')
    basis=RawBasis(8192);managed=ManagedFeatures(basis.anchors,4)
    arrays={M:{n:np.lib.format.open_memmap(folder/f'{n}_M{M}.npy',mode='w+',dtype=np.float64,shape=(4860 if n=='baseline' else 3240,8192)) for n in ENVIRONMENTS} for M in (128,512,2048)}
    sim=PairedHistory(original_seed('train',index),index);checkpoints={};started=time.perf_counter()
    p=parameters();clone=np.random.default_rng()
    for t in range(4860):
        clone.bit_generator.state=copy.deepcopy(sim.base.factor_rng.bit_generator.state)
        factors=np.asarray(p.mu_F)+sim.base._factor_root@clone.standard_normal(3)
        observations=sim.step(t<3240)
        names=['N1200','rho000','rho075'] if t<3240 else ['baseline']
        for name in names:
            z,r=observations[name];b0=beta(z,p)
            weights=np.column_stack([r,(beta_fast(z,p,512)-b0)@factors,(beta_fast(z,p,2048)-b0)@factors])
            cuts=[300,600,1200] if name=='N1200' else [len(r)]
            values=managed(z,weights,cuts)
            entries=[('N300',0),('baseline',1),('N1200',2)] if name=='N1200' else [(name,0)]
            for environment,k in entries:
                for j,M in enumerate((128,512,2048)):arrays[M][environment][t]=values[k,j]
        if t+1 in (1440,3240):checkpoints.update({n+f'_T{t+1}':h for n,h in sim.hashes().items()})
        if (t+1)%240==0:
            message=dict(index=index,stage='generation',T=t+1,maximum_T=4860,elapsed_seconds=time.perf_counter()-started,updated_utc=utc_now())
            json_write(DEST/'completion_progress.json',message)
            print(f'Completion path {index}: three paired Fourier designs T={t+1}/4860 ({message["elapsed_seconds"]:.1f}s)',flush=True)
    hashes={**sim.hashes(),**checkpoints}
    expected=json.loads(str(reference(index)['returns_hashes']))
    if hashes!=expected:raise ValueError('Original economic history hashes were not reproduced.')
    for case in arrays.values():
        for array in case.values():array.flush()
    receipt=dict(index=index,training_seed=original_seed('train',index),returns_hashes=hashes,
        elapsed_seconds=time.perf_counter()-started,maximum_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        Fourier_arrays='M128 raw managed returns; M512/M2048 loading-induced DIFFERENCES from M128.',
        source_sha256={str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('managed_native.c'),Path(__file__).with_name('fourier_evaluation.py')]},
        files={p.name:file_hash(p) for p in folder.glob('*.npy')},completed_utc=utc_now())
    json_write(receipt_path,receipt)
    return arrays,receipt


def complete_one(index):
    description=specification()
    path=FOURIER/f'rep_{index:03d}_full.npz'
    if path.exists():
        receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['sha256']!=file_hash(path) or receipt['protocol_hash']!=digest(description):raise ValueError('Changed Fourier completion checkpoint.')
        print(f'Completion path {index}: verified cached full audit reused',flush=True);return
    arrays,receipt=prepare(index)
    from simulations.closeout.numerical import fit
    fit(index,prepared=(dict(arrays[128]),receipt['returns_hashes'],receipt['elapsed_seconds']),fourier_deltas={M:arrays[M] for M in (512,2048)},generation_receipt=receipt)
    del arrays;gc.collect()
    shutil.rmtree(CACHE/'completion_history'/f'rep_{index:03d}')
    print(f'Completion path {index}: verified full rank/Fourier fits and cleaned owned history cache',flush=True)


def run():
    description=specification()
    lock=CACHE/'completion_worker.lock'
    if lock.exists():
        pid=int(lock.read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:lock.unlink()
        else:raise ValueError(f'Existing local worker {pid} is active; do not restart it.')
    lock.write_text(str(os.getpid()))
    try:
        order=[i for i in description['indices'] if i!=0]+[0]
        for index in order:complete_one(index)
        json_write(DEST/'completion_fit_coverage.json',dict(passed=True,indices=description['indices'],protocol_hash=digest(description),completed_utc=utc_now()))
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['plan','fit','run']);parser.add_argument('--index',type=int,default=1)
    args=parser.parse_args()
    if args.stage=='plan':specification()
    elif args.stage=='fit':complete_one(args.index)
    else:run()

if __name__=='__main__':main()
