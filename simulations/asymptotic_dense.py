"""Reuse unchanged pilots and collect dense confirmation one cell at a time.

The original scientific engine is unchanged. Cell CSV checkpoints bound disk
usage: coefficients are removed only after their full evaluated output is
saved atomically and hashed. Completed scientific outputs remain reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from . import asymptotic as engine

ALLOWED_GRID_CHANGES={'version','T_train','T_train_reference'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_upgrade(parent,config):
    if set(parent)!=set(config):
        raise ValueError('Dense design must preserve all pilot and scientific parameters')
    changed={key for key in parent if parent[key]!=config[key]}
    if not changed<=ALLOWED_GRID_CHANGES:
        raise ValueError(f'Cannot reuse pilots after changing {sorted(changed-ALLOWED_GRID_CHANGES)}')
    for key in ['T_train','T_train_reference']:
        if config[key]!=sorted(set(config[key])) or not set(parent[key])<=set(config[key]):
            raise ValueError('Dense grids must be ordered supersets of the original training sizes')
    return sorted(changed)


def reuse_pilots(parent_out,config,out):
    parent_out,out=Path(parent_out),Path(out)
    parent=json.loads((parent_out/'config.json').read_text())
    changed=validate_upgrade(parent,config)
    parent_signature=engine.scientific_hash(parent)
    if (parent_out/'cache/source_hash.txt').read_text()!=parent_signature:
        raise ValueError('Parent checkpoints have incompatible scientific provenance')
    # rank_selection is produced only after all fidelity policies and operators.
    selection=json.loads((parent_out/'rank_selection.json').read_text())
    assert selection['reference_rank']==max(config['candidate_ranks'])
    files=[Path(name) for name in ['pilot.csv','frozen_constants.json','fidelity_trials.csv','rank_selection.json']]
    files += [Path(f'cache/pilot_N{N}_trial{trial:03d}.npz')
              for N in config['N_values'] for trial in range(config['pilot_trials'])]
    N,rho=config['fixed_N'],config['fixed_rho']
    for kind,rank in [('nystrom',r) for r in config['candidate_ranks']]+[('rff',1000)]:
        files.append(Path(f'cache/evaluation_{kind}{rank}_N{N}.npz'))
        for trial in range(config['fidelity_trials']):
            relative=Path(f'cache/fidelity_{kind}{rank}_N{N}_rho{rho:g}_trial{trial:03d}.npz')
            with np.load(parent_out/relative) as data:
                np.testing.assert_array_equal(data['T_train'],config['fidelity_T_train'])
            files.append(relative)
    file_hashes={str(path):sha(parent_out/path) for path in files}
    engine.prepare(config,out)
    for relative in files:
        source,target=parent_out/relative,out/relative
        target.parent.mkdir(exist_ok=True,parents=True)
        if target.exists():
            if sha(target)!=file_hashes[str(relative)]:
                raise ValueError(f'Existing checkpoint differs: {target}')
        else:
            try:
                os.link(source,target)
            except OSError:
                shutil.copyfile(source,target)
    write_json(out/'reused_stages.json',dict(
        stages=['pilot','fidelity','evaluation at fixed N'],
        parent_design=parent,parent_source_hash=parent_signature,
        dense_source_hash=engine.scientific_hash(config),changed_parameters=changed,
        rationale='Only the subsequent confirmation training grid and design name changed. No confirmation policies were reused.',
        immutable_files_sha256=file_hashes))


def cell_times(config,N,rho):
    return config['T_train_reference'] if (N,rho)==(config['fixed_N'],config['fixed_rho']) else config['T_train']


def validate_cell(config,N,rho,frame,spectrum):
    times=cell_times(config,N,rho)
    keys=['N','rho','trial','lambda_multiplier','T_train']
    assert not frame.duplicated(keys).any()
    assert set(frame.N)=={N} and set(frame.rho)=={rho}
    assert len(frame)==config['trials']*len(times)*len(config['lambda_multipliers'])
    assert set(frame.trial)==set(range(config['trials']))
    assert set(frame.lambda_multiplier)==set(config['lambda_multipliers'])
    for _,group in frame.groupby(['trial','lambda_multiplier']):
        assert set(group.T_train)==set(times)
    assert np.isfinite(frame.select_dtypes('number')).all().all()
    assert len(spectrum)==65*config['trials']
    assert set(spectrum.T_train)=={config['spectrum_T_train']}
    assert not spectrum.duplicated(['trial','lambda_reg']).any()


def confirmation(config,out,workers):
    out=Path(out)
    cache=engine.prepare(config,out)
    signature=engine.scientific_hash(config)
    selection=json.loads((out/'rank_selection.json').read_text())
    rank=selection['selected_rank']
    constants=json.loads((out/'frozen_constants.json').read_text())
    engine.execute([(config,N,'nystrom',rank,str(cache)) for N in config['N_values']],
                   engine.evaluation_job,min(2,workers),'Confirmation evaluation operators')
    cells=out/'cells';cells.mkdir(exist_ok=True)
    reference=[config['fixed_N'],config['fixed_rho']]
    ordered=[reference]+[cell for cell in config['cells'] if cell!=reference]
    for N,rho in ordered:
        cell=cells/f'N{N}_rho{rho:g}';cell.mkdir(exist_ok=True)
        marker=cell/'completed.json'
        if marker.exists():
            state=json.loads(marker.read_text())
            assert state['source_hash']==signature and state['rank']==rank
            assert all(sha(cell/name)==checksum for name,checksum in state['files_sha256'].items())
            validate_cell(config,N,rho,pd.read_csv(cell/'confirmation_trials.csv'),pd.read_csv(cell/'complexity_lambda_trials.csv'))
            print(f'Reusing complete evaluated cell N={N}, rho={rho:g}',flush=True)
            continue
        times=cell_times(config,N,rho)
        jobs=[(config,N,rho,j,config['confirmation_seed'],times,'nystrom',rank,constants[str(N)],str(cache),'confirmation')
              for j in range(config['trials'])]
        engine.execute(jobs,engine.training_job,workers,f'Dense confirmation N={N}, rho={rho:g}')
        with tempfile.TemporaryDirectory(prefix='collect_',dir=cache) as temporary:
            staging=Path(temporary)
            (staging/'cache').symlink_to(cache.resolve(),target_is_directory=True)
            shutil.copyfile(out/'frozen_constants.json',staging/'frozen_constants.json')
            frame=engine.collect(config,staging,'confirmation',[(N,rho,'nystrom',rank)],
                {(N,rho):times},config['trials'],config['confirmation_seed'])
            spectrum=pd.read_csv(staging/'complexity_lambda_trials.csv')
            validate_cell(config,N,rho,frame,spectrum)
            names=['confirmation_trials.csv','complexity_lambda_trials.csv']
            for name in names:
                (staging/name).replace(cell/name)
            write_json(marker,dict(source_hash=signature,rank=rank,replications=config['trials'],
                       files_sha256={name:sha(cell/name) for name in names}))
        # These files were generated by this cell and have now been exported.
        for trial in range(config['trials']):
            (cache/f'confirmation_nystrom{rank}_N{N}_rho{rho:g}_trial{trial:03d}.npz').unlink()
        print(f'Exported and verified N={N}, rho={rho:g}; all {len(times)} actual training sizes retained',flush=True)
    for name in ['confirmation_trials.csv','complexity_lambda_trials.csv']:
        frame=pd.concat([pd.read_csv(cells/f'N{N}_rho{rho:g}'/name) for N,rho in config['cells']],ignore_index=True)
        temporary=out/(name+'.tmp')
        frame.to_csv(temporary,index=False);temporary.replace(out/name)
    write_json(out/'execution.json',dict(
        engine='simulations.asymptotic: unchanged training, evaluation and collect functions',
        collector_sha256=sha(Path(__file__)),source_hash=signature,
        collection='Atomic evaluated cell checkpoints; release only exported policy caches to bound disk usage',
        cell_execution_order=ordered))
    write_json(out/'completed.json',dict(completed_replications=len(config['cells'])*config['trials'],rank=rank,source_hash=signature))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=engine.ROOT/'config/asymptotic_dense.json')
    parser.add_argument('--parent-output',type=Path,default=engine.ROOT/'outputs/asymptotic_v2')
    parser.add_argument('--output',type=Path,default=engine.ROOT/'outputs/asymptotic_dense_v3')
    parser.add_argument('--stage',choices=['reuse','confirmation','all'],default='all')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.stage in ['reuse','all']:
        reuse_pilots(args.parent_output,config,args.output)
    if args.stage in ['confirmation','all']:
        confirmation(config,args.output,args.workers)


if __name__=='__main__':
    main()
