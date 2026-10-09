"""Portable population matrices in small, checksummed, lossless chunks."""
import json
import numpy as np

from simulations.extended.design import OUTPUT
from simulations.provenance import npz_write,json_write,file_hash


def archive_folder(name,rank,groups,quadrature_seed):
    economic_name='baseline' if name.startswith('rho') else name
    return OUTPUT/'population'/'operators'/f'{economic_name}_P{rank}_Q{groups}_S{quadrature_seed}'


def read_archived_operator(name,rank,groups,quadrature_seed):
    folder=archive_folder(name,rank,groups,quadrature_seed)
    manifest=json.loads((folder/'index.json').read_text())
    if manifest['rank']!=rank or manifest['groups']!=groups or manifest['seed']!=quadrature_seed:
        raise ValueError('Population archive identity mismatch.')
    S=np.empty((rank,rank))
    cursor=0
    for chunk in manifest['chunks']:
        path=folder/chunk['file']
        if file_hash(path)!=chunk['sha256'] or chunk['first_row']!=cursor:
            raise ValueError('Population archive chunk failed integrity validation.')
        with np.load(path) as z:
            block=z['second_rows']
            if block.shape!=(chunk['rows'],rank):
                raise ValueError('Population archive chunk has the wrong shape.')
            S[cursor:cursor+len(block)]=block
            cursor+=len(block)
    if cursor!=rank:
        raise ValueError('Population archive has missing rows.')
    path=folder/'moments.npz'
    if file_hash(path)!=manifest['moments_sha256']:
        raise ValueError('Population archive moments failed integrity validation.')
    with np.load(path) as z:
        m=z['mean'];floor=float(z['floor'])
    return m,S,floor


def export_population():
    from simulations.extended.compute import load_operator
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    cases=[(name,protocol['rank']) for name in ('baseline','N300','N1200')]
    cases.append(('baseline',512))  # Paired decomposition of the previous run.
    for name,rank in sorted(set(cases)):
        groups=protocol['population_groups'];qseed=protocol['population_seed']
        m,S,floor=load_operator(name,rank,groups,qseed)
        folder=archive_folder(name,rank,groups,qseed)
        rows=[]
        for start in range(0,rank,512):
            path=folder/f'rows_{start:05d}.npz'
            block=S[start:start+512]
            npz_write(path,second_rows=block)
            rows.append(dict(file=path.name,first_row=start,rows=len(block),sha256=file_hash(path)))
        npz_write(folder/'moments.npz',mean=m,floor=floor)
        json_write(folder/'index.json',dict(rank=rank,groups=groups,seed=qseed,
            economy=name,chunks=rows,moments_sha256=file_hash(folder/'moments.npz'),
            run_hash=protocol['run_hash']))
        restored=read_archived_operator(name,rank,groups,qseed)
        np.testing.assert_array_equal(restored[0],m)
        np.testing.assert_array_equal(restored[1],S)
        assert restored[2]==floor
        print('Archived and verified population operator',name,rank,flush=True)

if __name__=='__main__':
    export_population()
