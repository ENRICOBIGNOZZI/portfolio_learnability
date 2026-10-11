"""Cache verified population eigensystems for cheap targeted repeat fits."""
import numpy as np
from scipy.linalg import eigh
from simulations.closeout.numerical import CACHE,freeze
from simulations.closeout.results import DEST
from simulations.extended.compute import load_operator
from simulations.extended.design import parameters
from simulations.provenance import npz_write


def run():
    plan=freeze()
    for name in ('baseline','N300','N1200'):
        path=CACHE/f'eigensystem_{name}.npz'
        m,S,floor=load_operator(name,4096,plan['primary_groups'],plan['primary_seed'])
        if path.exists():
            with np.load(path) as z: values,vectors=z['values'],z['vectors']
        else:
            values,vectors=eigh(S)
            # Cached derivation; the original archive is hash-checked above.
            npz_write(path,values=values,vectors=vectors)
        projection=vectors.T@m
        npz_write(DEST/'population_spectral'/f'{name}.npz',values=values,mean_projection=projection,floor=floor)
        if name=='baseline':
            npz_write(DEST/'spectrum_energy.npz',eigenvalues=values[::-1],target_energy=(projection**2/values)[::-1],projection_floor=floor,q=parameters(name).q_star)
        print(f'Cached population eigensystem {name}',flush=True)
        del S,vectors

if __name__=='__main__':run()
