"""Bounded-memory evaluation of the canonical independent-group quadrature.

Only the order of floating-point accumulation changes. Random draws, balanced
triplets, group dependence and RKHS whitening follow population_moments.
Training and the frozen production evaluator do not import this module.
"""
import numpy as np

from simulations.dgp.balanced import balanced_triplets, beta


def population_moments_batched(parameters, basis, groups=8192, seed=20261008,
                               variant='baseline', batch_groups=256):
    if variant not in ('baseline', 'heteroskedastic'):
        raise ValueError('Batched quadrature requires independent population groups.')
    if groups < 1 or batch_groups < 1:
        raise ValueError('Quadrature and batch sizes must be positive.')
    p = parameters
    rng = np.random.default_rng(seed)
    h_sum = np.zeros((basis.rank, 3))
    factor_sum = np.zeros((basis.rank, basis.rank))
    noise_sum = np.zeros_like(factor_sum)
    for first in range(0, groups, batch_groups):
        size = min(batch_groups, groups-first)
        ranks = np.clip(rng.random((size, 6)), np.finfo(float).eps,
                        1-np.finfo(float).eps)
        z = balanced_triplets(ranks)
        raw = basis.raw(z)
        loading = beta(z, p).reshape(size, 3, 3)
        h = np.einsum('grm,grk->gmk', raw.reshape(size, 3, basis.rank), loading)/3
        h_sum += h.sum(axis=0)
        flattened = h.transpose(1, 0, 2).reshape(basis.rank, -1)
        factor_sum += flattened@flattened.T
        if variant == 'heteroskedastic':
            noise_sum += raw.T@(np.exp(.7*z[:, 1, None])*raw)
        else:
            noise_sum += raw.T@raw
    h_mean = h_sum/groups
    second_raw = (factor_sum/(p.G*groups)
                  +(1-1/p.G)*(h_mean@h_mean.T)
                  +p.sigma_eps**2*noise_sum/(p.N*3*groups))
    root = basis.inverse_root
    second = root@second_raw@root
    return root@h_mean@np.asarray(p.mu_F), (second+second.T)/2
