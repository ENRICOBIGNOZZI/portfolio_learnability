"""Independent population group quadrature for policy evaluation, never selection."""
import numpy as np
from scipy.linalg import eigh

from simulations.dgp.balanced import balanced_triplets, beta, population_ranks
from scipy.stats import rankdata


def population_moments(parameters, basis, groups=8192, seed=20261008, variant='baseline'):
    p = parameters
    rng = np.random.default_rng(seed)
    if variant == 'empirical_ranks':
        # Cross-sectional ranking couples all G groups: integrate full
        # independent population states, rather than pretend ranks are iid.
        second_raw = np.zeros((basis.rank, basis.rank))
        mean_raw = np.zeros(basis.rank)
        states = max(128, groups//8)
        for _ in range(states):
            z = balanced_triplets(population_ranks(rng.standard_normal((p.G, 6))))
            z = 2*(rankdata(z, axis=0)-.5)/p.N-1
            raw = basis.raw(z)
            exposure = beta(z, p).T@raw/p.N
            mean_raw += np.asarray(p.mu_F)@exposure
            second_raw += exposure.T@exposure+p.sigma_eps**2/p.N**2*(raw.T@raw)
        root = basis.inverse_root
        second = root@(second_raw/states)@root
        return root@(mean_raw/states), (second+second.T)/2
    z = balanced_triplets(np.clip(rng.random((groups, 6)), np.finfo(float).eps, 1-np.finfo(float).eps))
    raw = basis.raw(z)
    # Work in unwhitened coordinates, then apply the RKHS orthonormalizer once.
    loading = beta(z, p).reshape(groups, 3, 3)
    h = np.einsum('grm,grk->gmk', raw.reshape(groups, 3, basis.rank), loading)/3
    h_mean = h.mean(axis=0)
    flattened = h.transpose(1, 0, 2).reshape(basis.rank, -1)
    factor = flattened@flattened.T/(p.G*groups)+(1-1/p.G)*(h_mean@h_mean.T)
    variance = p.sigma_eps**2*(np.exp(.7*z[:, 1]) if variant == 'heteroskedastic' else np.ones(len(z)))
    second_raw = factor + (raw.T@(variance[:, None]*raw))/(p.N*len(z))
    root = basis.inverse_root
    second = root@second_raw@root
    mean = root@h_mean@np.asarray(p.mu_F)
    return mean, (second+second.T)/2


def population_ridge(mean, second, penalties, q_star):
    values, vectors = eigh(second)
    if values[0] <= 0:
        raise ValueError('Population quadrature second moment is not positive definite.')
    projection = vectors.T@mean
    coefficients = vectors@(projection[:, None]/(values[:, None]+penalties))
    complexity = np.sum(values[:, None]/(values[:, None]+penalties), axis=0)
    regret = q_star-2*(mean@coefficients)+np.sum(coefficients*(second@coefficients), axis=0)
    subspace_floor = float(q_star-np.sum(projection**2/values))
    if min(subspace_floor, regret.min()) < -1e-10:
        raise ValueError('Population quadrature violates the conditional optimum.')
    return coefficients, complexity, regret, subspace_floor
