"""A numerical subspace of the actual Matérn RKHS, not an economic factor model."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
from scipy.stats import qmc


def matern(x, y, nu=1.5, ell=1.0):
    if nu <= 0 or ell <= 0:
        raise ValueError('Matérn smoothness and length scale must be positive.')
    r = np.sqrt(2*nu)*cdist(np.atleast_2d(x), np.atleast_2d(y))/ell
    if nu == 1.5:
        return (1+r)*np.exp(-r)
    if nu == .5:
        return np.exp(-r)
    if nu == 2.5:
        return (1+r+r*r/3)*np.exp(-r)
    out = np.ones_like(r)
    positive = r > 0
    out[positive] = 2**(1-nu)/gamma(nu)*r[positive]**nu*kv(nu, r[positive])
    return out


@dataclass
class NystromBasis:
    rank: int = 256
    dimension: int = 6
    nu: float = 1.5
    ell: float = 1.0
    seed: int = 20261007

    def __post_init__(self):
        if self.rank < 4 or self.rank & (self.rank-1):
            raise ValueError('Use a power-of-two rank for nested Sobol inducing points.')
        self.anchors = 2*qmc.Sobol(self.dimension, scramble=True, seed=self.seed).random_base2(int(np.log2(self.rank)))-1
        gram = matern(self.anchors, self.anchors, self.nu, self.ell)
        values, vectors = eigh(gram)
        if values[0] <= values[-1]*1e-12:
            raise ValueError('Inducing Gram matrix is ill-conditioned; no hidden jitter is added.')
        self.inverse_root = (vectors/np.sqrt(values)) @ vectors.T
        self.gram = gram

    def raw(self, z):
        return matern(z, self.anchors, self.nu, self.ell)

    def features(self, z):
        return self.raw(z) @ self.inverse_root

    def managed(self, z, returns):
        # Sum before whitening: identical result, avoids N times rank² work.
        return (np.asarray(returns) @ self.raw(z) / len(z)) @ self.inverse_root

