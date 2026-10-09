"""Period-two Fourier target, with a fixed Parseval RMS normalization.

The ideal target is the infinite series. Every numerical truncation is smooth;
these simulations cannot establish an exact Sobolev/source condition.
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval


@lru_cache(maxsize=1)
def normalization():
    # Parseval: RMS(psi)^2 = sum a_m^2 / 2, RMS(sine)^2 = 1/2.
    # The omitted squared-coefficient sum is bounded by
    # L^-9 / (9 log(L+1)^2); this is below float64 precision here.
    m = np.arange(2, 65537, dtype=float)
    return float(1 / np.sqrt(np.sum(m**-10 / np.log1p(m)**2)))


@lru_cache(maxsize=12)
def coefficients(terms=128):
    if isinstance(terms, bool) or not isinstance(terms, (int, np.integer)) or terms < 2:
        raise ValueError('Fourier truncation must be an integer at least two.')
    result = np.zeros(terms+1)
    m = np.arange(2, terms+1, dtype=float)
    result[2:] = normalization() / (m**5 * np.log1p(m))
    result.setflags(write=False)
    return result


def normalized_psi(u, terms=128):
    """Vectorized Clenshaw evaluation; O(M n) work and O(n) memory."""
    return chebval(np.cos(np.pi*np.asarray(u)), coefficients(terms))


def normalized_psi_gradient(u, terms=128):
    u = np.asarray(u)
    return -np.pi*np.sin(np.pi*u)*chebval(np.cos(np.pi*u), chebder(coefficients(terms)))


def uniform_tail_bound(terms):
    """Absolute normalized value error bound by a decreasing integral majorant."""
    return normalization() / (4*terms**4*np.log(terms+1))
