"""User DGP, with explicit random streams and bounded-memory computations.

The equations and normalization are those of source/user_simulation.py.
No work runs at import. Apple Accelerate is an optional cosine implementation.
"""
from __future__ import annotations

import ctypes
import sys
from functools import lru_cache

import numpy as np
from scipy import linalg
from scipy.interpolate import CubicSpline
from scipy.special import ndtr
from scipy.stats import chi2, norm


@lru_cache(maxsize=1)
def _accelerate():
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        fun = lib.vvcos
        fun.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        return lib, fun
    except (OSError, AttributeError):
        return None


def cosine(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    accelerated = _accelerate()
    if accelerated is None:
        return np.cos(x)
    out = np.empty_like(x)
    size = ctypes.c_int(x.size)
    accelerated[1](out.ctypes.data, x.ctypes.data, ctypes.byref(size))
    return out


def matern_rff(D=3, s=3, M=1000, ell=1.0, seed=42):
    if s <= D / 2 or M < 1 or ell <= 0:
        raise ValueError("Require s > D/2, M >= 1 and ell > 0")
    rng = np.random.default_rng(seed)
    df = 2 * s - D
    omega = rng.normal(size=(D, M))
    omega /= np.sqrt(rng.chisquare(df, size=M))[None, :]
    omega *= np.sqrt(df) / ell
    return omega, rng.uniform(0, 2 * np.pi, M)


def phi(Z, omega, phase):
    return np.sqrt(2 / omega.shape[1]) * cosine(Z @ omega + phase)


def psi_direct(z, terms=1000):
    z = np.asarray(z)
    p = np.zeros_like(z, dtype=float)
    for m in range(2, terms + 1):
        p += np.cos(np.pi * m * z) / (m**3.5 * np.log(m + 1))
    return p


@lru_cache(maxsize=8)
def _psi_spline(terms):
    # FFT evaluates exactly the finite Fourier polynomial at grid nodes.
    # Cubic periodic interpolation is independently audited against direct sums.
    size = max(65536, 2 ** int(np.ceil(np.log2(4 * (terms + 1)))))
    coeff = np.zeros(size // 2 + 1, dtype=complex)
    m = np.arange(2, terms + 1)
    coeff[m] = size / (2 * m**3.5 * np.log(m + 1))
    values = np.fft.irfft(coeff, n=size)
    return CubicSpline(np.linspace(0, 2, size + 1),
                       np.r_[values, values[0]], bc_type="periodic")


def psi(z, terms=1000):
    return _psi_spline(terms)(np.mod(z, 2.0))


def calibrate(N, SR_ann=3.0, sigma_eps=0.04, lambda_factor=None, terms=1000,
              safety=1.01):
    lam = np.array([0.2, 0.1, 0.05]) if lambda_factor is None else np.asarray(lambda_factor)
    m = np.arange(2, terms + 1, dtype=float)
    coeff = 1 / (m**3.5 * np.log(m + 1))
    # Orthogonality gives the same variance as the user's 2M+1 grid.
    v = np.sum(coeff**2) / 2
    p = SR_ann**2 / (12 + SR_ann**2)
    L = lam @ lam
    H = (np.sqrt(2) + np.sqrt(3 * L) * coeff.sum())**2 + 3 * coeff.sum()**2
    den = 1 + v * L - safety * p * H
    if not (N > 0 and L < 1 and den > 0):
        raise ValueError("Inadmissible calibration")
    amplitude = np.sqrt(safety * p * sigma_eps**2 / (N * den))
    kappa = N * amplitude**2 * (1 + v * L) / p
    bound = sigma_eps**2 + N * amplitude**2 * H
    if kappa <= bound:
        raise ValueError("Covariance dominance failed")
    return dict(A_alpha=amplitude, A_beta=amplitude, kappa_0=kappa,
                upper_bound=bound, p_star=p, sigma_eps=sigma_eps,
                lambda_factor=lam, Sigma_factor=np.eye(len(lam)) - np.outer(lam, lam))


def characteristics(T, N, rho, rng, D=3):
    if T < 1 or not abs(rho) < 1:
        raise ValueError("Require T >= 1, |rho| < 1")
    states = rng.normal(size=(T, N, D))
    innovations = np.sqrt(1 - rho**2)
    for t in range(1, T):
        states[t] = rho * states[t - 1] + innovations * states[t]
    return 2 * ndtr(states) - 1


def gaussian(shape, rng, cutoff=4.0):
    x = rng.normal(size=shape)
    bad = np.abs(x) > cutoff
    while np.any(bad):
        x[bad] = rng.normal(size=bad.sum())
        bad = np.abs(x) > cutoff
    v = 1 - 2 * cutoff * norm.pdf(cutoff) / (2 * norm.cdf(cutoff) - 1)
    return x / np.sqrt(v)


def factors(T, cal, rng, cutoff=4.0):
    lam, cov = cal["lambda_factor"], cal["Sigma_factor"]
    root = linalg.cholesky(cov, lower=True)
    z = rng.normal(size=(T, len(lam)))
    bad = np.sum(z**2, axis=1) > cutoff**2
    while np.any(bad):
        z[bad] = rng.normal(size=(bad.sum(), len(lam)))
        bad = np.sum(z**2, axis=1) > cutoff**2
    v = chi2.cdf(cutoff**2, len(lam) + 2) / chi2.cdf(cutoff**2, len(lam))
    return lam + z @ root.T / np.sqrt(v)


def moments(Z, cal, terms=1000):
    A = cal["A_alpha"] * np.sqrt(2) * np.cos(np.pi * np.sum(Z, axis=-1))
    B = cal["A_beta"] * psi(Z, terms)
    mu = A + B @ cal["lambda_factor"]
    btu = np.einsum("tnk,tn->tk", B, mu)
    d = ((cal["kappa_0"] - cal["sigma_eps"]**2 - np.sum(mu**2, axis=1))[:, None] * mu
         - np.einsum("tnk,tk->tn", B, btu @ cal["Sigma_factor"]))
    denom = np.sum(mu * d, axis=1)
    if np.any(denom < -1e-20):
        raise ArithmeticError("Nonpositive shock covariance")
    h = np.divide(d, np.sqrt(np.maximum(denom, 0))[:, None],
                  out=np.zeros_like(d), where=denom[:, None] > 0)
    return A, B, mu, h


def returns(Z, cal, rng, terms=1000):
    T, N, _ = Z.shape
    A, B, mu, h = moments(Z, cal, terms)
    F = factors(T, cal, rng)
    R = (A + np.sum(B * F[:, None, :], axis=-1)
         + cal["sigma_eps"] * gaussian((T, N), rng)
         + h * gaussian((T, 1), rng))
    return R, mu, h


def managed_returns(Z, R, omega, phase):
    G = np.empty((len(Z), len(phase)))
    for t in range(len(Z)):
        G[t] = R[t] @ phi(Z[t], omega, phase) / Z.shape[1]
    return G


def fit_path(G, lambdas):
    """One spectral decomposition for the entire strictly positive ridge path."""
    T, M = G.shape
    lambdas = np.atleast_1d(lambdas)
    if np.any(lambdas <= 0):
        raise ValueError("Ridge penalties must be positive")
    if T <= M:
        eig, U = linalg.eigh(G @ G.T / T, check_finite=False)
        eig = np.maximum(eig, 0)
        theta = G.T @ (U @ ((U.T @ np.ones(T) / T)[:, None]
                           / (eig[:, None] + lambdas)))
    else:
        eig, U = linalg.eigh(G.T @ G / T, check_finite=False)
        eig = np.maximum(eig, 0)
        theta = U @ ((U.T @ G.mean(axis=0))[:, None] / (eig[:, None] + lambdas))
    complexity = np.sum(eig[:, None] / (eig[:, None] + lambdas), axis=0)
    return theta, complexity, eig


def evaluate_policies(thetas, Z, cal, omega, phase, terms=1000, block=8):
    """Integrate innovation moments exactly; integrate independent states by MC.

    This estimates marginal, annualized population Sharpe, without serial-return
    annualization. No evaluation outcomes enter fitting or penalty selection.
    """
    T, N, _ = Z.shape
    count = thetas.shape[1]
    mean = np.zeros(count)
    second = np.zeros(count)
    regret = np.zeros(count)
    oracle = []
    for start in range(0, T, block):
        z = Z[start:start + block]
        _, B, mu, h = moments(z, cal, terms)
        weights = (phi(z.reshape(-1, 3), omega, phase) @ thetas / N).reshape(len(z), N, count)
        m = np.einsum("bnl,bn->bl", weights, mu)
        exposure = np.einsum("bnl,bnk->blk", weights, B)
        common = np.einsum("bnl,bn->bl", weights, h)
        s = (m**2 + cal["sigma_eps"]**2 * np.sum(weights**2, axis=1)
             + np.einsum("blk,km,blm->bl", exposure, cal["Sigma_factor"], exposure)
             + common**2)
        p = np.sum(mu**2, axis=1) / cal["kappa_0"]
        mean += m.sum(axis=0)
        second += s.sum(axis=0)
        regret += (s - 2 * m + p[:, None]).sum(axis=0)
        oracle.extend(p)
    mean /= T
    second /= T
    variance = second - mean**2
    if np.any(variance <= 0):
        raise ArithmeticError("Nonpositive estimated payoff variance")
    sharpe = np.sqrt(12) * mean / np.sqrt(variance)
    # A shared finite state sample has its own oracle moment, recorded explicitly.
    p_eval = np.mean(oracle)
    return dict(SR_hat=sharpe, mean=mean, second=second,
                Q_hat=1 - 2 * mean + second, regret=regret / T,
                SR_oracle_eval=np.full(count, np.sqrt(12 * p_eval / (1 - p_eval))))
