"""Three-factor baseline from DGP_Baseline_Spec.pdf (October 2026).

Dates pair formation characteristics Z_t with returns R_{t+1}. No estimator,
target policy, or managed eigenvalues are inputs to this economy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
from scipy.special import ndtr

from kernels import exact_kernel


@dataclass(frozen=True)
class DGPParameters:
    N: int = 600
    D: int = 6
    K_F: int = 3
    rho: tuple[float, ...] = (0.95,) * 6
    nu: float = 1.5
    ell: float = 1.0
    c_beta: float = 0.04**2
    sigma_eps: float = 0.08
    mu_F: tuple[float, ...] = (0.10, 0.06, 0.03)

    def __post_init__(self):
        if isinstance(self.N, (bool, np.bool_)) or not isinstance(self.N, (int, np.integer)) or self.N < 3 or self.N % 3:
            raise ValueError('N must be a positive integer divisible by three.')
        if self.D != 6 or self.K_F != 3 or self.nu != 1.5:
            raise ValueError('The baseline requires D=6, K_F=3, nu=1.5.')
        rho, mu = np.asarray(self.rho, dtype=np.float64), np.asarray(self.mu_F, dtype=np.float64)
        if rho.shape != (6,) or not np.isfinite(rho).all() or np.any(np.abs(rho) >= 1):
            raise ValueError('rho must have six finite entries with absolute value < 1.')
        if mu.shape != (3,) or not np.isfinite(mu).all() or not 0 < np.linalg.norm(mu) < 1:
            raise ValueError('mu_F must have three finite entries and norm strictly inside (0,1).')
        for name in ('ell', 'c_beta', 'sigma_eps'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and strictly positive.')
        object.__setattr__(self, 'N', int(self.N))
        object.__setattr__(self, 'rho', tuple(float(x) for x in rho))
        object.__setattr__(self, 'mu_F', tuple(float(x) for x in mu))
        # Check before any simulation, including parameters near the boundary.
        if np.linalg.eigvalsh(self.factor_covariance).min() <= 0:
            raise ValueError('Factor covariance must be numerically positive definite.')
        np.linalg.cholesky(self.factor_covariance)

    @property
    def G(self) -> int:
        return self.N // 3

    @property
    def gamma_beta(self) -> np.ndarray:
        return np.eye(3, dtype=np.float64) * (self.c_beta / 3)

    @property
    def factor_covariance(self) -> np.ndarray:
        return np.eye(3, dtype=np.float64) - np.outer(self.mu_F, self.mu_F)

    @property
    def policy_coefficients(self) -> np.ndarray:
        """Woodbury plus B'B/N=Gamma: coefficients in the beta basis."""
        return np.linalg.solve(self.gamma_beta + self.sigma_eps**2 / self.N * np.eye(3), self.mu_F)

    @property
    def q_star(self) -> float:
        g = self.c_beta / 3
        return float(g / (g + self.sigma_eps**2 / self.N) * np.dot(self.mu_F, self.mu_F))

    @property
    def sr_star(self) -> float:
        return float(np.sqrt(self.q_star / (1 - self.q_star)))

    def constants(self) -> dict:
        return {
            'G': self.G, 'Gamma_beta': self.gamma_beta.tolist(),
            'factor_covariance': self.factor_covariance.tolist(),
            'factor_second_moment': np.eye(3).tolist(),
            'W_star_beta_coefficients': self.policy_coefficients.tolist(),
            'q_star': self.q_star, 'SR_star': self.sr_star,
            'optimal_loss': 1 - self.q_star, 'kappa_K': 1.0,
            'B_X_squared': 4 * (self.c_beta + self.sigma_eps**2),
            'K4': 3.0, 'b': 1.5, 'sobolev_order': 4.5,
            'spectrum_lower_multiplier': self.sigma_eps**2 / self.N,
            'spectrum_upper_multiplier': self.c_beta + self.sigma_eps**2 / self.N,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def population_ranks(state: np.ndarray) -> np.ndarray:
    """Population Phi ranks, never empirical cross-sectional ranks.

    Clipping only handles float64 CDF rounding to endpoints in extreme tails.
    The real-valued model is continuous; machine values are necessarily finite.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 6 or not np.isfinite(state).all():
        raise ValueError('State must be a finite groups-by-six matrix.')
    return np.clip(ndtr(state), np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)


def balanced_triplets(ranks: np.ndarray) -> np.ndarray:
    """Group-major stock ordering: (g,0), (g,1), (g,2); same phase in all D."""
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.ndim != 2 or ranks.shape[1] != 6 or not np.isfinite(ranks).all() or np.any((ranks <= 0) | (ranks >= 1)):
        raise ValueError('Population ranks must be a groups-by-six matrix inside (0,1).')
    shifted = (ranks[:, None, :] + np.arange(3, dtype=np.float64)[None, :, None] / 3) % 1
    z = (2 * shifted - 1).reshape(-1, 6)
    # A phase may land exactly on zero by rounding, a probability-zero event
    # in the continuous DGP. Keep the numerical representation in the open cube.
    return np.clip(z, np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0))


def beta(z: np.ndarray, parameters: DGPParameters) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim < 1 or z.shape[-1] != 6 or not np.isfinite(z).all() or np.any(np.abs(z) >= 1):
        raise ValueError('Characteristics must lie in the open six-dimensional cube.')
    angle = np.pi * (z[..., 0] + 1)
    return np.sqrt(parameters.c_beta) * np.stack((
        np.sqrt(2 / 3) * np.cos(angle), np.sqrt(2 / 3) * np.sin(angle),
        np.full_like(angle, 1 / np.sqrt(3))), axis=-1)


def w_star(z: np.ndarray, parameters: DGPParameters) -> np.ndarray:
    """Scalar characteristic policy W*, not the stock weights W*/N."""
    return beta(z, parameters) @ parameters.policy_coefficients


def kernel_gram(z: np.ndarray, parameters: DGPParameters) -> np.ndarray:
    return exact_kernel(z, z, 'matern32', parameters.ell)


def conditional_moments(loadings: np.ndarray, parameters: DGPParameters):
    b = np.asarray(loadings, dtype=np.float64)
    if b.shape != (parameters.N, 3) or not np.isfinite(b).all():
        raise ValueError('Loadings must be a finite N-by-three matrix.')
    mean = b @ parameters.mu_F
    second = b @ b.T + parameters.sigma_eps**2 * np.eye(parameters.N)
    covariance = b @ parameters.factor_covariance @ b.T + parameters.sigma_eps**2 * np.eye(parameters.N)
    return mean, covariance, second


@dataclass
class DGPDate:
    t: int
    state: np.ndarray
    z: np.ndarray
    loadings: np.ndarray
    factors: np.ndarray
    epsilon: np.ndarray
    returns: np.ndarray


class BalancedFactorDGP:
    """Memory-bounded simulator with independent, explicitly seeded RNG streams."""

    def __init__(self, parameters: DGPParameters = DGPParameters(), seed: int = 1729):
        self.parameters = parameters
        streams = np.random.SeedSequence(seed).spawn(3)
        self.characteristic_rng, self.factor_rng, self.epsilon_rng = (
            np.random.default_rng(stream) for stream in streams)
        self.state = self.characteristic_rng.standard_normal((parameters.G, parameters.D))
        self._factor_root = np.linalg.cholesky(parameters.factor_covariance)
        self._rho = np.asarray(parameters.rho)
        self._innovation_scale = np.sqrt(1 - self._rho**2)
        self.t = 0

    def step(self) -> DGPDate:
        p = self.parameters
        state = self.state.copy()
        z = balanced_triplets(population_ranks(state))
        b = beta(z, p)
        factors = np.asarray(p.mu_F) + self._factor_root @ self.factor_rng.standard_normal(3)
        epsilon = p.sigma_eps * self.epsilon_rng.standard_normal(p.N)
        result = DGPDate(self.t, state, z, b, factors, epsilon, b @ factors + epsilon)
        self.state = self._rho * state + self._innovation_scale * self.characteristic_rng.standard_normal(state.shape)
        self.t += 1
        return result

    def simulate(self, periods: int) -> Iterator[DGPDate]:
        if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods < 1:
            raise ValueError('periods must be a positive integer.')
        for _ in range(periods):
            yield self.step()
