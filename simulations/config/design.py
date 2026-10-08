"""Predeclared simulation design. Do not alter based on realized slopes or curves."""
from dataclasses import asdict, dataclass, field
import hashlib
import json

import numpy as np

from simulations.dgp.balanced import DGPParameters


@dataclass(frozen=True)
class Environment:
    name: str
    code: int
    N: int = 600
    rho: float = .95
    signal_multiplier: float = 1.
    sigma_eps: float = .08
    nu: float = 1.5
    rank: int = 256
    variant: str = 'baseline'
    replications: int = 200

    def parameters(self):
        return DGPParameters(N=self.N, rho=(self.rho,)*6, sigma_eps=self.sigma_eps,
                             mu_F=tuple(self.signal_multiplier*x for x in (.10, .06, .03)))

    @property
    def theoretical_b(self):
        return 1+2*self.nu/6


def environments():
    return [Environment('baseline', 0, replications=500),
            Environment('N99', 1, N=99), Environment('N300', 2, N=300),
            Environment('N501', 3, N=501), Environment('N999', 4, N=999),
            Environment('weak_high_noise', 5, signal_multiplier=.5, sigma_eps=.12),
            Environment('strong_low_noise', 6, signal_multiplier=2., sigma_eps=.04),
            Environment('rho070', 7, rho=.70), Environment('rho090', 8, rho=.90),
            Environment('rho097', 9, rho=.97),
            Environment('slow_spectrum', 10, nu=.5), Environment('fast_spectrum', 11, nu=2.5),
            # Same economic code pairs rank experiments with baseline innovations.
            Environment('rank128', 0, rank=128), Environment('rank512', 0, rank=512),
            Environment('empirical_ranks', 12, variant='empirical_ranks'),
            Environment('heteroskedastic', 13, variant='heteroskedastic')]


@dataclass(frozen=True)
class Design:
    profile: str = 'paper'
    master_seed: int = 202610071
    basis_seed: int = 202610072
    population_seed: int = 202610073
    bootstrap_seed: int = 202610074
    T: tuple[int, ...] = (60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440)
    penalties: tuple[float, ...] = field(default_factory=lambda: tuple(np.geomspace(1e-12, 1e-2, 96)))
    validation_fraction: float = .25
    oos_periods: int = 720
    population_groups: int = 8192
    bootstrap_replications: int = 1000
    workers: int = 2
    approximation_relative_tolerance: float = .10
    cases: tuple[Environment, ...] = field(default_factory=lambda: tuple(environments()))

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def design_for(profile):
    if profile == 'paper':
        return Design()
    if profile == 'smoke':
        return Design(profile='smoke', T=(60, 120, 240), oos_periods=120,
                      penalties=tuple(np.geomspace(1e-12, 1e-2, 32)), population_groups=1024,
                      bootstrap_replications=100,
                      cases=(Environment('baseline', 0, rank=64, replications=4),))
    raise ValueError('Unknown profile.')
