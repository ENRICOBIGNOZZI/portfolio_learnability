"""Predeclared simulation design. Do not alter based on realized slopes or curves."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

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
    loading_map: str = 'baseline_original'
    eta: float = 0.

    def parameters(self):
        return DGPParameters(N=self.N, rho=(self.rho,)*6, sigma_eps=self.sigma_eps,
                             mu_F=tuple(self.signal_multiplier*x for x in (.10, .06, .03)),
                             loading_map=self.loading_map, eta=self.eta)

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
    extra_T: tuple[int, ...] = ()
    extended_replications: int = 0
    protocol_hash: str = ''
    theory_scale: tuple[tuple[str, float], ...] = ()
    cases: tuple[Environment, ...] = field(default_factory=lambda: tuple(environments()))

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def design_for(profile):
    if profile == 'confirmation_v2':
        protocol = json.loads((Path(__file__).parent/'confirmation_v2.json').read_text())
        expected = protocol.pop('protocol_hash')
        actual = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if actual != expected or protocol['status'] != 'frozen_before_production':
            raise ValueError('Confirmation protocol is missing, unfrozen or modified.')
        cases = tuple(Environment(**case) for case in protocol['cases'])
        return Design(profile=profile, master_seed=protocol['master_seed'],
                      basis_seed=protocol['seeds']['anchor'], population_seed=protocol['seeds']['quadrature'],
                      bootstrap_seed=protocol['seeds']['bootstrap'], T=tuple(protocol['practical_T']),
                      extra_T=tuple(protocol['extra_T']), extended_replications=protocol['extended_replications'],
                      oos_periods=protocol['oos_periods'], population_groups=protocol['population_groups'],
                      bootstrap_replications=protocol['bootstrap_replications'], workers=protocol['workers'],
                      approximation_relative_tolerance=.05, cases=cases, protocol_hash=expected,
                      theory_scale=tuple((name, value) for name, value in protocol['theory_scale'].items()))
    if profile == 'paper':
        return Design()
    if profile == 'smoke':
        return Design(profile='smoke', T=(60, 120, 240), oos_periods=120,
                      penalties=tuple(np.geomspace(1e-12, 1e-2, 32)), population_groups=1024,
                      bootstrap_replications=100,
                      cases=(Environment('baseline', 0, rank=64, replications=4),))
    raise ValueError('Unknown profile.')
