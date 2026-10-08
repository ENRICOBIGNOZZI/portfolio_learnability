"""Observe every production date without changing any draw, policy or estimator.

Older valid checkpoints are audited by deterministic DGP-only replay; kernel
features and model fits are not recomputed. Newly simulated paths are checked
inline. This observer is separate from the scientific simulation engine.
"""
import hashlib
from pathlib import Path
import time

import numpy as np

from simulations.dgp.balanced import BalancedFactorDGP, beta
from simulations.dgp.robustness import transform_date
from simulations.experiments import monte_carlo


_ORIGINAL_STEP = BalancedFactorDGP.step
_CONTEXT = None
_AUDIT = None


def initialize_observed_worker(context):
    global _CONTEXT
    _CONTEXT = context
    monte_carlo.initialize_worker(context)
    BalancedFactorDGP.step = observed_step


def observed_step(self):
    date = _ORIGINAL_STEP(self)
    if _AUDIT is not None:
        p = self.parameters
        env = _CONTEXT['environment']
        z, _ = transform_date(date, p, env.variant)
        b = date.loadings if env.variant=='baseline' else beta(z, p)
        coefficients = _CONTEXT['integrity_coefficients']
        weights = b@coefficients/p.N
        mean = b@p.mu_F
        variance = p.sigma_eps**2*(np.exp(.7*z[:, 1]) if env.variant=='heteroskedastic' else np.ones(p.N))
        residual = b@(b.T@weights)+variance*weights-mean
        norm_error = np.max(np.abs(np.sum(b*b, axis=1)-p.c_beta))
        balance_error = np.linalg.norm(b.T@b/p.N-p.gamma_beta)
        equation_error = np.linalg.norm(residual)
        coordinates = beta(z, p)@np.asarray(p.mu_F)/(p.c_beta/3+p.sigma_eps**2/p.N)/p.N
        coordinate_error = np.max(np.abs(weights-coordinates))
        _AUDIT[0] += 1
        _AUDIT[1:] = np.maximum(_AUDIT[1:], [norm_error,balance_error,equation_error,coordinate_error])
        if norm_error>=1e-12 or balance_error>=1e-12 or coordinate_error>=1e-11:
            raise ValueError('A production date violated a loading/balance/coordinate invariant.')
        if env.variant!='heteroskedastic' and equation_error>=1e-11:
            raise ValueError('A production date violated exact conditional pricing.')
    return date


def observed_replication(index):
    global _AUDIT
    start = time.perf_counter()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _CONTEXT['integrity_coefficients'] = _CONTEXT['environment'].parameters().policy_coefficients
    _AUDIT = np.zeros(5, dtype=np.float64)
    path = Path(_CONTEXT['checkpoint_dir'])/f'{index:04d}.npz'
    existing = path.exists()
    result = monte_carlo.replication(index)
    if existing:
        with np.load(path, allow_pickle=False) as saved:
            if 'date_audit_source_hash' in saved and str(saved['date_audit_source_hash'])==fingerprint:
                _AUDIT = None
                return result
            train_seed, test_seed = int(saved['train_seed']), int(saved['test_seed'])
        p, design = _CONTEXT['environment'].parameters(), _CONTEXT['design']
        sizes = design.T+(design.extra_T if index < design.extended_replications and _CONTEXT['environment'].nu == 1.5 else ())
        training_periods = max(sizes)+(design.oos_periods if design.profile == 'confirmation_v2' else 0)
        for seed, periods in ((train_seed, training_periods), (test_seed, design.oos_periods)):
            for _ in BalancedFactorDGP(p, seed).simulate(periods):
                pass
    design = _CONTEXT['design']
    if design.profile == 'confirmation_v2':
        sizes = design.T+(design.extra_T if index < design.extended_replications and _CONTEXT['environment'].nu == 1.5 else ())
        expected = max(sizes)+2*design.oos_periods
    else:
        expected = max(design.T)+design.oos_periods
    if int(_AUDIT[0])!=expected:
        raise ValueError('Production date audit is incomplete.')
    with np.load(path, allow_pickle=False) as saved:
        contents = dict(saved)
    contents.update(date_audit=_AUDIT, date_audit_source_hash=fingerprint,
                    integrity_execution_seconds=time.perf_counter()-start,
                    date_audit_mode='deterministic_DGP_replay' if existing else 'inline')
    temporary = path.with_suffix('.audit-tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **contents)
    temporary.replace(path)
    _AUDIT = None
    return result
