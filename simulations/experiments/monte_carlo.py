"""Actual stock-return Monte Carlo, chronological validation and saved path surfaces."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from simulations.dgp.balanced import BalancedFactorDGP
from simulations.dgp.robustness import transform_date
from simulations.estimator.ridge import validation_select, moment_metrics, sample_metrics, ridge_path


_CONTEXT = None


def initialize_worker(context):
    global _CONTEXT
    _CONTEXT = context


def managed_path(p, basis, variant, seed, periods):
    output = np.empty((periods, basis.rank))
    linear = np.empty((periods, 7))
    equal = np.empty(periods)
    for t, date in enumerate(BalancedFactorDGP(p, seed).simulate(periods)):
        z, returns = transform_date(date, p, variant)
        output[t] = basis.managed(z, returns)
        linear[t, 0] = equal[t] = returns.mean()
        linear[t, 1:] = z.T@returns/p.N
    return output, linear, equal


def replication(replication_index):
    c = _CONTEXT
    design, environment, basis = c['design'], c['environment'], c['basis']
    destination = Path(c['checkpoint_dir'])/f'{replication_index:04d}.npz'
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            if str(saved['run_hash']) != c['run_hash']:
                raise ValueError('Checkpoint configuration/source mismatch; cannot silently reuse it.')
        return str(destination)
    start = time.perf_counter()
    p = environment.parameters()
    sequence = np.random.SeedSequence([design.master_seed, environment.code, replication_index])
    train_sequence, test_sequence = sequence.spawn(2)
    train_seed = int(train_sequence.generate_state(1, dtype=np.uint64)[0])
    test_seed = int(test_sequence.generate_state(1, dtype=np.uint64)[0])
    train, linear_train, _ = managed_path(p, basis, environment.variant, train_seed, max(design.T))
    test, linear_test, equal_test = managed_path(p, basis, environment.variant, test_seed, design.oos_periods)
    lam = np.asarray(design.penalties)
    shape = (len(design.T), len(lam))
    arrays = {name: np.empty(shape) for name in ('complexity', 'population_loss', 'population_sharpe',
                'oos_loss', 'oos_sharpe', 'regret', 'estimation_signed', 'estimation_norm', 'cross_term', 'validation_loss')}
    choices = np.empty(len(design.T), dtype=np.int64)
    linear_result = np.empty((len(design.T), 4))
    mean, second = c['population_mean'], c['population_second']
    population_coefficients = c['population_coefficients']
    for k, T in enumerate(design.T):
        coefficients, complexity, choice, validation, cut = validation_select(train[:T], lam, design.validation_fraction)
        loss, sr = moment_metrics(coefficients, mean, second)
        oos_loss, oos_sr = sample_metrics(test@coefficients)
        delta = coefficients-population_coefficients
        estimation_norm = np.sum(delta*(second@delta), axis=0)
        # Exact signed cross term in Q(W_hat)-Q(W_lambda). The subspace
        # approximation floor is included in population bias relative to W*.
        cross = 2*np.sum(delta*(second@population_coefficients-mean[:, None]), axis=0)
        regret = loss-c['optimal_loss']
        if regret.min() < -1e-9:
            raise ValueError('Population evaluation violates its declared optimum.')
        np.testing.assert_allclose(regret, c['population_bias']+estimation_norm+cross, atol=1e-10, rtol=1e-9)
        for name, value in [('complexity', complexity), ('population_loss', loss), ('population_sharpe', sr),
                            ('oos_loss', oos_loss), ('oos_sharpe', oos_sr), ('regret', regret),
                            ('estimation_signed', loss-c['population_loss']), ('estimation_norm', estimation_norm),
                            ('cross_term', cross), ('validation_loss', validation)]:
            arrays[name][k] = value
        choices[k] = choice
        linear_coef, linear_complexity, linear_choice, _, _ = validation_select(linear_train[:T], lam, design.validation_fraction)
        linear_loss, linear_sr = sample_metrics(linear_test@linear_coef[:, linear_choice])
        linear_result[k] = [linear_loss, linear_sr, linear_complexity[linear_choice], lam[linear_choice]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays, choice=choices, linear_benchmark=linear_result,
                            equal_benchmark=np.asarray(sample_metrics(equal_test)),
                            replication=replication_index, seed_entropy=np.asarray(sequence.entropy),
                            train_seed=np.uint64(train_seed), test_seed=np.uint64(test_seed),
                            run_hash=c['run_hash'], runtime_seconds=time.perf_counter()-start)
    temporary.replace(destination)
    return str(destination)
