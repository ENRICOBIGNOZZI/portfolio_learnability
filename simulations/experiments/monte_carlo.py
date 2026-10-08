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
from simulations.provenance import npz_write, require


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
    if design.profile == 'confirmation_v2':
        return confirmation_replication(replication_index, c, destination)
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


def evaluate_fit(fit, test, future, mean, second, optimal_loss, population_coefficients, bias):
    """Evaluation receives a completed fit and cannot modify training or selection."""
    a = fit['coefficients']
    loss, sr = moment_metrics(a, mean, second)
    independent_loss, independent_sr = sample_metrics(test@a)
    forward_loss, forward_sr = sample_metrics(future@a)
    delta = a-population_coefficients
    norm = np.sum(delta*(second@delta), axis=0)
    cross = 2*np.sum(delta*(second@population_coefficients-mean[:, None]), axis=0)
    regret = loss-optimal_loss
    require(np.allclose(regret, bias+norm+cross, rtol=1e-8, atol=1e-11), 'Signed risk decomposition failed')
    return {'complexity': fit['complexity'], 'population_loss': loss, 'population_sharpe': sr,
            'oos_loss': independent_loss, 'oos_sharpe': independent_sr,
            'forward_loss': forward_loss, 'forward_sharpe': forward_sr, 'regret': regret,
            'estimation_norm': norm, 'cross_term': cross, 'estimation_signed': norm+cross}


def confirmation_replication(index, c, destination):
    from simulations.diagnostics.confirmation_pilot import seed
    from simulations.estimator.selectors import fit_training
    from simulations.experiments.population import population_ridge
    start = time.perf_counter()
    d, env, basis = c['design'], c['environment'], c['basis']
    p = env.parameters()
    sizes = d.T+(d.extra_T if index < d.extended_replications and env.nu == 1.5 else ())
    train_seed, test_seed = seed('train', index), seed('test', index)
    trajectory, affine, equal = managed_path(p, basis, env.variant, train_seed, max(sizes)+d.oos_periods)
    test, affine_test, equal_test = managed_path(p, basis, env.variant, test_seed, d.oos_periods)
    lam = np.asarray(d.penalties)
    s_ref = dict(d.theory_scale)[env.name]
    exponent = env.theoretical_b/(env.theoretical_b+1)
    saved = {name: [] for name in ('complexity', 'population_loss', 'population_sharpe', 'oos_loss', 'oos_sharpe',
             'forward_loss', 'forward_sharpe', 'regret', 'estimation_norm', 'cross_term', 'estimation_signed',
             'holdout25_validation_loss', 'rolling3_validation_loss', 'penalties', 'population_bias',
             'population_complexity', 'choices', 'selected_coefficients', 'training_sha256', 'coefficients_sha256',
             'linear_benchmark', 'equal_benchmark', 'normal_equation_max_error')}
    saved['selected_independent_payoffs'] = []
    saved['selected_future_payoffs'] = []
    mean, second = c['population_mean'], c['population_second']
    for T in sizes:
        theoretical = np.asarray([.25, 1., 4.])*s_ref*T**(-exponent)
        fit = fit_training(trajectory[:T], lam, theoretical)
        popcoef, popC, bias, floor = population_ridge(mean, second, fit['penalties'], p.q_star)
        evaluation = evaluate_fit(fit, test, trajectory[T:T+d.oos_periods], mean, second,
                                  1-p.q_star, popcoef, bias)
        for name, value in evaluation.items():
            saved[name].append(value)
        choices = [fit['holdout25_choice'], fit['rolling3_choice'], *range(len(lam), len(lam)+3)]
        a = fit['coefficients']
        # Independent normal-equation residual on actual training managed returns.
        residual = trajectory[:T].T@(trajectory[:T]@a)/T+a*fit['penalties']-trajectory[:T].mean(axis=0)[:, None]
        error = float(np.max(np.abs(residual)))
        require(error < 1e-8, 'Fitted coefficients violate the response-one normal equation')
        for name in ('holdout25_validation_loss', 'rolling3_validation_loss', 'penalties', 'training_sha256', 'coefficients_sha256'):
            saved[name].append(fit[name])
        saved['population_bias'].append(bias)
        saved['population_complexity'].append(popC)
        saved['choices'].append(choices)
        saved['selected_coefficients'].append(a[:, choices])
        saved['selected_independent_payoffs'].append(test@a[:, choices])
        saved['selected_future_payoffs'].append(trajectory[T:T+d.oos_periods]@a[:, choices])
        saved['normal_equation_max_error'].append(error)
        affine_fit, affine_c, j, _, _ = validation_select(affine[:T], lam)
        aloss, asr = sample_metrics(affine_test@affine_fit[:, j])
        floss, fsr = sample_metrics(affine[T:T+d.oos_periods]@affine_fit[:, j])
        ploss, psr = moment_metrics(affine_fit[:, j], c['affine_population_mean'], c['affine_population_second'])
        saved['linear_benchmark'].append([aloss, asr, floss, fsr, affine_c[j], lam[j], ploss, psr, ploss-(1-p.q_star)])
        eloss, esr = sample_metrics(equal_test)
        efloss, efsr = sample_metrics(equal[T:T+d.oos_periods])
        saved['equal_benchmark'].append([eloss, esr, efloss, efsr])
    npz_write(destination, **{key: np.asarray(value) for key, value in saved.items()},
              T=np.asarray(sizes), replication=index, train_seed=np.uint64(train_seed), test_seed=np.uint64(test_seed),
              seed_entropy=np.asarray([d.master_seed, index]), run_hash=c['run_hash'], protocol_hash=d.protocol_hash,
              runtime_seconds=time.perf_counter()-start, s_ref=s_ref, subspace_floor=floor,
              loading_map=p.loading_map, eta=p.eta,
              oos_labels=np.asarray(['independent_stationary_test', 'contiguous_future_test']))
    return str(destination)
