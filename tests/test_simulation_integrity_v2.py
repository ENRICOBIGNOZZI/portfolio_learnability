"""Adversarial checks against stale caches and corrupted reported results."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from simulations.config.design import Environment, design_for
from simulations.provenance import (ROOT, digest, identity, json_write, output_lock,
                                    source_hashes)


def test_dependency_mutations_cover_orchestration_kernel_threshold_and_audits(tmp_path):
    names = ('kernels.py', 'simulations/run.py', 'simulations/provenance.py',
             'simulations/config/design.py', 'simulations/diagnostics/baseline.py',
             'simulations/diagnostics/path_integrity.py', 'simulations/plotting/figures.py')
    for name in names:
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('version = 1\n')
    initial = source_hashes(root=tmp_path)
    rendering = source_hashes('render', root=tmp_path)
    for name in names[:-1]:
        (tmp_path/name).write_text('version = 2\n')
        assert source_hashes(root=tmp_path) != initial, name
        (tmp_path/name).write_text('version = 1\n')
    (tmp_path/names[-1]).write_text('version = 2\n')
    assert source_hashes(root=tmp_path) == initial
    assert source_hashes('render', root=tmp_path) != rendering


def test_actual_kernel_and_threshold_invalidate_but_workers_do_not():
    d = design_for('smoke')
    e = d.cases[0]
    kernel = {'nu': e.nu, 'ell': 1., 'rank': e.rank, 'seed': d.basis_seed}
    key = identity(d, e, kernel, sources={'fixed': 'fixture'})
    assert identity(replace(d, workers=9), e, kernel, sources={'fixed': 'fixture'}) == key
    assert identity(d, e, {**kernel, 'ell': 2.}, sources={'fixed': 'fixture'}) != key
    assert identity(replace(d, approximation_relative_tolerance=.05), e, kernel, sources={'fixed': 'fixture'}) != key
    assert identity(replace(d, population_seed=3), e, kernel, sources={'fixed': 'fixture'}) != key


def test_cache_rejects_changed_context_and_preserves_previous_bytes(tmp_path):
    from simulations.run import context_for
    d = replace(design_for('smoke'), population_groups=32)
    e = Environment('test', 0, N=30, rank=8, replications=4)
    c, destination = context_for(d, e, tmp_path, 'original-source')
    before = (destination/'population.npz').read_bytes()
    again, _ = context_for(replace(d, workers=8), e, tmp_path, 'original-source')
    assert c['run_hash'] == again['run_hash']
    with pytest.raises(ValueError, match='Population cache differs'):
        context_for(d, e, tmp_path, 'changed-context_for-source')
    assert (destination/'population.npz').read_bytes() == before


def test_atomic_failure_and_concurrent_output_lock(tmp_path):
    from simulations.provenance import atomic_file
    p = tmp_path/'value.json'
    json_write(p, {'valid': True})
    with pytest.raises(RuntimeError):
        with atomic_file(p, 'w') as handle:
            handle.write('partial')
            raise RuntimeError('interrupted write')
    assert json.loads(p.read_text()) == {'valid': True}
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another process'):
            with output_lock(tmp_path):
                pass
    with pytest.raises(ValueError, match='Historical V1'):
        with output_lock(ROOT/'simulations/outputs/paper'):
            pass


def test_production_gate_is_active_under_optimized_python():
    r = subprocess.run([sys.executable, '-O', '-c',
                        'from simulations.provenance import require; require(False, "mutation rejected")'],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'mutation rejected' in r.stderr


@pytest.fixture(scope='module')
def aggregate_fixture(tmp_path_factory):
    from simulations.run import context_for
    from simulations.experiments.reporting import summarize
    from simulations.diagnostics import path_integrity as observer
    from simulations.dgp.balanced import BalancedFactorDGP
    from simulations.diagnostics.reconstruct import verify_environment
    root = tmp_path_factory.mktemp('actual_simulated_fixture')
    d = replace(design_for('smoke'), T=(12, 18, 24), oos_periods=12,
                population_groups=64, bootstrap_replications=12,
                penalties=(1e-7, 1e-5, .001))
    e = Environment('fixture', 0, N=30, rank=8, replications=6)
    c, directory = context_for(d, e, root, 'fixture-source')
    original = BalancedFactorDGP.step
    try:
        observer.initialize_observed_worker(c)
        for r in range(e.replications):
            observer.observed_replication(r)
    finally:
        BalancedFactorDGP.step = original
        observer._AUDIT = None
    summarize(c, directory)
    verify_environment(directory, d.to_dict(), e.__dict__)
    return directory, d.to_dict(), e.__dict__


@pytest.mark.parametrize('filename,column', [
    ('curves', 'oos_loss_mean'), ('curves', 'oos_loss_mcse'),
    ('curves', 'population_sharpe_median'), ('curves', 'regret_ci_high'),
    ('selections', 'lambda'), ('selections', 'pathwise_oracle_lambda'),
    ('benchmarks', 'oos_sharpe'), ('rates', 'OLS_slope'),
    ('rates', 'bootstrap_high'), ('main_results', 'selected_lambda_mcse')])
def test_independent_reconstruction_rejects_aggregate_mutations(aggregate_fixture, tmp_path, filename, column):
    from simulations.diagnostics.reconstruct import verify_environment
    directory, d, e = aggregate_fixture
    target = tmp_path/'data'
    shutil.copytree(directory, target)
    p = target/(filename+'.csv')
    f = pd.read_csv(p)
    f.loc[0, column] += .123
    f.to_csv(p, index=False)
    with pytest.raises(ValueError):
        verify_environment(target, d, e)


@pytest.mark.parametrize('mutation', ['nan', 'inf', 'missing', 'extra', 'wrong_seed', 'shape', 'order'])
def test_invalid_raw_or_order_fails(aggregate_fixture, tmp_path, mutation):
    from simulations.diagnostics.reconstruct import verify_environment
    directory, d, e = aggregate_fixture
    target = tmp_path/'data'
    shutil.copytree(directory, target)
    p = target/'replications/0000.npz'
    if mutation == 'missing':
        p.unlink()
    elif mutation == 'extra':
        shutil.copyfile(p, target/'replications/9999.npz')
    elif mutation == 'order':
        table = target/'curves.csv'
        pd.read_csv(table).iloc[::-1].to_csv(table, index=False)
    else:
        with np.load(p, allow_pickle=False) as saved:
            values = dict(saved)
        if mutation in ('nan', 'inf'):
            values['oos_loss'][0, 0] = np.nan if mutation == 'nan' else np.inf
        elif mutation == 'wrong_seed':
            values['train_seed'] = np.uint64(5)
        else:
            values['complexity'] = values['complexity'][:, :1]
        np.savez_compressed(p, **values)
    with pytest.raises(ValueError):
        verify_environment(target, d, e)


def test_end_to_end_evaluation_mutations_cannot_change_fitting(tmp_path, monkeypatch):
    from simulations.run import context_for
    from simulations.experiments import monte_carlo as mc
    from simulations.diagnostics.confirmation_pilot import seed
    from simulations.dgp import balanced
    d = replace(design_for('smoke'), profile='confirmation_v2', T=(12, 18, 24),
                oos_periods=12, population_groups=64, theory_scale=(('fixture', 1e-5),),
                penalties=(1e-7, 1e-5, .001), protocol_hash='test-protocol')
    e = Environment('fixture', 0, N=30, rank=8, replications=1)
    context, directory = context_for(d, e, tmp_path/'original', 'fixture')
    mc.initialize_worker(context)
    original = mc.replication(0)
    with np.load(original, allow_pickle=False) as saved:
        before = dict(saved)
    anchors = context['basis'].anchors.copy()
    managed = mc.managed_path
    def changed_returns(p, basis, variant, economic_seed, periods):
        if economic_seed != seed('test', 0):
            return managed(p, basis, variant, economic_seed, periods)
        # Change actual test stock returns before kernel/affine aggregation.
        x, linear, equal = [], [], []
        for date in balanced.BalancedFactorDGP(p, economic_seed).simulate(periods):
            returns = -2*date.returns+.003
            x.append(basis.managed(date.z, returns))
            linear.append(np.r_[returns.mean(), date.z.T@returns/p.N])
            equal.append(returns.mean())
        return np.asarray(x), np.asarray(linear), np.asarray(equal)
    monkeypatch.setattr(mc, 'managed_path', changed_returns)
    monkeypatch.setattr(balanced, 'w_star', lambda z, p: np.full(len(z), 999.))
    altered = {**context, 'design': replace(d, oos_periods=19),
               'checkpoint_dir': str(tmp_path/'mutated'),
               'population_mean': context['population_mean']*.9,
               'population_second': context['population_second']*1.1,
               'oracle': {'intentionally_changed': True}, 'diagnostic_W_star': 'altered'}
    mc.initialize_worker(altered)
    with np.load(mc.replication(0), allow_pickle=False) as saved:
        after = dict(saved)
    np.testing.assert_array_equal(context['basis'].anchors, anchors)
    for key in ('training_sha256', 'coefficients_sha256', 'penalties', 'choices',
                'selected_coefficients', 'holdout25_validation_loss', 'rolling3_validation_loss'):
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.allclose(before['oos_loss'], after['oos_loss'])
    assert not np.allclose(before['population_loss'], after['population_loss'])
    assert not np.allclose(before['forward_loss'], after['forward_loss'])


def test_dual_ridge_has_the_correct_T_normalization():
    from simulations.estimator.ridge import ridge_path
    rng = np.random.default_rng(440)
    X = rng.normal(size=(12, 30))+.2
    penalties = np.asarray([1e-5, .01, .5])
    coefficients, C = ridge_path(X, penalties)
    for j, lam in enumerate(penalties):
        primal = np.linalg.solve(X.T@X+len(X)*lam*np.eye(X.shape[1]), X.sum(axis=0))
        dual = np.linalg.solve(X@X.T+len(X)*lam*np.eye(len(X)), np.ones(len(X)))
        np.testing.assert_allclose(coefficients[:, j], primal, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(coefficients[:, j], X.T@dual, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(C[j], np.trace(np.linalg.solve(X@X.T+len(X)*lam*np.eye(len(X)), X@X.T)), rtol=1e-10)


def test_actual_audit_cache_rejects_mutated_kernel_and_threshold(tmp_path, monkeypatch):
    from simulations import run
    from simulations.diagnostics import variants
    version = {'kernel': 'original', 'threshold': 'original'}
    calls = []
    monkeypatch.setattr(run, 'source_hashes', lambda *args: dict(version))
    monkeypatch.setattr(run, 'audit_dates', lambda *args: {'passed': True})
    monkeypatch.setattr(run, 'audit_monte_carlo', lambda *args: {'passed': True})
    def spectrum(*args, **kwargs):
        calls.append(dict(version))
        return {'passed': True}, [{'index': 1, 'eigenvalue': .1}]
    monkeypatch.setattr(run, 'audit_spectrum', spectrum)
    monkeypatch.setattr(variants, 'audit_variant', lambda *args: {'passed': True, 'states': list(range(24))})
    env = Environment('kernel_fixture', 0, N=30, nu=.5)
    run.environment_audit(env, tmp_path)
    run.environment_audit(env, tmp_path)
    assert len(calls) == 1
    version['kernel'] = 'changed actual quadrature kernel'
    run.environment_audit(env, tmp_path)
    assert len(calls) == 2
    version['threshold'] = 'changed audit threshold'
    run.environment_audit(env, tmp_path)
    assert len(calls) == 3


def test_actual_worker_count_does_not_change_scientific_checkpoint(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from simulations.run import context_for
    from simulations.diagnostics.path_integrity import initialize_observed_worker, observed_replication
    d = replace(design_for('smoke'), T=(12, 18, 24), oos_periods=12, population_groups=64,
                penalties=(1e-7, 1e-5, .001))
    e = Environment('workers_fixture', 0, N=30, rank=8, replications=2)
    results = []
    for workers in (1, 2):
        c, destination = context_for(replace(d, workers=workers), e, tmp_path/str(workers), 'fixture')
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_observed_worker, initargs=(c,)) as executor:
            files = list(executor.map(observed_replication, range(2)))
        records = []
        for file in files:
            with np.load(file, allow_pickle=False) as saved:
                records.append(dict(saved))
        results.append(records)
    for left, right in zip(*results):
        for key in left:
            if key not in ('runtime_seconds', 'integrity_execution_seconds'):
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)
