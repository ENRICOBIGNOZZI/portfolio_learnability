from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from simulations.config.design import Environment, design_for


@pytest.fixture(scope='module')
def confirmation_fixture(tmp_path_factory):
    from simulations.run import context_for
    from simulations.dgp.balanced import BalancedFactorDGP
    from simulations.diagnostics import path_integrity as observer
    from simulations.experiments.reporting import summarize_confirmation
    from simulations.diagnostics.reconstruct_confirmation import raw_check, verify_aggregates, verify_rates
    root = tmp_path_factory.mktemp('confirmation_report')
    env = Environment('fixture', 0, N=30, rank=8, replications=10)
    design = replace(design_for('smoke'), profile='confirmation_v2', master_seed=2026100801,
                     T=(12, 18, 24), extra_T=(30, 36, 48), extended_replications=10,
                     oos_periods=12, population_groups=64, bootstrap_replications=12,
                     penalties=(1e-7, 1e-5, .001), theory_scale=(('fixture', 1e-5),), protocol_hash='test')
    context, directory = context_for(design, env, root, 'fixture')
    original = BalancedFactorDGP.step
    try:
        observer.initialize_observed_worker(context)
        for r in range(env.replications):
            observer.observed_replication(r)
    finally:
        BalancedFactorDGP.step = original
        observer._AUDIT = None
    summarize_confirmation(context, directory)
    config = design.to_dict()
    raw, q = raw_check(directory, config, env.__dict__)
    verify_aggregates(directory, config, env.__dict__, raw, q)
    verify_rates(directory, config, env.__dict__, raw, q)
    return directory, config, env.__dict__


def test_new_metrics_and_bootstraps_are_independently_reconstructed(confirmation_fixture):
    directory, _, _ = confirmation_fixture
    methods = pd.read_csv(directory/'methods.csv')
    assert set(methods.method) == {'holdout25', 'rolling3', 'theory_0.25', 'theory_1', 'theory_4',
                                   'ensemble_plugin_loss_oracle', 'crossfit_loss_oracle', 'zero',
                                   'linear_validation', 'equal_weight'}


@pytest.mark.parametrize('filename,column', [('curves', 'forward_loss_mcse'),
                                           ('methods', 'population_sharpe_mean'),
                                           ('path_methods', 'lambda_index'),
                                           ('selection_diagnostics', 'upper_frequency'),
                                           ('rates', 'bootstrap_high'),
                                           ('oracle_optimism_bootstrap', 'bootstrap_high')])
def test_confirmation_report_mutations_fail(confirmation_fixture, tmp_path, filename, column):
    from simulations.diagnostics.reconstruct_confirmation import raw_check, verify_aggregates, verify_rates
    directory, config, env = confirmation_fixture
    target = tmp_path/'case'
    shutil.copytree(directory, target)
    p = target/(filename+'.csv')
    data = pd.read_csv(p)
    data.loc[0, column] += 1
    data.to_csv(p, index=False)
    raw, q = raw_check(target, config, env)
    with pytest.raises(ValueError):
        if filename in ('rates', 'oracle_optimism_bootstrap'):
            verify_rates(target, config, env, raw, q)
        else:
            verify_aggregates(target, config, env, raw, q)


def test_saved_payoff_mutation_fails_coefficient_evaluation_check(confirmation_fixture, tmp_path):
    from simulations.diagnostics.reconstruct_confirmation import raw_check
    directory, config, env = confirmation_fixture
    target = tmp_path/'case'
    shutil.copytree(directory, target)
    p = target/'replications/0000.npz'
    with np.load(p, allow_pickle=False) as saved:
        v = dict(saved)
    v['selected_future_payoffs'][0, 0, 0] += 1
    np.savez_compressed(p, **v)
    with pytest.raises(ValueError, match='future payoff'):
        raw_check(target, config, env)
