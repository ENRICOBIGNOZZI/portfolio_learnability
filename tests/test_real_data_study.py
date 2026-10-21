"""Numerical and timing contracts; fixtures are tests, never empirical results."""
from collections import deque

import numpy as np
import pandas as pd

from empirical_final.real_data_core import (fit_path, chronological_fit, match_groups,
    nested_universe, past_stock_betas, projected_statistics)
from empirical_final.real_data_study import months, definition
from empirical_final.memory_guard import run as guarded_run
import sys
import json


def test_ridge_matches_direct_normal_equations():
    g = np.arange(70, dtype=float).reshape(14, 5)/83
    g[:, 2] = np.sin(g[:, 2]*7)
    result = fit_path(g, [1e-2, 1.])
    for j, penalty in enumerate(result['penalty']):
        direct = np.linalg.solve(g.T@g/len(g)+penalty*np.eye(5), g.mean(axis=0))
        np.testing.assert_allclose(result['beta'][:, j], direct, atol=1e-10)


def test_complexity_and_pcr_are_economic_spectral_filters():
    g = np.diag([5., 3., 2., 1.])
    fit = fit_path(g, [1., 2., 4.], 'complexity')
    np.testing.assert_allclose(fit['complexity'][:2], [1., 2.], atol=1e-10)
    assert fit['valid'].tolist() == [True, True, False]
    pcr = fit_path(g, [.5, 1.], 'pcr')
    np.testing.assert_allclose(g@pcr['beta'], [[1., 1.], [1., 1.], [0., 1.], [0., 1.]], atol=1e-12)


def test_test_payoffs_cannot_change_tuning_or_coefficients():
    x = np.arange(480, dtype=float).reshape(120, 4)/100
    g = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 1]), np.sin(x[:, 2]*3)])
    first = chronological_fit(g, 96, 60, 12, [.001, .1, 10.])
    g[96:] = 1e9
    second = chronological_fit(g, 96, 60, 12, [.001, .1, 10.])
    np.testing.assert_array_equal(first['beta'], second['beta'])
    assert first['choice'] == second['choice']
    assert len(first['refit']) == 60 and first['validation'].max() == 95


def test_group_match_obeys_fixed_sharpe_tolerance():
    rows = [dict(variant=name, validation_sharpe=sr, top_decile_mass=m, top_decile_signal_share=a)
            for name, sr, m, a in [('a', 1., .2, .1), ('b', 1.1, .9, .8), ('c', 4., .99, .99)]]
    assert match_groups(rows) == ('a', 'b')
    assert match_groups(rows, .01) is None


def test_universe_is_nested_and_invariant_to_row_order():
    ids = np.arange(100, 170)
    small = ids[nested_universe(ids, 10, 42)]
    large = ids[nested_universe(ids, 30, 42)]
    backwards = ids[::-1]
    np.testing.assert_array_equal(small, backwards[nested_universe(backwards, 10, 42)])
    assert set(small) <= set(large)


def test_neutralization_preserves_managed_payoffs_and_missing_basis():
    x = np.column_stack([np.ones(8), np.linspace(-.5, .5, 8), np.arange(8)**2])
    b = np.column_stack([np.ones(8), np.cos(np.arange(8))])
    known = np.linspace(-.1, .2, 8)
    missing = np.zeros(8)
    missing[[1, 5]] = 1
    known[missing == 1] = 0
    observed, unknown_basis = projected_statistics(x, known, missing, b)
    beta = np.array([.2, .3, -.1])
    weights = (x@beta-b@(np.linalg.pinv(b)@(x@beta)))/len(x)
    np.testing.assert_allclose(b.T@weights, 0., atol=1e-13)
    for imposed in [-1., 0., 1.]:
        np.testing.assert_allclose((observed+imposed*unknown_basis)@beta,
                                   weights@(known+imposed*missing), atol=1e-12)


def test_past_betas_use_observed_history_and_fixed_new_listing_prior():
    ids = np.array([10, 11, 12])
    f = np.linspace(-.03, .04, 24)[:, None]
    history = [(np.array([10, 11]), np.array([2*f[t, 0], np.nan])) for t in range(24)]
    betas = past_stock_betas(ids, history, f)
    assert 1 < betas[0, 0] < 2
    np.testing.assert_allclose(betas[1:], 1.)
    np.testing.assert_allclose(past_stock_betas(ids, [], np.empty((0, 1))), 1.)


def test_streaming_keeps_unknown_payoffs_and_all_formation_stocks(tmp_path):
    frame = pd.DataFrame(dict(id=[10, 11, 10, 11], eom=pd.to_datetime(['2000-01-31']*2+['2000-02-29']*2),
                              r=[.1, np.nan, -.1, .2], return_known=[True, False, True, True], z=[-.5, .5, -.5, .5]))
    frame.to_parquet(tmp_path/'jkp_2000.parquet', index=False)
    result = list(months(tmp_path, [{'name': 'jkp_2000.parquet'}], ['z']))
    assert len(result) == 2
    assert len(result[0][1]) == 2
    assert np.isnan(result[0][1].r.iloc[1])


def test_guard_propagates_failure_and_enforces_memory_limit(tmp_path):
    assert guarded_run([sys.executable, '-c', 'raise SystemExit(7)'],
                       tmp_path/'failure', minimum_available_mib=0, interval=.05) == 7
    assert guarded_run([sys.executable, '-c', 'import time; time.sleep(10)'],
                       tmp_path/'limited', limit_mib=.1, minimum_available_mib=0, interval=.05) == 75
    result = json.loads((tmp_path/'limited/memory_summary.json').read_text())
    assert result['stopped_reason'] == 'process_tree_rss_limit'
