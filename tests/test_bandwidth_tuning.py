"""Independent checks of RFF equivalence, joint selection, and temporal isolation."""
import numpy as np
import pandas as pd
import pytest

from bandwidth_tuning import managed_bandwidths, select_bandwidth, refit_selected
from kernels import FeatureBank
from portfolio import managed_matrix, ridge_path, complexity_grid, annual_splits, fit_windows


@pytest.mark.parametrize('kernel', ['gaussian', 'matern32'])
def test_shared_projections_equal_independent_feature_banks(kernel):
    rng = np.random.default_rng(67)
    panel = {'x': rng.normal(size=(270, 5)), 'r': rng.normal(size=270)}
    multipliers = [.25, .5, 1., 2., 4.]
    bank = FeatureBank(kernel, 5, 90, 1.7, 12)
    actual = managed_bandwidths(panel, bank, multipliers)
    for i, multiplier in enumerate(multipliers):
        independent = FeatureBank(kernel, 5, 90, 1.7*multiplier, 12)
        expected = managed_matrix([panel], independent)[0]
        np.testing.assert_allclose(actual[i], expected, atol=1e-14, rtol=1e-12)


def test_selection_matches_primal_and_ignores_test_and_future():
    rng = np.random.default_rng(39)
    gs = [rng.normal(size=(744, 18))*.15 + .015 for _ in range(3)]
    dates = pd.date_range('1963-01-31', periods=744, freq='ME')
    split = list(annual_splits(dates))[10]
    _, train, validation, test = split
    multipliers = [.5, 1., 2.]
    grids = [complexity_grid(g[:120], 12) for g in gs]
    grams = [g@g.T for g in gs]
    chosen, choices, losses = select_bandwidth(grams, grids, split, multipliers)
    expected_losses = []
    for g, penalties in zip(gs, grids):
        beta, _, _ = ridge_path(g[train], penalties)
        expected_losses.append(np.mean((1-g[validation]@beta)**2, axis=0))
    np.testing.assert_allclose(losses, expected_losses, atol=1e-12)
    assert losses[chosen][choices[chosen]] == np.min(losses)
    assert losses[chosen][choices[chosen]] <= losses[1][choices[1]]
    changed = [g.copy() for g in gs]
    for g in changed:
        g[test[0]:] = rng.normal(size=g[test[0]:].shape)*100
    after = select_bandwidth([g@g.T for g in changed], grids, split, multipliers)
    assert chosen == after[0] and choices == after[1]
    np.testing.assert_array_equal(losses, after[2])
    g, penalty = gs[chosen], grids[chosen][choices[chosen]]
    returns, complexity = refit_selected(grams[chosen], penalty, split)
    history = np.r_[train, validation]
    beta, _, expected_c = ridge_path(g[history], [penalty])
    np.testing.assert_allclose(returns, (g[test]@beta)[:, 0], atol=1e-12)
    assert complexity == pytest.approx(expected_c[0])


def test_single_bandwidth_reproduces_existing_pipeline():
    rng = np.random.default_rng(123)
    g = rng.normal(size=(744, 20))*.1 + .01
    dates = pd.date_range('1963-01-31', periods=744, freq='ME')
    split = next(annual_splits(dates))
    penalties = complexity_grid(g[split[1]], 12)
    original = next(fit_windows(g, dates, penalties))
    gram = g@g.T
    b, choices, losses = select_bandwidth([gram], [penalties], split, [1.])
    assert b == 0 and choices[0] == original['choice']
    np.testing.assert_array_equal(losses[0], original['validation_loss'])
    returns, complexity = refit_selected(gram, penalties[choices[0]], split)
    np.testing.assert_allclose(returns, original['test_returns'][:, original['choice']], atol=1e-12)
    assert complexity == pytest.approx(original['complexity'][original['choice']])


def test_exact_bandwidth_ties_prefer_median():
    rng = np.random.default_rng(8)
    g = rng.normal(size=(744, 8))
    gram = g@g.T
    split = next(annual_splits(pd.date_range('1963-01-31', periods=744, freq='ME')))
    chosen, _, _ = select_bandwidth([gram]*3, [np.array([.1, 1.])]*3, split, [.5, 1., 2.])
    assert chosen == 1
