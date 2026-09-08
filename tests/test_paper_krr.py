import numpy as np
import pandas as pd

from Kernels.kernel_function import PortfolioKernel
from Utils.utils import (
    _evaluate_oos_panels,
    _response_one_loss,
    _ridge_path_from_managed_payoffs,
)


def test_response_one_dual_equals_primal_solution():
    rng = np.random.default_rng(7)
    managed_payoffs = rng.normal(size=(12, 5))
    lambdas = np.array([1e-4, 0.03, 2.0])

    path = _ridge_path_from_managed_payoffs(
        managed_payoffs,
        lambdas=lambdas,
    )
    rhs = managed_payoffs.T @ np.ones(len(managed_payoffs))
    expected = np.column_stack(
        [
            np.linalg.solve(
                managed_payoffs.T @ managed_payoffs
                + len(managed_payoffs) * lambda_ * np.eye(managed_payoffs.shape[1]),
                rhs,
            )
            for lambda_ in lambdas
        ]
    )

    np.testing.assert_allclose(path["coefficients"], expected, rtol=1e-9, atol=1e-10)


def test_effective_dimension_uses_managed_payoff_spectrum():
    rng = np.random.default_rng(11)
    managed_payoffs = rng.normal(size=(10, 20))
    path = _ridge_path_from_managed_payoffs(
        managed_payoffs,
        rho_grid=np.logspace(-8, 3, 17),
    )
    complexity = path["effective_dimensions"]

    assert np.all(np.diff(complexity) <= 1e-12)
    assert np.all(complexity >= 0.0)
    assert np.all(complexity <= min(managed_payoffs.shape) + 1e-10)


def test_matern32_random_features_approximate_exact_kernel():
    x = np.array([[0.0, 0.0], [0.5, -0.25], [1.0, 1.0]])
    kernel = PortfolioKernel(
        kernel="matern32",
        ell=0.8,
        n_random_features=20000,
        random_state=4,
    )
    features = kernel.feature_map(x)
    approximate = features @ features.T
    exact = PortfolioKernel.matern32_gram(x, x, ell=0.8)

    np.testing.assert_allclose(approximate, exact, atol=0.03)


def test_response_one_loss_prefers_returns_near_one():
    assert _response_one_loss([0.9, 1.0, 1.1]) < _response_one_loss([0.0, 0.1, 0.2])


def test_monthly_managed_loading_has_no_division_by_number_of_stocks():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(9, 3))
    returns = rng.normal(size=9)
    kernel = PortfolioKernel(
        kernel="gaussian",
        ell=1.3,
        n_random_features=16,
        random_state=2,
        batch_size=4,
    )
    statistics = kernel.gaussian_rff_managed_statistics(x, returns)
    features = kernel.gaussian_rff_features(x)

    np.testing.assert_allclose(statistics["return_loading"], features.T @ returns)
    np.testing.assert_allclose(statistics["net_loading"], features.sum(axis=0))

    beta = rng.normal(size=features.shape[1])
    weights = features @ beta
    assert np.isclose(statistics["return_loading"] @ beta, returns @ weights)
    assert np.isclose(statistics["net_loading"] @ beta, weights.sum())


def test_oos_records_save_free_net_and_gross_exposure():
    x = np.array([[-0.4], [0.1], [0.5]])
    returns = np.array([0.02, -0.01, 0.03])
    kernel = PortfolioKernel(
        kernel="gaussian",
        ell=0.4,
        n_random_features=8,
        random_state=3,
        batch_size=2,
    )
    statistics = kernel.gaussian_rff_managed_statistics(x, returns)
    coefficients = np.column_stack(
        [np.linspace(-0.2, 0.3, 8), np.linspace(0.5, -0.1, 8)]
    )
    panel = {
        "date": pd.Timestamp("2020-01-31"),
        "id": np.array([1, 2, 3]),
        "permno": np.array([10, 20, 30]),
        "x": x,
        "r": returns,
    }
    month = {
        "date": panel["date"],
        "n_assets": 3,
        "return_loading": statistics["return_loading"],
        "net_loading": statistics["net_loading"],
    }

    records, weights, _, _ = _evaluate_oos_panels(
        [panel],
        [month],
        kernel,
        coefficients,
        np.array([0.1, 1.0]),
        np.array([0.01, 0.1]),
        2020,
        selected_index=1,
        max_gross_exposure=0.5,
        save_weights=True,
    )
    raw_selected_weights = kernel.gaussian_rff_features(x) @ coefficients[:, 1]
    scale = min(1.0, 0.5 / np.abs(raw_selected_weights).sum())
    selected_weights = raw_selected_weights * scale
    selected = records[1]

    assert np.isclose(selected["sum_weights"], selected_weights.sum())
    assert np.isclose(selected["sum_absolute_weights"], np.abs(selected_weights).sum())
    assert np.isclose(selected["portfolio_return"], returns @ selected_weights)
    assert selected["sum_absolute_weights"] <= 0.5 + 1e-12
    assert np.isclose(selected["leverage_scale"], scale)
    assert np.isnan(selected["turnover"])
    assert len(weights) == len(x)
