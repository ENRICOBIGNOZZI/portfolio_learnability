"""Numerical checks for the new experimental design."""

import numpy as np
import pytest

from Kernels.kernel_function import PortfolioKernel, fit_lambda_grid
from lambda_scaling import paper_alpha, profile_learning_rate, scaled_lambda, spectral_fit


def test_scaling_reference_refit_and_equivalent_unnormalized_constant():
    lam0, n0, alpha = 2e-4, 120, .6
    assert scaled_lambda(lam0, n0, n0, alpha) == lam0
    months = np.array([180, 360, 732])
    np.testing.assert_allclose(scaled_lambda(lam0, months, n0, alpha),
                               (lam0 * n0**alpha) * months.astype(float)**(-alpha))
    assert scaled_lambda(lam0, 180, n0, alpha) < lam0
    assert np.all(np.diff(scaled_lambda(lam0, months, n0, alpha)) < 0)


def test_ridge_uses_average_loss_normalization():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(19, 7))
    lam = float(scaled_lambda(.1, len(matrix), 12, .6))
    betas, _ = fit_lambda_grid(matrix, [lam])
    direct = np.linalg.solve(matrix.T @ matrix / len(matrix) + lam * np.eye(7), matrix.mean(axis=0))
    np.testing.assert_allclose(betas[lam], direct, atol=1e-12)


def test_paper_exponent_and_distinct_sharpe_exponent():
    assert paper_alpha(2., 1.5) == .5
    assert 1.5 * paper_alpha(2., 1.5) == .75
    assert paper_alpha(np.inf, 1.5) == 1 / 1.5
    for b, r in [(1., 1.5), (2., 1.), (2., 2.1)]:
        with pytest.raises(ValueError):
            paper_alpha(b, r)


def test_rate_profile_recovers_synthetic_unknown_asymptote():
    sizes = np.array([36, 60, 90, 120, 180, 240, 360])
    truth = 3.2 - 1.7 * (sizes / 36)**(-.7)
    profile = profile_learning_rate(sizes, truth)
    winner = profile.loc[profile.sse.idxmin()]
    assert winner.p == pytest.approx(.7)
    assert winner.sharpe_limit == pytest.approx(3.2)
    assert winner.amplitude == pytest.approx(1.7)


def test_decreasing_sharpe_does_not_become_positive_learning_by_sign_flip():
    sizes = np.array([36, 60, 90, 120, 180, 240, 360])
    curve = 2. - .1 * np.log(sizes)
    profile = profile_learning_rate(sizes, curve)
    assert (profile.amplitude == 0).all()
    np.testing.assert_allclose(profile.sse, profile.sse.iloc[0])


def test_negative_empirical_lambda_exponent_is_preserved():
    assert scaled_lambda(.1, 240, 120, -.5) > .1


def test_rank_two_spectrum_has_no_estimable_tail():
    matrix = np.diag([2., 1.])
    spectrum, fits = spectral_fit(matrix)
    np.testing.assert_allclose(spectrum.eigenvalue, [2., .5])
    assert fits.empty
    assert "b" in fits.columns


def test_short_spectrum_fits_only_observed_ranks():
    ranks = np.arange(1, 9)
    spectrum, fits = spectral_fit(np.diag(ranks.astype(float)**(-1.5)))
    assert len(fits) == 2
    assert (fits.rank_end <= len(spectrum)).all()
    assert (fits.rank_end - fits.rank_start >= 2).all()
    np.testing.assert_allclose(fits.b, 3.)


def test_one_characteristic_ntk_has_only_two_independent_coordinates():
    x = np.array([[-.8], [-.2], [0.], [.3], [.9]])
    kernel = PortfolioKernel(kernel="ntk", n_random_features=1000)
    features = kernel.features(x)
    basis = np.column_stack([np.maximum(x[:, 0], 0), np.minimum(x[:, 0], 0)])
    coefficients, _, _, _ = np.linalg.lstsq(basis, features, rcond=None)
    np.testing.assert_allclose(basis @ coefficients, features, atol=1e-15)
    assert np.linalg.matrix_rank(features) == 2
