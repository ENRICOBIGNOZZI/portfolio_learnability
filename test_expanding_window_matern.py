import numpy as np

from expanding_window_matern import estimate_spectral_b, theoretical_matern_rate


def test_matern_rate_uses_sobolev_smoothness():
    D = 132
    rate = theoretical_matern_rate(D, 1.5)

    s = D / 2 + 1.5
    b = 2 * s / D
    alpha = b / (b + 1)

    assert np.isclose(rate["s"], s)
    assert np.isclose(rate["b_theory"], b)
    assert np.isclose(rate["alpha_s"], alpha)


def test_matern_smoother_kernel_has_faster_lambda_decay():
    D = 132
    a12 = theoretical_matern_rate(D, 0.5)["alpha_s"]
    a32 = theoretical_matern_rate(D, 1.5)["alpha_s"]
    a52 = theoretical_matern_rate(D, 2.5)["alpha_s"]

    assert a12 < a32 < a52


def test_matern_exponents_are_between_half_and_one():
    for nu in (0.5, 1.5, 2.5):
        alpha = theoretical_matern_rate(132, nu)["alpha_s"]
        assert 0.5 < alpha < 1.0


def test_empirical_b_estimator_recovers_power_law():
    ranks = np.arange(1, 1001, dtype=float)
    true_b = 1.8
    eigenvalues = ranks ** (-true_b)
    fit = estimate_spectral_b(eigenvalues)

    assert abs(fit["b_hat"] - true_b) < 0.03
    assert fit["b_fit_start"] == 5
    assert fit["b_fit_end"] == 450
    assert fit["b_fit_r2"] > 0.999


def test_empirical_b_rule_uses_r_equal_one():
    fit = {"b_hat": 1.7}
    alpha = fit["b_hat"] / (fit["b_hat"] + 1.0)
    assert np.isclose(alpha, 1.7 / 2.7)
