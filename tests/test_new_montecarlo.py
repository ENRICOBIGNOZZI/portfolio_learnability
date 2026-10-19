"""Numerical and economic identities for the replacement DGP."""
import numpy as np
import pytest

from simulations.model import (calibrate, characteristics, cosine, evaluate_policies,
                               factors, fit_path, gaussian, matern_rff, moments,
                               phi, psi, psi_direct, returns)


def test_covariance_identity_and_oracle_conditional_moments():
    cal = calibrate(12)
    Z = characteristics(5, 12, .95, np.random.default_rng(81))
    _, B, mu, h = moments(Z, cal)
    for t in range(len(Z)):
        second = (np.outer(mu[t], mu[t]) + B[t] @ cal["Sigma_factor"] @ B[t].T
                  + cal["sigma_eps"]**2 * np.eye(12) + np.outer(h[t], h[t]))
        np.testing.assert_allclose(second @ mu[t], cal["kappa_0"] * mu[t], rtol=2e-14)
        assert np.linalg.eigvalsh(second - np.outer(mu[t], mu[t])).min() > 0
        w = mu[t] / cal["kappa_0"]
        np.testing.assert_allclose(w @ second @ w, w @ mu[t], rtol=2e-14)


def test_fft_interpolation_and_accelerated_cosine_match_original():
    rng = np.random.default_rng(19)
    z = np.r_[rng.uniform(-1, 1, 2000), -1, 0, 1]
    np.testing.assert_allclose(psi(z), psi_direct(z), atol=2e-13, rtol=0)
    x = rng.normal(size=(30, 41)) * 100
    np.testing.assert_allclose(cosine(x), np.cos(x), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("shape", [(8, 19), (24, 9)])
def test_ridge_path_matches_direct_primal_and_effective_trace(shape):
    G = np.random.default_rng(17).normal(size=shape)
    theta, complexity, _ = fit_path(G, [1e-4, .1, 2.])
    S = G.T @ G / len(G)
    for j, lam in enumerate([1e-4, .1, 2.]):
        inverse = np.linalg.inv(S + lam * np.eye(G.shape[1]))
        np.testing.assert_allclose(theta[:, j], inverse @ G.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(complexity[j], np.trace(S @ inverse), atol=1e-10)


def test_conditional_evaluator_matches_repeated_innovation_simulation():
    rng = np.random.default_rng(500)
    cal = calibrate(20)
    omega, phase = matern_rff(M=16)
    state = rng.uniform(-1, 1, (1, 20, 3))
    theta = rng.normal(size=(16, 2)) * 1000
    evaluated = evaluate_policies(theta, state, cal, omega, phase)
    Z = np.repeat(state, 40000, axis=0)
    R, _, _ = returns(Z, cal, rng)
    weights = phi(state[0], omega, phase) @ theta / 20
    payoff = R @ weights
    # Absolute tolerance based on the independent innovation Monte Carlo SE.
    for j in range(2):
        mean_se = payoff[:, j].std() / np.sqrt(len(payoff))
        second_se = (payoff[:, j]**2).std() / np.sqrt(len(payoff))
        assert abs(evaluated["mean"][j] - payoff[:, j].mean()) < 5 * mean_se
        assert abs(evaluated["second"][j] - np.mean(payoff[:, j]**2)) < 5 * second_se


def test_factor_truncation_preserves_the_user_covariance():
    cal = calibrate(30)
    F = factors(100000, cal, np.random.default_rng(66))
    np.testing.assert_allclose(F.mean(axis=0), cal["lambda_factor"], atol=.012)
    np.testing.assert_allclose(np.cov(F.T), cal["Sigma_factor"], atol=.02)
    assert cal["kappa_0"] > cal["upper_bound"]
    # Same calibrated optimum for all requested N.
    for N in [300, 600, 1200]:
        np.testing.assert_allclose(calibrate(N)["p_star"], 3/7)


def test_reproducible_stationary_characteristics_and_argument_checks():
    a = characteristics(20000, 1, .5, np.random.default_rng(42))
    b = characteristics(20000, 1, .5, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean()) < .02
    assert abs(a.var() - 1/3) < .02
    with pytest.raises(ValueError):
        characteristics(10, 2, 1, np.random.default_rng(1))
    with pytest.raises(ValueError):
        fit_path(np.ones((2, 3)), [0])


def test_rate_fit_identifies_known_power_and_separates_a_floor():
    from simulations.report import rate_fit
    x = np.array([100, 200, 400, 800, 1600])
    values = np.repeat((9 * x**(-2/3))[None, :], 20, axis=0)
    fit = rate_fit(x, values, -2/3, 20)
    assert fit["r_squared"] == pytest.approx(1)
    assert fit["exponent"] == pytest.approx(-2/3)
    shifted = rate_fit(x, values + .1, -2/3, 20)
    assert shifted["floor"] == pytest.approx(.1)
    assert shifted["floor_r_squared"] == pytest.approx(1)
    assert shifted["r_squared"] < 1
    assert shifted["all_points_used"]
