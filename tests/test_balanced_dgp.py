from dataclasses import replace

import numpy as np
import pytest

from simulations.diagnostics.baseline import (
    AuditSettings, audit_dates, gaussian_fourth_ratio,
    gaussian_quadratic_log_mgf, population_operators,
)
from simulations.dgp.balanced import (
    BalancedFactorDGP, DGPParameters, balanced_triplets, beta,
    conditional_moments, kernel_gram, population_ranks, w_star,
)


@pytest.mark.parametrize('overrides', [
    {'N': 601}, {'N': 0}, {'N': 6.0}, {'N': True}, {'D': 5}, {'K_F': 4}, {'nu': 2.5},
    {'rho': [0.5]*5}, {'rho': [1.0]*6}, {'rho': [-1.0]*6}, {'rho': [np.nan]*6},
    {'mu_F': [0, 0, 0]}, {'mu_F': [1, 0, 0]}, {'mu_F': [np.nan, 0, 0]},
    {'sigma_eps': 0}, {'sigma_eps': -0.1}, {'c_beta': -1}, {'ell': np.inf},
])
def test_invalid_dgp_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        DGPParameters(**overrides)


def test_population_ranks_are_not_sample_ranks_and_phases_share_all_coordinates():
    state = np.array([[0, 1, -1, .2, -.2, 2], [.5, -.5, 0, 2, -2, 1.]])
    ranks = population_ranks(state)
    np.testing.assert_allclose(ranks[0, :3], [.5, .8413447460685429, .15865525393145707])
    np.testing.assert_array_equal(population_ranks(state[:1]), ranks[:1])
    z = balanced_triplets(ranks).reshape(2, 3, 6)
    for role in range(3):
        np.testing.assert_allclose(z[:, role], 2*((ranks + role/3) % 1)-1)
    extreme = balanced_triplets(population_ranks(np.array([[-100, 100, 0, 0, 0, 0]])))
    assert np.all(np.abs(extreme) < 1)


@pytest.mark.parametrize('n', [3, 30, 600])
def test_exact_balance_conditional_optimizer_and_pricing(n):
    p = DGPParameters(N=n, rho=(-.9, -.5, 0, .2, .7, .99), mu_F=(.2, -.1, .04))
    for date in BalancedFactorDGP(p, 24).simulate(3):
        b = date.loadings
        np.testing.assert_allclose(np.sum(b*b, axis=1), p.c_beta, atol=1e-12, rtol=0)
        np.testing.assert_allclose(b.T @ b / n, p.gamma_beta, atol=1e-12, rtol=0)
        # Stronger group-by-group identity, not just cancellation across groups.
        for group in b.reshape(p.G, 3, 3):
            np.testing.assert_allclose(group.T @ group, p.c_beta*np.eye(3), atol=1e-12, rtol=0)
        mean, covariance, second = conditional_moments(b, p)
        np.testing.assert_allclose(covariance + np.outer(mean, mean), second, atol=1e-15)
        optimum = np.linalg.solve(second, mean)
        np.testing.assert_allclose(w_star(date.z, p)/n, optimum, atol=1e-11, rtol=0)
        assert np.linalg.norm(second @ optimum-mean) < 1e-11
        np.testing.assert_allclose([optimum @ mean, optimum @ second @ optimum], p.q_star, atol=1e-12)
        trial = np.random.default_rng(4).normal(size=n)
        loss = lambda w: 1 - 2*w @ mean + w @ second @ w
        assert loss(optimum) < loss(trial)
        assert abs(trial @ (mean-second @ optimum)) < 1e-11
        np.testing.assert_allclose(date.returns, b @ date.factors + date.epsilon)


def test_deterministic_streams_stationary_initialization_and_timing():
    p = DGPParameters(N=30)
    left, right = BalancedFactorDGP(p, 13), BalancedFactorDGP(p, 13)
    stationary_seed = np.random.SeedSequence(13).spawn(3)[0]
    expected = np.random.default_rng(stationary_seed).normal(size=(p.G, 6))
    np.testing.assert_array_equal(left.state, expected)
    for t in range(4):
        before = left.state.copy()
        a, b = left.step(), right.step()
        assert a.t == b.t == t
        for field in ('state', 'z', 'loadings', 'factors', 'epsilon', 'returns'):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
            assert getattr(a, field).dtype == np.float64
        np.testing.assert_array_equal(a.state, before)
        np.testing.assert_array_equal(a.z, balanced_triplets(population_ranks(before)))
    # Consuming extra factor shocks cannot move the characteristic stream.
    left.factor_rng.normal(size=17)
    np.testing.assert_array_equal(left.step().z, right.step().z)


def test_kernel_normalization_and_scalar_policy():
    p = DGPParameters(N=12)
    z = BalancedFactorDGP(p).step().z
    gram = kernel_gram(z, p)
    np.testing.assert_array_equal(np.diag(gram), np.ones(p.N))
    assert np.linalg.eigvalsh(gram)[0] > 0
    np.testing.assert_allclose(w_star(z[0], p), beta(z[0], p) @ p.mu_F / (p.c_beta/3+p.sigma_eps**2/p.N))
    np.testing.assert_allclose(p.q_star, p.c_beta/3 * np.dot(p.mu_F, p.mu_F)/(p.c_beta/3+p.sigma_eps**2/p.N))


def test_quadratic_mgf_noncentral_diagonal_rotated_and_singular():
    eigen_a = np.array([0., .2, .7])
    var = np.array([.3, .8, 1.2])
    mean = np.array([.1, -.4, .7])
    scale = 4.0
    expected = np.sum(-.5*np.log(1-2*eigen_a*var/scale) + eigen_a*mean**2/(scale-2*eigen_a*var))
    rotation, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    actual = gaussian_quadratic_log_mgf(rotation @ np.diag(eigen_a) @ rotation.T,
                                      rotation @ mean, rotation @ np.diag(var) @ rotation.T, scale)
    np.testing.assert_allclose(actual['log_mgf'], expected, atol=1e-14)
    assert np.isinf(gaussian_quadratic_log_mgf(np.eye(2), np.zeros(2), np.eye(2), 1)['log_mgf'])
    with pytest.raises(ValueError, match='PSD'):
        gaussian_quadratic_log_mgf(-np.eye(2), np.zeros(2), np.eye(2), 1)


def test_fourth_moment_bound_includes_zero_mean_and_zero_policy():
    assert gaussian_fourth_ratio(0, 2) == 3
    assert gaussian_fourth_ratio(2, 0) == 1
    assert gaussian_fourth_ratio(0, 0) == 0
    for mean in (-4, -.1, 0, .1, 4):
        assert gaussian_fourth_ratio(mean, .5) <= 3


def test_managed_operator_matches_enumeration_of_economic_group_draws():
    # N=6 has G=2 independently drawn groups. Enumerate every ordered pair
    # from a small population quadrature to verify its cross-group term.
    p, groups = DGPParameters(N=6), 4
    kernel_values, managed, z = population_operators(p, groups, 9)
    gram = kernel_gram(z, p)
    values, vectors = np.linalg.eigh(gram)
    phi = np.sqrt(values)[:, None] * vectors.T
    b = beta(z, p)
    accumulated = np.zeros_like(managed)
    for g in range(groups):
        for h in range(groups):
            idx = np.r_[3*g:3*g+3, 3*h:3*h+3]
            exposures = phi[:, idx] @ b[idx] / p.N
            # Repeated characteristic groups still have independent stock eps.
            accumulated += exposures @ exposures.T + p.sigma_eps**2/p.N**2*(phi[:, idx] @ phi[:, idx].T)
    accumulated /= groups**2
    # Eigenvector signs may differ between LAPACK drivers; compare spectra.
    np.testing.assert_allclose(np.linalg.eigvalsh(managed), np.linalg.eigvalsh(accumulated), rtol=1e-11, atol=1e-16)
    np.testing.assert_allclose(kernel_values, values/len(z), atol=1e-15)
    lower, upper = p.sigma_eps**2/p.N, p.c_beta+p.sigma_eps**2/p.N
    eig = np.linalg.eigvalsh(managed / np.sqrt(kernel_values[:, None]*kernel_values[None, :]))
    assert eig.min() >= lower-1e-12
    assert eig.max() <= upper+1e-12


def test_conditional_audit_runs_and_detects_broken_policy(monkeypatch):
    p = DGPParameters(N=12)
    s = replace(AuditSettings(), date_pool=3, audited_dates=2)
    result = audit_dates(p, s)
    assert result['passed']
    assert len(result['dates']) == 2
    monkeypatch.setattr('simulations.diagnostics.baseline.w_star', lambda z, p: np.zeros(len(z)))
    assert not audit_dates(p, s)['passed']


def test_bad_predeclared_spectral_ranges_are_rejected():
    with pytest.raises(ValueError):
        AuditSettings(slope_first=1)
    with pytest.raises(ValueError):
        AuditSettings(slope_last=4000)
