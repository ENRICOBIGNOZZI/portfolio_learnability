import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular

from simulations.estimator.kernel import NystromBasis
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.extended.design import parameters, BASIS_SEED
from simulations.extended.population import raw_integrals, operators
from simulations.extended.rank_followup import RawBasis, evaluate
from simulations.extended.followup_diagnostics import oracle_diagnostic


def test_cholesky_coordinates_preserve_rkhs_ridge_and_population_metrics():
    p = parameters()
    basis = NystromBasis(rank=32, seed=BASIS_SEED)
    raw_basis = RawBasis(32)
    np.testing.assert_array_equal(raw_basis.anchors, basis.anchors)
    rng = np.random.default_rng(10)
    raw_history = rng.normal(size=(47, 32))
    penalties = np.array([.03, .7])
    symmetric, _ = ridge_path(raw_history@basis.inverse_root, penalties)
    L = cholesky(basis.gram, lower=True)
    alternate, _ = ridge_path(solve_triangular(L, raw_history.T, lower=True).T, penalties)
    c = solve_triangular(L.T, alternate)
    np.testing.assert_allclose(c, basis.inverse_root@symmetric, atol=1e-11)
    stats = raw_integrals(p, basis, 1024, 891)
    m, S = operators(p, basis, stats)
    loss, sr = moment_metrics(symmetric, m, S)
    streamed = evaluate(raw_basis, c, 1024, 891)
    np.testing.assert_allclose(streamed['regret'], loss-(1-p.q_star), atol=1e-12)
    np.testing.assert_allclose(streamed['annual_SR'], sr*np.sqrt(12), atol=1e-11)
    raw_S = stats['hh']/p.G+(1-1/p.G)*(stats['h_mean']@stats['h_mean'].T)+p.sigma_eps**2/p.N*stats['kk']
    raw_m = stats['h_mean']@np.asarray(p.mu_F)
    raw_optimum = cho_solve((cholesky(raw_S, lower=True), True), raw_m)
    np.testing.assert_allclose(raw_m@raw_optimum, m@np.linalg.solve(S, m), atol=1e-11)


def test_oracle_uses_common_grid_choice_and_disjoint_evaluation_indices():
    # The two halves prefer opposite penalties; in-sample max is optimistic.
    sr = np.array([[4., 0., 1.], [0., 3., 1.], [4., 0., 1.], [0., 3., 1.]])
    row, crossfit = oracle_diagnostic(sr, 5-sr)
    assert row['grid_SR_index'] == 0
    assert row['grid_max_mean_SR'] == 2
    assert row['theory_SR'] == 1
    np.testing.assert_array_equal(crossfit, np.zeros(4))
    assert row['crossfit_even_selected_index'] == 0
    assert row['crossfit_odd_selected_index'] == 1


def test_paired_bound_retains_pairing_instead_of_independent_rank_errors():
    from simulations.extended.followup_report import paired_bound
    low = np.array([[1., 3.], [2., 6.], [3., 9.]])
    high = low-np.array([.1, .2])
    mean, mcse, bound = paired_bound(low-high)
    np.testing.assert_allclose(mean, [.1, .2])
    np.testing.assert_allclose(mcse, 0, atol=1e-15)
    np.testing.assert_allclose(bound, [.1, .2])


def test_paired_slope_change_is_invariant_to_common_gap_scale():
    from simulations.extended.followup_report import paired_slope_change
    rng = np.random.default_rng(19)
    times = np.array([1440, 2160, 3240, 4860])
    low = rng.uniform(.9, 1.1, (12, 4))*times[None, :]**(-.6)
    result = paired_slope_change(times, low, 2*low)
    np.testing.assert_allclose(result['slope_change'], 0, atol=1e-14)
    np.testing.assert_allclose(result['paired_MCSE'], 0, atol=1e-14)
