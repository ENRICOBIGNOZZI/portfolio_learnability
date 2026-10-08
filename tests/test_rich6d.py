from dataclasses import replace

import numpy as np

from simulations.dgp.balanced import (BalancedFactorDGP, DGPParameters, beta,
                                     w_star, w_star_gradient, balanced_triplets)


def test_eta_zero_reproduces_baseline_bytes_and_all_loading_coordinates():
    p = DGPParameters(N=30)
    z = np.random.default_rng(71).uniform(-.95, .95, (100, 6))
    angle = np.pi*(z[:, 0]+1)
    independent = np.sqrt(p.c_beta)*np.column_stack((np.sqrt(2/3)*np.cos(angle), np.sqrt(2/3)*np.sin(angle), np.full(len(z), 1/np.sqrt(3))))
    np.testing.assert_array_equal(beta(z, p), independent)
    np.testing.assert_array_equal(beta(z, replace(p, loading_map='rich6d', eta=0)), independent)
    first = BalancedFactorDGP(p, 22).step()
    second = BalancedFactorDGP(replace(p, loading_map='rich6d', eta=0), 22).step()
    np.testing.assert_array_equal(first.returns, second.returns)


def test_rich6d_group_balance_and_independent_dense_conditional_solve():
    p = DGPParameters(N=60, loading_map='rich6d', eta=.35)
    for date in BalancedFactorDGP(p, 729).simulate(8):
        u = (date.z+1)/2
        theta = 2*np.pi*u[:, 0]+.35*np.sin(2*np.pi*(u[:, 1:]-u[:, :1])).sum(axis=1)
        independent = np.sqrt(p.c_beta/3)*np.column_stack((np.sqrt(2)*np.cos(theta), np.sqrt(2)*np.sin(theta), np.ones(p.N)))
        np.testing.assert_allclose(independent, date.loadings, atol=1e-15)
        np.testing.assert_allclose(np.sum(independent**2, axis=1), p.c_beta, atol=1e-15)
        np.testing.assert_allclose(independent.T@independent/p.N, np.eye(3)*p.c_beta/3, atol=1e-15)
        S = independent@independent.T+p.sigma_eps**2*np.eye(p.N)
        m = independent@p.mu_F
        solved = np.linalg.solve(S, m)
        np.testing.assert_allclose(w_star(date.z, p)/p.N, solved, atol=1e-12)
        np.testing.assert_allclose(m@solved, p.q_star, atol=1e-13)
    assert p.q_star == replace(p, loading_map='baseline_original', eta=0).q_star
    assert p.sr_star == replace(p, loading_map='baseline_original', eta=0).sr_star


def test_rich6d_analytic_derivatives_and_nonzero_six_coordinate_dependence():
    p = DGPParameters(loading_map='rich6d', eta=.35)
    z = np.random.default_rng(888).uniform(-.9, .9, (1000, 6))
    gradient = w_star_gradient(z, p)
    for d in range(6):
        step = np.eye(6)[d]*1e-6
        fd = (w_star(z+step, p)-w_star(z-step, p))/2e-6
        np.testing.assert_allclose(gradient[:, d], fd, rtol=1e-5, atol=2e-8)
    assert np.all((gradient**2).mean(axis=0) > 0)
