from dataclasses import replace
import numpy as np
import pytest

from simulations.dgp.balanced import BalancedFactorDGP, DGPParameters, beta, w_star, w_star_gradient
from simulations.dgp.rough import normalization, normalized_psi, uniform_tail_bound


def test_fourier_direct_sum_periodicity_and_parseval():
    u = np.random.default_rng(431).uniform(-4,4,501)
    m = np.arange(2,129,dtype=float)
    direct = normalization()*np.sum(np.cos(np.pi*u[:,None]*m)/(m**5*np.log1p(m)),axis=1)
    np.testing.assert_allclose(normalized_psi(u),direct,atol=4e-14)
    np.testing.assert_allclose(normalized_psi(u+2),direct,atol=4e-14)
    grid = np.arange(8192)*2/8192
    np.testing.assert_allclose(np.mean(normalized_psi(grid)**2),.5,atol=1e-14)
    assert np.max(np.abs(normalized_psi(u,128)-normalized_psi(u,2048))) < uniform_tail_bound(128)


def test_rough_balanced_optimum_and_gaussian_innovations_preserved():
    p = DGPParameters(N=60,loading_map='rich6d_rough',eta=.35)
    old = BalancedFactorDGP(replace(p,loading_map='rich6d'),42)
    new = BalancedFactorDGP(p,42)
    for _ in range(5):
        date,original = new.step(),old.step()
        np.testing.assert_array_equal(date.z,original.z)
        np.testing.assert_array_equal(date.factors,original.factors)
        np.testing.assert_array_equal(date.epsilon,original.epsilon)
        B = date.loadings
        np.testing.assert_allclose(B.T@B/p.N,p.gamma_beta,atol=1e-16)
        optimum = np.linalg.solve(B@B.T+p.sigma_eps**2*np.eye(p.N),B@p.mu_F)
        np.testing.assert_allclose(w_star(date.z,p)/p.N,optimum,atol=1e-13)
        np.testing.assert_allclose((B@p.mu_F)@optimum,p.q_star,atol=1e-14)
    assert p.sr_star == replace(p,loading_map='rich6d').sr_star


def test_rough_gradient_all_coordinates():
    p = DGPParameters(loading_map='rich6d_rough',eta=.35)
    z = np.random.default_rng(678).uniform(-.9,.9,(71,6))
    g = w_star_gradient(z,p)
    for d in range(6):
        step = np.eye(6)[d]*1e-6
        fd = (w_star(z+step,p)-w_star(z-step,p))/2e-6
        np.testing.assert_allclose(g[:,d],fd,atol=2e-8,rtol=1e-5)
    assert np.all(np.mean(g**2,axis=0)>0)


@pytest.mark.parametrize('M',[True,1,2.5])
def test_invalid_truncation_rejected(M):
    with pytest.raises(ValueError):
        DGPParameters(loading_map='rich6d_rough',fourier_terms=M)
