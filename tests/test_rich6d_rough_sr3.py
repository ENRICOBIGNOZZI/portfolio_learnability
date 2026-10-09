"""SR=3 is an economic recalibration followed by explicit monthly reporting."""
import numpy as np

from simulations.dgp.balanced import BalancedFactorDGP, DGPParameters, w_star
from simulations.diagnostics.low_memory import population_moments_batched
from simulations.estimator.kernel import NystromBasis
from simulations.run_rough_sr3 import calibrated_parameters


def test_monthly_optimum_and_conditional_portfolio_moments():
    p, scale = calibrated_parameters()
    np.testing.assert_allclose(p.mu_F, [0.5490699705067791,0.32944198230406746,0.16472099115203373],rtol=1e-14)
    np.testing.assert_allclose(p.q_star,3/7,rtol=1e-14)
    date = BalancedFactorDGP(p,81).step()
    weights = w_star(date.z,p)/p.N
    exposure = weights @ date.loadings
    mean = exposure @ np.asarray(p.mu_F)
    variance = exposure @ p.factor_covariance @ exposure + p.sigma_eps**2*(weights@weights)
    np.testing.assert_allclose(np.sqrt(12)*mean/np.sqrt(variance),3.,rtol=1e-13)
    assert np.linalg.eigvalsh(p.factor_covariance).min() > 0


def test_returns_are_regenerated_not_rescaled():
    base = DGPParameters(loading_map='rich6d_rough',eta=.35)
    p, scale = calibrated_parameters()
    old = BalancedFactorDGP(base,199)
    new = BalancedFactorDGP(p,199)
    for _ in range(3):
        a,b = old.step(),new.step()
        np.testing.assert_array_equal(a.z,b.z)
        np.testing.assert_array_equal(a.loadings,b.loadings)
        np.testing.assert_array_equal(a.epsilon,b.epsilon)
        assert not np.allclose(b.returns,scale*a.returns)
        assert not np.allclose(b.factors,scale*a.factors)
        np.testing.assert_allclose(w_star(b.z,p),scale*w_star(a.z,base),rtol=1e-13,atol=2e-14)


def test_population_second_moment_invariant_but_mean_recalibrated():
    p, scale = calibrated_parameters()
    base = DGPParameters(loading_map='rich6d_rough',eta=.35)
    basis = NystromBasis(rank=16,seed=91)
    m0,S0 = population_moments_batched(base,basis,257,28)
    m1,S1 = population_moments_batched(p,basis,257,28)
    np.testing.assert_array_equal(S1,S0)
    np.testing.assert_allclose(m1,scale*m0,atol=1e-16,rtol=1e-13)
