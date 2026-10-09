"""Cross-check reused numerical integrals and paired economic innovations."""
from dataclasses import replace
import numpy as np

from simulations.dgp.balanced import BalancedFactorDGP
from simulations.estimator.kernel import NystromBasis
from simulations.diagnostics.low_memory import population_moments_batched
from simulations.extended.design import parameters
from simulations.extended.population import raw_integrals, operators, spectral_path
from simulations.extended.histories import PairedHistory, raw_managed_history
from simulations.run_rough import seed


def test_paired_baseline_is_existing_simulation():
    legacy=BalancedFactorDGP(parameters(),seed('train',7))
    paired=PairedHistory(seed('train',7),7)
    for t in range(12):
        old=legacy.step()
        got=paired.step(robustness=t<8)
        np.testing.assert_array_equal(got['baseline'][0],old.z)
        np.testing.assert_array_equal(got['baseline'][1],old.returns)
        if t<8:
            np.testing.assert_array_equal(got['N300'][1],old.returns[:300])
            np.testing.assert_array_equal(got['N1200'][1][:600],old.returns)


def test_raw_integrals_match_canonical_population_for_every_N():
    basis=NystromBasis(rank=32,seed=4)
    stats=raw_integrals(parameters(),basis,127,27,batch_groups=19)
    for name in ('baseline','N300','N1200','rho000','rho075'):
        p=parameters(name)
        m,S,parts=operators(p,basis,stats,components=True)
        expected=population_moments_batched(p,basis,127,27,batch_groups=19)
        np.testing.assert_allclose(m,expected[0],rtol=2e-13,atol=1e-16)
        np.testing.assert_allclose(S,expected[1],rtol=2e-12,atol=1e-18)
        np.testing.assert_allclose(parts['factor']+parts['idiosyncratic'],S,atol=1e-18)


def test_nested_prefixes_match_separate_bases():
    large=NystromBasis(rank=64,seed=9)
    small=NystromBasis(rank=32,seed=9)
    a=raw_integrals(parameters(),large,99,29,batch_groups=17)
    b=raw_integrals(parameters(),small,99,29,batch_groups=17)
    for got,want in zip(operators(parameters(),small,a),operators(parameters(),small,b)):
        np.testing.assert_allclose(got,want,rtol=1e-11,atol=1e-16)


def test_population_elasticity_exact_limits_and_identity():
    values=np.geomspace(1e-8,1e-3,100)
    C,E,local=spectral_path(values,np.array([1e-13,1e-6,100.]))
    assert C[0]>99.99 and E[0]<1e-5
    assert E[-1]>.9999
    np.testing.assert_allclose(local,.6*E)


def test_raw_managed_baseline_matches_canonical():
    basis=NystromBasis(rank=16,seed=21)
    arrays,_=raw_managed_history(basis,seed('train',4),4,6,4)
    old=BalancedFactorDGP(parameters(),seed('train',4))
    reference=np.array([basis.managed(d.z,d.returns) for d in old.simulate(6)])
    np.testing.assert_allclose(arrays['baseline']@basis.inverse_root,reference,atol=1e-16)


def test_slope_uncertainty_respects_common_multiplicative_shock():
    from simulations.extended.statistics import gap_slopes
    T=np.array([60,120,360,720,1440,2160,3240,4860])
    rng=np.random.default_rng(91)
    gap=np.exp(rng.normal(size=(50,1)))*T[None,:]**(-.6)
    for row in gap_slopes(T,gap):
        if row['available']:
            np.testing.assert_allclose(row['slope'],-.6,atol=1e-14)
            assert row['MCSE_delta']<1e-8
            assert row['MCSE_jackknife']<1e-12


def test_population_rho_invariance_is_exact():
    basis=NystromBasis(rank=16,seed=13)
    stats=raw_integrals(parameters(),basis,97,57)
    m,S=operators(parameters(),basis,stats)
    for name in ('rho000','rho075'):
        m2,S2=operators(parameters(name),basis,stats)
        np.testing.assert_array_equal(m,m2)
        np.testing.assert_array_equal(S,S2)
        assert parameters(name).sr_star==parameters().sr_star
