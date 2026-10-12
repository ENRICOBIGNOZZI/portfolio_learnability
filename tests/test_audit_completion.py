import numpy as np
import pytest
from simulations.closeout.perturb_ridge import RidgeFactor
from simulations.estimator.ridge import ridge_path


@pytest.mark.parametrize('T,P',[(9,15),(15,15),(21,15)])
def test_reused_ridge_factor_and_perturbation_match_independent_direct_solves(T,P):
    rng=np.random.default_rng(778+T);x=rng.normal(size=(T,P))+.2
    delta=rng.normal(size=x.shape)*1e-9
    penalties=np.array([1e-8,.02])
    factor=RidgeFactor(x)
    a,C=factor.path(penalties);original,complexity=ridge_path(x,penalties)
    np.testing.assert_allclose(a,original,atol=1e-10)
    np.testing.assert_allclose(C,complexity,atol=1e-12)
    changed,receipt=factor.perturb(delta,penalties)
    for j,lam in enumerate(penalties):
        target=x+delta
        direct=target.T@np.linalg.solve(target@target.T/T+lam*np.eye(T),np.ones(T))/T
        np.testing.assert_allclose(target@changed[:,j],target@direct,atol=2e-7,rtol=2e-7)
    assert receipt['absolute_residual']<1e-8 and receipt['relative_residual']<1e-8


def test_compiled_managed_features_preserve_stock_prefixes_and_kernel():
    from simulations.closeout.managed_native import ManagedFeatures
    from simulations.extended.rank_followup import RawBasis
    rng=np.random.default_rng(121);z=rng.uniform(-1,1,(36,6));r=rng.normal(size=(36,3))
    b=RawBasis(64);got=ManagedFeatures(b.anchors,4)(z,r,[12,24,36])
    expected=np.array([r[:n].T@b.raw(z[:n])/n for n in [12,24,36]])
    np.testing.assert_allclose(got,expected,atol=2e-15,rtol=1e-12)


@pytest.mark.parametrize('M',[512,2048])
def test_fast_fourier_evaluation_matches_canonical_and_has_uniform_error_bound(M):
    from simulations.closeout.fourier_evaluation import psi,table,beta_fast
    from simulations.dgp.rough import normalized_psi
    from simulations.dgp.balanced import beta,balanced_triplets
    from simulations.extended.design import parameters
    from dataclasses import replace
    rng=np.random.default_rng(554)
    points=np.r_[rng.uniform(-2,2,1000),[-2.,-1.,0.,1.,2.],np.linspace(-1e-10,1e-10,100)]
    np.testing.assert_allclose(psi(points,M),normalized_psi(points,M),atol=4e-15,rtol=4e-15)
    assert table(M)[2]<1e-15
    z=balanced_triplets(rng.uniform(.01,.99,(20,6)))
    np.testing.assert_allclose(beta_fast(z,parameters(),M),beta(z,replace(parameters(),fourier_terms=M)),atol=1e-15,rtol=1e-13)
