import numpy as np
import pytest
from simulation_characteristic_factor import population, factor_path, path_estimates, stock_audit


def test_covariance_square_root_is_second_moment_not_covariance():
    mu,theta,mean,_=population(30,1.5)
    h=np.sqrt(mu)*theta
    root=np.diag(np.sqrt(mu))@(np.eye(len(mu))-np.outer(h,h)/(1+np.sqrt(1-h@h)))
    np.testing.assert_allclose(root@root.T+np.outer(mean,mean),np.diag(mu),atol=1e-18)
    np.testing.assert_allclose(mean,mu*theta)


@pytest.mark.parametrize("j",[23,53])
def test_dual_path_matches_primal_regret_and_chronological_validation(j):
    rng=np.random.default_rng(33);mu,theta,_,_=population(j,1.5)
    f=factor_path(rng,40,mu,theta);penalties=np.geomspace(1e-7,.01,15)
    regret,c,choice=path_estimates(f,penalties,mu,theta)
    losses=[]
    for k,penalty in enumerate(penalties):
        estimate=np.linalg.solve(f.T@f/len(f)+penalty*np.eye(j),f.mean(axis=0))
        expected=np.dot(mu,(estimate-theta)**2)
        np.testing.assert_allclose(regret[k],expected,rtol=1e-9)
        eig=np.linalg.eigvalsh(f.T@f/len(f))
        np.testing.assert_allclose(c[k],np.sum(eig/(eig+penalty)),rtol=1e-9)
        inner=f[:-13]
        beta=np.linalg.solve(inner.T@inner/len(inner)+penalty*np.eye(j),inner.mean(axis=0))
        losses.append(np.mean((1-f[-13:]@beta)**2))
    assert choice==np.argmin(losses)


@pytest.mark.parametrize('economy',['NL','R1'])
def test_stock_economy_recovers_managed_payoff(economy):
    audit=stock_audit(40,1.5,economy,months=3,n=100)
    assert audit['passed']
    assert audit['min_eigenvalue_V_F']>0
    assert all(x['stock_residual_std']>.05 for x in audit['months'])


def test_boundary_target_normalization_and_distinction_from_smooth():
    j=np.arange(1,2001,dtype=float)
    mu,theta,mean,scale=population(len(j),1.5,'R1')
    np.testing.assert_allclose(theta/scale,1/(np.sqrt(j)*np.log1p(j)))
    np.testing.assert_allclose(np.dot(mu,theta**2),.16)
    np.testing.assert_allclose(mean,mu*theta)
    _,smooth,_,smooth_scale=population(len(j),1.5,'NL')
    np.testing.assert_allclose(smooth/smooth_scale,1/j)
    assert theta[-1]/theta[0] > smooth[-1]/smooth[0]
    with pytest.raises(ValueError,match='Unknown economy'):
        population(10,1.5,'typo')
