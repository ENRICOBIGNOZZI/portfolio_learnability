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


def test_stock_economy_recovers_managed_payoff():
    audit=stock_audit(40,1.5,'NL',months=3,n=100)
    assert audit['passed']
    assert audit['min_eigenvalue_V_F']>0
    assert all(x['stock_residual_std']>.05 for x in audit['months'])
