import numpy as np
import simulation_performance as performance
from simulation_characteristic_factor import population,factor_path,path_estimates,SEED


def test_population_sharpe_scale_invariance_sign_and_risk_identity():
    mu,theta,_,_=population(40,1.5,'R1')
    estimate=np.column_stack([theta,2*theta,-theta])
    mean,second,loss,sharpe=performance.population_performance(estimate,mu,theta)
    np.testing.assert_allclose(loss,[.84,1.,1.48])
    np.testing.assert_allclose(sharpe,np.sqrt(12*.16/.84)*np.array([1,1,-1]))
    np.testing.assert_allclose(loss,1-2*mean+second)
    np.testing.assert_allclose(loss,.84+np.sum(mu[:,None]*(estimate-theta[:,None])**2,axis=0))


def test_recomputed_moments_use_same_fitted_paths(monkeypatch):
    monkeypatch.setattr(performance,'J',13)
    monkeypatch.setattr(performance,'T_GRID',(8,16))
    mu,theta,_,_=population(13,1.5,'R1')
    rng=np.random.default_rng(np.random.SeedSequence([SEED,150,13,2,3]))
    f=factor_path(rng,16,mu,theta);lam=np.geomspace(1e-7,.01,7)
    ref=[path_estimates(f[:t],lam,mu,theta) for t in [8,16]]
    risks=np.array([x[0] for x in ref]);complexity=np.array([x[1] for x in ref])
    r,means,seconds,error=performance.performance_replication((3,lam,risks,complexity))
    assert r==3 and error<1e-10
    np.testing.assert_allclose(1-2*means+seconds,.84+risks)
    assert np.all(seconds-means**2>0)
