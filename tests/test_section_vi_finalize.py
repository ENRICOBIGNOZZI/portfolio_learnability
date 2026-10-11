"""Independent checks of evaluated limits and historical information boundaries."""
import numpy as np
import pandas as pd
import pytest
from empirical_final.section_vi_finalize import ridge_evaluation
from bandwidth_tuning import select_bandwidth, refit_selected
from portfolio import annual_splits


@pytest.mark.parametrize('rank_deficient',[False,True])
def test_zero_endpoint_matches_feature_space_minimum_norm(rank_deficient):
    rng=np.random.default_rng(192)
    g=rng.normal(size=(18,40))
    if rank_deficient:
        g=rng.normal(size=(18,7))@rng.normal(size=(7,40))
    test=rng.normal(size=(12,40))
    penalties=np.array([1e2,1e-1,1e-5,1e-12,0.])
    returns,C,history,_,rank,_=ridge_evaluation(g@g.T,test@g.T,penalties)
    beta=np.linalg.lstsq(g,np.ones(len(g)),rcond=1e-6)[0]
    np.testing.assert_allclose(returns[:,-1],test@beta,atol=1e-11)
    np.testing.assert_allclose(history[:,-1],g@beta,atol=1e-11)
    assert rank==(7 if rank_deficient else 18)
    assert C[-1]==rank and (C<=rank).all() and (np.diff(C)>=0).all()
    np.testing.assert_allclose(returns[:,-2],returns[:,-1],atol=1e-9)


def test_positive_policies_are_actual_oos_predictions():
    rng=np.random.default_rng(713)
    g=rng.normal(size=(15,24));test=rng.normal(size=(12,24))
    penalties=np.array([10.,.3,1e-4,0.])
    returns,C,*_=ridge_evaluation(g@g.T,test@g.T,penalties)
    for j,penalty in enumerate(penalties[:-1]):
        beta=np.linalg.solve(g.T@g+len(g)*penalty*np.eye(g.shape[1]),g.T@np.ones(len(g)))
        np.testing.assert_allclose(returns[:,j],test@beta,rtol=1e-8,atol=1e-10)
    changed,*_=ridge_evaluation(g@g.T,(2*test)@g.T,penalties)
    np.testing.assert_allclose(changed,2*returns)


def test_joint_validation_and_refit_ignore_all_future_payoff_values():
    rng=np.random.default_rng(803)
    gs=[rng.normal(size=(744,25))*.1 for _ in range(3)]
    dates=pd.date_range('1963-01-31',periods=744,freq='ME')
    split=list(annual_splits(dates))[13];_,train,val,test=split
    grids=[np.array([1e-5,.01,1.]) for _ in gs];multipliers=[.5,1.,2.]
    grams=[g@g.T for g in gs]
    chosen,choices,losses=select_bandwidth(grams,grids,split,multipliers)
    perturbed=[g.copy() for g in gs]
    for g in perturbed:g[test[0]:]=rng.normal(size=g[test[0]:].shape)*100
    changed=[g@g.T for g in perturbed]
    after=select_bandwidth(changed,grids,split,multipliers)
    assert chosen==after[0] and choices==after[1]
    np.testing.assert_array_equal(losses,after[2])
    penalty=grids[chosen][choices[chosen]]
    _,C=refit_selected(grams[chosen],penalty,split)
    _,after_C=refit_selected(changed[chosen],penalty,split)
    assert C==after_C


def test_numerical_rank_cutoff_bounds_positive_and_zero_complexity():
    # Very weak numerical eigenvalues must not create spurious directions
    # or enormous pseudoinverse returns beyond the documented effective rank.
    K=np.diag([3.,1.,1e-16])
    returns,C,_,zero,rank,_=ridge_evaluation(K,np.eye(3),np.array([1e-20,0.]))
    assert rank==2 and (C<=2).all()
    np.testing.assert_allclose(zero,[1/3,1.,0.])
    assert np.isfinite(returns).all()
