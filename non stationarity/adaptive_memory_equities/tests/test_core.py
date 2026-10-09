import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize
from adaptive_memory_equities.calendar import ages, next_month, eligible
from adaptive_memory_equities.memory import theory_weights, taper, matched_horizon, Memory, uniform, grid
from adaptive_memory_equities.data_adapter import Panel, transform, History
from adaptive_memory_equities.networks import Policy
from adaptive_memory_equities.training import two_pass_gradient, fit, calibrate
from adaptive_memory_equities.portfolio import payoff, holdings, ensemble, PayoffCompletenessError
from adaptive_memory_equities.guard import select_guard, mature_records, block_indices
from adaptive_memory_equities.accounting import turnover, wealth, net_excess
from adaptive_memory_equities.metrics import summary, paired_comparisons


OPT = dict(steps=2,learning_rate=.01,weight_decay=0.,stock_block=3,batch_dates=0,threads=1)


def panels(n=6, dimension=3):
    rng = np.random.default_rng(41)
    dates = pd.date_range('2000-01-31',periods=n,freq='ME')
    return [Panel(d,np.arange(5+j%4),rng.normal(size=(5+j%4,dimension)),next_month(d),
                  rng.normal(.02,.08,5+j%4),next_month(d)) for j,d in enumerate(dates)]


@pytest.mark.parametrize('tau',[0,.0001,.1,1,10,100,1e8,1e16])
def test_simplex_kkt(tau):
    d = (np.arange(1,121)/120)**.5
    w,a,b = theory_weights(d,tau)
    assert np.all(w>=0)
    assert abs(w.sum()-1)<1e-12
    if tau == 0:
        np.testing.assert_array_equal(w,uniform(len(d)))
    if tau<=100:
        gradient = w + tau*(d@w)*d
        np.testing.assert_allclose(gradient[w>0],gradient[w>0][0],rtol=1e-8,atol=1e-10)
        assert np.all(gradient[w==0] >= gradient[w>0][0]-1e-10)
    assert np.isfinite(w).all()


@pytest.mark.parametrize('tau',[0,.01,1,100])
def test_generic_qp_parity(tau):
    d = np.random.default_rng(1).uniform(.01,1,20)
    w,_,_ = theory_weights(d,tau)
    solution = minimize(lambda x:x@x+tau*(d@x)**2,uniform(len(d)),
        jac=lambda x:2*x+2*tau*(d@x)*d, bounds=[(0,None)]*len(d),
        constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(len(x)))],
        method='SLSQP',options=dict(ftol=1e-13,maxiter=1000))
    assert solution.success
    np.testing.assert_allclose(w,solution.x,atol=2e-7)


@pytest.mark.parametrize('gamma',[.25,.5,1])
@pytest.mark.parametrize('tau',[.0001,.1,1,100])
def test_matched_weights(gamma,tau):
    age = np.arange(1,181)
    w,h = matched_horizon(age,180,tau,gamma)
    np.testing.assert_allclose(w,taper(age,h,gamma),atol=1e-13)
    if tau == .0001:
        assert h>180


def test_calendar_holes_and_availability():
    a,b = ages('2000-05-31',['2000-02-29','2000-05-31'],'2000-01-31')
    np.testing.assert_array_equal(a,[4,1])
    assert b==5
    p = panels(1)[0]
    assert p.return_realization_date == pd.Timestamp('2000-02-29')
    assert not eligible(p,'2000-01-31')
    p.available_at = pd.Timestamp('2000-04-30')
    assert not eligible(p,'2000-03-31')
    with pytest.raises(ValueError):
        ages('2000-01-31',['2000-02-29'],'2000-01-31')
    with pytest.raises(ValueError):
        taper([1,2],1,1)


def test_gradient_and_loss_sum_before_square():
    data = panels()
    omega = np.arange(1,len(data)+1,dtype=float)
    omega /= omega.sum()
    model = Policy(3,2,8,1)
    exact = copy.deepcopy(model)
    loss = two_pass_gradient(model,data,omega,block_size=2)
    payoffs = torch.stack([torch.dot(exact(torch.tensor(p.Z)),torch.tensor(p.forward_excess_returns))/len(p.Z) for p in data])
    reference = torch.dot(torch.tensor(omega),(1-payoffs)**2)
    reference.backward()
    assert abs(loss-reference.item())<1e-12
    for a,b in zip(model.parameters(),exact.parameters(),strict=True):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-10)
        assert a.grad.abs().sum()>0  # every hidden/output weight and bias receives gradient
    stock_loss = sum(w*np.mean((1-holdings(model,p.Z)[0]*p.forward_excess_returns)**2) for w,p in zip(omega,data))
    assert abs(stock_loss-loss)>1e-6


def test_permutation_variable_n_and_no_stockmonth_reweighting():
    p = panels(1)[0]
    model = Policy(3,2,8)
    score,w = holdings(model,p.Z)
    perm = np.arange(len(w))[::-1]
    wp = holdings(model,p.Z[perm])[1]
    assert payoff(w,p.forward_excess_returns) == pytest.approx(payoff(wp,p.forward_excess_returns[perm]))
    doubled = copy.deepcopy(p)
    doubled.Z = np.tile(p.Z,(2,1))
    doubled.forward_excess_returns = np.tile(p.forward_excess_returns,2)
    expected = (1-payoff(w,p.forward_excess_returns))**2
    assert two_pass_gradient(model,[p,doubled],np.array([.5,.5])) == pytest.approx(expected)


def test_scalar_training_only_and_matched_fit():
    data = panels(8)
    omega,h = matched_horizon(np.arange(1,9),8,1,1)
    model,a = fit(data,omega,dict(depth=2,width=8),0,OPT,'2000-09-30')
    same,b = fit(data,taper(np.arange(1,9),h,1),dict(depth=2,width=8),0,OPT,'2000-09-30')
    assert a['loss_after']<=a['loss_before']+1e-12
    assert a['weight_decay']==0 and a['steps']==2
    for x,y in zip(model.parameters(),same.parameters()):
        torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)
    with pytest.raises(ArithmeticError):
        calibrate(model,np.zeros(8),uniform(8))


def test_missing_payoffs_never_imputed_or_renormalized():
    with pytest.raises(PayoffCompletenessError):
        payoff([.1,.2],[.1,np.nan])
    assert payoff([.1,0],[.1,np.nan])==pytest.approx(.01)
    data = panels(1)
    data[0].forward_excess_returns[0]=np.nan
    with pytest.raises(PayoffCompletenessError):
        fit(data,uniform(1),dict(depth=2,width=8),0,OPT)


def test_future_returns_invariant_universe_preprocessing_fit_decision():
    from data_pipeline import FLAGS
    f = pd.DataFrame(dict(id=[1,2,3],eom=pd.Timestamp('2000-01-31'),excntry='USA',
        crsp_shrcd=10,crsp_exchcd=1,size_grp='large',ret_exc_lead1m=[.1,.2,.3],signal=[.2,.1,.4]))
    for flag in FLAGS:
        f[flag]=1
    a = transform(f,['signal'],'formation_audit')
    f.ret_exc_lead1m=[np.nan,300,-100]
    b = transform(f,['signal'],'formation_audit')
    pd.testing.assert_frame_equal(a[['id','signal']],b[['id','signal']])
    all_data = panels(10)
    changed = copy.deepcopy(all_data)
    cutoff = pd.Timestamp('2000-06-30')
    for p in changed:
        if not eligible(p,cutoff):
            p.forward_excess_returns[:]=999
    original = [p for p in all_data if eligible(p,cutoff)]
    altered = [p for p in changed if eligible(p,cutoff)]
    age,span = ages(cutoff,[p.return_realization_date for p in original],'2000-02-29')
    omega = Memory('theory',1).weights(age,span)
    m,_ = fit(original,omega,dict(depth=2,width=8),0,OPT,cutoff)
    n,_ = fit(altered,omega,dict(depth=2,width=8),0,OPT,cutoff)
    np.testing.assert_array_equal(holdings(m,all_data[5].Z)[1],holdings(n,changed[5].Z)[1])


def test_guard_fallback_and_maturity_and_simultaneous_advantage():
    c=dict(window=60,replicates=2000,block=12,seed=11)
    assert select_guard(np.zeros(59),np.zeros((59,3)),c)['index'] is None
    assert select_guard(np.zeros(60),np.zeros((60,3)),c)['index'] is None
    rng = np.random.default_rng(5)
    baseline = rng.normal(0,.02,60)
    good = rng.normal(.7,.02,(60,2))
    assert select_guard(baseline,good,c)['active']
    rec=[dict(decision_date='2000-01-31',return_date='2000-02-29',available_at='2000-03-31')]
    assert mature_records(rec,'2000-02-29')==[]
    assert mature_records(rec,'2000-03-31')==rec


def test_ensemble_costs_on_actual_aggregated_holdings():
    a,b = np.array([1.,-1.]),np.array([-1.,1.])
    w=ensemble([a,b])
    r=np.array([.1,.2])
    assert payoff(w,r)==pytest.approx((payoff(a,r)+payoff(b,r))/2)
    traded,_=turnover([1,2],w)
    assert traded==0
    assert net_excess(0,traded,0,25)==0


def test_cash_wealth_sharpe_and_drifted_turnover():
    r=np.array([.1,-.2,.3])
    rf=np.array([.01,.01,.01])
    v,dd,insolvent=wealth(r,rf)
    np.testing.assert_allclose(v,np.cumprod(1+r+rf))
    assert dd[1]==pytest.approx(-.19)
    assert insolvent is None
    assert summary(r)['sharpe']==pytest.approx(np.sqrt(12)*r.mean()/r.std(ddof=1))
    assert wealth([-1.1],[.01])[2]==0
    t,k=turnover([2,3],[.5,.1],[1,2],[.2,.4],[.1,.2],.1)
    assert t==pytest.approx(.2+abs(.5-.4*1.2/1.1)+.1)
    assert k=='drifted_self_financing_turnover'
    assert summary([0.,0.])['sharpe'] is None


def test_date_sampler_expectation_and_paired_bootstrap():
    data=panels(3);omega=np.array([.1,.2,.7]);model=Policy(3,2,8)
    two_pass_gradient(model,data,omega)
    reference=torch.cat([p.grad.flatten() for p in model.parameters()])
    expectation=torch.zeros_like(reference)
    # Enumerate the actual one-date stochastic gradient, including stock blocking.
    for j in range(3):
        coefficients=np.zeros(3);coefficients[j]=1
        two_pass_gradient(model,data,coefficients,block_size=2)
        expectation+=omega[j]*torch.cat([p.grad.flatten() for p in model.parameters()])
    torch.testing.assert_close(expectation,reference,atol=1e-12,rtol=1e-10)
    x=np.random.default_rng(9).normal(size=36)
    out=paired_comparisons(pd.DataFrame(dict(a=x,b=x)),'a',replicates=50)
    assert all(r['difference']==0 and r['paired_low']==0 and r['paired_high']==0 for r in out)


def test_tau_zero_grid_deduplicated():
    memories=grid(dict(theory=[[0,.25],[0,.5],[0,1],[1,1]],taper=[[240,1]]))
    assert sum(m.family=='uniform' for m in memories)==1
    assert len(memories)==3
