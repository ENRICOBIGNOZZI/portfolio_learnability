"""Deterministic accounting/algebra checks; no simulated research outcomes."""
import numpy as np
import pytest
from empirical_final.core import accounting_step, execution_cost, incremental_loss, performance, spectral_policy


def test_entry_cost_solves_exact_post_cost_notional():
    cost,turnover,error=execution_cost([.8,-.4],[0.,0.],.0025)
    expected=.0025*1.2/(1+.0025*1.2)
    assert cost==pytest.approx(expected)
    assert turnover==pytest.approx((1-expected)*1.2)
    assert error<1e-12


def test_cash_counted_once_and_borrow_fee_uses_short():
    result=accounting_step([.8,-.4],[.1,-.05],.01,[.8,-.4],borrow_rate=.003)
    assert result['total_return']==pytest.approx(.01+.8*.1+.4*.05-.003*.4/12)
    assert result['short_notional']==pytest.approx(.4)
    assert result['excess_return']==pytest.approx(result['total_return']-.01)


def test_net_return_drag_reconciles_including_cross_term():
    result=accounting_step([.8,-.4],[.1,-.05],.01,[.7,-.5],.0025,.003)
    gross=.01+.8*.1+.4*.05
    assert gross-result['total_return']==pytest.approx(result['trading_return_drag']+result['borrowing_fee'])


def test_exits_charged_without_inventing_next_returns():
    result=accounting_step([1.,0.],[.03,np.nan],.01,[.6,.4],.0025)
    assert result['turnover']>.79
    assert result['next_drift'][1]==0
    with pytest.raises(ValueError):accounting_step([1.],[np.nan],.01,[0.])


def test_drawdown_includes_starting_capital():
    result=performance([-.1,.02],[-.1,.02])
    assert result['maximum_drawdown']==pytest.approx(-.1)


def test_nested_full_ridge_matches_independent_primal_solve():
    g=np.array([[2.,.1],[1.,.3],[.5,-.2],[.4,.6]])
    penalty=.02;spec=spectral_policy(g,penalty)
    expected=np.linalg.solve(g.T@g+len(g)*penalty*np.eye(2),g.T@np.ones(len(g)))
    np.testing.assert_allclose(spec['beta'][:,-1],expected,rtol=1e-12,atol=1e-12)
    assert spec['feature_operator_relative_residual']<1e-12
    assert spec['complexity']==pytest.approx(np.trace(np.linalg.solve(g.T@g/len(g)+penalty*np.eye(2),g.T@g/len(g))))


def test_equal_eigenvalue_cluster_is_not_split():
    spec=spectral_policy(np.eye(4),.1)
    assert spec['counts']==[4,4,4,4]


def test_incremental_loss_retains_cross_terms():
    previous=np.array([.3,.1,-.2]);added=np.array([.1,-.1,.2])
    change,cross,square=incremental_loss(previous,added)
    assert change==pytest.approx(cross+square)
    assert not np.isclose(change,square)
