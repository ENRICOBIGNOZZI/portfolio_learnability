"""Small numerical fixtures only; these are not empirical backtest results."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import kv, gamma
from data import (REFERENCE_SELECTION_END, FLAGS, characteristic_selection, formation_mask,
                  rank_months, sample_initial)
from model import (FeatureBank, annual_splits, complexity_grid, exact_kernel,
                   l1_target_turnover, managed_matrix, median_distance, ridge_path, sharpe)


def fixture_panel():
    frame = pd.DataFrame({'id':[1,2,3,1,2,3],
        'eom':pd.to_datetime(['1963-01-31']*3+['2024-01-31']*3),
        'excntry':'USA','crsp_shrcd':10,'crsp_exchcd':1,'size_grp':'large',
        'ret_exc_lead1m':[.1,.2,.3,.2,.1,.0],
        'a':[1,2,3,4,5,6],'b':[2,3,4,5,6,7],
        'c':[np.nan,np.nan,3,4,5,6]})
    for flag in FLAGS:
        frame[flag]=1
    return frame


@pytest.mark.parametrize('kernel',['gaussian','matern32'])
@pytest.mark.parametrize('dimension',[1,4,130])
@pytest.mark.parametrize('ell',[0.5,4.25])
def test_kernel_formula(kernel,dimension,ell):
    from scipy.spatial.distance import cdist
    x=np.random.default_rng(5).normal(size=(7,dimension))
    distance=cdist(x,x)/ell
    if kernel=='gaussian':
        target=np.exp(-distance**2/2)
    else:
        nu=1.5
        arg=np.sqrt(2*nu)*distance
        target=np.ones_like(arg)
        positive=arg>0
        target[positive]=2**(1-nu)/gamma(nu)*arg[positive]**nu*kv(nu,arg[positive])
    actual=exact_kernel(x,x,kernel,ell)
    np.testing.assert_allclose(actual,target,rtol=1e-12,atol=1e-12)
    assert np.linalg.eigvalsh(actual).min()>-1e-10
    np.testing.assert_allclose(np.diag(actual),1)


@pytest.mark.parametrize('kernel',['gaussian','matern32'])
def test_nested_prefix_and_batching(kernel):
    x=np.random.default_rng(1).normal(size=(20,5))
    small=FeatureBank(kernel,5,100,2.0,3)
    large=FeatureBank(kernel,5,1000,2.0,3)
    np.testing.assert_array_equal(small.frequencies,large.frequencies[:100])
    np.testing.assert_array_equal(small.phases,large.phases[:100])
    np.testing.assert_array_equal(small.features(x),large.features(x,100))
    np.testing.assert_allclose(large.features(x,100),
        large.features(x,1000)[:,:100]*np.sqrt(10),atol=1e-15)
    np.testing.assert_allclose(large.features(x),
        np.vstack([large.features(x[:7]),large.features(x[7:])]),atol=1e-14)


@pytest.mark.parametrize('kernel',['gaussian','matern32'])
def test_rff_targets_kernel(kernel):
    x=np.random.default_rng(2).normal(size=(18,4))
    bank=FeatureBank(kernel,4,60000,2.0,1)
    phi=bank.features(x)
    error=phi@phi.T-exact_kernel(x,x,kernel,2.0)
    assert np.sqrt(np.mean(error**2))<.015


@pytest.mark.parametrize('shape',[(20,7),(12,60)])
def test_ridge_primal_dual_and_complexity(shape):
    g=np.random.default_rng(3).normal(size=shape)
    penalties=np.array([.001,.1,2.])
    b,mu,c=ridge_path(g,penalties)
    for j,lam in enumerate(penalties):
        primal=np.linalg.solve(g.T@g/len(g)+lam*np.eye(g.shape[1]),g.mean(axis=0))
        dual=g.T@np.linalg.solve(g@g.T+len(g)*lam*np.eye(len(g)),np.ones(len(g)))
        np.testing.assert_allclose(b[:,j],primal,atol=1e-11)
        np.testing.assert_allclose(b[:,j],dual,atol=1e-11)
        hat=g@np.linalg.solve(g.T@g+len(g)*lam*np.eye(g.shape[1]),g.T)
        np.testing.assert_allclose(c[j],np.trace(hat),atol=1e-11)
    assert (np.diff(c)<0).all()
    assert ((c>=0)&(c<=min(shape))).all()


def test_penalty_grid_uses_initial_spectrum_only():
    g=np.random.default_rng(0).normal(size=(25,8))
    penalties=complexity_grid(g,120)
    _,mu,c=ridge_path(g,penalties)
    assert len(penalties)==120
    assert np.all(np.diff(penalties)>0)
    np.testing.assert_allclose(c[::-1],np.linspace(.25,.995*8,120),atol=1e-9)


@pytest.mark.parametrize('kernel',['linear','gaussian','matern32'])
def test_managed_return_is_actual_stock_return(kernel):
    rng=np.random.default_rng(0)
    x=rng.normal(size=(11,4));r=rng.normal(size=11)
    bank=FeatureBank(kernel,4,30,2.0,1)
    beta=rng.normal(size=5 if kernel=='linear' else 30)
    g=managed_matrix([{'x':x,'r':r}],bank)[0]
    weights=bank.features(x)@beta/len(r)
    np.testing.assert_allclose(g@beta,weights@r,atol=1e-14)


def test_2024_extension_does_not_change_reference_feature_set():
    original=fixture_panel()
    before,_=characteristic_selection(original,['a','b','c'],count=2)
    mutated=original.copy()
    mutated.loc[mutated.eom>REFERENCE_SELECTION_END,['a','b']]=np.nan
    after,_=characteristic_selection(mutated,['a','b','c'],count=2)
    assert before==after==['a','b']


def test_future_returns_never_change_formation_universe_or_ranks():
    original=fixture_panel()
    mutated=original.copy();mutated.loc[0,'ret_exc_lead1m']=np.nan
    np.testing.assert_array_equal(formation_mask(original),formation_mask(mutated))
    a=rank_months(original.loc[formation_mask(original)],['a','b'])
    b=rank_months(mutated.loc[formation_mask(mutated)],['a','b'])
    np.testing.assert_array_equal(a[['id','a','b']].to_numpy(),b[['id','a','b']].to_numpy())


def test_row_threshold_is_thirty_percent_not_one_third():
    names=[f'x{i}' for i in range(130)]
    frame=pd.DataFrame(np.ones((2,130)),columns=names)
    frame['id']=[1,2];frame['eom']=pd.Timestamp('1963-01-31')
    frame.loc[0,names[:39]]=np.nan
    frame.loc[1,names[:40]]=np.nan
    result=rank_months(frame,names)
    assert result.id.tolist()==[1]
    assert np.isfinite(result[names].to_numpy()).all()


def test_average_ranks_ties_and_neutral_imputation():
    frame=pd.DataFrame({'eom':pd.Timestamp('1963-01-31'),
                        'a':[0.,0.,0.,2.,np.nan], 'b':[1.,2.,3.,4.,5.],
                        'c':[1.,2.,3.,4.,5.], 'd':[1.,2.,3.,4.,5.]})
    result=rank_months(frame,['a','b','c','d'])
    np.testing.assert_allclose(result.a,[-1/6,-1/6,-1/6,.5,0.])
    # Zero is a chosen neutral point, not generally the empirical median.
    assert result.a.median()!=0


def test_initial_distance():
    sample=np.array([[0.,0.],[3.,0.],[0.,4.]])
    assert median_distance(sample)==4.0
    panels=[np.random.default_rng(i).normal(size=(20,3)) for i in range(120)]
    a=sample_initial(panels);b=sample_initial(panels)
    assert a.shape==(1000,3)
    np.testing.assert_array_equal(a,b)


def test_annual_calendar():
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    splits=list(annual_splits(dates))
    assert len(splits)==47
    assert len(splits[0][1])==120
    year,tr,va,te=splits[-1]
    assert year==2024 and len(tr)+len(va)==732
    assert dates[te[0]]+pd.offsets.MonthEnd(1)==pd.Timestamp('2024-02-29')
    assert dates[te[-1]]+pd.offsets.MonthEnd(1)==pd.Timestamp('2025-01-31')


def test_turnover_union_and_initial_entry():
    first,previous=l1_target_turnover([1,2],[.7,-.2])
    second,_=l1_target_turnover([2,3],[-.1,.5],previous)
    assert first==pytest.approx(.9)
    assert second==pytest.approx(1.3)


def test_missing_return_is_not_silently_zero_or_dropped():
    bank=FeatureBank('linear',2,2)
    with pytest.raises(ValueError,match='unresolved'):
        managed_matrix([{'x':np.ones((3,2)), 'r':np.array([.1,np.nan,.2])}],bank)
    with pytest.raises(ValueError,match='missing'):
        sharpe([.1,np.nan,.2])


def test_report_module_and_latex_generation(tmp_path):
    import figures
    data=[]
    for kernel in ('linear','gaussian','matern32'):
        data.append({'kernel':kernel,'annual_mean_excess':.1,'annual_volatility':.1,
            'sharpe':1.,'max_drawdown':.1,'mean_gross':1.8,'annual_target_turnover':12.,
            'net_sharpe_25bps_target_turnover_proxy':.7})
    figures.make_text(tmp_path,pd.DataFrame(data),[{'interior':False}],4.0)
    text=(tmp_path/'empirics.tex').read_text()
    assert '\\subsection' not in text
    assert 'do not establish an interior optimum' in text
    assert 'ELL' not in text and 'FINDING' not in text
    assert all((tmp_path/p).exists() for p in ['main.tex','empirics.tex','performance.tex','references.bib'])
