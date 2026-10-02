"""Temporal separation, fixed C0, spectral estimation and nested-grid checks."""
import numpy as np
import pandas as pd
import pytest
from complexity_schedule import spectral_fit, nested_grid, calibrate, window, heatmap_values
from portfolio import annual_splits, complexity_grid, ridge_path


@pytest.fixture
def managed():
    rng=np.random.default_rng(318)
    g=(rng.normal(size=(744,60))+.1)*np.arange(1,61)[None,:]**-1.1
    return g,pd.date_range('1963-01-31','2024-12-31',freq='ME')


def test_spectral_power_law_and_domain():
    fit=spectral_fit(np.arange(1,301,dtype=float)**-1.8)
    assert fit['b']==pytest.approx(1.8)
    assert fit['r_squared']==pytest.approx(1.)
    with pytest.raises(ValueError,match='domain'):
        spectral_fit(np.arange(1,301,dtype=float)**-.7)


def test_grid_preserves_all_old_candidates_and_endpoints(managed):
    g,_=managed
    old=complexity_grid(g[:120]);new=nested_grid(g[:120])
    assert len(new)==len(np.unique(new))==200
    assert np.isin(old,new).all()
    assert new[0]==old[0] and new[-1]==old[-1]


def test_initial_c0_uses_no_oos_information(managed):
    g,dates=managed
    before=calibrate(g,dates)
    changed=g.copy();changed[180:]*=-20
    after=calibrate(changed,dates)
    for key in ['c0','loss','lambda']:
        np.testing.assert_array_equal(before[key],after[key])
    assert before['choice']==after['choice']


@pytest.mark.parametrize('position',[0,-1])
def test_annual_history_only_and_primal_equivalence(managed,position):
    g,dates=managed;c=calibrate(g,dates)
    split=list(annual_splits(dates))[position]
    a=window(g,dates,split,c)
    changed=g.copy();changed[split[3]]*=8
    b=window(changed,dates,split,c)
    for key in ['lambda','complexity','selected_beta']:
        np.testing.assert_array_equal(a[key],b[key])
    assert a['spectrum']==b['spectrum']
    history=g[np.r_[split[1],split[2]]]
    beta,_,comp=ridge_path(history,a['lambda'])
    np.testing.assert_allclose(a['returns'],g[split[3]]@beta,atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(a['complexity'],comp,atol=1e-6)
    np.testing.assert_allclose(a['lambda'],c['c0']*len(history)**(-a['spectrum']['b']/(a['spectrum']['b']+1)))
    assert a['last_known_payoff']==f'{split[0]}-01-31'


def test_heatmap_never_extrapolates():
    p=pd.DataFrame({'year':[2000,2000,2001,2001],
        'complexity':[1.,10.,5.,100.],'oos_loss':[1.,2.,3.,4.]})
    x,y,z=heatmap_values(p,50)
    assert np.isnan(z[0,x>10]).all()
    assert np.isnan(z[1,x<5]).all()
    assert np.isfinite(z[0,x<=10]).all()


def test_normalization_uses_each_years_historical_month_count():
    from complexity_schedule import normalized_inputs
    paths=pd.DataFrame({'year':[1978,1978,2024], 'complexity':[18.,90.,73.2],
                        'oos_loss':[.2,.4,.3]})
    selected=pd.DataFrame({'year':[1978,2024], 'T':[180,732], 'complexity':[90.,73.2]})
    p,s=normalized_inputs(paths,selected)
    np.testing.assert_allclose(p.complexity_over_T,[.1,.5,.1])
    np.testing.assert_allclose(s.complexity_over_T,[.5,.1])
    np.testing.assert_array_equal(p.oos_loss,paths.oos_loss)
    np.testing.assert_array_equal(p.complexity,paths.complexity)
    assert 'T' not in paths.columns
    with pytest.raises(ValueError,match='Missing'):
        normalized_inputs(paths,selected.iloc[:1])
