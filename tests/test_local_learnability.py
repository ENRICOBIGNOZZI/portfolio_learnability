"""Rolling chronology, zero-ridge rank, OOS isolation and observed-cell heatmaps."""
import numpy as np
import pandas as pd
import pytest
from local_learnability import rolling_split,fit_local,bin_observations,T_GRID
from portfolio import dual_path,ridge_path

DATES=pd.date_range('1963-02-28','2025-01-31',freq='ME')


def test_exact_calendar_cutoffs_and_validation():
    s,reason=rolling_split(DATES,2000,60)
    assert reason is None and s['V']==20
    assert DATES[s['history'][0]]==pd.Timestamp('1995-01-31')
    assert DATES[s['inner'][-1]]==pd.Timestamp('1998-04-30')
    assert DATES[s['validation'][0]]==pd.Timestamp('1998-05-31')
    assert DATES[s['validation'][-1]]==pd.Timestamp('1999-12-31')
    assert DATES[s['test'][0]]==pd.Timestamp('2000-01-31')
    assert DATES[s['test'][-1]]==pd.Timestamp('2000-12-31')
    assert DATES[s['history']].max()<pd.Timestamp('2000-01-01')


def test_long_windows_skip_early_years_and_never_shorten_T():
    assert rolling_split(DATES,1983,240)[0] is None
    assert rolling_split(DATES,1984,240)[0] is not None
    assert rolling_split(DATES,1993,360)[0] is None
    assert rolling_split(DATES,1994,360)[0] is not None
    assert rolling_split(DATES,1978,180)[0] is None
    assert rolling_split(DATES,1979,180)[0] is not None
    assert rolling_split(DATES,2025,60)[1]=='incomplete 12-month OOS block'
    s,_=rolling_split(DATES,2000,84,validation_months=12)
    assert s['V']==12 and len(s['inner'])==72
    with pytest.raises(ValueError):rolling_split(DATES,2000,60,validation_months=60)


@pytest.mark.parametrize('t',T_GRID)
def test_oos_and_future_cannot_change_overlay(t):
    g=np.random.default_rng(31).normal(.03,.1,size=(744,24));s,_=rolling_split(DATES,2005,t)
    a=fit_local(g@g.T,s,count=80)
    other=g.copy();other[DATES>s['cutoff']]*=-9
    b=fit_local(other@other.T,s,count=80)
    for key in ['lambda','validation_loss','complexity','relative_complexity','selected_alpha']:
        np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12)
    assert a['choice']==b['choice']
    assert not np.allclose(a['returns'],b['returns'])


def test_validation_does_not_set_candidate_grid_and_test_metric_is_held_out():
    g=np.random.default_rng(22).normal(.02,.1,size=(744,30));s,_=rolling_split(DATES,2000,60)
    a=fit_local(g@g.T,s,count=80)
    changed=g.copy();changed[s['validation']]*=-4
    b=fit_local(changed@changed.T,s,count=80)
    np.testing.assert_array_equal(a['lambda'],b['lambda'])
    assert not np.allclose(a['validation_loss'],b['validation_loss'])
    np.testing.assert_allclose(a['oos_loss'],np.mean((1-a['returns'])**2,axis=0))
    beta,_,c=ridge_path(g[s['history']],a['lambda'][1:])
    np.testing.assert_allclose(a['returns'][:,1:],g[s['test']]@beta,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(a['complexity'][1:],c,atol=1e-6)
    expected=np.linalg.lstsq(g[s['history']],np.ones(60),rcond=None)[0]
    np.testing.assert_allclose(a['returns'][:,0],g[s['test']]@expected,atol=1e-9)
    assert a['complexity'][0]==30


def test_empty_heatmap_cells_stay_missing_and_every_filled_cell_is_observed():
    f=pd.DataFrame({'decision_year':[2000]*4,'candidate':[0,1,2,3],
        'relative_complexity':[0.,.12,.19,1.],'oos_loss':[1.,.5,.8,2.]})
    edges,z,cells=bin_observations(f,bins=5,years=[1999,2000])
    assert np.isnan(z[0]).all()
    assert z[1,0]==.5 and z[1,-1]==2.
    assert np.isnan(z[1,1:4]).all()
    assert set(cells.oos_loss).issubset(set(f.oos_loss))
    assert ((cells.relative_complexity>=cells.bin_left)&(cells.relative_complexity<=cells.bin_right)).all()


def test_optional_repository_timing_is_explicit():
    s,_=rolling_split(DATES,2000,60,timing='january-close')
    assert s['cutoff']==pd.Timestamp('2000-01-31')
    assert DATES[s['test'][0]]==pd.Timestamp('2000-02-29')
    assert DATES[s['test'][-1]]==pd.Timestamp('2001-01-31')


def test_dual_zero_handles_rank_deficiency_without_inventing_dimensions():
    g=np.random.default_rng(7).normal(size=(20,4));a,mu,c=dual_path(g@g.T,np.array([0.,.1]))
    b=np.linalg.lstsq(g,np.ones(20),rcond=None)[0]
    np.testing.assert_allclose(g.T@a[:,0],b,atol=1e-10)
    assert c[0]==4
    with pytest.raises(ValueError):dual_path(g@g.T,[-1.])
