"""Adversarial leakage, accounting and source integrity checks; no empirical outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from data_pipeline import prepare, load_panels, digest, rank_months, characteristic_selection
from portfolio import fit_windows, ridge_path, dual_path, annual_splits
from figures import scaled_history, wealth_drawdown, drawdown_episodes, final_path


def test_dual_matches_primal_and_selected_payoffs():
    rng=np.random.default_rng(8)
    g=rng.normal(size=(50,12));lam=np.array([.0001,.1,1.])
    b,mu,c=ridge_path(g,lam)
    a,dual_mu,dual_c=dual_path(g@g.T,lam)
    np.testing.assert_allclose(g.T@a,b,atol=1e-10)
    np.testing.assert_allclose(dual_c,c,atol=1e-9)


def test_future_test_returns_cannot_choose_lambda_or_coefficients():
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    g=np.random.default_rng(42).normal(.01,.1,size=(744,5))
    a=next(fit_windows(g,dates))
    g[180:]*=-10
    b=next(fit_windows(g,dates))
    assert a['choice']==b['choice']
    for key in ['penalties','validation_loss','beta','selected_train_beta']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-10)
    assert not np.allclose(a['test_returns'],b['test_returns'])


def test_every_oos_month_exists_once():
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    indices=np.concatenate([te for y,tr,va,te in annual_splits(dates)])
    assert len(indices)==len(set(indices))==564
    assert pd.DatetimeIndex(dates[indices]).equals(pd.date_range('1978-01-31','2024-12-31',freq='ME'))
    with pytest.raises(ValueError):
        list(annual_splits(dates.delete(200)))


def sample_history():
    dates=pd.date_range('1978-01-31','2024-12-31',freq='ME')
    n=len(dates)
    frame=pd.DataFrame({'formation_date':dates,'return_date':dates+pd.offsets.MonthEnd(1),
        'raw_excess_return':np.tile([.04,-.08,.02],188),
        'raw_gross':np.linspace(1,50,n),'raw_net':np.linspace(-1,3,n)})
    rf=pd.DataFrame({'return_date':frame.return_date,'rf':.003})
    return frame,rf


def test_common_fixed_scale_no_cap_no_volatility_target_cash_once():
    frame,rf=sample_history()
    histories=[scaled_history(frame,rf,.5) for _ in range(3)]
    for h in histories:
        assert h.kappa.nunique()==1
        np.testing.assert_allclose(h.gross,frame.raw_gross*.5)
        np.testing.assert_allclose(h.net,frame.raw_net*.5)
        np.testing.assert_allclose(h.excess_return,frame.raw_excess_return*.5)
        np.testing.assert_allclose(h.total_return,frame.raw_excess_return*.5+.003)
        assert h.gross.max()==25
        np.testing.assert_allclose(h.wealth,np.cumprod(1+h.total_return))
        np.testing.assert_allclose(h.drawdown,h.wealth/np.maximum.accumulate(np.r_[1,h.wealth])[1:]-1)
    bad=frame.copy();bad.loc[0,'formation_date']=bad.loc[1,'formation_date']
    with pytest.raises(ValueError,match='564'):
        scaled_history(bad,rf,.5)


def test_drawdown_sign_baseline_and_percent_units_failure():
    w,d=wealth_drawdown([-.1,.2,-.2])
    np.testing.assert_allclose(w,[.9,1.08,.864])
    np.testing.assert_allclose(d,[-.1,0,-.2])
    with pytest.raises(ValueError,match='decimal'):
        wealth_drawdown([-10,20,-20])
    with pytest.raises(ValueError):
        wealth_drawdown([-.01,np.nan])


@pytest.fixture
def raw_snapshot(tmp_path):
    candidates=json.loads(Path('characteristics.json').read_text())
    dates=pd.date_range('1963-01-31','2025-01-31',freq='ME')
    frame=pd.DataFrame({'eom':np.repeat(dates,10),'id':np.tile(np.arange(1,11),len(dates))})
    frame['permno']=frame.id
    for key,value in {'excntry':'USA','size_grp':'large','me':1.,'crsp_shrcd':10,
        'crsp_exchcd':1,'common':1,'primary_sec':1,'obs_main':1,'exch_main':1,
        'ret_exc_lead1m':.01,'current_excess_return':.01,'current_total_return':.013}.items():
        frame[key]=value
    values=np.random.default_rng(0).normal(size=(len(frame),153))
    values[:,130:]=np.nan
    frame=pd.concat([frame,pd.DataFrame(values,columns=candidates)],axis=1)
    raw=tmp_path/'raw';raw.mkdir()
    path=raw/'jkp_1963.parquet';frame.to_parquet(path,index=False)
    manifest={'status':'complete','return_units':'decimal','source':'synthetic test fixture only',
        'characteristics':candidates,'files':[{'name':path.name,'sha256':digest(path),'rows':len(frame)}]}
    (raw/'manifest.json').write_text(json.dumps(manifest))
    return raw,frame,manifest


def overwrite_raw(raw,frame,manifest):
    p=raw/'jkp_1963.parquet';frame.to_parquet(p,index=False)
    manifest['files'][0]['sha256']=digest(p)
    manifest['files'][0]['rows']=len(frame)
    (raw/'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('bad_return',[np.nan,np.inf,-np.inf])
def test_drop_before_ranking_without_fallback_recovery(raw_snapshot,tmp_path,bad_return):
    raw,frame,meta=raw_snapshot
    frame.loc[0,'ret_exc_lead1m']=bad_return
    # A finite contemporaneous return next month must not undo the chosen drop.
    assert np.isfinite(frame.loc[10,'current_excess_return'])
    overwrite_raw(raw,frame,meta)
    m=prepare(raw,tmp_path/'clean')
    assert m['feature_count']==130 and m['recovered_returns']==0
    assert m['dropped_returns']==1 and m['unresolved_returns']==0
    assert m['coverage_end']=='1972-12-31'
    panels,_=load_panels(tmp_path/'clean')
    assert len(panels)==744 and len(panels[0]['ids'])==9
    assert 1 not in panels[0]['ids'] and 1 in panels[1]['ids']
    names=m['characteristics']
    expected=rank_months(frame.iloc[1:10],names)
    np.testing.assert_allclose(panels[0]['x'],expected[names].to_numpy())
    np.testing.assert_allclose(panels[0]['r'],.01)
    assert np.load(tmp_path/'clean/initial_sample.npy').shape==(1000,130)
    excluded=pd.read_csv(tmp_path/'clean/excluded_returns.csv')
    assert len(excluded)==1 and excluded.id.iloc[0]==1
    counts=pd.read_csv(tmp_path/'clean/universe_counts.csv')
    assert counts.iloc[0].stocks==9 and counts.iloc[0].stocks_before_payoff_filter==10
    assert counts.dropped_returns.sum()==1
    for panel in panels:
        assert np.isfinite(panel['x']).all() and np.abs(panel['x']).max()<=.5
        assert np.isfinite(panel['r']).all()
    assert len(json.loads((tmp_path/'clean/characteristic_provenance.json').read_text())['coverage'])==153


def test_payoff_drop_cannot_silently_remove_whole_month(raw_snapshot,tmp_path):
    raw,frame,meta=raw_snapshot
    frame.loc[:9,'ret_exc_lead1m']=np.nan
    overwrite_raw(raw,frame,meta)
    with pytest.raises(ValueError,match='entire formation month'):
        prepare(raw,tmp_path/'clean')


def test_incomplete_old_manifest_still_blocks_fit(tmp_path):
    clean=tmp_path/'clean';clean.mkdir()
    (clean/'manifest.json').write_text(json.dumps({'status':'unresolved_returns','unresolved_returns':1}))
    with pytest.raises(ValueError,match='Unresolved'):
        load_panels(clean)


def test_duplicate_raw_stops_even_when_payoff_missing(raw_snapshot,tmp_path):
    raw,frame,meta=raw_snapshot
    duplicate=frame.iloc[[0]].copy();duplicate['current_excess_return']=np.nan
    frame=pd.concat([frame,duplicate],ignore_index=True)
    overwrite_raw(raw,frame,meta)
    with pytest.raises(ValueError,match='duplicated'):
        prepare(raw,tmp_path/'clean')


def test_exact_final_test_path_not_pooled():
    dates=pd.date_range('2024-01-31','2024-12-31',freq='ME')
    lambdas=np.geomspace(1e-8,1,120)
    rows=[];diags=[]
    from portfolio import sharpe
    for i,lam in enumerate(lambdas):
        r=np.sin(np.arange(12))*.01+.001*i
        diags.append({'test_year':2024,'lambda':lam,'oos_sharpe':float(sharpe(r)),
            'historical_sharpe':1.,'selected':i==50,'effective_complexity':120-i})
        rows.extend({'test_year':2024,'lambda':lam,'formation_date':d,
            'return_date':d+pd.offsets.MonthEnd(1),'excess_return':v} for d,v in zip(dates,r))
    diag,path=pd.DataFrame(diags),pd.DataFrame(rows)
    assert len(final_path(diag,path))==120
    path.loc[0,'formation_date']=pd.Timestamp('2023-01-31')
    with pytest.raises(ValueError,match='Pooled'):
        final_path(diag,path)


def test_empirical_text_uses_new_numbers_and_exactly_five_figures(tmp_path):
    from figures import make_text, STEMS, LABELS
    performance=pd.DataFrame([{'Policy':v,'Sharpe ratio':1.234,'Maximum drawdown':-.234}
                              for v in LABELS.values()])
    make_text(tmp_path,performance,[],3.14159)
    text=(tmp_path/'empirics.tex').read_text()
    assert text.count('\\includegraphics')==5
    assert all(stem+'.pdf' in text for stem in STEMS)
    assert '\\subsection' not in text
    assert '1.23' in text and '-23.4' in text and '3.1416' in text
    assert 'response-one' not in text
    assert 'SUMMARY' not in text and 'ELL' not in text


def test_cash_returns_from_same_raw_snapshot_not_added_twice(raw_snapshot,tmp_path):
    from run_empirics import get_risk_free
    raw,frame,manifest=raw_snapshot
    destination=tmp_path/'rf.csv'
    get_risk_free(raw,destination)
    rf=pd.read_csv(destination)
    np.testing.assert_allclose(rf.rf,.003)
    assert len(rf)==745
    frame.loc[0,'current_total_return']=.5
    overwrite_raw(raw,frame,manifest)
    with pytest.raises(ValueError,match='Inconsistent cash'):
        get_risk_free(raw,tmp_path/'wrong_rf.csv')


def test_final_refit_cannot_use_final_test_payoffs():
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    g=np.random.default_rng(91).normal(.01,.1,size=(744,5))
    before=list(fit_windows(g,dates))[-1]
    g[-12:]=np.random.default_rng(92).normal(100,10,size=(12,5))
    after=list(fit_windows(g,dates))[-1]
    assert before['choice']==after['choice']
    for key in ['penalties','validation_loss','beta','selected_train_beta']:
        np.testing.assert_allclose(before[key],after[key],rtol=1e-10,atol=1e-10)
    assert not np.allclose(before['test_returns'],after['test_returns'])


def test_refit_labels_available_by_trade_close_and_test_payoffs_strictly_later():
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    for year,train,validation,test in annual_splits(dates):
        trade=dates[test[0]]
        assert (dates[np.r_[train,validation]]+pd.offsets.MonthEnd(1)).max() <= trade
        assert (dates[test]+pd.offsets.MonthEnd(1)).min() > trade
        assert dates[train].max()<dates[validation].min()<dates[test].min()


def test_streamed_panels_preserve_values_and_random_access(tmp_path):
    from data_pipeline import PanelSequence
    records=[]
    for year in (1963,1964):
        dates=pd.date_range(f'{year}-01-31',periods=2,freq='ME')
        f=pd.DataFrame({'eom':np.repeat(dates,2),'id':[2,1,2,1],
            'x':[.2,.1,.4,.3],'r':[.02,.01,.04,.03]})
        f['return_date']=f.eom+pd.offsets.MonthEnd(1)
        filename=f'{year}.parquet';f.to_parquet(tmp_path/filename,index=False)
        records.extend((date,filename) for date in dates)
    panels=PanelSequence(tmp_path,['x'],records)
    assert len(panels)==4
    for index in [0,3,1,2,0]:
        p=panels[index]
        np.testing.assert_array_equal(p['ids'],[1,2])
        np.testing.assert_allclose(p['x'].ravel(),[.1,.2] if index%2==0 else [.3,.4])
        np.testing.assert_allclose(p['r'],p['x'].ravel()/10)
    assert [p['date'] for p in panels]==[d for d,_ in records]
