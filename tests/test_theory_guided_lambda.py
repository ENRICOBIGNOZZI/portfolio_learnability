"""Algebra, timing, empirical reconciliation, and actual-weight cost checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from data_pipeline import digest
from kernels import FeatureBank
from portfolio import annual_splits, dual_path, fit_windows, ridge_path, sharpe
from empirical_final.run import ROOT, clean_manifest, load_managed
from empirical_final.theory_guided_lambda import (
    DEFAULT_OUTPUT, account_targets, anchor_from_history, calibrate_rho,
    choose_lambda, complexity, sensitivity, spectrum,
)
from empirical_final.three_experiments import block_counts, boot_metrics


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def test_gram_feature_primal_spectra_and_existing_complexity():
    g = np.array([[2., .1], [1., .3], [.5, -.2], [.4, .6]])
    mu, _ = spectrum(g@g.T)
    primal = np.linalg.eigvalsh(g.T@g/len(g))[::-1]
    np.testing.assert_allclose(mu[:2], primal, rtol=1e-13)
    np.testing.assert_allclose(mu[2:], 0, atol=1e-15)
    _, expected_mu, c = dual_path(g@g.T, [.02, .1, 2.])
    np.testing.assert_allclose(mu, expected_mu, atol=1e-15)
    np.testing.assert_allclose([complexity(mu, x) for x in [.02, .1, 2.]], c, atol=1e-12)


def test_psd_roundoff_clipped_but_meaningful_negatives_rejected():
    mu, audit = spectrum(np.diag([1., .1, -1e-12, 0.]))
    assert (mu >= 0).all() and audit['clipped_eigenvalues'] == 1
    assert complexity(mu, .1) == pytest.approx(complexity(mu[:2], .1))
    with pytest.raises(ValueError, match='Meaningfully negative'):
        spectrum(np.diag([1., -.001]))


def test_monotonicity_uniqueness_rho_and_proxy_minimum():
    mu = np.array([.12, .035, .009, .003, .0005])
    grid = np.geomspace(1e-5, .25, 100)
    rho = calibrate_rho(mu, 180, .004, (grid.min(), grid.max()))
    cs = np.array([complexity(mu, x) for x in grid])
    derivatives = np.array([1-rho/180*sensitivity(mu, x) for x in grid])
    assert (np.diff(cs) < 0).all() and (np.diff(derivatives) > 0).all()
    assert sum(np.diff(np.sign(derivatives)) != 0) == 1
    root = choose_lambda(mu, 180, rho, grid.min(), grid.max())
    more = choose_lambda(mu, 180, 2*rho, grid.min(), grid.max())
    assert more['penalty'] > root['penalty'] and more['C'] < root['C']
    objective = grid+rho/180*cs
    assert root['objective'] <= objective.min()
    assert abs(root['derivative_residual']) < 1e-11


def test_known_spectrum_analytic_root_and_both_boundaries():
    # One nonzero eigenvalue: lambda = sqrt(rho*mu/T) - mu = 4 - 2.
    mu = np.array([2., 0., 0.])
    r = choose_lambda(mu, 10, 80, .01, 10.)
    assert r['penalty'] == pytest.approx(2., rel=1e-12)
    assert r['C'] == pytest.approx(.5)
    low = choose_lambda(mu, 10, .001, .01, 10.)
    high = choose_lambda(mu, 10, 10000., .01, 10.)
    assert low['boundary'] == 'lower' and low['penalty'] == .01
    assert high['boundary'] == 'upper' and high['penalty'] == 10.
    assert low['derivative_residual'] >= 0 and high['derivative_residual'] <= 0
    assert low['kkt_residual'] == high['kkt_residual'] == 0
    zero = choose_lambda(np.zeros(3), 10, 80, .01, 10.)
    assert zero['boundary'] == 'lower' and zero['C'] == 0


@pytest.mark.parametrize('mu,T,rho,lower,upper', [
    ([-1.],10,1.,.01,1.), ([np.nan],10,1.,.01,1.), ([1.],0,1.,.01,1.),
    ([1.],10,-1.,.01,1.), ([1.],10,np.inf,.01,1.), ([1.],10,1.,0.,1.),
    ([1.],10,1.,1.,.01),
])
def test_invalid_selector_inputs_raise(mu,T,rho,lower,upper):
    with pytest.raises(ValueError):
        choose_lambda(mu,T,rho,lower,upper)


def test_boundary_anchor_reports_identification_limitation():
    for anchor in (.01, 1.):
        with pytest.raises(ValueError, match='point-identify'):
            calibrate_rho([1.], 180, anchor, (.01,1.))


def test_all_annual_formation_and_return_dates():
    dates = pd.date_range('1963-01-31','2024-12-31',freq='ME')
    splits = list(annual_splits(dates))
    assert [s[0] for s in splits] == list(range(1978,2025))
    months = np.concatenate([s[3] for s in splits])
    assert len(months) == len(set(months)) == 564
    assert dates[months[0]]+pd.offsets.MonthEnd(1) == pd.Timestamp('1978-02-28')
    assert dates[months[-1]]+pd.offsets.MonthEnd(1) == pd.Timestamp('2025-01-31')
    for y,tr,va,te in splits:
        assert dates[tr[-1]].year == y-6 and len(va) == 60
        assert len(np.r_[tr,va]) == (y-1963)*12
        assert (dates[np.r_[tr,va]]+pd.offsets.MonthEnd(1)).max() == dates[te[0]]
        assert (dates[te]+pd.offsets.MonthEnd(1)).min() > dates[te[0]]


def test_identical_scaling_and_portfolio_aggregation():
    g = np.array([[2.,.1],[1.,.3],[.5,-.2],[.4,.6]])
    penalties = [.02, .08]
    beta, _, _ = ridge_path(g, penalties)
    alpha, mu, _ = dual_path(g@g.T, penalties)
    test = np.array([[.1,.3],[-.2,.4]])
    raw = test@g.T@alpha
    np.testing.assert_allclose(test@beta, raw, atol=1e-13)
    kappa, rf = .0374, .003
    np.testing.assert_allclose((test@beta)*kappa+rf, raw*kappa+rf, atol=1e-13)
    assert not np.isclose(complexity(mu, .02), complexity(mu*kappa*kappa, .02))


def test_paired_bootstrap_identical_paths_and_literal_sample():
    r = np.array([.1,-.1,.2,.01,-.05,.03,.06,-.02])
    counts = block_counts(len(r),3,19,10)
    b = boot_metrics(np.column_stack([r,r]),counts)
    np.testing.assert_array_equal(b['sharpe'][:,0]-b['sharpe'][:,1],0)
    for j,c in enumerate(counts):
        assert b['sharpe'][j,0] == pytest.approx(sharpe(np.repeat(r,c.astype(int))))


def test_target_costs_include_entry_exit_and_signed_short_notional():
    target = pd.Series([.8,-.4],index=[1,2])
    returns = pd.Series([.1,-.05],index=[1,2])
    previous = pd.Series([.6,.3],index=[1,3])
    state = pd.Series([.7,.2],index=[1,3])
    rows, drift = account_targets(target,returns,.01,state,previous)
    scenarios = {r['scenario']:r for r in rows}
    turnover = .2+.4+.3
    net = scenarios['target_trade25_borrow30']
    assert net['target_turnover'] == pytest.approx(turnover)
    assert net['excess_return'] == pytest.approx(.1-.0025*turnover-.003/12*.4)
    assert scenarios['gross']['total_return'] == pytest.approx(.11)
    assert scenarios['trade25_borrow30']['accounting_residual'] < 1e-12
    assert set(drift.index) == {1,2}


@pytest.fixture(scope='module')
def real_cache():
    with threadpool_limits(limits=1):
        g, dates = load_managed('gaussian', clean_manifest())
        config = json.loads((DEFAULT_OUTPUT/'audit/config.json').read_text())
    return g, dates, np.array(config['lambda_grid']), config


def test_real_anchor_no_1978_test_return_and_later_spectra_past_only(real_cache):
    g, dates, grid, _ = real_cache
    anchor = anchor_from_history(g,dates,grid)
    changed = np.asarray(g).copy()
    changed[180:] = np.random.default_rng(12).normal(100.,10.,changed[180:].shape)
    assert anchor == anchor_from_history(changed,dates,grid)
    assert anchor['T0'] == 180 and anchor['last_known_payoff'] == '1978-01-31'
    assert anchor['first_test_payoff'] == '1978-02-28'
    # Independent existing annual CV reproduces the historical anchor.
    result = next(fit_windows(g, dates, grid))
    assert grid[result['choice']] == pytest.approx(anchor['lam0'], rel=1e-12)
    assert not np.allclose(next(fit_windows(changed,dates,grid))['test_returns'],result['test_returns'])
    cutoff = 324  # January 1990 decision: every subsequent label is perturbed.
    changed = np.asarray(g).copy()
    changed[cutoff:] *= -1e4
    mu,_ = spectrum(np.asarray(g[:cutoff])@np.asarray(g[:cutoff]).T)
    mu2,_ = spectrum(changed[:cutoff]@changed[:cutoff].T)
    assert choose_lambda(mu,cutoff,anchor['rho0'],grid.min(),grid.max()) == choose_lambda(mu2,cutoff,anchor['rho0'],grid.min(),grid.max())


def test_real_outputs_reproduce_both_original_baselines_and_grid():
    out = DEFAULT_OUTPUT
    m = pd.read_csv(out/'tables/monthly_oos.csv')
    a = pd.read_csv(out/'tables/annual_selection.csv')
    old = pd.read_csv(ROOT/'results/final/public/gaussian/seed_0/p_10000/monthly.csv')
    legacy = pd.read_csv(ROOT/'outputs/empirical_three_experiments_20261009/tables/e1_selected_monthly.csv')
    paths = pd.read_csv(ROOT/'outputs/empirical_three_experiments_20261009/tables/e1_monthly_paths.csv')
    eigenvalues = pd.read_csv(out/'tables/empirical_eigenvalues.csv')
    assert len(m) == m.return_date.nunique() == 564 and len(a) == 47
    assert m.decision_year.ge(1979).sum() == 552
    assert a.rho0.nunique() == 1
    np.testing.assert_array_equal(m.return_date,old.return_date)
    np.testing.assert_array_equal(m.return_date,legacy.return_date)
    np.testing.assert_allclose(m.cv_raw,old.raw_excess_return,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(m.cv_raw,legacy.raw_return,rtol=1e-7,atol=1e-9)
    for row in a.itertuples():
        mu = eigenvalues[eigenvalues.year.eq(row.year)].mu.to_numpy()
        assert len(mu) == row.T and (mu >= 0).all()
        assert complexity(mu,row.lambda_spectral) == pytest.approx(row.C_spectral,abs=1e-8)
        assert complexity(mu,row.lambda_cv) == pytest.approx(row.C_cv,abs=1e-8)
        observed = m[m.decision_year.eq(row.year)].spectral_grid_raw
        expected = paths[paths.decision_year.eq(row.year)][f'lambda_{row.spectral_grid_choice:03}']
        np.testing.assert_allclose(observed,expected,rtol=1e-7,atol=1e-9)
    assert a.iloc[0].lambda_spectral == pytest.approx(a.iloc[0].lambda_cv,rel=1e-11)
    assert a.kkt_residual.max() < 1e-10
    for f in (m,a):
        assert np.isfinite(f.select_dtypes('number').to_numpy()).all()
    for p in ('cv','spectral','spectral_grid'):
        np.testing.assert_allclose(m[f'{p}_excess'],m.kappa*m[f'{p}_raw'],atol=1e-14)
        np.testing.assert_allclose(m[f'{p}_total'],m[f'{p}_excess']+m.rf,atol=1e-14)


def test_real_weight_aggregation_and_all_cost_conventions(real_cache):
    g, _, grid, config = real_cache
    meta = clean_manifest()
    data = pd.read_parquet(ROOT/'data/clean/jkp_1978.parquet')
    data = data[data.eom.eq('1978-01-31')].sort_values('id').set_index('id')
    bank = FeatureBank('gaussian',130,10000,config['feature_bank']['ell'],0)
    alpha, _, _ = dual_path(np.asarray(g[:180])@np.asarray(g[:180]).T,[json.loads((DEFAULT_OUTPUT/'audit/anchor.json').read_text())['lam0']])
    beta = np.asarray(g[:180]).T@alpha[:,0]
    raw = bank.scores(data[meta['characteristics']].to_numpy(),beta)/len(data)
    target = pd.Series(config['kappa']*raw,index=data.index)
    m = pd.read_csv(DEFAULT_OUTPUT/'tables/monthly_oos.csv').iloc[0]
    np.testing.assert_allclose(raw@data.r,m.cv_raw,rtol=1e-7,atol=1e-9)
    actual,_ = account_targets(target,data.r,m.rf,pd.Series(dtype=float),pd.Series(dtype=float))
    stored = pd.read_csv(DEFAULT_OUTPUT/'tables/costs.csv').query("policy=='cv' and formation_date=='1978-01-31'").set_index('scenario')
    for row in actual:
        for key in ('excess_return','total_return','target_turnover','trading_fee','borrowing_fee','gross_exposure','net_exposure'):
            assert row[key] == pytest.approx(stored.loc[row['scenario'],key],rel=2e-6,abs=1e-9)
    illustrative = next(r for r in actual if r['scenario']=='target_trade25_borrow30')
    assert illustrative['excess_return'] == pytest.approx(target@data.r-.0025*target.abs().sum()-.003/12*np.maximum(-target,0).sum())


def test_frozen_audit_hashes_summary_and_actual_weight_verification():
    out = DEFAULT_OUTPUT
    for name,expected in json.loads((out/'audit/input_hashes.json').read_text()).items():
        assert digest(ROOT/name) == expected
    config = json.loads((out/'audit/config.json').read_text())
    assert digest(ROOT/'empirical_final/theory_guided_lambda.py') == config['module_sha256']
    for name,expected in json.loads((out/'audit/output_hashes.json').read_text()).items():
        assert digest(out/name) == expected
    v = json.loads((out/'audit/verification.json').read_text())
    assert v['passed'] and v['rho_frozen'] and v['net_performance_reconstructed']
    c = json.loads((out/'audit/cost_verification.json').read_text())
    assert c['actual_signed_weights'] and c['cv_stock_weights_reproduced'] and c['original_drift_cost_account_reproduced']
    assert c['maximum_accounting_residual'] < 1e-10
    m = pd.read_csv(out/'tables/monthly_oos.csv').query('decision_year>=1979')
    s = pd.read_csv(out/'tables/summary.csv').query("period=='1979-2024' and scenario=='gross'").set_index('policy')
    assert s.loc['spectral','sharpe'] == pytest.approx(sharpe(m.spectral_excess))
    assert s.loc['cv','sharpe'] == pytest.approx(sharpe(m.cv_excess))
