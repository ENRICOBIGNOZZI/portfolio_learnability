"""Numerical and chronology checks on actual, persisted empirical observations."""
import numpy as np
import pandas as pd
import pytest
from empirical_final.three_experiments import OUT, block_counts, boot_metrics, hac_regression
from portfolio import sharpe


def read(name):
    path = OUT/'tables'/f'{name}.csv'
    if not path.exists():
        pytest.skip('Run the three real-data experiments first')
    return pd.read_csv(path)


def test_bootstrap_statistics_match_literal_resampling_of_real_returns():
    r = read('e1_selected_monthly').raw_return.to_numpy()[:120]
    counts = block_counts(len(r), seed=19, replicates=7)
    np.testing.assert_array_equal(counts.sum(axis=1), len(r))
    result = boot_metrics(r, counts)
    for j, count in enumerate(counts):
        sample = np.repeat(r, count.astype(int))
        np.testing.assert_allclose(result['sharpe'][j, 0], sharpe(sample))
        np.testing.assert_allclose(result['loss'][j, 0], np.mean((1-sample)**2))


def test_selected_policy_is_the_corresponding_annual_candidate_not_an_ex_post_peak():
    paths, annual, selected = read('e1_monthly_paths'), read('e1_annual_paths'), read('e1_selected_monthly')
    for year, a in annual.groupby('decision_year'):
        choice = int(a.loc[a.validation_loss.idxmin(), 'candidate'])
        s = selected[selected.decision_year == year]
        assert s.candidate.eq(choice).all()
        p = paths[paths.decision_year == year]
        np.testing.assert_allclose(s.raw_return, p[f'lambda_{choice:03}'])
    curves = read('e1_curves')
    assert len(curves.period.unique()) == 5
    assert curves.groupby('period').descriptive_peak.sum().eq(1).all()


def test_history_windows_are_exact_chronological_and_share_all_test_months():
    annual, monthly = read('e2_annual'), read('e2_monthly')
    assert sorted(annual['T'].unique()) == [60, 120, 240, 360]
    assert annual.V.eq(20).all()
    for r in annual.itertuples():
        dates = pd.date_range(r.first_training_return, r.last_training_return, freq='ME')
        assert len(dates) == r.T
        assert pd.Timestamp(r.last_training_return) < pd.Timestamp(r.first_test_return)
        assert pd.Timestamp(r.last_training_return).month == 1
        assert pd.Timestamp(r.first_test_return).month == 2
    wide = monthly.pivot(index='return_date', columns='T', values='raw_return')
    assert wide.notna().all().all()
    assert len(wide) == annual.decision_year.nunique()*12


def test_spectral_additions_reconstruct_selected_policy_and_oos_risk():
    m, selected = read('e3_monthly'), read('e1_selected_monthly')
    c = m.pivot(index='return_date', columns='group', values='raw_contribution')
    nested = m.pivot(index='return_date', columns='group', values='raw_nested_return')
    np.testing.assert_allclose(c.cumsum(axis=1), nested, atol=1e-10)
    np.testing.assert_allclose(c.sum(axis=1), selected.set_index('return_date').loc[c.index].raw_return, atol=2e-7)
    summary = read('e3_nested_summary')
    np.testing.assert_allclose(summary.delta_loss, summary.cross_loss_term+summary.own_square_term, atol=1e-12)
    np.testing.assert_allclose(summary.delta_variance, summary.added_variance+summary.twice_covariance, atol=1e-12)
    # No fictitious OOS orthogonality: observed cross terms must be retained.
    assert np.max(np.abs(summary.twice_covariance)) > 1e-8


def test_hac_regression_is_ols_with_factor_units_in_decimal_monthly_returns():
    factors = read('french_factors').set_index('return_date')
    monthly = read('e3_monthly').query('group == 1').set_index('return_date')
    names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    f = factors.loc[monthly.index, names].to_numpy()
    y = monthly.diagnostic_return.to_numpy()
    beta, se, r2 = hac_regression(y, f)
    x = np.column_stack([np.ones(len(y)), f])
    np.testing.assert_allclose(x.T@(y-x@beta), 0, atol=1e-10)
    assert np.isfinite(se).all() and (se > 0).all() and 0 <= r2 <= 1
    assert np.max(np.abs(f)) < 1


@pytest.mark.parametrize('year', [1993, 2024])
def test_history_payoffs_match_an_independent_direct_ridge_solve(year):
    from empirical_final.run import clean_manifest, load_managed
    from local_learnability import rolling_split
    g, dates = load_managed('gaussian', clean_manifest())
    ret_dates = dates+pd.offsets.MonthEnd(1)
    split, reason = rolling_split(ret_dates, year, 60, 20, 'january-close')
    assert reason is None
    h, o = split['history'], split['test']
    row = read('e2_annual').query('decision_year == @year and T == 60').iloc[0]
    gram = np.asarray(g[h])@np.asarray(g[h]).T
    alpha = np.linalg.solve(gram/60+row.penalty*np.eye(60), np.ones(60))/60
    direct = np.asarray(g[o])@np.asarray(g[h]).T@alpha
    observed = read('e2_monthly').query('decision_year == @year and T == 60').raw_return.to_numpy()
    np.testing.assert_allclose(direct, observed, rtol=2e-6, atol=2e-7)
