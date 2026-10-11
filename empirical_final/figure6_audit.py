"""Reconstruct fixed-penalty paths and the distinct annual CV policy from exports.

No model fitting, policy selection, return modification, or interpolation occurs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import digest
from portfolio import sharpe
from empirical_final.section_vi_finalize import BASE, OUT, BLOCKS


def scientific_audit(section=OUT, original=BASE):
    section, original = Path(section), Path(original)
    inputs = {}

    def load(folder, name):
        path = folder / 'tables' / f'{name}.csv'
        inputs[str(path)] = digest(path)
        return pd.read_csv(path, float_precision='round_trip')

    annual = load(section, 'ridge_annual')
    monthly = load(section, 'ridge_monthly_paths').sort_values('return_date')
    curves = load(section, 'nine_panel_curves')
    summary = load(section, 'nine_panel_summary').set_index('period')
    single = load(section, 'single_window_complexity')
    endpoints = load(section, 'ridge_endpoint_checks')
    original_annual = load(original, 'kernel_annual_paths').query("policy == 'gaussian'")
    original_monthly = load(original, 'e1_monthly_paths').sort_values('return_date')
    selected_monthly = load(original, 'kernel_selected_monthly').query("policy == 'gaussian'").sort_values('return_date')
    accounts = load(section, 'monthly_accounting').query("policy == 'gaussian' and scenario == 'gross'").sort_values('return_date')
    protocol_path = section / 'audit/ridge_protocol.json'
    inputs[str(protocol_path)] = digest(protocol_path)
    grid = np.array(json.loads(protocol_path.read_text())['penalties'])
    cols = [f'lambda_{j:04}' for j in range(1522)]
    assert len(grid) == 1522 and grid[-1] == 0 and (grid[:-1] > 0).all()
    assert len(annual) == 47 * 1522 and len(curves) == 9 * 1522
    assert len(monthly) == len(selected_monthly) == len(accounts) == 564
    expected_dates = pd.date_range('1978-02-28', '2025-01-31', freq='ME')
    for frame in [monthly, original_monthly, selected_monthly, accounts]:
        assert pd.DatetimeIndex(frame.return_date).equals(expected_dates)
    assert set(annual.decision_year) == set(range(1978, 2025))
    assert len(endpoints) == 47
    assert endpoints.zero_training_relative_residual.max() < 1e-6
    assert endpoints.primal_endpoint_error.dropna().max() < 2e-7
    assert endpoints.near_zero_endpoint_error.max() < 1e-7

    years = []
    reconstructed = []
    complexities = []
    for year in range(1978, 2025):
        old = original_annual[original_annual.decision_year.eq(year)].sort_values('candidate')
        chosen = old[old.selected]
        assert len(old) == 120 and len(chosen) == 1
        choice = chosen.iloc[0]
        # Stored selection must be the original validation-loss minimizer.
        assert int(choice.candidate) == int(old.loc[old.validation_Q.idxmin(), 'candidate'])
        assert pd.Timestamp(choice.last_known_payoff) < pd.Timestamp(choice.first_test_payoff)
        a = annual[annual.decision_year.eq(year)].sort_values('candidate')
        assert np.array_equal(a.candidate, np.arange(1522))
        np.testing.assert_allclose(a.penalty, grid, rtol=1e-13, atol=0)
        selected = a[a.selected]
        assert len(selected) == 1 and a.zero_penalty.sum() == 1
        item = selected.iloc[0]
        np.testing.assert_allclose(item.penalty, choice.penalty, rtol=1e-12, atol=0)
        pp = monthly[monthly.decision_year.eq(year)]
        ss = selected_monthly[selected_monthly.decision_year.eq(year)]
        assert len(pp) == len(ss) == 12
        np.testing.assert_allclose(ss.selected_penalty, choice.penalty, rtol=1e-12, atol=0)
        np.testing.assert_allclose(ss.C, choice.C, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(item.C, choice.C, rtol=2e-6, atol=2e-7)
        rr = pp[cols[int(item.candidate)]].to_numpy()
        np.testing.assert_allclose(rr, ss.raw_return, rtol=2e-6, atol=2e-7)
        # Verify ALL original positive-penalty observations against their exports.
        indices = [int(np.argmin(abs(grid - penalty))) for penalty in old.penalty]
        np.testing.assert_allclose(grid[indices], old.penalty, rtol=1e-12, atol=0)
        legacy = original_monthly[original_monthly.decision_year.eq(year)].filter(like='lambda_')
        np.testing.assert_allclose(pp[cols].to_numpy()[:, indices], legacy, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(a.iloc[indices].C, old.C, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(sharpe(pp[cols]), a.oos_sharpe, rtol=1e-11, atol=1e-11)
        zero = a[a.zero_penalty].iloc[0]
        assert zero.penalty == 0 and zero.C == zero['rank']
        assert (a.C <= zero['rank'] + 1e-8).all()
        reconstructed.extend(rr)
        complexities.extend([item.C] * len(pp))
        years.append(dict(year=year, original_candidate=int(choice.candidate),
                          extended_candidate=int(item.candidate), penalty=float(choice.penalty),
                          validation_Q=float(choice.validation_Q), months=len(pp),
                          max_selected_payoff_error=float(np.max(abs(rr - ss.raw_return.to_numpy()))),
                          selected_complexity_error=float(abs(item.C - choice.C))))

    reconstructed = np.array(reconstructed)
    complexities = np.array(complexities)
    blocks = []
    for lo, hi in BLOCKS:
        label = f'{lo}-{hi}'
        pp = monthly[monthly.decision_year.between(lo, hi)]
        aa = annual[annual.decision_year.between(lo, hi)]
        ss = selected_monthly[selected_monthly.decision_year.between(lo, hi)]
        mask = monthly.decision_year.between(lo, hi).to_numpy()
        sizes = pp.groupby('decision_year').size()
        values = aa.pivot(index='decision_year', columns='candidate', values='C').loc[sizes.index]
        mean_C = np.average(values, weights=sizes, axis=0)
        sr = sharpe(pp[cols])
        cc = curves[curves.period.eq(label)].sort_values('candidate')
        row = summary.loc[label]
        assert len(cc) == 1522 and cc.zero_penalty.sum() == 1
        np.testing.assert_allclose(cc.mean_C, mean_C, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(cc.oos_sharpe, sr, rtol=1e-11, atol=1e-11)
        selected_sr = float(sharpe(ss.raw_return))
        selected_C = float(np.average(complexities[mask]))
        np.testing.assert_allclose(row.selected_sharpe, selected_sr, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(row.selected_mean_C, selected_C, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(sharpe(reconstructed[mask]), selected_sr, rtol=2e-6, atol=2e-7)
        peak = int(np.argmax(sr))
        assert peak == int(row.maximum_candidate)
        assert cc.retrospective_maximum.sum() == 1 and bool(cc.iloc[peak].retrospective_maximum)
        np.testing.assert_allclose([row.maximum_C, row.maximum_sharpe, row.zero_sharpe],
                                   [mean_C[peak], sr[peak], sr[-1]], rtol=1e-11, atol=1e-11)
        choices = [y for y in years if lo <= y['year'] <= hi]
        matching = np.isclose(mean_C, row.selected_mean_C, rtol=1e-8, atol=1e-8) & np.isclose(sr, selected_sr, rtol=1e-8, atol=1e-8)
        blocks.append(dict(period=label, months=len(pp), candidate_observations=len(cc),
                           annual_selected_candidates=[y['extended_candidate'] for y in choices],
                           distinct_selected_penalties=len({y['penalty'] for y in choices}),
                           selected_sharpe=selected_sr, recorded_selected_sharpe=float(row.selected_sharpe),
                           reconstructed_selected_sharpe=float(sharpe(reconstructed[mask])),
                           selected_mean_C=selected_C, recorded_selected_mean_C=float(row.selected_mean_C),
                           diamond_matches_actual_cv=True, matches_fixed_candidate=bool(matching.any()),
                           numerical_or_indexing_inconsistency=False, maximum_candidate=peak,
                           maximum_sharpe=float(sr[peak]), zero_sharpe=float(sr[-1]),
                           mean_effective_rank=float(mean_C[-1])))

    chosen = single[single.selected]
    assert len(chosen) == 1
    item = chosen.iloc[0]
    a2024 = annual[(annual.decision_year == 2024) & (annual.candidate == item.candidate)].iloc[0]
    np.testing.assert_allclose(single.sort_values('candidate').C,
                               annual[annual.decision_year.eq(2024)].sort_values('candidate').C, rtol=1e-12)
    plotted_sr = float(sharpe(accounts[accounts.decision_year.eq(2024)].excess_return))
    np.testing.assert_allclose([item.C, item.oos_sharpe], [a2024.C, plotted_sr], rtol=2e-6, atol=2e-7)
    return dict(passed=True, reason='The blue curve uses one common candidate penalty across annual refits. '
                'The original annual CV policy uses different past-selected penalties within a block. '
                'Its diamond correctly reports that dynamic policy and need not lie on the fixed-penalty curve.',
                sharpe_convention='sqrt(12) * mean(concatenated monthly excess returns) / sample SD; not mean annual Sharpe',
                complexity_convention='Annual effective complexity weighted by evaluated month counts',
                comparison_tolerances={'legacy_eigensystem_rtol': 2e-6, 'legacy_eigensystem_atol': 2e-7,
                                       'aggregate_rtol': 1e-11, 'aggregate_atol': 1e-11},
                oos_months=564, annual_refits=47, candidates_per_refit=1522,
                historical_candidate_observations=len(curves), annual_choices=years, blocks=blocks,
                figure5=dict(year=2024, candidate=int(item.candidate), penalty=float(item.penalty),
                             selected_C=float(item.C), curve_sharpe=float(item.oos_sharpe),
                             plotted_selected_sharpe=plotted_sr, on_own_path=True, diamond_retained=True),
                input_sha256=inputs, investment_results_modified=False)


if __name__ == '__main__':
    result = scientific_audit()
    print(json.dumps(result, indent=2, allow_nan=False))
