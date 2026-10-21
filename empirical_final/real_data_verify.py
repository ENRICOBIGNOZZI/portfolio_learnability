"""Independent temporal, aggregate accounting, and artifact integrity checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import digest


def verify(out):
    audit, tables = out/'audit', out/'tables'
    execution = json.loads((audit/'execution.json').read_text())
    protocol = json.loads((audit/'protocol.json').read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in execution['identity']['code'].items():
        assert digest(root/'empirical_final'/name) == expected, name
    for name, expected in execution['identity']['dependencies'].items():
        assert digest(root/name) == expected, name
    for name, expected in execution['artifact_sha256'].items():
        assert digest(out/name) == expected, name
    if (audit/'analysis.json').exists():
        analysis = json.loads((audit/'analysis.json').read_text())
        assert analysis['code_sha256'] == digest(root/'empirical_final/real_data_analysis.py')
        assert analysis['source_execution_sha256'] == digest(audit/'execution.json')
        for name, expected in analysis['outputs'].items():
            assert digest(out/name) == expected, name
    monthly = pd.read_csv(tables/'monthly_policies.csv', parse_dates=['formation_date', 'return_date'],
                          dtype={name: 'category' for name in ['scenario', 'experiment', 'policy', 'variant']})
    annual = pd.read_csv(tables/'annual_policies.csv', parse_dates=['last_train_payoff', 'first_test_payoff'])
    summary = pd.read_csv(tables/'summary.csv')
    universe = pd.read_csv(tables/'universe.csv')
    assert universe.unknown_returns.sum() == execution['unresolved_source_payoffs'] == 7489
    assert universe.stocks.sum() == 2504402
    assert not monthly.duplicated(['scenario', 'experiment', 'policy', 'formation_date']).any()
    assert (annual.last_train_payoff < annual.first_test_payoff).all()
    assert np.array_equal(monthly.return_date, monthly.formation_date+pd.offsets.MonthEnd(1))
    np.testing.assert_allclose(monthly.Q, (1-monthly.raw_return)**2, rtol=1e-11, atol=1e-10)
    scale = json.loads((audit/'scale.json').read_text())['scale']
    np.testing.assert_allclose(monthly.scaled_gross_excess, scale*monthly.raw_return, rtol=1e-11, atol=1e-10)
    assert (monthly.unknown_abs_weight >= 0).all()
    assert (monthly.unknown_abs_weight <= monthly.gross_exposure+1e-9).all()
    cash = pd.read_csv(root/'data/risk_free.csv', parse_dates=['return_date']).set_index('return_date').rf
    assert digest(root/'data/risk_free.csv') == execution['identity']['cash_sha256']
    assert digest(root/'outputs/empirical_three_experiments_20261009/tables/french_factors.csv') == execution['identity']['factors_sha256']
    valid = np.isfinite(monthly.net_excess)
    m = monthly.loc[valid]
    rf = cash.reindex(m.return_date).to_numpy()
    expected_net = m.scaled_gross_excess-m.trading_fee*(1+rf+m.scaled_gross_excess)-m.borrowing_fee
    np.testing.assert_allclose(m.net_excess, expected_net, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(m.trading_fee, protocol['trade_bps']/1e4*m.turnover, rtol=1e-9, atol=1e-10)
    complete_dates = pd.date_range(f"{protocol['start_year']}-01-31", f"{protocol['end_year']}-12-31", freq='ME')
    summary_index = summary.set_index(['scenario', 'experiment', 'policy'])
    for scenario in monthly.scenario.cat.categories:
        baseline = monthly[(monthly.scenario == scenario) & (monthly.experiment == 'E') &
                           (monthly.policy == 'ridge_all')].set_index('return_date')
        for experiment, policy in [('B', 'Nall_T240'), ('C', 'G130')]:
            equivalent = monthly[(monthly.scenario == scenario) & (monthly.experiment == experiment) &
                                 (monthly.policy == policy)].set_index('return_date')
            assert baseline.index.equals(equivalent.index)
            np.testing.assert_allclose(baseline[['raw_return', 'net_excess']],
                                       equivalent[['raw_return', 'net_excess']], rtol=1e-12, atol=1e-12, equal_nan=True)
    for key, group in monthly.groupby(['scenario', 'experiment', 'policy'], observed=True, sort=False):
        row = summary_index.loc[key]
        assert len(group) == row.months
        np.testing.assert_allclose(group.Q.mean(), row.raw_Q, rtol=1e-9, atol=1e-10)
        if row.complete_common_period:
            assert np.array_equal(group.formation_date, complete_dates)
        if row.net_status == 'complete':
            assert row.complete_common_period and np.isfinite(group.net_excess).all()
        else:
            assert np.isnan(row.net_sharpe)
    increments = pd.read_csv(tables/'E_incremental_directions.csv')
    np.testing.assert_allclose(increments.delta_OOS_loss,
                               increments.OOS_cross_term+increments.OOS_added_square, rtol=1e-9, atol=1e-9)
    match = json.loads((audit/'A_match.json').read_text())
    if match['pair']:
        candidates = pd.read_csv(tables/'A_candidates.csv').set_index('variant')
        sharpes = candidates.loc[match['pair'], 'validation_sharpe']
        assert sharpes.max()-sharpes.min() <= match['tolerance_annualized_sharpe']+1e-12
    expected_experiments = set('ABCDE') if match['pair'] else set('BCDE')
    assert set(monthly.experiment) == expected_experiments
    result = dict(passed=True, code_sha256=digest(Path(__file__)), monthly_policy_observations=len(monthly),
                  annual_policy_fits=len(annual), policy_scenario_paths=len(summary),
                  common_test_months=len(complete_dates), formation_stock_months=int(universe.stocks.sum()),
                  unresolved_payoffs=int(universe.unknown_returns.sum()), identified_performance=False,
                  checks=['source and artifact identities', 'chronology', 'stock-versus-managed payoffs (main execution)',
                          'full common-period calendars', 'response-one loss', 'common scale',
                          'independent net return accounting', 'cost equals rate times turnover',
                          'identical baseline policies across B/C/E',
                          'unknown-payoff exposure', 'infeasible/invalid paths not ranked',
                          'direction incremental loss identity', 'historical matching tolerance'])
    (audit/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/real_data_ABCDE_20261021')
    verify(parser.parse_args().out)
