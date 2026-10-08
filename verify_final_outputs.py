"""Independent consistency checks of saved scientific outputs (no model refitting)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import write_json,digest


def verify(output='outputs'):
    out=Path(output);tables=out/'tables'
    timing=json.loads((tables/'formation_timing_audit.json').read_text())
    assert timing['future_availability_perturbation_rank_error']==0
    assert timing['selected_names_exactly_match']
    common=pd.read_csv(tables/'common_period_window_comparison.csv')
    monthly=pd.read_csv(tables/'selected_monthly_payoffs_matern32.csv',parse_dates=['return_date'])
    for row in common.itertuples():
        r=monthly[(monthly.T_months==row.T_months)&monthly.return_date.between('1994-01-31','2024-12-31')].raw_excess_return
        assert len(r)==row.months==372
        np.testing.assert_allclose(row.response_one_loss,np.mean((1-r)**2))
        np.testing.assert_allclose(row.annualized_sharpe,np.sqrt(12)*r.mean()/r.std(ddof=1))
    year=pd.read_csv(tables/'validation_vs_oracle_by_year.csv')
    assert (year.validation_regret>=-1e-12).all()
    for metric in ['delta','log_ratio']:
        cells=pd.read_csv(tables/f'empirical_heatmap_cells_{metric}.csv')
        assert cells.plot_value.min()>=-1e-12
    expected_figures=['empirical_E1_managed_payoff_spectrum','empirical_E2_row_normalized_heatmap',
        'empirical_E3_common_period_window_tradeoff','empirical_appendix_log_ratio_heatmap','empirical_appendix_raw_heatmap']
    for name in expected_figures:
        assert (out/'figures'/(name+'.png')).stat().st_size>10000
    report=dict(empirical_common_months=372, formation_only_reconstruction_passed=True,
        unresolved_stock_payoffs=timing['unresolved_payoffs'],
        empirical_evidence_status='Explicitly isolated complete-payoff sensitivity; pristine real-time evidence unavailable.',
        figure_files={name:digest(out/'figures'/(name+'.png')) for name in expected_figures})
    write_json(out/'final_acceptance_checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':verify()
