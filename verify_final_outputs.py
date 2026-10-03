"""Independent consistency checks of saved scientific outputs (no model refitting)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import write_json,digest


def verify(output='outputs'):
    out=Path(output);sim=out/'simulations';tables=out/'tables'
    surface=pd.read_parquet(sim/'characteristic_factor_full_surface.parquet')
    oracle=pd.read_csv(sim/'characteristic_factor_oracle_by_T.csv')
    validation=pd.read_csv(sim/'characteristic_factor_validation_by_T.csv')
    audits=json.loads((sim/'characteristic_factor_dgp_audit.json').read_text())
    keys=['economy','b','J','R','T']
    assert len(surface)==6*10*360 and len(oracle)==60 and len(validation)==60
    assert (surface.R>=200).all()
    assert (surface[(surface.economy=='NL')&(surface.b==1.5)&(surface.J==2000)].R>=500).all()
    assert np.isfinite(surface.select_dtypes('number')).all().all()
    assert (surface.mean_regret>0).all() and (surface.regret_mcse>0).all()
    expected=[]
    for key,d in surface.groupby(keys):
        d=d.sort_values('lambda');idx=int(d.mean_regret.to_numpy().argmin())
        assert 0<idx<len(d)-1 and len(d)>=300 and d['lambda'].min()>0
        assert (np.diff(d.population_C)<0).all()
        assert d.mean_empirical_C.max()<=min(key[2],key[4])+1e-7
        np.testing.assert_allclose(d.population_C_over_T,d.population_C/key[4],rtol=1e-12)
        expected.append(dict(zip(keys,key),lambda_expected=d['lambda'].iloc[idx],regret_expected=d.mean_regret.iloc[idx]))
    matched=oracle.merge(pd.DataFrame(expected),on=keys,validate='one_to_one')
    np.testing.assert_allclose(matched.lambda_oracle,matched.lambda_expected,rtol=1e-12)
    np.testing.assert_allclose(matched.regret_oracle,matched.regret_expected,rtol=1e-12)
    assert all(x['passed'] and x['long_path']['passed'] for x in audits)
    for a in audits:
        assert a['theta_S_theta']<1 and a['min_eigenvalue_V_F']>0
        assert max(max(m['orthogonality_error'],m['residual_projection_error'],m['payoff_error']) for m in a['months'])<1e-10
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
    expected_figures=['simulation_learnability_law_b150','simulation_heatmap_absolute_complexity_b150',
        'simulation_spectrum_comparison_b125_b150_b200','simulation_linear_sanity_check',
        'empirical_E1_managed_payoff_spectrum','empirical_E2_row_normalized_heatmap',
        'empirical_E3_common_period_window_tradeoff','empirical_appendix_log_ratio_heatmap','empirical_appendix_raw_heatmap']
    for name in expected_figures:assert (out/'figures'/(name+'.png')).stat().st_size>10000
    rank=json.loads((sim/'characteristic_factor_finite_J_check.json').read_text())
    report=dict(aggregate_consistency_passed=True,finite_J_stability_passed=rank['passed'],
        simulation_surface_rows=len(surface),simulation_configurations=6,simulation_T_values=10,
        positive_lambda_values=360,minimum_replications=200,headline_replications=500,
        empirical_common_months=372,formation_only_reconstruction_passed=True,
        unresolved_stock_payoffs=timing['unresolved_payoffs'],
        empirical_evidence_status='Explicitly isolated complete-payoff sensitivity; pristine real-time evidence unavailable.',
        figure_files={name:digest(out/'figures'/(name+'.png')) for name in expected_figures})
    write_json(out/'final_acceptance_checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':verify()
