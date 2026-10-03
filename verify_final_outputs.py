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
    run=json.loads((sim/'characteristic_factor_run.json').read_text())
    expected_configs={tuple(c) for c in run['configs']}
    actual_configs=set(surface[['economy','b','J','R']].itertuples(index=False,name=None))
    assert expected_configs==actual_configs and len(actual_configs)==9
    assert run['headline_economy']=='R1'
    assert run['source_sha256']==digest(Path(__file__).with_name('simulation_characteristic_factor.py'))
    assert len(surface)==9*10*360 and len(oracle)==90 and len(validation)==90
    assert not surface.duplicated(keys+['lambda']).any()
    assert not oracle.duplicated(keys).any() and not validation.duplicated(keys).any()
    assert (surface.R>=200).all()
    for economy in ['R1','NL']:
        headline=surface[(surface.economy==economy)&(surface.b==1.5)&(surface.J==2000)]
        assert len(headline)==3600 and (headline.R==500).all()
    assert np.isfinite(surface.select_dtypes('number')).all().all()
    assert (surface.mean_regret>0).all() and (surface.regret_mcse>0).all()
    expected=[]
    for key,d in surface.groupby(keys):
        d=d.sort_values('lambda');idx=int(d.mean_regret.to_numpy().argmin())
        assert 0<idx<len(d)-1 and len(d)>=300 and d['lambda'].min()>0
        assert (np.diff(d.population_C)<0).all()
        assert d.mean_empirical_C.max()<=min(key[2],key[4])+1e-7
        if key[0]=='R1':
            j=np.arange(1,key[2]+1,dtype=float);mu=.0004*j**(-key[1])
            shape=1/(np.sqrt(j)*np.log1p(j))
            theta=shape*np.sqrt(.16/np.dot(mu,shape**2))
            lam=d['lambda'].to_numpy()
            expected_bias=np.sum((mu*theta**2)[:,None]*(lam/(mu[:,None]+lam))**2,axis=0)
            # Subtraction in the saved near-zero ridge bias loses relative precision.
            np.testing.assert_allclose(d.population_ridge_regret,expected_bias,rtol=1e-10,atol=1e-22)
            np.testing.assert_allclose(d.population_C,np.sum(mu[:,None]/(mu[:,None]+lam),axis=0),rtol=1e-12)
        np.testing.assert_allclose(d.population_C_over_T,d.population_C/key[4],rtol=1e-12)
        expected.append(dict(zip(keys,key),lambda_expected=d['lambda'].iloc[idx],regret_expected=d.mean_regret.iloc[idx]))
    matched=oracle.merge(pd.DataFrame(expected),on=keys,validate='one_to_one')
    np.testing.assert_allclose(matched.lambda_oracle,matched.lambda_expected,rtol=1e-12)
    np.testing.assert_allclose(matched.regret_oracle,matched.regret_expected,rtol=1e-12)
    assert all(x['passed'] and x['long_path']['passed'] for x in audits)
    for a in audits:
        if a['economy']=='R1':
            assert a['a']==.5 and a['log_power']==1
            assert a['target_shape']=='1/(sqrt(j)*log(j+1))'
        assert a['theta_S_theta']<1 and a['min_eigenvalue_V_F']>0
        assert max(max(m['orthogonality_error'],m['residual_projection_error'],m['payoff_error']) for m in a['months'])<1e-10
    performance=pd.read_parquet(sim/'characteristic_factor_performance_surface.parquet')
    performance_audit=json.loads((sim/'characteristic_factor_performance_audit.json').read_text())
    assert len(performance)==3600 and performance.R.eq(500).all()
    assert np.isfinite(performance.select_dtypes('number')).all().all()
    assert performance_audit['source_sha256']==digest(Path(__file__).with_name('simulation_performance.py'))
    assert performance_audit['all_replicates_matched_existing_ridge_risk']
    assert performance_audit['max_loss_identity_error']<1e-10
    matched_performance=performance.merge(surface[keys+['lambda','mean_regret']],on=keys+['lambda'],validate='one_to_one')
    assert len(matched_performance)==3600
    np.testing.assert_allclose(matched_performance.mean_population_loss,.84+matched_performance.mean_regret,rtol=1e-11)
    assert (abs(performance.mean_population_sharpe)<=np.sqrt(12*.16/.84)+1e-12).all()
    assert (performance.mean_population_loss>=.84).all()
    selection=pd.read_csv(sim/'characteristic_factor_performance_selection.csv')
    selected=selection.merge(validation[(validation.economy=='R1')&(validation.b==1.5)&(validation.J==2000)],on='T',validate='one_to_one')
    np.testing.assert_allclose(selected.lambda_median,selected.median_lambda_validation,rtol=1e-12)
    np.testing.assert_allclose(selected.C_median/selected['T'],selected.median_population_C_over_T,rtol=1e-12)
    np.testing.assert_allclose(selected.mean_selected_loss,.84+selected.mean_validation_regret,rtol=1e-11)
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
        'simulation_appendix_smooth_nonlinear_b150',
        'simulation_dgp_01_loss_complexity','simulation_dgp_02_sharpe_complexity',
        'simulation_dgp_03_loss_heatmap','simulation_dgp_04_selected_complexity_lambda',
        'simulation_dgp_four_panel_overview',
        'empirical_E1_managed_payoff_spectrum','empirical_E2_row_normalized_heatmap',
        'empirical_E3_common_period_window_tradeoff','empirical_appendix_log_ratio_heatmap','empirical_appendix_raw_heatmap']
    for name in expected_figures:assert (out/'figures'/(name+'.png')).stat().st_size>10000
    manifest=json.loads((out/'manifest_simulation_figures.json').read_text())
    assert manifest['source_sha256']==digest(Path(__file__).with_name('render_characteristic_factor.py'))
    for name,checksum in manifest['inputs'].items():
        assert digest(sim/name)==checksum
    for name,checksum in manifest['outputs'].items():
        assert digest(out/'figures'/name)==checksum
    performance_manifest=json.loads((out/'manifest_simulation_performance.json').read_text())
    assert performance_manifest['source_sha256']==digest(Path(__file__).with_name('render_simulation_performance.py'))
    for section in ['inputs','outputs']:
        for name,checksum in performance_manifest[section].items():assert digest(out/name)==checksum
    rank=json.loads((sim/'characteristic_factor_finite_J_check.json').read_text())
    report=dict(aggregate_consistency_passed=True,finite_J_stability_passed=rank['passed'],
        simulation_surface_rows=len(surface),simulation_configurations=len(actual_configs),simulation_T_values=10,
        positive_lambda_values=360,minimum_replications=200,headline_replications=500,
        headline_economy='R1',performance_figures_verified=True,total_independent_paths=sum(c[3] for c in actual_configs),
        finite_J_by_economy=rank['economies'],
        empirical_common_months=372,formation_only_reconstruction_passed=True,
        unresolved_stock_payoffs=timing['unresolved_payoffs'],
        empirical_evidence_status='Explicitly isolated complete-payoff sensitivity; pristine real-time evidence unavailable.',
        figure_files={name:digest(out/'figures'/(name+'.png')) for name in expected_figures})
    write_json(out/'final_acceptance_checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':verify()
