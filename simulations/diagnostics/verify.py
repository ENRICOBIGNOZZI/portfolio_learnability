"""Independent verification of saved Monte Carlo artifacts; no refitting or tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.config.design import design_for


def verify(output, profile='paper'):
    output = Path(output)
    design = design_for(profile)
    manifest = json.loads((output/'run_manifest.json').read_text())
    assert manifest['scope']=='full', 'A partial run cannot certify the full requested experiment.'
    assert manifest['config_hash']==design.digest
    for name, expected in manifest['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
    for name, expected in manifest.get('derived_output_source_hashes', {}).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
    expected_cases = {e.name for e in design.cases}
    assert set(manifest['completed_environments'])==expected_cases
    baseline = json.loads(Path('simulations/outputs/audit/baseline/dgp_audit.json').read_text())
    assert baseline['passed'] and all(baseline['acceptance'].values())
    cond = baseline['conditional_audit']
    assert cond['beta_norm_max_error']<1e-12 and cond['balance_operator_max_error']<1e-12
    assert cond['normal_equation_max_absolute_error']<1e-11 and cond['E2a_max_log_mgf']<=.5+1e-12
    assert baseline['E5']['operator_and_eigenvalue_sandwich_passed']
    assert baseline['E5']['fits']['managed']['compatible']
    records = []
    penalties = np.asarray(design.penalties)
    for env in design.cases:
        directory = output/'data'/env.name
        pop = json.loads((directory/'population.json').read_text())
        curves = pd.read_csv(directory/'curves.csv')
        main = pd.read_csv(directory/'main_results.csv')
        rates = pd.read_csv(directory/'rates.csv')
        assert len(curves)==len(design.T)*len(penalties)
        assert len(main)==len(design.T)
        assert not curves.duplicated(['T', 'lambda']).any()
        if env.name!='baseline':
            audit = json.loads((directory/'environment_audit.json').read_text())
            assert audit['passed']
            if env.nu != 1.5:
                assert audit['actual_kernel_conditional_audit']['passed']
                assert len(audit['actual_kernel_conditional_audit']['states'])>=24
            if env.variant!='baseline':
                assert not audit['baseline_theorem_claim']
                assert len(audit['states'])>=24
        regret, choice, selected_loss, selected_sr = [], [], [], []
        empirical_complexity = []
        date_count = 0
        date_maxima = np.zeros(4)
        for r in range(env.replications):
            path = directory/'replications'/f'{r:04d}.npz'
            with np.load(path, allow_pickle=False) as saved:
                assert int(saved['replication'])==r
                assert str(saved['run_hash'])==pop['run_hash']
                assert int(saved['train_seed'])!=int(saved['test_seed'])
                assert int(saved['date_audit'][0])==max(design.T)+design.oos_periods
                assert str(saved['date_audit_source_hash'])==hashlib.sha256(Path('simulations/diagnostics/path_integrity.py').read_bytes()).hexdigest()
                date_count += int(saved['date_audit'][0])
                date_maxima = np.maximum(date_maxima, saved['date_audit'][1:])
                assert saved['date_audit'][1]<1e-12 and saved['date_audit'][2]<1e-12
                assert saved['date_audit'][4]<1e-11
                if env.variant!='heteroskedastic':
                    assert saved['date_audit'][3]<1e-11
                for key in ('regret','complexity','population_loss','population_sharpe','oos_loss','oos_sharpe','validation_loss'):
                    assert saved[key].shape==(len(design.T), len(penalties))
                    assert np.isfinite(saved[key]).all()
                expected_choice = np.argmin(saved['validation_loss'], axis=1)
                np.testing.assert_array_equal(saved['choice'], expected_choice)
                assert np.all(np.diff(saved['complexity'], axis=1)<=1e-8)
                assert np.all(saved['complexity']<=np.minimum(design.T, env.rank)[:, None]+1e-6)
                assert saved['regret'].min()>=-1e-9
                assert np.abs(saved['population_sharpe']).max()<=pop['reference_sharpe']+1e-9
                for t, T in enumerate(design.T):
                    population_bias = curves[curves['T']==T].sort_values('lambda').population_bias.to_numpy()
                    np.testing.assert_allclose(saved['regret'][t], population_bias+saved['estimation_norm'][t]+saved['cross_term'][t], atol=1e-9)
                regret.append(saved['regret'])
                empirical_complexity.append(saved['complexity'])
                choice.append(expected_choice)
                selected_loss.append(saved['population_loss'][np.arange(len(design.T)), expected_choice])
                selected_sr.append(saved['population_sharpe'][np.arange(len(design.T)), expected_choice])
        regret, choices = np.stack(regret), np.stack(choice)
        oracle = regret.mean(axis=0).argmin(axis=1)
        np.testing.assert_allclose(main.oracle_lambda, penalties[oracle], rtol=1e-12)
        np.testing.assert_allclose(main.oracle_regret, regret.mean(axis=0)[np.arange(len(design.T)), oracle], rtol=1e-10)
        np.testing.assert_allclose(main.selected_population_loss, np.mean(selected_loss, axis=0), rtol=1e-10)
        np.testing.assert_allclose(main.selected_population_sharpe, np.mean(selected_sr, axis=0), rtol=1e-10)
        np.testing.assert_allclose(main.boundary_selection_frequency, np.mean((choices==0)|(choices==len(penalties)-1), axis=0))
        assert (rates.bootstrap_draws==design.bootstrap_replications).all()
        assert (rates.bootstrap_low<=rates.bootstrap_high).all()
        sharpe_rates = rates.loc[rates.quantity.str.endswith('sharpe_gap'), 'theoretical_slope']
        np.testing.assert_allclose(sharpe_rates, -env.theoretical_b/(env.theoretical_b+1), rtol=1e-12)
        expected_windows = {'full','upper_half','largest_four'} if profile=='paper' else {'full','largest_four'}
        assert set(rates.window)==expected_windows
        for name in ('four_panel','oos_loss_uncertainty','oos_sharpe_uncertainty'):
            for suffix in ('pdf','png'):
                assert (output/'figures'/f'{env.name}_{name}.{suffix}').stat().st_size>1000
        records.append({'environment':env.name,'replications_verified':env.replications,'T_values':len(design.T),
                        'penalties':len(penalties),'oracle_boundary_T_count':int(main.oracle_on_boundary.sum()),
                        'all_production_dates_verified': date_count,
                        'beta_norm_max_error': float(date_maxima[0]),
                        'balance_max_error': float(date_maxima[1]),
                        'baseline_policy_normal_equation_max_error': float(date_maxima[2]),
                        'policy_coordinate_max_error': float(date_maxima[3]),
                        'pricing_exact_claim': env.variant!='heteroskedastic', 'passed':True})
    for table in ('table1_calibration','table2_assumptions','table3_main_results','table4_rates','table5_robustness'):
        for suffix in ('csv','tex'):
            assert (output/'tables'/f'{table}.{suffix}').stat().st_size>0
    if profile=='paper':
        approximation = json.loads((output/'audit/approximation_rank.json').read_text())
        assert approximation['passed']
        quadrature = json.loads((output/'audit/quadrature_sensitivity.json').read_text())
        assert quadrature['passed'] and quadrature['replications']==20
        assert quadrature['population_groups']==[8192,32768]
        assert quadrature['relative_tolerance']==.05
        assert quadrature['max_absolute_relative_difference']<=.05
        for name in ('main_robustness','appendix_N_robustness','appendix_persistence','appendix_SNR',
                     'appendix_spectral_b','appendix_approximation_rank','appendix_empirical_ranks','appendix_heteroskedastic'):
            assert (output/'figures'/f'{name}.pdf').stat().st_size>1000
    cleanup = json.loads(Path('simulations/outputs/audit/cleanup_inventory.json').read_text())
    for item in cleanup:
        assert not Path(item['path']).exists(), f"Legacy file survived: {item['path']}"
    report = {'passed':True,'profile':profile,'environments':records,'total_replications':sum(x['replications_verified'] for x in records),
              'legacy_cleanup_verified':True,'paper_integration': 'separately verified; not implied by this numerical audit'}
    destination = output/'audit'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'final_numerical_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','paper'],default='paper')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.output or Path('simulations/outputs')/args.profile,args.profile),indent=2))
