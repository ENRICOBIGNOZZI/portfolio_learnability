"""Independent verification of saved Monte Carlo artifacts; no refitting or tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.config.design import design_for
from simulations.provenance import ROOT, require, json_write


def verify(output, profile='paper'):
    output = Path(output)
    design = design_for(profile)
    manifest = json.loads((output/'run_manifest.json').read_text())
    require(manifest['scope']=='full', 'A partial run cannot certify the full requested experiment.')
    require(manifest['config_hash']==design.digest, 'Verification gate at verify.py:18')
    for name, expected in manifest['source_hashes'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name)
    for name, expected in manifest.get('derived_output_source_hashes', {}).items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name)
    expected_cases = {e.name for e in design.cases}
    require(set(manifest['completed_environments'])==expected_cases, 'Verification gate at verify.py:24')
    baseline = json.loads((output/'audit/baseline/dgp_audit.json').read_text())
    require(baseline['passed'] and all(baseline['acceptance'].values()), 'Verification gate at verify.py:26')
    cond = baseline['conditional_audit']
    require(cond['beta_norm_max_error']<1e-12 and cond['balance_operator_max_error']<1e-12, 'Verification gate at verify.py:28')
    require(cond['normal_equation_max_absolute_error']<1e-11 and cond['E2a_max_log_mgf']<=.5+1e-12, 'Verification gate at verify.py:29')
    require(baseline['E5']['operator_and_eigenvalue_sandwich_passed'], 'Verification gate at verify.py:30')
    records = []
    penalties = np.asarray(design.penalties)
    for env in design.cases:
        directory = output/'data'/env.name
        pop = json.loads((directory/'population.json').read_text())
        curves = pd.read_csv(directory/'curves.csv')
        main = pd.read_csv(directory/'main_results.csv')
        rates = pd.read_csv(directory/'rates.csv')
        require(len(curves)==len(design.T)*len(penalties), 'Verification gate at verify.py:40')
        require(len(main)==len(design.T), 'Verification gate at verify.py:41')
        require(not curves.duplicated(['T', 'lambda']).any(), 'Verification gate at verify.py:42')
        if env.name!='baseline':
            audit = json.loads((directory/'environment_audit.json').read_text())
            require(audit['passed'], 'Verification gate at verify.py:45')
            if env.nu != 1.5:
                require(audit['actual_kernel_conditional_audit']['passed'], 'Verification gate at verify.py:47')
                require(len(audit['actual_kernel_conditional_audit']['states'])>=24, 'Verification gate at verify.py:48')
            if env.variant!='baseline':
                require(not audit['baseline_theorem_claim'], 'Verification gate at verify.py:50')
                require(len(audit['states'])>=24, 'Verification gate at verify.py:51')
        regret, choice, selected_loss, selected_sr = [], [], [], []
        empirical_complexity = []
        date_count = 0
        date_maxima = np.zeros(4)
        for r in range(env.replications):
            path = directory/'replications'/f'{r:04d}.npz'
            with np.load(path, allow_pickle=False) as saved:
                require(int(saved['replication'])==r, 'Verification gate at verify.py:59')
                require(str(saved['run_hash'])==pop['run_hash'], 'Verification gate at verify.py:60')
                require(int(saved['train_seed'])!=int(saved['test_seed']), 'Verification gate at verify.py:61')
                require(int(saved['date_audit'][0])==max(design.T)+design.oos_periods, 'Verification gate at verify.py:62')
                require(str(saved['date_audit_source_hash'])==hashlib.sha256(Path('simulations/diagnostics/path_integrity.py').read_bytes()).hexdigest(), 'Verification gate at verify.py:63')
                date_count += int(saved['date_audit'][0])
                date_maxima = np.maximum(date_maxima, saved['date_audit'][1:])
                require(saved['date_audit'][1]<1e-12 and saved['date_audit'][2]<1e-12, 'Verification gate at verify.py:66')
                require(saved['date_audit'][4]<1e-11, 'Verification gate at verify.py:67')
                if env.variant!='heteroskedastic':
                    require(saved['date_audit'][3]<1e-11, 'Verification gate at verify.py:69')
                for key in ('regret','complexity','population_loss','population_sharpe','oos_loss','oos_sharpe','validation_loss'):
                    require(saved[key].shape==(len(design.T), len(penalties)), 'Verification gate at verify.py:71')
                    require(np.isfinite(saved[key]).all(), 'Verification gate at verify.py:72')
                expected_choice = np.argmin(saved['validation_loss'], axis=1)
                np.testing.assert_array_equal(saved['choice'], expected_choice)
                require(np.all(np.diff(saved['complexity'], axis=1)<=1e-8), 'Verification gate at verify.py:75')
                require(np.all(saved['complexity']<=np.minimum(design.T, env.rank)[:, None]+1e-6), 'Verification gate at verify.py:76')
                require(saved['regret'].min()>=-1e-9, 'Verification gate at verify.py:77')
                require(np.abs(saved['population_sharpe']).max()<=pop['reference_sharpe']+1e-9, 'Verification gate at verify.py:78')
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
        require((rates.bootstrap_draws==design.bootstrap_replications).all(), 'Verification gate at verify.py:94')
        require((rates.bootstrap_low<=rates.bootstrap_high).all(), 'Verification gate at verify.py:95')
        sharpe_rates = rates.loc[rates.quantity.str.endswith('sharpe_gap'), 'theoretical_slope']
        np.testing.assert_allclose(sharpe_rates, -env.theoretical_b/(env.theoretical_b+1), rtol=1e-12)
        expected_windows = {'full','upper_half','largest_four'} if profile=='paper' else {'full','largest_four'}
        require(set(rates.window)==expected_windows, 'Verification gate at verify.py:99')
        for name in ('four_panel','oos_loss_uncertainty','oos_sharpe_uncertainty'):
            for suffix in ('pdf','png'):
                require((output/'figures'/f'{env.name}_{name}.{suffix}').stat().st_size>1000, 'Verification gate at verify.py:102')
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
            require((output/'tables'/f'{table}.{suffix}').stat().st_size>0, 'Verification gate at verify.py:113')
    if profile=='paper':
        approximation = json.loads((output/'audit/approximation_rank.json').read_text())
        require(approximation['passed'], 'Verification gate at verify.py:116')
        quadrature = json.loads((output/'audit/quadrature_sensitivity.json').read_text())
        require(quadrature['passed'] and quadrature['replications']==20, 'Verification gate at verify.py:118')
        require(quadrature['population_groups']==[8192,32768], 'Verification gate at verify.py:119')
        require(quadrature['relative_tolerance']==.05, 'Verification gate at verify.py:120')
        require(quadrature['max_absolute_relative_difference']<=.05, 'Verification gate at verify.py:121')
        for name in ('main_robustness','appendix_N_robustness','appendix_persistence','appendix_SNR',
                     'appendix_spectral_b','appendix_approximation_rank','appendix_empirical_ranks','appendix_heteroskedastic'):
            require((output/'figures'/f'{name}.pdf').stat().st_size>1000, 'Verification gate at verify.py:124')
    cleanup = json.loads(Path('simulations/outputs/audit/cleanup_inventory.json').read_text())
    for item in cleanup:
        require(not Path(item['path']).exists(), f"Legacy file survived: {item['path']}")
    from simulations.diagnostics.reconstruct import verify as reconstruct
    independent = reconstruct(output, full=True)
    report = {'independent_reconstruction': independent, 'passed':True,'profile':profile,'environments':records,'total_replications':sum(x['replications_verified'] for x in records),
              'legacy_cleanup_verified':True,'paper_integration': 'separately verified; not implied by this numerical audit'}
    destination = output/'audit'
    destination.mkdir(parents=True, exist_ok=True)
    json_write(destination/'final_numerical_verification.json', report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','paper'],default='paper')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.output or Path('simulations/outputs')/(args.profile+'_v2'),args.profile),indent=2))
