"""Requested publication tables built exclusively from completed saved results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.config.design import design_for


def emit(frame, directory, name, caption, label):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory/f'{name}.csv', index=False)
    display = frame.copy()
    if name=='appendix_spectral_rates':
        display['environment'] = display['environment'].replace(
            {'baseline':'nu = 1.5', 'slow_spectrum':'nu = 0.5', 'fast_spectrum':'nu = 2.5'})
        display.columns = ['Kernel', 'Theory', 'Fitted', 'Theory', 'Fitted', 'Theory', 'Fitted']
    tabular = display.to_latex(index=False, escape=True, float_format=lambda x: f'{x:.4g}', na_rep='--')
    if name=='appendix_spectral_rates':
        tabular = tabular.replace('\\toprule\n', '\\toprule\n'
            r' & \multicolumn{2}{c}{Spectral decay} & \multicolumn{2}{c}{Regret exponent} & \multicolumn{2}{c}{Complexity exponent} \\'+'\n'
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'+'\n')
    if name=='table1_calibration':
        for source, target in [(r'c\_beta', r'$c_\beta$'), (r'mu\_F', r'$\mu_F$'),
                               (r'sigma\_epsilon', r'$\sigma_\varepsilon$'), ('rho', r'$\rho$'),
                               ('Matern nu', r'Mat\'ern $\nu$')]:
            tabular = tabular.replace(source, target)
    body = ('\\begingroup\\small\n\\setlength{\\tabcolsep}{4pt}\n'
            '\\ifdefined\\simtablebox\\else\\newsavebox{\\simtablebox}\\fi\n'
            '\\sbox{\\simtablebox}{%\n'+tabular+'}\n'
            '\\ifdim\\wd\\simtablebox>\\linewidth\n'
            '\\resizebox{\\linewidth}{!}{\\usebox{\\simtablebox}}\n'
            '\\else\\usebox{\\simtablebox}\\fi\n\\endgroup\n')
    text = ('\\begin{table}[p]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
            +body+'\\end{table}\n')
    (directory/f'{name}.tex').write_text(text)


def emit_main_results(frame, directory):
    """Two readable panels retain every requested result in one numbered table."""
    directory = Path(directory)
    frame.to_csv(directory/'table3_main_results.csv', index=False)
    panels = [('A. Population performance', ['T', 'Oracle loss', 'Selected loss', 'Oracle SR', 'Selected SR', 'Regret ratio']),
              ('B. Effective complexity and regularization', ['T', 'Oracle C', 'Selected C', 'Oracle lambda', 'Selected lambda'])]
    text = (r'\begin{table}[p]\centering\small'+'\n'
            r'\caption{Monte Carlo mean population performance. Validation uses no population information; the oracle minimizes mean population regret over the fixed grid.}'+'\n'
            r'\label{tab:sim_main}'+'\n')
    for title, columns in panels:
        text += r'\textit{'+title+r'}\par\smallskip'+'\n'
        text += frame[columns].to_latex(index=False, escape=True, float_format=lambda x: f'{x:.4g}')
        text += r'\par\medskip'+'\n'
    (directory/'table3_main_results.tex').write_text(text+r'\end{table}'+'\n')


def build_tables(output, profile='paper'):
    output = Path(output)
    design = design_for(profile)
    table_dir = output/'tables'
    cases = [env for env in design.cases if (output/'data'/env.name/'main_results.csv').exists()]
    if len(cases) != len(design.cases):
        raise ValueError('Cannot publish full-profile tables with missing environments.')
    p = cases[0].parameters()
    calibration = pd.DataFrame({'Parameter': ['N', 'T', 'D', 'Economic factors', 'rho', 'c_beta', 'mu_F',
                       'sigma_epsilon', 'Matern nu', 'Length scale', 'Theoretical b', 'Population Sharpe',
                       'Nystrom rank', 'Headline replications', 'Independent OOS length'],
                       'Value': [str(p.N), ', '.join(map(str, design.T)), '6', '3', '.95', str(p.c_beta),
                                 str(p.mu_F), str(p.sigma_eps), '1.5', '1', '1.5', f'{p.sr_star:.8f}',
                                 str(cases[0].rank), str(cases[0].replications), str(design.oos_periods)]})
    emit(calibration, table_dir, 'table1_calibration', 'Baseline calibration and frozen Monte Carlo design.', 'tab:sim_calibration')
    audit = json.loads(Path('simulations/outputs/audit/baseline/dgp_audit.json').read_text())
    if not audit['passed']:
        raise ValueError('Cannot publish an assumption table from a failed audit.')
    cond = audit['conditional_audit']
    audit_rows = [
        ['E1', 'Exact: stationary stable Gaussian AR', f"spectral radius {audit['E1']['spectral_radius']:.3f}", '< 1', 'Pass'],
        ['E2(a)', 'Exact Gaussian quadratic MGF', f"max log MGF {cond['E2a_max_log_mgf']:.5g}", '<= 0.5', 'Pass'],
        ['E2(b)', 'Conditional scalar Gaussian', f"max ratio {max(x['fourth_to_second_squared_max'] for x in cond['dates']):.5g}", '<= 3', 'Pass'],
        ['E3', 'Smooth beta in Sobolev H^(9/2)', 'fixed finite linear combination', 'analytic', 'Pass'],
        ['E4', 'Exact conditional normal equation', f"max residual {cond['normal_equation_max_absolute_error']:.3g}", '< 1e-11', 'Pass'],
        ['E5', 'Loewner sandwich; theoretical b=1.5', f"slope {audit['E5']['fits']['managed']['slope']:.4f}", '-1.5 +/- 0.35', 'Pass'],
        ['E6', 'Analytic optimal Sharpe', f"MC {audit['E6']['SR_estimate']:.5f}; exact {p.sr_star:.5f}", f"abs error <= {audit['E6']['predeclared_absolute_tolerances']['SR']:.4f}", 'Pass']]
    emit(pd.DataFrame(audit_rows, columns=['Condition', 'Theoretical status', 'Diagnostic', 'Tolerance', 'Result']),
         table_dir, 'table2_assumptions', 'Baseline assumption audit. The spectral band is a finite-resolution diagnostic.', 'tab:sim_assumptions')
    main = pd.read_csv(output/'data/baseline/main_results.csv')
    selected_columns = ['T', 'oracle_population_loss', 'selected_population_loss', 'oracle_population_sharpe',
                        'selected_population_sharpe', 'oracle_complexity', 'selected_complexity', 'oracle_lambda',
                        'selected_lambda_mean', 'selected_oracle_regret_ratio']
    display = main[selected_columns].rename(columns={'oracle_population_loss':'Oracle loss', 'selected_population_loss':'Selected loss',
         'oracle_population_sharpe':'Oracle SR', 'selected_population_sharpe':'Selected SR', 'oracle_complexity':'Oracle C',
         'selected_complexity':'Selected C', 'oracle_lambda':'Oracle lambda', 'selected_lambda_mean':'Selected lambda',
         'selected_oracle_regret_ratio':'Regret ratio'})
    for column in ('Oracle loss', 'Selected loss'):
        display[column] = display[column].map(lambda value: f'{value:.6f}')
    for column in ('Oracle SR', 'Selected SR'):
        display[column] = display[column].map(lambda value: f'{value:.5f}')
    emit_main_results(display, table_dir)
    rates = pd.read_csv(output/'data/baseline/rates.csv')
    rate_display = rates[['quantity','window','theoretical_slope','OLS_slope','OLS_SE','bootstrap_low','bootstrap_high']].copy()
    rate_display['quantity'] = rate_display['quantity'].replace({
        'oracle_lambda':'Oracle lambda', 'selected_lambda':'Selected lambda',
        'oracle_complexity':'Oracle C (pop.)', 'selected_complexity':'Selected C (pop.)',
        'oracle_empirical_complexity':'Oracle C (sample)', 'selected_empirical_complexity':'Selected C (sample)',
        'oracle_regret':'Oracle regret', 'selected_regret':'Selected regret',
        'oracle_sharpe_gap':'Oracle SR gap', 'selected_sharpe_gap':'Selected SR gap'})
    rate_display['window'] = rate_display['window'].replace({'full':'Full','upper_half':'Upper half','largest_four':'Largest four'})
    rate_display = rate_display.rename(columns={'quantity':'Quantity','window':'Window','theoretical_slope':'Theory',
        'OLS_slope':'Slope','OLS_SE':'OLS SE','bootstrap_low':'CI low','bootstrap_high':'CI high'})
    emit(rate_display,
         table_dir, 'table4_rates', 'Fixed-window rate regressions with whole-replication bootstrap intervals. Theoretical slopes are benchmarks, not fitted targets.', 'tab:sim_rates')
    robustness, calibrations, spectral = [], [], []
    all_main, all_rates, all_benchmarks = [], [], []
    for env in cases:
        case = output/'data'/env.name
        result = pd.read_csv(case/'main_results.csv')
        fit = pd.read_csv(case/'rates.csv')
        all_main.append(result); all_rates.append(fit); all_benchmarks.append(pd.read_csv(case/'benchmarks.csv'))
        last = result.iloc[-1]
        pop = json.loads((case/'population.json').read_text())
        robustness.append({'Environment': env.name, 'N': env.N, 'rho': env.rho, 'nu': env.nu,
                           'Oracle regret': last.oracle_regret, 'Selected regret': last.selected_regret,
                           'Oracle C': last.oracle_complexity, 'Selected C': last.selected_complexity,
                           'Selected SR': last.selected_population_sharpe, 'Boundary frequency': last.boundary_selection_frequency,
                           'Reference': pop['reference_kind']})
        calibrations.append({**env.__dict__, 'theoretical_b': env.theoretical_b, 'reference_sharpe': pop['reference_sharpe'],
                             'T_grid': str(design.T), 'D': 6, 'K_F': 3, 'ell': 1., 'c_beta': p.c_beta,
                             'mu_F': str(env.parameters().mu_F)})
        if env.name in ('baseline', 'slow_spectrum', 'fast_spectrum'):
            spec = audit['E5'] if env.name=='baseline' else json.loads((case/'environment_audit.json').read_text())['spectrum']
            slope = lambda quantity: float(fit.loc[(fit.quantity==quantity)&(fit.window=='full'), 'OLS_slope'].iloc[0])
            spectral.append({'environment': env.name, 'theoretical_b': env.theoretical_b,
                             'fitted_spectral_b': -spec['fits']['managed']['slope'],
                             'theoretical_regret_exponent': env.theoretical_b/(env.theoretical_b+1),
                             'fitted_regret_exponent': -slope('oracle_regret'),
                             'theoretical_complexity_exponent': 1/(env.theoretical_b+1),
                             'fitted_complexity_exponent': slope('oracle_complexity')})
    robust_display = pd.DataFrame(robustness).drop(columns=['Reference','N','rho','nu'])
    robust_display['Environment'] = robust_display['Environment'].replace({
        'weak_high_noise':'Weak signal', 'strong_low_noise':'Strong signal',
        'slow_spectrum':'nu = 0.5', 'fast_spectrum':'nu = 2.5',
        'rho070':'rho = 0.70', 'rho090':'rho = 0.90', 'rho097':'rho = 0.97',
        'empirical_ranks':'Empirical ranks', 'heteroskedastic':'Heteroskedastic'})
    robust_display['Boundary frequency'] *= 100
    robust_display = robust_display.rename(columns={'Boundary frequency':'Boundary (%)'})
    emit(robust_display, table_dir, 'table5_robustness',
         'Robustness at the largest sample size. Heteroskedastic regret uses a finite-subspace reference; it is not a theorem-baseline result.', 'tab:sim_robustness')
    emit(pd.DataFrame(spectral), table_dir, 'appendix_spectral_rates', 'Spectral and learning-rate sensitivity across Matérn smoothness values. Fitted learning exponents use the full predeclared T grid.', 'tab:sim_spectral_rates')
    pd.DataFrame(calibrations).to_csv(table_dir/'all_calibrations.csv', index=False)
    pd.DataFrame(robustness).to_csv(table_dir/'robustness_reference_status.csv', index=False)
    pd.concat(all_main).to_csv(table_dir/'all_main_results.csv', index=False)
    pd.concat(all_rates).to_csv(table_dir/'all_rate_regressions.csv', index=False)
    pd.concat(all_benchmarks).to_csv(table_dir/'all_benchmarks.csv', index=False)
