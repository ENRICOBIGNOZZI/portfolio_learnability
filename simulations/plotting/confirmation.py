"""Confirmation figures and paired comparisons from saved, verified CSVs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from simulations.plotting.figures import save
from simulations.provenance import ROOT, file_hash, json_write, require

LABELS = {'holdout25': 'Holdout 25%', 'rolling3': 'Rolling 3', 'theory_0.25': 'Theory a=0.25',
          'theory_1': 'Theory a=1', 'theory_4': 'Theory a=4',
          'ensemble_plugin_loss_oracle': 'Plugin loss oracle', 'crossfit_loss_oracle': 'Crossfit loss oracle'}
COLORS = dict(zip(LABELS, ['#245b86', '#c06b32', '#aa83ac', '#755092', '#493a6d', '#458462', '#71988b']))


def band(axis, x, frame, metric, label, color, scale=1., style='-'):
    x = np.asarray(x, dtype=float)
    axis.plot(x, frame[metric+'_mean'].to_numpy()*scale, style, color=color, label=label)
    axis.fill_between(x, frame[metric+'_ci_low'].to_numpy()*scale,
                      frame[metric+'_ci_high'].to_numpy()*scale, color=color, alpha=.10)


def render_environment(output, name):
    output = Path(output)
    directory, figures = output/'data'/name, output/'figures'
    curves = pd.read_csv(directory/'curves.csv')
    methods = pd.read_csv(directory/'methods.csv')
    selections = pd.read_csv(directory/'selection_diagnostics.csv')
    population = json.loads((directory/'population.json').read_text())
    q = population['q_reference']
    display = [60, 240, 720, 1440]
    colors = plt.cm.viridis(np.linspace(.08, .85, len(display)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for T, color in zip(display, colors):
        frame = curves[curves['T'] == T].sort_values('complexity_mean')
        for axis, metric in zip(axes[0], ('oos_loss', 'oos_sharpe')):
            band(axis, frame.complexity_mean, frame, metric, f'T={T}', color)
            axis.set_xscale('log'); axis.set_xlabel('Mean sample complexity')
    axes[0, 0].set(title='(a) Independent OOS loss', ylabel='Response-one loss', yscale='log')
    axes[0, 1].set(title='(b) Independent OOS Sharpe', ylabel='Sharpe per period')
    sizes = sorted(T for T in curves['T'].unique() if T <= 1440)
    grid = np.geomspace(curves.complexity_mean.min(), curves.complexity_mean.max(), 150)
    heat = []
    for T in sizes:
        row = curves[curves['T'] == T].sort_values('complexity_mean')
        heat.append(np.interp(np.log(grid), np.log(row.complexity_mean), row.oos_loss_mean, left=np.nan, right=np.nan))
    artist = axes[1, 0].pcolormesh(grid, np.arange(len(sizes)), heat, shading='auto', cmap='magma',
                                  norm=LogNorm(), rasterized=True)
    axes[1, 0].set(xscale='log', title='(c) OOS loss inside computed support', xlabel='Mean sample complexity', ylabel='T')
    axes[1, 0].set_yticks(np.arange(len(sizes)), sizes)
    fig.colorbar(artist, ax=axes[1, 0], label='OOS loss')
    penalty_axis = axes[1, 1].twinx()
    for method in ('holdout25', 'rolling3'):
        frame = methods[(methods.method == method)&(methods['T'] <= 1440)]
        band(axes[1, 1], frame['T'], frame, 'complexity', LABELS[method]+' mean C', COLORS[method])
        penalty_axis.plot(frame['T'], frame.lambda_median, '--', color=COLORS[method], label=LABELS[method]+' median lambda')
    axes[1, 1].set(title='(d) Training-only selection', xlabel='T', ylabel='Mean sample C')
    penalty_axis.set(yscale='log', ylabel='Median lambda (dashed)')
    for axis in axes[0]:
        axis.legend(fontsize=7)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=7)
    fig.suptitle(name.replace('_', ' '))
    save(fig, figures, name+'_four_panel')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for T, color in zip(display, colors):
        frame = curves[curves['T'] == T].sort_values('complexity_mean')
        band(axes[0, 0], frame.complexity_mean, frame, 'population_loss', f'T={T}', color)
        band(axes[0, 1], frame.complexity_mean, frame, 'regret', f'T={T}', color)
        band(axes[1, 0], frame.complexity_mean, frame, 'regret', f'T={T}', color, scale=1/q)
        band(axes[1, 1], frame.complexity_mean/T, frame, 'regret', f'T={T}', color, scale=1/q)
    for axis in axes.flat:
        axis.set_xscale('log'); axis.legend(fontsize=7)
        axis.set_xlabel('Mean sample complexity')
    axes[0, 0].set(ylabel='Population loss', title='Raw risk (linear vertical scale)')
    axes[0, 1].set(ylabel='Population regret', yscale='symlog', title='Regret relative to analytic optimum')
    axes[1, 0].set(ylabel='Regret / q*', yscale='symlog', title='Normalized population regret')
    axes[1, 1].set(ylabel='Regret / q*', yscale='symlog', xlabel='Mean sample C / T', title='Complexity relative to sample size')
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set_yscale('symlog', linthresh=1e-5)
    save(fig, figures, name+'_risk_complexity')

    last = curves[curves['T'] == curves['T'].max()].sort_values('complexity_mean')
    fig, axis = plt.subplots(figsize=(7, 4.5), layout='constrained')
    for column, label in [('population_bias', 'Population bias including floor'), ('estimation_norm_mean', 'Estimation norm'),
                          ('cross_term_mean', 'Signed cross term'), ('regret_mean', 'Total regret')]:
        axis.plot(last.complexity_mean, last[column], label=label)
    axis.axhline(population['subspace_regret_floor'], color='gray', linestyle=':', label='Projection floor')
    axis.set(xscale='log', xlabel='Mean sample complexity', ylabel='Population loss relative to optimum', title=f'{name}: T={int(last["T"].iloc[0])}')
    axis.set_yscale('symlog', linthresh=1e-5)
    axis.legend(fontsize=7)
    save(fig, figures, name+'_decomposition')

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    metrics = [('lambda', 'Actual penalty'), ('complexity_over_T', 'Sample C / T'), ('scaled_regret', r'$T^{b/(b+1)}$ regret'),
               ('scaled_population_complexity', r'Population $C/T^{1/(b+1)}$'), ('regret', 'Population regret'), ('sharpe_gap', 'Population Sharpe gap')]
    for method in LABELS:
        frame = methods[methods.method == method]
        for axis, (metric, label) in zip(axes.flat, metrics):
            band(axis, frame['T'], frame, metric, LABELS[method], COLORS[method], style='--' if 'oracle' in method else '-')
            axis.set(xscale='log', xlabel='T', ylabel=label)
    displayed = methods[methods.method.isin(LABELS)]
    for axis, (metric, label) in zip(axes.flat, metrics):
        if metric in ('lambda', 'regret', 'sharpe_gap'):
            if (displayed[metric+'_ci_low'] <= 0).any():
                axis.set_yscale('symlog', linthresh=1e-5)
                axis.set_ylabel(label+' (symlog)')
            else:
                axis.set_yscale('log')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=7)
    fig.suptitle(name.replace('_', ' ')+': distinct theory, oracle and validation paths')
    save(fig, figures, name+'_theory_paths')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for method in ('holdout25', 'rolling3'):
        frame = selections[selections.selector == method]
        color = COLORS[method]
        for metric, style, label in [('lambda_mean', '-', 'mean'), ('lambda_median', '--', 'median'), ('lambda_geometric_mean', ':', 'geometric mean')]:
            axes[0, 0].plot(frame['T'], frame[metric], style, color=color, label=LABELS[method]+' '+label)
        axes[0, 1].plot(frame['T'], frame.lower_frequency, ':', color=color, label=LABELS[method]+' lower')
        axes[0, 1].plot(frame['T'], frame.upper_frequency, '-', color=color, label=LABELS[method]+' upper')
        for axis, metric in zip(axes[1], ('complexity', 'regret')):
            axis.plot(frame['T'], frame[metric+'_median'], color=color, label=LABELS[method]+' median')
            axis.fill_between(frame['T'], frame[metric+'_q25'], frame[metric+'_q75'], color=color, alpha=.18)
    axes[0, 0].set(yscale='log', ylabel='Selected lambda', title='Distinct location summaries')
    axes[0, 1].set(ylabel='Selection frequency', title='Lower and upper grid endpoints')
    axes[1, 0].set(ylabel='Selected sample complexity', title='Median and interquartile range')
    axes[1, 1].set(ylabel='Selected population regret', title='Median and interquartile range')
    for axis in axes.flat:
        axis.set(xscale='log', xlabel='T'); axis.legend(fontsize=7)
    save(fig, figures, name+'_selection_distributions')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for method in ('holdout25', 'rolling3', 'theory_1'):
        frame = methods[methods.method == method]
        for suffix, style in [('oos', '-'), ('forward', '--')]:
            excess = frame.copy()
            for stat in ('mean', 'ci_low', 'ci_high'):
                excess[suffix+'_loss_'+stat] -= 1-q
            label = LABELS[method]+(' independent' if suffix == 'oos' else ' future')
            band(axes[0], frame['T'], excess, suffix+'_loss', label, COLORS[method], style=style)
            band(axes[1], frame['T'], frame, suffix+'_sharpe', label, COLORS[method], style=style)
    axes[0].axhline(0, color='gray', linewidth=.7)
    axes[0].set(ylabel='OOS loss minus Q* (signed)', title='Independent versus contiguous OOS')
    axes[1].set(ylabel='Sample Sharpe per period', title='H=720')
    for axis in axes:
        axis.set(xscale='log', xlabel='T'); axis.legend(fontsize=7)
    save(fig, figures, name+'_oos_types')


def paired_comparisons(output):
    output = Path(output)
    frames = {name: pd.read_csv(output/'data'/name/'path_methods.csv') for name in ('baseline_original', 'rich6d')}
    rows = []
    for method in ('holdout25', 'rolling3', 'theory_1'):
        left, right = [f[f.method == method] for f in frames.values()]
        paired = left.merge(right, on=['T', 'replication'], suffixes=('_base', '_rich'), validate='one_to_one')
        for T, frame in paired.groupby('T'):
            for metric in ('regret', 'oos_loss', 'forward_loss', 'population_sharpe'):
                delta = frame[metric+'_rich']-frame[metric+'_base']
                mean, se = delta.mean(), delta.std(ddof=1)/np.sqrt(len(delta))
                rows.append({'comparison': 'rich6d_minus_baseline', 'method': method, 'T': T, 'metric': metric,
                             'replications': len(delta), 'mean': mean, 'mcse': se, 'ci_low': mean-1.96*se, 'ci_high': mean+1.96*se})
    for name, frame in frames.items():
        left = frame[frame.method == 'holdout25']
        right = frame[frame.method == 'rolling3']
        paired = left.merge(right, on=['T', 'replication'], suffixes=('_hold', '_rolling'), validate='one_to_one')
        for T, f in paired.groupby('T'):
            for metric in ('regret', 'oos_loss', 'forward_loss', 'population_sharpe'):
                delta = f[metric+'_rolling']-f[metric+'_hold']
                mean, se = delta.mean(), delta.std(ddof=1)/np.sqrt(len(delta))
                rows.append({'comparison': name+'_rolling_minus_holdout', 'method': 'rolling3', 'T': T, 'metric': metric,
                             'replications': len(delta), 'mean': mean, 'mcse': se, 'ci_low': mean-1.96*se, 'ci_high': mean+1.96*se})
    result = pd.DataFrame(rows)
    (output/'tables').mkdir(parents=True, exist_ok=True)
    result.to_csv(output/'tables/paired_comparisons.csv', index=False)
    for comparison in result.comparison.unique():
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        for axis, metric in zip(axes.flat, ('regret', 'oos_loss', 'forward_loss', 'population_sharpe')):
            for method in result[result.comparison == comparison].method.unique():
                f = result[(result.comparison == comparison)&(result.metric == metric)&(result.method == method)]
                axis.plot(f['T'], f['mean'], color=COLORS[method], label=LABELS[method])
                axis.fill_between(f['T'], f.ci_low, f.ci_high, color=COLORS[method], alpha=.15)
            axis.axhline(0, color='gray', linewidth=.7)
            axis.set(xscale='log', xlabel='T', ylabel='Paired difference: '+metric.replace('_', ' '))
            axis.legend(fontsize=7)
        fig.suptitle(comparison.replace('_', ' '))
        save(fig, output/'figures', comparison)


def numerical_appendix(output):
    output = Path(output)
    figures = output/'figures'
    for name in ('baseline_original', 'rich6d'):
        rank_path = output/'audit'/f'{name}_rank_resolution.csv'
        quadrature_path = output/'audit'/f'{name}_quadrature_resolution.csv'
        require(rank_path.exists() and quadrature_path.exists(), 'Numerical resolution audits missing')
        rank = pd.read_csv(rank_path)
        quad = pd.read_csv(quadrature_path)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
        for T, color in zip((60, 240, 720, 1440), plt.cm.viridis(np.linspace(.08, .85, 4))):
            f = rank[(rank['T'] == T)&(rank.lambda_index < 96)].sort_values('complexity512')
            axes[0, 0].plot(f.complexity512, f.regret512, color=color, label=f'T={T}, rank 512')
            axes[0, 0].plot(f.complexity512, f.regret1024, '--', color=color)
            axes[0, 1].plot(f.complexity512, 100*f.paired_difference/f.regret512, color=color, label=f'T={T}')
            axes[0, 1].fill_between(f.complexity512, 100*f.paired_ci_low/f.regret512, 100*f.paired_ci_high/f.regret512, color=color, alpha=.1)
            axes[1, 0].plot(f.complexity512, 100*f.projection_floor512/f.regret512, color=color, label=f'T={T}')
        axes[0, 0].set(yscale='log', ylabel='Mean population regret', title='Risk at fixed lambda\nSolid 512; dashed 1024')
        axes[0, 1].set(ylabel=r'$(R_{512}-R_{1024})/R_{512}$ (%)', title='Paired rank difference\n95% Monte Carlo interval')
        axes[0, 1].axhline(5, color='gray', linestyle=':'); axes[0, 1].axhline(-5, color='gray', linestyle=':')
        axes[1, 0].set(yscale='log', ylabel='Floor / mean regret (%)', title='Projection floor relative to risk')
        axes[1, 0].axhline(5, color='gray', linestyle=':')
        for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set(xscale='log', xlabel='Mean sample C (rank 512)'); axis.legend(fontsize=7)
        for method in ('holdout25', 'rolling3', 'theory_1'):
            f = quad[quad.method == method]
            axes[1, 1].plot(f['T'], 100*f.difference8192_minus32768/f.mean_regret32768, color=COLORS[method], label=LABELS[method])
            axes[1, 1].fill_between(f['T'], 100*f.ci_low/f.mean_regret32768, 100*f.ci_high/f.mean_regret32768, color=COLORS[method], alpha=.12)
        axes[1, 1].axhline(5, color='gray', linestyle=':'); axes[1, 1].axhline(-5, color='gray', linestyle=':')
        axes[1, 1].set(xscale='log', xlabel='T', ylabel=r'$(R_{8192}-R_{32768})/R_{32768}$ (%)', title='Quadrature resolution\nFour independent seeds')
        axes[1, 1].legend(fontsize=7)
        save(fig, figures, name+'_numerical_resolution')
    spectrum = pd.read_csv(output/'audit/spectrum/eigenvalues.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for axis, operator in zip(axes, ('kernel', 'baseline_original', 'rich6d')):
        for groups, color in zip((1024, 2048, 4096), ('#33658a', '#c47b44', '#558467')):
            f = spectrum[(spectrum.operator == operator)&(spectrum.groups == groups)]
            values = f.pivot(index='index', columns='seed_index', values='eigenvalue')
            require(set(values.columns) == {0, 1, 2}, 'Missing spectrum seed')
            resolved = f.pivot(index='index', columns='seed_index', values='resolved')
            usable = values.notna().all(axis=1)&resolved.fillna(False).all(axis=1)
            visible = values.where(usable, np.nan)
            axis.loglog(values.index, visible.mean(axis=1), color=color, label=f'{groups} groups')
            axis.fill_between(values.index, visible.min(axis=1), visible.max(axis=1), color=color, alpha=.15)
            if not usable.all() or len(values) < 800:
                axis.plot([], [], ':', color=color, label=f'{groups}: unresolved indices omitted')
        axis.set(xlabel='Eigenvalue index', ylabel='Eigenvalue', title=operator.replace('_', ' '))
        axis.legend(fontsize=7)
    save(fig, figures, 'spectrum_resolution')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for axis, loading in zip(axes, ('baseline_original', 'rich6d')):
        for suffix, label, color in [('_nu05', 'nu=0.5', '#b97540'), ('', 'nu=1.5', '#35658b'), ('_nu25', 'nu=2.5', '#65845c')]:
            f = pd.read_csv(output/'data'/(loading+suffix)/'methods.csv')
            f = f[(f.method == 'rolling3')&(f['T'] <= 1440)]
            band(axis, f['T'], f, 'regret', label, color)
        axis.set(xscale='log', yscale='log', xlabel='T', ylabel='Rolling3 population regret', title=loading.replace('_', ' '))
        axis.legend(fontsize=7)
    save(fig, figures, 'kernel_sensitivity')


def render(output):
    output = Path(output)
    evidence = json.loads((output/'audit/confirmation_reconstruction.json').read_text())
    require(evidence['passed'] and not evidence['partial'] and not evidence['raw_only'], 'Full numerical reconstruction required before publication figures')
    numerical = json.loads((output/'audit/numerical_reconstruction.json').read_text())
    require(numerical['passed'] and 'spectrum' in numerical['details']
            and all('quadrature' in numerical['details'].get(name, {}) for name in ('baseline_original', 'rich6d')),
            'Complete numerical audit reconstruction required before figures')
    for report in (evidence, numerical):
        for name, expected in report['input_hashes'].items():
            require(file_hash(output/name) == expected, 'Changed verified figure input: '+name)
        for name, expected in report['verifier_sources'].items():
            require(file_hash(ROOT/name) == expected, 'Changed figure verifier: '+name)
    for name in ('baseline_original', 'rich6d'):
        render_environment(output, name)
    paired_comparisons(output)
    numerical_appendix(output)
    json_write(output/'audit/figure_provenance.json', {'schema': 'confirmation-figures/2.0',
               'renderer_sha256': file_hash(__file__), 'sources': {
                   str(p.relative_to(output)): file_hash(p) for p in output.rglob('*.csv')},
               'figures': {str(p.relative_to(output)): file_hash(p) for p in (output/'figures').glob('*')},
               'heatmap_interpolation': 'linear in log mean complexity, only within each T support; outside is NaN',
               'visual_review': 'pending'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    render(parser.parse_args().output)
