"""Plots from saved Monte Carlo output only; no DGP tuning or curve smoothing."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18})


def save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Figures are included at the manuscript's 6.5-inch text width. Keep
    # labels readable after reduction, including multi-panel legends.
    from matplotlib.text import Text
    from matplotlib.ticker import AutoLocator, MaxNLocator
    for axis in fig.axes:
        if axis.get_xscale()=='linear' and isinstance(axis.xaxis.get_major_locator(), AutoLocator):
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.canvas.draw()
    scale = max(1., fig.get_figwidth()/6.5)
    for item in fig.findobj(Text):
        item.set_fontfamily('serif')
        item.set_fontsize(max(8., item.get_fontsize())*scale)
    fig.savefig(directory/f'{name}.pdf', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'Balanced-factor simulation pipeline'})
    fig.savefig(directory/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def render_environment(directory, figure_directory, name='baseline'):
    directory, figure_directory = Path(directory), Path(figure_directory)
    curves = pd.read_csv(directory/'curves.csv')
    main = pd.read_csv(directory/'main_results.csv')
    sizes = sorted(curves['T'].unique())
    display = list(dict.fromkeys(sizes[j] for j in np.linspace(0, len(sizes)-1, min(4, len(sizes))).astype(int)))
    colors = plt.cm.viridis(np.linspace(.1, .85, len(display)))
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for size, color in zip(display, colors):
        d = curves[curves['T']==size].sort_values('complexity_mean')
        ax[0, 0].plot(d.complexity_mean, d.oos_loss_mean, color=color, label=f'T={size}')
        ax[0, 1].plot(d.complexity_mean, d.oos_sharpe_mean, color=color, label=f'T={size}')
    for a in ax[0]:
        a.set_xscale('log')
        a.set_xlabel('Empirical effective complexity')
        a.legend(fontsize=8)
    ax[0, 0].set(title='(a) Independent OOS loss', ylabel='OOS loss')
    ax[0, 0].set_yscale('log')
    ax[0, 1].set(title='(b) Independent OOS Sharpe', ylabel='Sharpe per period')
    xmin, xmax = curves.complexity_mean.min(), curves.complexity_mean.max()
    grid = np.geomspace(xmin, xmax, 150)
    heat = []
    for size in sizes:
        d = curves[curves['T']==size].sort_values('complexity_mean')
        heat.append(np.interp(np.log(grid), np.log(d.complexity_mean), d.oos_loss_mean, left=np.nan, right=np.nan))
    from matplotlib.colors import LogNorm
    values = np.asarray(heat)
    artist = ax[1, 0].pcolormesh(grid, np.arange(len(sizes))+.5, values,
                               shading='auto', cmap='magma', rasterized=True, edgecolors='face', linewidth=0, norm=LogNorm(vmin=np.nanmin(values), vmax=np.nanmax(values)))
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yticks(np.arange(len(sizes))+.5, sizes)
    ax[1, 0].set(title='(c) OOS loss by T', xlabel='Empirical effective complexity', ylabel='T')
    fig.colorbar(artist, ax=ax[1, 0], label='OOS loss (log color scale)')
    selected_ax = ax[1, 1]
    selected_ax.plot(main['T'], main.selected_complexity, 'o-', color='#24597b')
    selected_ax.set(title='(d) Validation selection', xlabel='Training observations T', ylabel='Mean selected complexity')
    penalty_ax = selected_ax.twinx()
    penalty_ax.plot(main['T'], main.selected_lambda_median, 's--', color='#a44732')
    penalty_ax.set_yscale('log')
    penalty_ax.set_ylabel('Median selected lambda', color='#a44732')
    fig.suptitle(f'Balanced three-factor economy: {name.replace("_", " ")}')
    save(fig, figure_directory, f'{name}_four_panel')

    largest = curves[curves['T']==max(sizes)].sort_values('complexity_mean')
    for metric, label in [('oos_loss', 'Independent OOS loss'), ('oos_sharpe', 'Independent OOS Sharpe')]:
        fig, a = plt.subplots(figsize=(6.5, 4.2), layout='constrained')
        a.plot(largest.complexity_mean, largest[metric+'_mean'], label='Mean')
        a.plot(largest.complexity_mean, largest[metric+'_median'], '--', label='Median')
        a.fill_between(largest.complexity_mean, largest[metric+'_ci_low'], largest[metric+'_ci_high'], alpha=.2, label='Pointwise 95% MC CI for mean')
        a.set(xscale='log', xlabel='Empirical effective complexity', ylabel=label, title=f'{name}, T={max(sizes)}')
        if metric=='oos_loss':a.set_yscale('log')
        a.legend(fontsize=8)
        save(fig, figure_directory, f'{name}_{metric}_uncertainty')
    if name != 'baseline':
        return
    fig, a = plt.subplots(figsize=(6.5, 4.2), layout='constrained')
    for column, label in [('population_bias', 'Population regularization + approximation bias'),
                          ('estimation_signed_mean', 'Signed estimation contribution'), ('regret_mean', 'Total regret'),
                          ('estimation_norm_mean', 'Nonnegative estimation norm'), ('cross_term_mean', 'Cross term')]:
        a.plot(largest.complexity_mean, largest[column], label=label)
    a.set(xscale='log', yscale='symlog', xlabel='Empirical effective complexity', ylabel='Loss relative to Q(W*)', title=f'Exact decomposition, T={max(sizes)}')
    a.set_yscale('symlog', linthresh=1e-4)
    a.legend(fontsize=7)
    save(fig, figure_directory, 'bias_estimation_regret')
    fig, a = plt.subplots(figsize=(6.5, 4.2), layout='constrained')
    a.plot(largest.population_complexity, largest.population_ridge_loss)
    a.set(xscale='log', xlabel='Population effective complexity', ylabel='Population ridge loss', title='Population regularization path')
    save(fig, figure_directory, 'population_loss_complexity')
    fig, a = plt.subplots(figsize=(6.5, 4.2), layout='constrained')
    a.loglog(largest['lambda'], largest.population_complexity, label='Population')
    for size in display:
        d=curves[curves['T']==size]
        a.loglog(d['lambda'], d.complexity_mean, label=f'Empirical T={size}')
    a.set(xlabel='Ridge penalty lambda', ylabel='Effective complexity')
    a.legend(fontsize=8)
    save(fig, figure_directory, 'lambda_complexity')
    selections = pd.read_csv(directory/'selections.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].boxplot([selections.loc[selections['T']==t, 'complexity'] for t in sizes], tick_labels=sizes, showfliers=True)
    axes[0].set(xlabel='T', ylabel='Selected complexity', title='Selection distribution across replications')
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[1].plot(main['T'], main.oracle_complexity, 'o-', label='Oracle')
    axes[1].plot(main['T'], main.selected_complexity, 's-', label='Validation')
    axes[1].set(xlabel='T', ylabel='Mean empirical complexity', title='Oracle versus validation')
    axes[1].legend()
    save(fig, figure_directory, 'selected_complexity_distributions')
    render_rates(directory, figure_directory)


def render_rates(directory, figure_directory):
    main = pd.read_csv(Path(directory)/'main_results.csv')
    rates = pd.read_csv(Path(directory)/'rates.csv')
    metadata = __import__('json').loads((Path(directory)/'population.json').read_text())
    srstar = metadata['reference_sharpe']
    quantities = [('lambda', main.oracle_lambda, main.selected_lambda_mean),
                  ('complexity', main.oracle_population_complexity, main.selected_population_complexity),
                  ('regret', main.oracle_regret, main.selected_regret),
                  ('sharpe_gap', srstar-main.oracle_population_sharpe, srstar-main.selected_population_sharpe)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    def draw(a, key, oracle, selected):
        slope = rates.loc[(rates.quantity=='oracle_'+key)&(rates.window=='full'), 'theoretical_slope'].iloc[0]
        a.loglog(main['T'], oracle, 'o-', label='Oracle')
        if selected is not None:a.loglog(main['T'], selected, 's-', label='Validation')
        anchor = len(main)//2
        reference = float(oracle.iloc[anchor])*(main['T']/main['T'].iloc[anchor])**slope
        a.loglog(main['T'], reference, '--', color='gray', label=f'Theory slope: {slope:.3f}')
        a.set(xlabel='T', ylabel=key.replace('_', ' ').capitalize())
        a.legend(fontsize=7)
    for a, (key, oracle, selected) in zip(axes.flat, quantities):
        draw(a, key, oracle, selected)
        single, sa = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
        draw(sa, key, oracle, selected)
        save(single, figure_directory, 'rate_'+key)
    fig.suptitle('Monte Carlo rate paths and theoretical slopes')
    save(fig, figure_directory, 'rate_laws')
