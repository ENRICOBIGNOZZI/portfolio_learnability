"""Cross-environment comparisons retaining every predeclared configuration."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulations.plotting.figures import save


def comparison(output, names, filename, title):
    output = Path(output)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    colors = plt.cm.viridis(np.linspace(.08, .9, len(names)))
    for name, color in zip(names, colors):
        directory = output/'data'/name
        if not (directory/'curves.csv').exists():
            raise ValueError(f'Missing requested environment {name}.')
        d = pd.read_csv(directory/'curves.csv')
        d = d[d['T']==d['T'].max()].sort_values('complexity_mean')
        main = pd.read_csv(directory/'main_results.csv')
        label = name.replace('_', ' ')
        ax[0, 0].loglog(d.complexity_mean, d.oos_loss_mean, label=label, color=color)
        ax[0, 1].semilogx(d.complexity_mean, d.oos_sharpe_mean, label=label, color=color)
        ax[1, 0].loglog(main['T'], main.selected_regret, 'o-', label=label, color=color)
        ax[1, 1].plot(main['T'], main.selected_complexity, 'o-', label=label, color=color)
        ax[1, 1].plot(main['T'], main.oracle_complexity, '--', color=color)
    ax[0, 0].set(xlabel='Empirical effective complexity', ylabel='Independent OOS loss', title='Largest T: loss path')
    ax[0, 1].set(xlabel='Empirical effective complexity', ylabel='Independent OOS Sharpe', title='Largest T: Sharpe path')
    ax[1, 0].set(xlabel='T', ylabel='Selected population regret')
    ax[1, 1].set(xlabel='T', ylabel='Mean empirical complexity', title='Solid: selected; dashed: oracle')
    for a in ax.flat:a.legend(fontsize=7)
    fig.suptitle(title)
    save(fig, output/'figures', filename)


def render_comparisons(output, profile='paper'):
    output = Path(output)
    spectrum = pd.read_csv(output/'audit/baseline/dgp_spectrum.csv')
    audit = json.loads((output/'audit/baseline/dgp_audit.json').read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for a, kind in zip(ax, ('kernel', 'managed')):
        y = spectrum[kind+'_eigenvalue']
        a.loglog(spectrum['index'], y, label='Population quadrature')
        a.loglog(spectrum['index'], y.iloc[99]*(spectrum['index']/100)**(-1.5), '--', label='Theoretical slope -1.5')
        for row in audit['E5']['predeclared_window_sensitivity']:
            lo, hi = row['range']
            if not row['resolved']:continue
            a.axvspan(lo, hi, alpha=.07, label=f"{lo}-{hi}: {row[kind]['slope']:.3f}")
        a.set(xlabel='Eigenvalue rank', ylabel='Eigenvalue', title=kind.capitalize()+' operator')
        a.legend(fontsize=7)
    save(fig, output/'figures', 'population_spectrum')
    if profile=='smoke':return
    groups = [(['weak_high_noise', 'baseline', 'strong_low_noise'], 'main_robustness', 'Signal-to-noise sensitivity'),
              (['N99', 'N300', 'N501', 'baseline', 'N999'], 'appendix_N_robustness', 'Cross-sectional size at fixed training lengths'),
              (['rho070', 'rho090', 'baseline', 'rho097'], 'appendix_persistence', 'Characteristic persistence'),
              (['weak_high_noise', 'baseline', 'strong_low_noise'], 'appendix_SNR', 'Signal and idiosyncratic noise'),
              (['slow_spectrum', 'baseline', 'fast_spectrum'], 'appendix_spectral_b', 'Matérn smoothness and spectral decay'),
              (['rank128', 'baseline', 'rank512'], 'appendix_approximation_rank', 'Numerical RKHS rank; three economic factors throughout'),
              (['baseline', 'empirical_ranks'], 'appendix_empirical_ranks', 'Empirical ranks: finite-support robustness'),
              (['baseline', 'heteroskedastic'], 'appendix_heteroskedastic', 'Heteroskedasticity: distinct population reference')]
    for names, filename, title in groups:
        comparison(output, names, filename, title)
