"""Publication figures drawn exclusively from the three empirical result tables."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from empirical_final.three_experiments import OUT, FACTORS

BLUE, RED, GREY, GREEN = '#174e75', '#ad413c', '#78828c', '#337c65'
GROUPS = ['0-10%', '10-50%', '50-90%', '90-100%']


def read(name):
    return pd.read_csv(OUT/'tables'/f'{name}.csv')


def save(fig, name):
    fig.savefig(OUT/'figures'/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close(fig)


def error(ax, x, value, low, high, color=BLUE, **kwargs):
    ax.errorbar(x, value, yerr=np.vstack([np.asarray(value)-low, high-np.asarray(value)]),
                fmt='o-', color=color, lw=1.4, markersize=4, capsize=3, **kwargs)


def main():
    plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral'], 'mathtext.fontset':'stix',
        'font.size':10, 'axes.titlesize':11, 'axes.labelsize':10, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.grid':True, 'grid.alpha':.15, 'grid.linewidth':.6,
        'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white'})
    curves, selected = read('e1_curves'), read('e1_selected_summary').set_index('period')
    fig, axes = plt.subplots(5, 2, figsize=(8.5, 10.5), sharex=True, layout='constrained')
    for i, (period, p) in enumerate(curves.groupby('period', sort=False)):
        p = p.sort_values('mean_C'); s = selected.loc[period]
        left, right = axes[i]
        left.plot(p.mean_C, p.oos_sharpe, color=BLUE)
        left.fill_between(p.mean_C, p.sr_low, p.sr_high, color=BLUE, alpha=.16)
        peak = p[p.descriptive_peak].iloc[0]
        left.scatter([peak.mean_C], [peak.oos_sharpe], marker='x', color=GREY, s=28, zorder=5)
        right.plot(p.mean_C, p.is_sharpe, color=GREY)
        for ax in (left, right):
            ax.axvline(s.mean_C, color=RED, ls='--', lw=1)
            ax.set_xscale('log'); ax.set_xlim(.25, 750)
        left.set_ylim(-1.5, 9); left.set_ylabel('OOS Sharpe')
        right.set_yscale('log'); right.set_ylim(.3, 600); right.set_ylabel('Mean refit IS Sharpe (log)')
        left.set_title(f'{period} decision years | {int(s.months)} OOS months', loc='left')
        right.set_title(f'{period} | overlapping training windows', loc='left')
    for ax in axes[-1]:
        ax.set_xlabel(r'Mean effective complexity $\overline{\widehat C}(\lambda)$ (log scale)')
    fig.suptitle('1. Fixed Gaussian representation: complexity and performance\n'
                 'Blue bands: pointwise 95% block bootstrap. Red line: mean validation-selected complexity.\n'
                 'Grey crosses: descriptive test-period maxima; selected-policy Sharpe is reported separately.', fontsize=11)
    save(fig, 'E1_complexity_by_period')

    summary = read('e2_summary'); summary = summary[summary.period != 'all']
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.8), sharex=True, layout='constrained')
    for j, (period, p) in enumerate(summary.groupby('period', sort=False)):
        p = p.sort_values('T'); x = np.arange(4)
        error(axes[0,j], x, p.mean_C, p.C_low, p.C_high)
        error(axes[1,j], x, p.sharpe, p.sr_low, p.sr_high)
        axes[0,j].set_title(f'{period}\n{int(p.months.iloc[0])} common months')
        for i, label in enumerate(['Mean active complexity', 'OOS annualized Sharpe']):
            axes[i,j].set_xticks(x, ['60','120','240','360'])
            if j == 0: axes[i,j].set_ylabel(label)
        axes[1,j].set_xlabel('History T (months)')
    for row in axes:
        lo = min(ax.get_ylim()[0] for ax in row); hi = max(ax.get_ylim()[1] for ax in row)
        for ax in row: ax.set_ylim(lo, hi)
    fig.suptitle('2. Does more history activate more useful directions?\n'
                 'Same evaluation months, features, candidate grid and 20-month validation for every T.\n'
                 'Intervals: 12-month OOS blocks; 3-year blocks for mean complexity.', fontsize=11)
    save(fig, 'E2_history_comparison')

    local, annual = read('e2_local_comparison'), read('e2_annual')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    for t, color in [(60, BLUE),(120, '#61a6bf'),(240, GREEN),(360, RED)]:
        a = annual[annual['T'] == t]
        axes[0].plot(a.decision_year, a.top10_share*100, label=f'T={t}', color=color, lw=1.3)
    axes[0].set(xlabel='January decision year', ylabel='Top 10 second-moment share (%)', title='Local spectral concentration, estimated before trading')
    axes[0].legend(ncol=2, fontsize=9)
    points = axes[1].scatter(local.top10_share_T60*100, local.delta_C_360_60,
                             c=local.decision_year, cmap='viridis', s=35, edgecolor='white', linewidth=.3)
    axes[1].axhline(0, color=GREY, lw=.8)
    axes[1].set(xlabel='Pre-test top 10 share, T=60 (%)', ylabel=r'$\widehat C_{360}-\widehat C_{60}$',
                title='History gain is not ordered by this concentration measure')
    fig.colorbar(points, ax=axes[1], label='Decision year', shrink=.8)
    fig.suptitle('2. Local economic conditions and the history comparison', fontsize=12)
    save(fig, 'E2_local_spectrum')

    gs = read('e3_group_summary').query("period == 'all'").sort_values('group')
    regs = read('e3_factor_regressions').query("period == 'all' and kind == 'diagnostic'")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    x = np.arange(4)
    axes[0,0].bar(x, gs.mean_spectral_share*100, color=GREY, width=.55)
    axes[0,0].set_yscale('log'); axes[0,0].set_ylim(.0005, 200)
    axes[0,0].set_ylabel('Training second-moment share (%, log)')
    axes[0,0].set_title('A. Most payoff variation is in the leading group')
    for j, value in enumerate(gs.mean_spectral_share*100):
        axes[0,0].text(j, value*1.22, f'{value:.3f}%', ha='center', fontsize=9)
    matrix = regs[regs.factor.isin(FACTORS)].pivot(index='group', columns='factor', values='coefficient')[FACTORS]
    im = axes[0,1].imshow(matrix, cmap='RdBu_r', vmin=-.4, vmax=.4, aspect='auto')
    axes[0,1].set_xticks(range(6), ['Market','Size','Value','Profit.','Invest.','Mom.'])
    axes[0,1].set_yticks(x, GROUPS); axes[0,1].grid(False)
    for i in range(4):
        for j, factor in enumerate(FACTORS):
            r = regs[(regs.group == i+1) & (regs.factor == factor)].iloc[0]
            star = '*' if r.low*r.high > 0 else ''
            axes[0,1].text(j, i, f'{r.coefficient:.2f}{star}', ha='center', va='center', fontsize=9)
    axes[0,1].set_title('B. Diagnostic eigenportfolio factor loadings')
    fig.colorbar(im, ax=axes[0,1], shrink=.75, label='Loading; * HAC 95% interval excludes zero')
    error(axes[1,0], x-.07, gs.contribution_annual_mean*100,
          gs.contribution_mean_low*100, gs.contribution_mean_high*100, label='Ridge contribution')
    error(axes[1,0], x+.07, gs.contribution_hedged_annual_mean*100,
          gs.contribution_hedged_low*100, gs.contribution_hedged_high*100, color=GREEN,
          label='After hedge fitted on past data')
    axes[1,0].axhline(0, color=GREY, lw=.8)
    axes[1,0].set(ylabel='Annual mean payoff contribution (%)', title='C. Mean contributions retain the fitted ridge magnitudes')
    axes[1,0].legend(fontsize=8)
    error(axes[1,1], x-.07, gs.diagnostic_annual_mean*100,
          gs.diagnostic_mean_low*100, gs.diagnostic_mean_high*100, label='Diagnostic basket')
    error(axes[1,1], x+.07, gs.diagnostic_hedged_annual_mean*100,
          gs.diagnostic_hedged_low*100, gs.diagnostic_hedged_high*100, color=GREEN,
          label='After hedge fitted on past data')
    axes[1,1].axhline(0, color=GREY, lw=.8)
    axes[1,1].set(ylabel='Annual diagnostic mean payoff (%)', title='D. Equal training risk reveals weak tail mean estimates')
    axes[1,1].legend(fontsize=8)
    for ax in [axes[0,0], *axes[1]]:
        ax.set_xticks(x, GROUPS); ax.set_xlabel('Disjoint positive-eigenvalue rank group')
    fig.suptitle('3. Economic content of the managed-payoff spectrum | February 1978-January 2025\n'
                 'Diagnostic baskets: equal 10% annualized training RMS, past-only signs and eigenvectors.', fontsize=11)
    save(fig, 'E3_economic_spectrum')

    ns = read('e3_nested_summary').query("period == 'all'").sort_values('group')
    cov = read('e3_oos_covariance').query("period == 'all'").pivot(index='group_i', columns='group_j', values='annual_covariance').to_numpy()
    corr = cov/np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout='constrained')
    error(axes[0,0], x, ns.sharpe, ns.sr_low, ns.sr_high)
    axes[0,0].set(ylabel='Annualized OOS Sharpe', title='A. Nested ridge-filtered policies')
    error(axes[0,1], x, ns.loss, ns.loss_low, ns.loss_high)
    axes[0,1].set(ylabel='OOS response-one loss', title='B. Estimation criterion out of sample')
    p = ns.iloc[1:]
    error(axes[1,0], np.arange(3), p.delta_loss, p.delta_low, p.delta_high)
    axes[1,0].axhline(0, color=GREY, lw=.8)
    axes[1,0].set_xticks(np.arange(3), GROUPS[1:])
    axes[1,0].set(xlabel='Additional disjoint rank group', ylabel='Incremental loss (negative improves)', title='C. Paired changes include all covariance terms')
    im = axes[1,1].imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1,1].set_xticks(x, GROUPS); axes[1,1].set_yticks(x, GROUPS); axes[1,1].grid(False)
    for i in range(4):
        for j in range(4):
            axes[1,1].text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', color='white' if abs(corr[i,j])>.65 else 'black')
    axes[1,1].set_title('D. OOS correlations of ridge contributions')
    fig.colorbar(im, ax=axes[1,1], shrink=.8, label='Correlation')
    for ax in axes[0]:
        ax.set_xticks(x, ['10%','50%','90%','100%']); ax.set_xlabel('Cumulative share of positive spectral ranks')
    fig.suptitle('3. Do weaker directions add portfolio value?\n'
                 'Lambda stays at its chronologically selected value; rank groups are attribution diagnostics.\n'
                 'Intervals are pointwise; comparisons are exploratory and not adjusted for multiplicity.', fontsize=11)
    save(fig, 'E3_incremental_value')


if __name__ == '__main__':
    main()
