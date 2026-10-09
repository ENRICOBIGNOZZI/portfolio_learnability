"""Rich6D review sequence, Figures 0--4; no manuscript or simulation changes.

All population-complexity axes use the SAME fixed 800-eigenpair audited
Rich6D spectrum. This is a resolved partial trace, not the full infinite trace,
the rank-512 production trace, or sample effective complexity. The actual
100-replication policy outcomes remain those of the verified rank-512 fits.

The only policy rule is theory_1, calibrated using the independent population
pilot. It is a theoretical benchmark, not fully feasible historical tuning.

    python -m simulations.plotting.journal_rich6d_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from simulations.plotting import journal_rich6d as checked
from simulations.plotting.journal_rich6d import (
    BLUE, COLORS, DEFAULT, DISPLAY_T, GRAY, STYLE, close, decorate, digest,
    resolution_curve, save_figure, time_ticks, verified_inputs, with_rank_status,
)

COMPLEXITY_LABEL = r'Population complexity, $\mathcal{C}_{800}(\lambda)$'
STEMS = ('figure_0_economic_spectrum', 'figure_1_sharpe_learnability',
         'figure_2_complexity_over_time', 'figure_3_sharpe_versus_complexity',
         'figure_4_regret_decomposition')


def complexity(d, penalties):
    penalties = np.atleast_1d(np.asarray(penalties, dtype=float))
    mu = d['spectrum'].eigenvalue.to_numpy()
    return (mu[:, None] / (mu[:, None] + penalties)).sum(axis=0)


def prepare(d):
    theory = d['theory'].copy()
    theory['resolved_population_complexity_800'] = complexity(d, theory.lambda_mean)
    theory['resolved_relative_complexity_800'] = theory.resolved_population_complexity_800 / theory['T']
    theory['complexity_guide_T_0p4'] = theory.resolved_population_complexity_800.iloc[0] * (theory['T'] / 60)**.4
    theory['relative_complexity_guide_T_minus_0p6'] = theory.resolved_relative_complexity_800.iloc[0] * (theory['T'] / 60)**(-.6)
    theory['sharpe_gap_guide_T_minus_0p6'] = theory.sharpe_gap_mean.iloc[0] * (theory['T'] / 60)**(-.6)
    assert np.all(np.diff(theory.lambda_mean) < 0)
    assert np.all(np.diff(theory.resolved_population_complexity_800) > 0)
    assert np.all(np.diff(theory.resolved_relative_complexity_800) < 0)
    spectral = d['spectrum'].copy()
    h_previous = None
    for T in DISPLAY_T:
        row = theory[theory['T'] == T].iloc[0]
        h = spectral.eigenvalue.to_numpy() / (spectral.eigenvalue.to_numpy() + row.lambda_mean)
        assert np.all((h > 0) & (h < 1)) and np.all(np.diff(h) <= 0)
        if h_previous is not None:
            assert np.all(h > h_previous)
        h_previous = h
        cumulative = np.cumsum(h)
        close(cumulative[-1], row.resolved_population_complexity_800, 'Cumulative trace endpoint')
        spectral[f'lambda_T{T}'] = row.lambda_mean
        spectral[f'filter_T{T}'] = h
        spectral[f'cumulative_T{T}'] = cumulative
    frames = []
    for T in DISPLAY_T:
        f = with_rank_status(d, T)
        f['resolved_population_complexity_800'] = complexity(d, f['lambda'])
        f = f.sort_values('resolved_population_complexity_800')
        assert len(f) == 96 and f['lambda'].nunique() == 96
        close(f.regret_mean, f.population_bias + f.estimation_norm_mean + f.cross_term_mean,
              'Mean signed decomposition at every displayed penalty')
        frames.append(f)
    d['final_theory'], d['final_spectrum'] = theory, spectral
    d['final_paths'] = pd.concat(frames, ignore_index=True)
    d['resolved_slopes'] = {
        'lambda': float(np.polyfit(np.log(theory['T']), np.log(theory.lambda_mean), 1)[0]),
        'population_complexity_800': float(np.polyfit(np.log(theory['T']), np.log(theory.resolved_population_complexity_800), 1)[0]),
        'relative_population_complexity_800': float(np.polyfit(np.log(theory['T']), np.log(theory.resolved_relative_complexity_800), 1)[0]),
    }
    return d


def spectrum_ticks(ax, letter='j'):
    ax.set_xscale('log')
    ax.set_xticks([1, 10, 100, 800], ['1', '10', '100', '800'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'Eigenvalue rank, $' + letter + '$')


def time_legend(fig, bottom=.075):
    handles = [Line2D([0], [0], color=color, label=f'$T={T}$') for T, color in zip(DISPLAY_T, COLORS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, bottom), ncol=4, frameon=False)


def figure_zero(d, folder):
    s = d['final_spectrum']
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.65))
    fig.subplots_adjust(left=.075, right=.985, bottom=.27, top=.88, wspace=.42)
    ax = axes[0]
    ax.loglog(s['index'], s.eigenvalue, color=BLUE)
    spectrum_ticks(ax)
    ax.set_ylabel(r'Population eigenvalue, $\mu_j$')
    ax.yaxis.set_minor_locator(NullLocator())
    decorate(ax, '(a) The economic spectrum')
    ax = axes[1]
    for T, color in zip(DISPLAY_T, COLORS):
        ax.loglog(s['index'], s[f'filter_T{T}'], color=color)
    spectrum_ticks(ax)
    ax.set_ylabel(r'Spectral filter, $h_j(T)$')
    ax.yaxis.set_minor_locator(NullLocator())
    decorate(ax, '(b) Deeper spectral activation')
    ax = axes[2]
    for T, color in zip(DISPLAY_T, COLORS):
        ax.semilogx(s['index'], s[f'cumulative_T{T}'], color=color)
    spectrum_ticks(ax, 'm')
    ax.set_ylabel(r'Cumulative complexity, $\sum_{j\leq m} h_j(T)$')
    ax.set_ylim(0, s.cumulative_T1440.iloc[-1] * 1.07)
    decorate(ax, '(c) Effective number of directions')
    time_legend(fig)
    fig.text(.53, .02, 'One fixed Rich6D population spectrum; all 800 resolved eigenpairs. No extrapolated tail.',
             ha='center', color=GRAY, fontsize=8)
    save_figure(fig, folder, STEMS[0])


def figure_one(d, folder):
    f = d['final_theory']
    T = f['T'].to_numpy()
    sr_scale = d.get('sharpe_display_scale', 1.)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65), layout='constrained')
    ax = axes[0]
    ax.plot(T, f.population_sharpe_mean * sr_scale, '-o', color=BLUE, markersize=3.5, label='Monte Carlo mean')
    ax.fill_between(T, f.population_sharpe_ci_low * sr_scale, f.population_sharpe_ci_high * sr_scale, color=BLUE, alpha=.12, linewidth=0)
    ax.axhline(d['sr_star'] * sr_scale, color=GRAY, linestyle='--', linewidth=1.2, label=r'Analytic $SR(W^\star)$')
    ax.set(ylabel=d.get('sharpe_axis_label', 'Population Sharpe ratio'), ylim=(0, d['sr_star'] * sr_scale * 1.10))
    time_ticks(ax)
    decorate(ax, '(a) Sharpe recovery')
    ax.legend(loc='lower right', frameon=False)
    ax = axes[1]
    ax.loglog(T, f.sharpe_gap_mean * sr_scale, '-o', color=BLUE, markersize=3.5, label='Measured Sharpe gap')
    ax.fill_between(T, f.sharpe_gap_ci_low * sr_scale, f.sharpe_gap_ci_high * sr_scale, color=BLUE, alpha=.12, linewidth=0)
    ax.plot(T, f.sharpe_gap_guide_T_minus_0p6 * sr_scale, '--', color=GRAY, linewidth=1.2, label=r'$T^{-0.6}$ theoretical benchmark')
    ax.set_ylabel(d.get('sharpe_gap_axis_label', r'$SR(W^\star)-\mathbb{E}_{MC}[SR(\widehat W_{\lambda_T,T})]$'))
    if sr_scale == 1:
        ax.set_yticks([.01, .02, .05, .1], ['0.01', '0.02', '0.05', '0.10'])
    ax.yaxis.set_minor_locator(NullLocator())
    time_ticks(ax, logarithmic=True)
    decorate(ax, '(b) Sharpe learning gap')
    ax.legend(loc='lower left', frameon=False)
    save_figure(fig, folder, STEMS[1])


def figure_two(d, folder):
    f = d['final_theory']
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.55))
    fig.subplots_adjust(left=.09, right=.985, bottom=.22, top=.88, wspace=.48)
    titles = ('(a) Regularization over time', '(b) Effective complexity', '(c) Relative complexity')
    values = ('lambda_mean', 'resolved_population_complexity_800', 'resolved_relative_complexity_800')
    labels = (r'Regularization, $\lambda_T$', r'Population complexity, $\mathcal{C}_{800}(\lambda_T)$',
              r'Relative complexity, $\mathcal{C}_{800}(\lambda_T)/T$')
    for ax, title, value, label in zip(axes, titles, values, labels):
        ax.loglog(f['T'], f[value], '-o', color=BLUE, markersize=3, label='Computed sequence')
        time_ticks(ax, logarithmic=True)
        ax.set_ylabel(label)
        ax.yaxis.set_minor_locator(NullLocator())
        decorate(ax, title)
    axes[0].set_yticks([1e-6, 2e-6, 5e-6, 1e-5],
                       [r'$10^{-6}$', r'$2\!\times\!10^{-6}$', r'$5\!\times\!10^{-6}$', r'$10^{-5}$'])
    axes[0].text(.96, .90, r'$\lambda_T=aT^{-0.6}$', transform=axes[0].transAxes, ha='right')
    axes[1].plot(f['T'], f.complexity_guide_T_0p4, '--', color=GRAY, linewidth=1.2, label=r'$T^{0.4}$ theory guide')
    axes[1].set_yticks([3, 5, 10], ['3', '5', '10'])
    axes[1].legend(frameon=False, loc='upper left', fontsize=7.7, handlelength=1.5)
    axes[2].plot(f['T'], f.relative_complexity_guide_T_minus_0p6, '--', color=GRAY, linewidth=1.2, label=r'$T^{-0.6}$ theory guide')
    axes[2].set_yticks([.005, .01, .02, .05], ['0.005', '0.01', '0.02', '0.05'])
    axes[2].legend(frameon=False, loc='lower left', fontsize=7.7, handlelength=1.5)
    fig.text(.53, .02, 'Population-pilot theory rule. Complexity uses the same 800 resolved eigenvalues as Figure 0.',
             ha='center', color=GRAY, fontsize=8)
    save_figure(fig, folder, STEMS[2])


def resolution_legend(fig, bottom=.04):
    handles = [Line2D([0], [0], color=GRAY, marker='D', linestyle='None', label=r'Theory policy, $\lambda_T$'),
               Line2D([0], [0], color=GRAY, label='Rank audit passed'),
               Line2D([0], [0], color=GRAY, linestyle=(0, (4, 2.5)), label='Rank sensitivity unresolved')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, bottom), ncol=3,
               frameon=False, fontsize=8)


def figure_three(d, folder):
    sr_scale = d.get('sharpe_display_scale', 1.)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.95))
    fig.subplots_adjust(left=.075, right=.985, top=.87, bottom=.24, wspace=.22)
    offsets = {60: (14, 10), 240: (14, 0), 720: (14, -18), 1440: (14, 9)}
    annual_labels = {60: (.53, .59), 240: (.56, .70), 720: (.60, .80), 1440: (.64, .94)}
    for T, color in zip(DISPLAY_T, COLORS):
        f = d['final_paths'][d['final_paths']['T'] == T]
        row = d['final_theory'][d['final_theory']['T'] == T].iloc[0]
        x, y = row.resolved_population_complexity_800, row.population_sharpe_mean * sr_scale
        for index, ax in enumerate(axes):
            start = len(ax.lines)
            resolution_curve(ax, f.resolved_population_complexity_800, f.population_sharpe_mean * sr_scale,
                             f.rank_audit_passed, color)
            if index == 1:
                for line in ax.lines[start:]:
                    line.set_alpha(.27)
            ax.plot(x, y, 'D', color=color, markersize=5.8 if index == 0 else 6.8,
                    markeredgecolor='white', markeredgewidth=.6, zorder=5)
        axes[1].annotate(f'$T={T}$, $\\mathcal{{C}}_{{800}}={x:.2f}$', xy=(x, y),
                         xytext=annual_labels[T] if sr_scale != 1 else offsets[T],
                         textcoords='axes fraction' if sr_scale != 1 else 'offset points', color=color,
                         fontsize=8, va='center', arrowprops={'arrowstyle': '-', 'color': color, 'lw': .65})
    for ax, title in zip(axes, ('(a) Full diagnostic penalty paths', '(b) The theory-scaled policy')):
        ax.set(xscale='log', xlabel=COMPLEXITY_LABEL, ylabel=d.get('sharpe_axis_label', 'Population Sharpe ratio'),
               ylim=d.get('sharpe_ylim', (0, .13)))
        decorate(ax, title)
    axes[0].legend(handles=[Line2D([0], [0], color=c, label=f'$T={T}$') for T, c in zip(DISPLAY_T, COLORS)],
                   loc='upper left', ncol=2, frameon=False, handlelength=1.6)
    resolution_legend(fig)
    fig.text(.52, .01, 'Both panels retain all 96 penalties; diamonds use actual theory penalties, without grid snapping.',
             ha='center', fontsize=8, color=GRAY)
    save_figure(fig, folder, STEMS[3])


def figure_four(d, folder):
    f = d['final_paths'][d['final_paths']['T'] == 1440]
    row = d['final_theory'][d['final_theory']['T'] == 1440].iloc[0]
    threshold = f.loc[~f.rank_audit_passed, 'resolved_population_complexity_800'].min()
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    fig.subplots_adjust(left=.115, right=.985, top=.91, bottom=.35)
    ax.axvspan(threshold, f.resolved_population_complexity_800.max() * 1.07, color='#EFEFEF', zorder=0)
    specs = [('population_bias', 'Population bias (including approximation floor)', COLORS[0]),
             ('estimation_norm_mean', 'Squared estimation error', COLORS[1]),
             ('cross_term_mean', 'Signed cross term', COLORS[3]),
             ('regret_mean', 'Total population regret', '#202020')]
    for column, label, color in specs:
        resolution_curve(ax, f.resolved_population_complexity_800, f[column], f.rank_audit_passed,
                         color, linewidth=1.85 if column == 'regret_mean' else 1.45)
        y = row.population_bias_mean if column == 'population_bias' else row[column]
        ax.plot(row.resolved_population_complexity_800, y, 'D', color=color,
                markeredgecolor='white', markeredgewidth=.55, markersize=5, zorder=5)
    ax.axvline(row.resolved_population_complexity_800, ymax=.77, linestyle=':', color=GRAY, linewidth=1.)
    ax.axhline(0, color=GRAY, linewidth=.55)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1e-5, linscale=.8)
    ax.set(xlabel=COMPLEXITY_LABEL, ylabel='Population regret and signed components',
           ylim=(min(f.cross_term_mean.min() * 1.45, -1.2e-5), f.regret_mean.max() * 1.4))
    # Stronger signal calibrations can produce larger signed cross terms.
    lower, upper = ax.get_ylim()
    negative = [-10.**power for power in range(1, -6, -1) if lower <= -10.**power]
    positive = [10.**power for power in range(-4, 3) if 10.**power <= upper]
    ticks = negative + [0.] + positive
    labels = [('0' if value == 0 else
               rf'${"-" if value < 0 else ""}10^{{{int(round(np.log10(abs(value))))}}}$')
              for value in ticks]
    ax.set_yticks(ticks, labels)
    ax.yaxis.set_minor_locator(NullLocator())
    decorate(ax, r'(a) Exact regret decomposition, $T=1440$')
    fig.legend(handles=[Line2D([0], [0], color=c, label=label) for _, label, c in specs],
               loc='lower center', bbox_to_anchor=(.54, .155), frameon=False,
               ncol=2, fontsize=8, handlelength=1.5, columnspacing=1.1)
    resolution_legend(fig, .065)
    fig.text(.54, .022, r'Signed cross term retained. Symmetric-log scale, linear within $\pm10^{-5}$.',
             ha='center', color=GRAY, fontsize=8)
    save_figure(fig, folder, STEMS[4])


def write_documentation(d, folder):
    f = d['final_theory']
    rate = d['rates'][(d['rates'].quantity == 'sharpe_gap') & (d['rates'].window == 'full')].iloc[0]
    common = f"""COMMON DESIGN AND NUMERICAL SCOPE
Rich6D only; N=600 assets, three economic factors, six characteristics, rho=0.95, eta=0.35. Kernel: normalized Matern-3/2, length scale 1. Policy means use 100 replications at every practical T=60,90,120,180,240,360,540,720,1080,1440, using the verified rank-512 Nystrom estimator. Population performance uses the saved independent 8,192-group quadrature (seed {d['pop']['population_seed']}; anchor seed {d['pop']['basis_seed']}). The reference is the exact analytical optimum SR(W*)={d['sr_star']:.12f}, q*={d['q_star']:.12f}. Sharpe ratios are per period.
The single rule is theory_1: lambda_T=a*T^(-0.6), a={d['scale']:.15g}. The protocol's dimensionless multiplier is 1. The scale is frozen from an independent population pilot, using 8,192 groups, seed {d['pilot']['population_seed']}; it is a theoretical benchmark, not fully feasible historical tuning. No test observation determines this scale or a selected policy.
ALL complexity axes in Figures 0,2,3,4 use C_800(lambda)=sum_(j=1)^800 mu_j/(mu_j+lambda), with ONE fixed audited Rich6D managed-payoff spectrum (4,096 groups, 12,288 nodes, seed index 0, seed 4101162095, all 800 available eigenpairs resolved). This is a finite resolved partial trace, not the complete infinite-dimensional population trace, not empirical complexity, and not the production rank-512 population trace. The earlier three-figure draft used the production C_512 for its time-complexity panel and empirical complexity for the trade-off: these have been replaced explicitly to follow the new population-complexity specification and keep a common spectral coordinate. The saved policy outcomes, pilot scale, exact optimum and rank-sensitivity flags are unchanged. Directions beyond rank 800 are neither imputed nor extrapolated; eigenpair residual resolution does not certify the quadrature or omitted spectrum uniformly.
In Figures 3 and 4, the population spectral diagnostic C_800 reparametrizes the existing penalty grid; it does not assert that the rank-512 fitted policy uses the same number of effective directions. Production C_512 and sample complexity remain available as separately named source columns in the exported data.
All 96 penalties 10^(-12) to 10^(-2) are retained. Rank flags use the predeclared comparison of 512 vs 1024 on 50 matched paths, aligned by T and exact penalty index. A point passes only if both the projection floor and |paired mean difference|+1.96*paired MCSE are <=5% of mean rank-512 regret. Segments touching a failed point are dashed. The criterion is regret-based, not a separate Sharpe-error certificate; the entire infinite kernel and high-complexity tail are not certified. The theory policies pass the saved practical-grid rank and quadrature tests. Pointwise Monte Carlo bands condition on the fixed basis and quadrature and omit numerical approximation uncertainty.
Every theoretical rate guide is normalized at T=60, with its prescribed exponent. No guide scale or plotted policy is optimized against the remaining performance observations. Full-grid descriptive slopes: Sharpe gap {rate.OLS_slope:.6f} (1,000-whole-path-bootstrap 95% interval [{rate.bootstrap_low:.6f}, {rate.bootstrap_high:.6f}]); C_800 {d['resolved_slopes']['population_complexity_800']:.6f}; C_800/T {d['resolved_slopes']['relative_population_complexity_800']:.6f}; lambda {d['resolved_slopes']['lambda']:.6f}. These are descriptive finite-grid values, not required asymptotic equalities. The spectral complexity is deterministic conditional on this fixed quadrature: no Monte Carlo interval is attached to its slope. Rich6D's target is smoother than the minimum theorem regularity.
"""
    captions = [
        "Figure 0. The economic spectrum. Panel (a) plots the actual Rich6D population managed-payoff eigenvalues from one independent 4,096-group full-kernel audit; all 800 resolved eigenpairs are retained. Panel (b) applies the independently population-calibrated theory penalties to this same spectrum, h_j(T)=mu_j/(mu_j+lambda_T). Panel (c) accumulates these filters through rank m, ending at the finite resolved complexity C_800(lambda_T). Increasing history lowers the penalty and increases the weights of weaker directions. This is shrinkage activation, not proof that each direction has been statistically identified; the figure neither extrapolates the unresolved tail nor establishes an exact power-law spectrum.",
        f"Figure 1. Sharpe learnability. Panel (a) reports mean population Sharpe and the analytical optimum {d['sr_star']:.6f}; shading is a pointwise 95% Monte Carlo interval. Panel (b) plots the measured Sharpe gap and a T^(-0.6) theoretical benchmark normalized at T=60. All ten sample sizes use the same 100-replication cohort and the independently population-pilot-calibrated theory_1 policy. The descriptive full-grid slope is {rate.OLS_slope:.3f}, with whole-path-bootstrap 95% interval [{rate.bootstrap_low:.3f}, {rate.bootstrap_high:.3f}]. The observed recovery illustrates learnability over available histories, not an exact asymptotic rate equality or a fully feasible historical tuning procedure.",
        "Figure 2. Effective complexity over time. Panel (a) displays the actual theory_1 penalty sequence lambda_T=a*T^(-0.6), with a frozen by the independent population pilot. Panels (b) and (c) evaluate C_800(lambda_T) and C_800(lambda_T)/T on exactly the same fixed 800-eigenpair Rich6D spectrum used in Figure 0. Dashed T^(0.4) and T^(-0.6) theory guides are normalized at T=60, not fitted. Absolute resolved complexity increases while relative complexity falls. This is a finite-spectrum population diagnostic; it neither replaces the production estimator's rank-512 trace nor proves an infinite-dimensional power-law exponent. The rule remains a population-calibrated theoretical benchmark.",
        "Figure 3. Sharpe versus complexity. Both panels plot all 96 diagnostic penalties at T=60,240,720,1440, using the fixed population spectral coordinate C_800(lambda) and the actual 100-replication mean population Sharpe of the rank-512 fitted policies. Panel (b) repeats the full paths with lighter lines and labels the independently population-calibrated theory points; diamonds use actual unsnapped penalties. Solid segments pass the predeclared regret-based rank audit, while dashed segments retain unresolved observations. The interior trade-off and theory-rule location are descriptive evidence; no test-performance optimization, additional selector, or numerical certification of the unresolved tail is implied.",
        f"Figure 4. Why the trade-off exists. At T=1440, all 96 penalties are shown against the same C_800(lambda) coordinate as Figure 3. Total population regret equals population bias plus squared estimation error plus the signed cross term. Bias includes regularization and the finite rank-512 approximation floor ({d['pop']['subspace_regret_floor']:.8g}); squared estimation error is not labeled as the entire variance contribution. Diamonds mark the population-pilot theory policy. Dashed segments and shading identify unresolved rank sensitivity. The symmetric-log vertical scale is linear within +/-10^(-5). The exact decomposition explains the measured trade-off without proving that the theoretical penalty is performance-optimal or certifying the unresolved tail.",
    ]
    (folder / 'captions.txt').write_text('\n\n'.join(captions) + '\n', encoding='utf-8')
    (folder / 'numerical_provenance.txt').write_text(common + "\nEXACT DECOMPOSITION\nR(W_hat)-R(W*) = B_lambda + ||W_hat-W_lambda^(512)||_S^2 + 2<W_hat-W_lambda^(512), W_lambda^(512)-W*>_S. It is preserved for each raw replication and each penalty, and therefore for the plotted means.\n\nREPRODUCTION\npython -m simulations.plotting.journal_rich6d_final\nThe generator verifies historical SHA256 identities, reconstructs all 100 paths and policy Sharpe from saved coefficients, and checks spectral sums and the signed regret identity. No manuscript file is written.\n", encoding='utf-8')
    data = folder / 'curve_data'
    d['final_spectrum'].to_csv(data / 'figure_0_spectrum_filters_cumulative.csv', index=False)
    f.to_csv(data / 'figures_1_2_theory_sequences.csv', index=False)
    d['final_paths'].to_csv(data / 'figure_3_full_paths.csv', index=False)
    f[f['T'].isin(DISPLAY_T)].to_csv(data / 'figure_3_theory_points.csv', index=False)
    d['final_paths'][d['final_paths']['T'] == 1440].to_csv(data / 'figure_4_decomposition.csv', index=False)
    f[f['T'] == 1440].to_csv(data / 'figure_4_theory_point.csv', index=False)
    d['rates'][d['rates'].quantity == 'sharpe_gap'].to_csv(data / 'sharpe_gap_slopes_prespecified_windows.csv', index=False)
    provenance = {
        'scope': 'Figures 0--4; Rich6D only, Matern-3/2, theory_1; practical grid, 100 replications',
        'source_sha256': d['hashes'],
        'generator_sha256': {str(Path(__file__).relative_to(checked.ROOT)): digest(__file__),
                              str(Path(checked.__file__).relative_to(checked.ROOT)): digest(checked.__file__)},
        'protocol_hash': d['protocol']['protocol_hash'], 'run_hash': d['pop']['run_hash'],
        'parameters': d['pop']['parameters'], 'population_evaluation': d['pop'],
        'rule': {'formula': 'lambda_T=a*T^(-0.6)', 'a': d['scale'], 'protocol_multiplier': 1,
                 'pilot': d['pilot'], 'fully_feasible_historical_tuning': False},
        'common_complexity': {'definition': 'sum_{j=1}^{800} mu_j/(mu_j+lambda)',
                             'operator': 'rich6d', 'groups': 4096, 'seed_index': 0, 'seed': 4101162095,
                             'finite_partial_trace': True, 'synthetic_or_extrapolated_tail': False,
                             'policy_estimator_rank': 512, 'not_empirical_complexity': True},
        'numerical_checks': d['checks'], 'descriptive_resolved_spectrum_slopes': d['resolved_slopes'],
        'sharpe_slope': rate.to_dict(), 'guide_normalization': 'Anchor at T=60; no fit',
        'panels': {
            '0a': 'audit/spectrum/eigenvalues.csv, operator=rich6d, groups=4096, seed_index=0, indices 1..800, eigenvalue; all resolved.',
            '0b': 'Same fixed eigenvalues and theory_1 actual penalties at T=60,240,720,1440; mu/(mu+lambda_T).',
            '0c': 'Cumulative sum of each filter vector in 0b, in unchanged eigenvalue order, through all 800 ranks.',
            '1a': 'data/rich6d/methods.csv, theory_1, practical grid: population_sharpe_mean, ci_low, ci_high; analytic optimum from DGPParameters formula.',
            '1b': 'Same rows: sharpe_gap_mean and pointwise 95% MC intervals; theoretical guide uses gap(T=60)*(T/60)^(-0.6). Descriptive slope/CI from rates.csv practical/full.',
            '2a': 'Existing theory_1 lambda_mean, checked against independent pilot scale*T^(-0.6), ten practical sample sizes.',
            '2b': 'Sum of the fixed 800-eigenvalue filters at each actual theory_1 penalty; guide C800(60)*(T/60)^0.4.',
            '2c': 'The same C800(lambda_T) divided by T; guide [C800(60)/60]*(T/60)^(-0.6).',
            '3a': 'data/rich6d/curves.csv, full 96-penalty paths at T=60,240,720,1440: population_sharpe_mean vs the common C800(lambda). Diamonds from methods.csv actual theory_1 rows.',
            '3b': 'Exactly the same complete data as 3a, with lighter curves and individually annotated theory_1 points; no zoom, filtering, or optimized selection.',
            '4': 'curves.csv T=1440: population_bias, estimation_norm_mean, cross_term_mean, regret_mean vs common C800(lambda); theory values directly from methods.csv.',
            'resolution': 'audit/rich6d_rank_resolution.csv, same T and lambda_index (0..95), certified boolean independently recomputed; 50 matched audit paths, not the 100-path plotted mean cohort.'},
        'output_sha256': {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'provenance.json'},
    }
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    folder = args.output or args.input / 'figures/journal_rich6d_final'
    (folder / 'curve_data').mkdir(parents=True, exist_ok=True)
    print('Verifying existing Rich6D confirmation and fixed population spectrum...', flush=True)
    d = prepare(verified_inputs(args.input))
    print(json.dumps(d['checks'], indent=2), flush=True)
    with plt.rc_context(STYLE):
        for render in (figure_zero, figure_one, figure_two, figure_three, figure_four):
            render(d, folder)
    write_documentation(d, folder)
    print('Resolved-spectrum slopes:', d['resolved_slopes'], flush=True)
    print('Five PDF/PNG figures, English captions and numerical provenance:', folder, flush=True)


if __name__ == '__main__':
    main()
