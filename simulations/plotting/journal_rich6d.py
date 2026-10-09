"""Three review figures: Rich6D, Matérn-3/2, population-pilot theory_1 only.

Render from the frozen 100-path practical-grid confirmation, without simulation,
test-performance tuning, manuscript edits, or replacing earlier figures.

    python -m simulations.plotting.journal_rich6d
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'simulations/outputs/confirmation_v2'
DISPLAY_T = (60, 240, 720, 1440)
COLORS = ('#294F72', '#387F75', '#B17C35', '#87576E')
BLUE, GRAY = COLORS[0], '#6B6B6B'
STYLE = {
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9.5,
    'axes.titlesize': 10, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'lines.linewidth': 1.55,
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, *, atol=2e-12, rtol=2e-9):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)


def verified_inputs(base):
    """Bind the plots to existing independent reconstructions and raw paths."""
    directory = base / 'data/rich6d'
    protocol = json.loads((base / 'audit/protocol_freeze.json').read_text())
    pop = json.loads((directory / 'population.json').read_text())
    metadata = json.loads((directory / 'summary_metadata.json').read_text())
    assert pop['parameters']['loading_map'] == 'rich6d'
    assert pop['parameters']['D'] == 6 and pop['parameters']['K_F'] == 3
    assert pop['parameters']['nu'] == 1.5 and pop['theoretical_b'] == 1.5
    assert pop['environment']['rank'] == 512 and pop['environment']['replications'] == 100
    assert metadata['protocol_hash'] == protocol['protocol_hash']
    practical = np.asarray(protocol['practical_T'], dtype=int)
    assert practical.tolist() == [60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440]
    scale = protocol['theory_scale']['rich6d']
    pilot = json.loads((base / 'audit/cost_accuracy_pilot.json').read_text())
    pilot_row = next(r for r in pilot['records'] if r['loading_map'] == 'rich6d'
                     and r['nu'] == 1.5 and r['rank'] == 512)
    close(scale, pilot_row['s_ref'], 'Independent pilot scale', atol=0, rtol=0)
    assert pilot_row['population_seed'] != pop['population_seed']

    files = [directory / name for name in (
        'population.json', 'population.npz', 'curves.csv', 'methods.csv',
        'path_methods.csv', 'rates.csv', 'summary_metadata.json')]
    files += [base / 'audit' / name for name in (
        'protocol_freeze.json', 'cost_accuracy_pilot.json',
        'rich6d_rank_resolution.csv', 'rich6d_quadrature_resolution.csv',
        'spectrum/eigenvalues.csv', 'spectrum/groups4096_seed0.json',
        'spectrum/spectrum_resolution.json')]
    raw_files = sorted((directory / 'replications').glob('*.npz'))
    assert [p.stem for p in raw_files] == [f'{i:04d}' for i in range(100)]
    files += raw_files
    audit_hashes = {}
    for name in ('confirmation_reconstruction.json', 'numerical_reconstruction.json'):
        path = base / 'audit' / name
        report = json.loads(path.read_text())
        assert report['passed'] is True
        audit_hashes.update(report['input_hashes'])
        files.append(path)
    hashes = {str(p.relative_to(base)): digest(p) for p in files}
    checked = []
    for relative, actual in hashes.items():
        if relative in audit_hashes:
            assert actual == audit_hashes[relative], f'Verified input changed: {relative}'
            checked.append(relative)
    assert len(checked) >= 110

    with np.load(directory / 'population.npz', allow_pickle=False) as saved:
        mean, second = saved['mean'], saved['second']
        assert str(saved['run_hash']) == pop['run_hash']
    mu, U = np.linalg.eigh(second)
    assert mu.min() > 0
    projection = U.T @ mean
    p = pop['parameters']
    g = p['c_beta'] / 3
    q = g / (g + p['sigma_eps']**2 / p['N']) * np.dot(p['mu_F'], p['mu_F'])
    sr_star = np.sqrt(q / (1 - q))
    close(sr_star, pop['reference_sharpe'], 'Analytic population SR optimum')
    floor = q - (projection**2 / mu).sum()
    close(floor, pop['subspace_regret_floor'], 'Projection floor')

    curves = pd.read_csv(directory / 'curves.csv')
    curves = curves[curves['T'].isin(practical)].copy()
    methods = pd.read_csv(directory / 'methods.csv')
    theory = methods[(methods.method == 'theory_1') & methods['T'].isin(practical)].sort_values('T').copy()
    paths = pd.read_csv(directory / 'path_methods.csv')
    paths = paths[(paths.method == 'theory_1') & paths['T'].isin(practical)].copy()
    assert curves.environment.eq('rich6d').all() and theory.environment.eq('rich6d').all()
    assert curves.replications.eq(100).all() and theory.replications.eq(100).all()
    assert len(curves) == 960 and len(theory) == 10 and len(paths) == 1000
    assert protocol['penalties']['grid'] == 'geomspace(1e-12,1e-2,96)'
    penalties = np.geomspace(1e-12, 1e-2, 96)
    theory_lam = scale * practical ** (-.6)
    close(theory.lambda_mean, theory_lam, 'Actual unsnapped theory penalties', atol=1e-18)

    keys = ('population_sharpe', 'complexity', 'regret', 'estimation_norm', 'cross_term',
            'population_bias', 'population_complexity')
    raw = {key: [] for key in keys}
    max_decomposition_error, max_coefficient_sr_error = 0., 0.
    for path in raw_files:
        with np.load(path, allow_pickle=False) as saved:
            assert str(saved['run_hash']) == pop['run_hash']
            close(saved['T'][:10], practical, 'Practical cohort', atol=0, rtol=0)
            close(saved['penalties'][:10, :96], np.broadcast_to(penalties, (10, 96)), 'Full penalty path', atol=1e-18)
            close(saved['penalties'][:10, 97], theory_lam, 'Raw theory_1 penalties', atol=1e-18)
            for key in keys:
                raw[key].append(saved[key][:10])
            coef = saved['selected_coefficients'][:10, :, 3]
            payoff_mean = coef @ mean
            payoff_second = np.einsum('ti,ij,tj->t', coef, second, coef)
            sharpe = payoff_mean / np.sqrt(payoff_second - payoff_mean**2)
            close(sharpe, saved['population_sharpe'][:10, 97], 'Population Sharpe from policy coefficients')
            max_coefficient_sr_error = max(max_coefficient_sr_error, float(np.max(np.abs(sharpe - saved['population_sharpe'][:10, 97]))))
            close(q - 2 * payoff_mean + payoff_second, saved['regret'][:10, 97], 'Regret from coefficients')
    raw = {k: np.stack(v) for k, v in raw.items()}
    decomposition = raw['population_bias'] + raw['estimation_norm'] + raw['cross_term']
    close(raw['regret'], decomposition, 'Exact signed decomposition on every raw grid point')
    max_decomposition_error = float(np.max(np.abs(raw['regret'] - decomposition)))
    for t, T in enumerate(practical):
        frame = curves[curves['T'] == T].sort_values('lambda')
        close(frame['lambda'], penalties, 'Published penalty path', atol=1e-18)
        selected = theory[theory['T'] == T].iloc[0]
        for key in keys:
            actual = raw[key][:, t, :96]
            published = key if key in ('population_bias', 'population_complexity') else key + '_mean'
            close(frame[published], actual.mean(axis=0), 'Reconstructed curve ' + key)
            close(selected[key + '_mean'], raw[key][:, t, 97].mean(), 'Reconstructed theory curve ' + key)
        for key in ('population_sharpe', 'complexity', 'regret', 'estimation_norm', 'cross_term'):
            se = raw[key][:, t, 97].std(ddof=1) / 10
            close(selected[key + '_mcse'], se, 'MCSE ' + key)
        close(paths[paths['T'] == T].sort_values('replication').population_sharpe,
              raw['population_sharpe'][:, t, 97], 'Replication-level policy paths')
    close(theory.sharpe_gap_mean, sr_star - theory.population_sharpe_mean, 'Sharpe gap')
    pop_complexity = (mu[:, None] / (mu[:, None] + theory_lam)).sum(axis=0)
    close(pop_complexity, theory.population_complexity_mean, 'Population complexity from fixed eigenvalues')

    rates = pd.read_csv(directory / 'rates.csv')
    rates = rates[(rates.cohort == 'practical') & (rates.method == 'theory_1')
                  & rates.quantity.isin(['sharpe_gap', 'population_complexity'])].copy()
    for quantity, values in [('sharpe_gap', theory.sharpe_gap_mean), ('population_complexity', pop_complexity)]:
        slope = np.polyfit(np.log(practical), np.log(values), 1)[0]
        close(slope, rates[(rates.quantity == quantity) & (rates.window == 'full')].OLS_slope.iloc[0], 'Full-window descriptive slope')

    spectrum = pd.read_csv(base / 'audit/spectrum/eigenvalues.csv')
    spectrum = spectrum[(spectrum.operator == 'rich6d') & (spectrum.groups == 4096)
                        & (spectrum.seed_index == 0)].sort_values('index').copy()
    assert len(spectrum) == 800 and spectrum.resolved.all()
    assert (spectrum.eigenvalue > 0).all() and np.all(np.diff(spectrum.eigenvalue) <= 0)
    close(spectrum['index'], np.arange(1, 801), 'Contiguous resolved spectrum', atol=0, rtol=0)
    rank = pd.read_csv(base / 'audit/rich6d_rank_resolution.csv')
    assert rank.environment.eq('rich6d').all()
    rank_bound = abs(rank.paired_difference) + 1.96 * rank.paired_MCSE
    certified = (rank.projection_floor512 <= .05 * rank.regret512) & (rank_bound <= .05 * rank.regret512)
    assert np.array_equal(certified, rank.certified)
    assert rank[rank.lambda_index == 97].certified.all()
    quadrature = pd.read_csv(base / 'audit/rich6d_quadrature_resolution.csv')
    assert quadrature[(quadrature.method == 'theory_1') & quadrature['T'].isin(practical)].certified.all()
    checked_data = {'curves': curves, 'theory': theory, 'paths': paths, 'rates': rates,
                    'mu': mu, 'spectrum': spectrum, 'rank': rank, 'T': practical,
                    'scale': scale, 'sr_star': float(sr_star), 'q_star': float(q), 'pop': pop,
                    'pilot': pilot_row, 'protocol': protocol, 'hashes': hashes,
                    'checks': {'verified_historical_hashes': len(checked), 'raw_replications': 100,
                               'raw_grid_cells_checked': int(raw['regret'].size),
                               'maximum_decomposition_error': max_decomposition_error,
                               'maximum_theory_coefficient_sharpe_error': max_coefficient_sr_error}}
    return checked_data


def decorate(ax, title):
    ax.set_title(title, loc='left', pad=9)
    ax.grid(axis='y', which='major', color='#E4E4E4', linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(which='both', direction='out')


def time_ticks(ax, *, logarithmic=False, compact=False):
    values = [60, 720, 1440] if compact else [60, 240, 720, 1440]
    if logarithmic:
        ax.set_xscale('log')
    ax.set_xticks(values, [str(v) for v in values])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'Sample size, $T$')


def save_figure(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (stem + '.pdf'), metadata={'Title': stem.replace('_', ' '), 'Creator': 'Rich6D verified confirmation'})
    fig.savefig(folder / (stem + '.png'), dpi=450)
    fig.canvas.draw()
    bounds = fig.bbox
    for ax in fig.axes:
        box = ax.get_tightbbox(fig.canvas.get_renderer())
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1, 'Clipped axes or labels'
    plt.close(fig)


def figure_one(d, folder):
    f = d['theory']
    T = f['T'].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout='constrained')
    ax = axes[0]
    ax.plot(T, f.population_sharpe_mean, '-o', color=BLUE, markersize=3.5, label='Monte Carlo mean')
    ax.fill_between(T, f.population_sharpe_ci_low, f.population_sharpe_ci_high, color=BLUE, alpha=.12, linewidth=0)
    ax.axhline(d['sr_star'], color=GRAY, linestyle='--', linewidth=1.2, label=r'Analytic $SR(W^\star)$')
    ax.set_ylabel(r'Population Sharpe ratio')
    ax.set_ylim(0, d['sr_star'] * 1.10)
    time_ticks(ax)
    decorate(ax, '(a) Sharpe ratio recovery')
    ax.legend(loc='lower right', frameon=False)
    ax = axes[1]
    ax.loglog(T, f.sharpe_gap_mean, '-o', color=BLUE, markersize=3.5, label='Measured Sharpe gap')
    ax.fill_between(T, f.sharpe_gap_ci_low, f.sharpe_gap_ci_high, color=BLUE, alpha=.12, linewidth=0)
    reference = f.sharpe_gap_mean.iloc[0] * (T / T[0])**(-.6)
    ax.plot(T, reference, '--', color=GRAY, linewidth=1.2, label=r'$T^{-0.6}$ rate reference')
    ax.set_ylabel(r'$SR(W^\star)-\mathbb{E}_{MC}[SR(\widehat W_{\lambda_T,T})]$')
    ax.set_yticks([.01, .02, .05, .1], ['0.01', '0.02', '0.05', '0.10'])
    ax.yaxis.set_minor_locator(NullLocator())
    time_ticks(ax, logarithmic=True)
    decorate(ax, '(b) Sharpe learning rate')
    ax.legend(loc='lower left', frameon=False)
    save_figure(fig, folder, 'figure_1_sharpe_learnability')
    export = f.copy()
    export['analytic_SR_star'] = d['sr_star']
    export['gap_rate_reference_T_minus_0p6'] = reference
    export.to_csv(folder / 'curve_data/figure_1.csv', index=False)


def figure_two(d, folder):
    f, mu, spec = d['theory'], d['mu'], d['spectrum']
    lam = d['curves'][d['curves']['T'] == 60]['lambda'].sort_values().to_numpy()
    C = (mu[:, None] / (mu[:, None] + lam)).sum(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.55))
    fig.subplots_adjust(left=.075, right=.99, top=.88, bottom=.27, wspace=.45)
    ax = axes[0]
    ax.loglog(lam, C, color=BLUE)
    ax.set(xlabel=r'Regularization, $\lambda$', ylabel=r'Population complexity, $\mathcal{C}_{512}(\lambda)$')
    ax.set_xticks([1e-12, 1e-8, 1e-4, 1e-2])
    ax.xaxis.set_minor_locator(NullLocator())
    decorate(ax, '(a) Complexity and regularization')
    for T, color in zip(DISPLAY_T, COLORS):
        row = f[f['T'] == T].iloc[0]
        ax.plot(row.lambda_mean, row.population_complexity_mean, 'o', color=color,
                markeredgecolor='white', markeredgewidth=.5, markersize=5, zorder=4)
    ax = axes[1]
    ax.plot(f['T'], f.population_complexity_mean, '-o', color=BLUE, markersize=3,
            label=r'Computed $\mathcal{C}_{512}(\lambda_T)$')
    reference = f.population_complexity_mean.iloc[0] * (f['T'].to_numpy() / 60)**.4
    ax.plot(f['T'], reference, '--', color=GRAY, linewidth=1.2, label=r'$T^{0.4}$ rate reference')
    time_ticks(ax, compact=True)
    ax.set_ylabel(r'Population complexity, $\mathcal{C}_{512}(\lambda_T)$')
    decorate(ax, '(b) Complexity and history')
    ax.legend(frameon=False, loc='upper left', fontsize=7.8, handlelength=1.5)
    ax = axes[2]
    filters = spec.copy()
    for T, color in zip(DISPLAY_T, COLORS):
        penalty = d['scale'] * T**(-.6)
        h = spec.eigenvalue / (spec.eigenvalue + penalty)
        filters[f'lambda_T{T}'] = penalty
        filters[f'h_T{T}'] = h
        ax.loglog(spec['index'], h, color=color)
    ax.set(xlabel=r'Eigenvalue rank, $j$', ylabel=r'Spectral filter, $h_j(T)$', ylim=(4e-5, 1.2))
    ax.set_xticks([1, 10, 100, 800], ['1', '10', '100', '800'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    decorate(ax, '(c) Deeper spectral activation')
    handles = [Line2D([0], [0], color=c, marker='o', markersize=4, label=f'$T={T}$') for T, c in zip(DISPLAY_T, COLORS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .075), ncol=4, frameon=False)
    fig.text(.53, .015, '(a–b): fixed rank-512 population operator.  (c): fixed full-kernel audit, 800 resolved eigenpairs.',
             ha='center', fontsize=8, color=GRAY)
    save_figure(fig, folder, 'figure_2_effective_complexity')
    pd.DataFrame({'lambda': lam, 'population_complexity_rank512': C}).to_csv(folder / 'curve_data/figure_2a.csv', index=False)
    pd.DataFrame({'T': f['T'], 'lambda_T': f.lambda_mean, 'population_complexity_rank512': f.population_complexity_mean,
                  'rate_reference_T_0p4': reference, 'C_over_T_secondary_not_plotted': f.population_complexity_mean / f['T']}).to_csv(folder / 'curve_data/figure_2b.csv', index=False)
    filters.to_csv(folder / 'curve_data/figure_2c.csv', index=False)
    pd.DataFrame({'rank': np.arange(1, len(mu) + 1), 'eigenvalue': mu[::-1]}).to_csv(folder / 'curve_data/population_rank512_eigenvalues.csv', index=False)


def with_rank_status(d, T):
    frame = d['curves'][d['curves']['T'] == T].sort_values('lambda').copy()
    audit = d['rank'][(d['rank']['T'] == T) & (d['rank'].lambda_index < 96)].sort_values('lambda_index')
    close(frame['lambda'], audit['lambda'], 'Rank-audit lambda alignment', atol=1e-18)
    assert len(frame) == len(audit) == 96
    frame['lambda_index'] = np.arange(96)
    frame['rank_audit_passed'] = audit.certified.to_numpy()
    frame['rank_audit_replications'] = audit.replications.to_numpy()
    return frame.sort_values('complexity_mean')


def resolution_curve(ax, x, y, passed, color, linewidth=1.55):
    """Every point retained; intervals touching a failed endpoint are dashed."""
    points = np.column_stack((x, y))
    flags = np.asarray(passed, bool)
    solid = flags[:-1] & flags[1:]
    # One continuous line per run preserves visible dashes; a separate
    # two-point artist per interval would restart the dash pattern each time.
    boundaries = np.r_[0, np.flatnonzero(solid[1:] != solid[:-1]) + 1, len(solid)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        run = points[start:stop + 1]
        ax.plot(run[:, 0], run[:, 1], color=color, linewidth=linewidth,
                linestyle='solid' if solid[start] else (0, (4, 2.5)), zorder=3)


def figure_three(d, folder):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.95))
    fig.subplots_adjust(left=.075, right=.985, top=.87, bottom=.24, wspace=.30)
    ax = axes[0]
    exports, starts = [], {}
    for T, color in zip(DISPLAY_T, COLORS):
        frame = with_rank_status(d, T)
        exports.append(frame)
        resolution_curve(ax, frame.complexity_mean, frame.population_sharpe_mean, frame.rank_audit_passed, color)
        row = d['theory'][d['theory']['T'] == T].iloc[0]
        ax.plot(row.complexity_mean, row.population_sharpe_mean, 'D', color=color,
                markeredgecolor='white', markeredgewidth=.6, markersize=5.8, zorder=5)
        starts[str(T)] = float(frame.loc[~frame.rank_audit_passed, 'complexity_mean'].min())
    ax.set(xscale='log', xlabel=r'Mean empirical complexity, $\mathbb{E}_{MC}[\widehat{\mathcal{C}}_T(\lambda)]$',
           ylabel='Population Sharpe ratio', ylim=(0, .125))
    decorate(ax, '(a) Sharpe and effective complexity')
    ax.legend(handles=[Line2D([0], [0], color=c, label=f'$T={T}$') for T, c in zip(DISPLAY_T, COLORS)],
              loc='upper left', ncol=2, frameon=False, handlelength=1.6, columnspacing=1.)
    ax = axes[1]
    frame = with_rank_status(d, 1440)
    specs = [('population_bias', 'Regularization bias', COLORS[0]),
             ('estimation_norm_mean', 'Squared estimation error', COLORS[1]),
             ('cross_term_mean', 'Signed cross term', COLORS[3]),
             ('regret_mean', 'Total regret', '#202020')]
    threshold = starts['1440']
    ax.axvspan(threshold, frame.complexity_mean.max() * 1.07, color='#EFEFEF', zorder=0)
    for column, label, color in specs:
        resolution_curve(ax, frame.complexity_mean, frame[column], frame.rank_audit_passed, color,
                         linewidth=1.85 if column == 'regret_mean' else 1.45)
    selected = d['theory'][d['theory']['T'] == 1440].iloc[0]
    ax.axvline(selected.complexity_mean, ymax=.80, color=GRAY, linestyle=':', linewidth=1.)
    for column, label, color in specs:
        value = selected.population_bias_mean if column == 'population_bias' else selected[column]
        ax.plot(selected.complexity_mean, value, 'D', color=color, markeredgecolor='white',
                markeredgewidth=.45, markersize=4.5, zorder=5)
    ax.axhline(0, color=GRAY, linewidth=.55, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=1e-5, linscale=.8)
    ax.set(xlabel=r'Mean empirical complexity, $\mathbb{E}_{MC}[\widehat{\mathcal{C}}_{1440}(\lambda)]$',
           ylabel='Population regret and signed components')
    ax.set_ylim(min(frame.cross_term_mean.min() * 1.45, -1.2e-5), frame.regret_mean.max() * 1.4)
    ax.set_yticks([-1e-5, 0, 1e-4, 1e-3, 1e-2, 1e-1],
                 [r'$-10^{-5}$', '0', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    ax.yaxis.set_minor_locator(NullLocator())
    decorate(ax, r'(b) Regret decomposition, $T=1440$')
    ax.legend(handles=[Line2D([0], [0], color=c, label=label) for _, label, c in specs],
              loc='upper left', frameon=False, fontsize=7.5, handlelength=1.5, ncol=2, columnspacing=.9)
    fig.legend(handles=[Line2D([0], [0], color=GRAY, marker='D', linestyle='None', label=r'Theory policy, $\lambda_T$'),
                        Line2D([0], [0], color=GRAY, label='Rank audit passed'),
                        Line2D([0], [0], color=GRAY, linestyle=(0, (4, 2.5)), label='Rank sensitivity unresolved')],
               loc='lower center', bbox_to_anchor=(.52, .045), ncol=3, frameon=False, fontsize=8)
    fig.text(.52, .01, r'All 96 penalties retained.  Panel (b): symmetric-log scale, linear within $\pm10^{-5}$.',
             ha='center', fontsize=8, color=GRAY)
    save_figure(fig, folder, 'figure_3_learning_tradeoff')
    pd.concat(exports).to_csv(folder / 'curve_data/figure_3a_full_paths.csv', index=False)
    frame.to_csv(folder / 'curve_data/figure_3b_decomposition.csv', index=False)
    d['theory'][d['theory']['T'].isin(DISPLAY_T)].to_csv(folder / 'curve_data/figure_3_theory_markers.csv', index=False)
    return starts


def documentation(d, folder, unresolved):
    rate = d['rates'][(d['rates'].quantity == 'sharpe_gap') & (d['rates'].window == 'full')].iloc[0]
    c_rate = d['rates'][(d['rates'].quantity == 'population_complexity') & (d['rates'].window == 'full')].iloc[0]
    captions = f"""COMMON DESIGN AND INTERPRETATION
Rich6D only: N=600 assets, three economic factors, six persistent characteristics (rho=0.95), eta=0.35; normalized Matern-3/2 kernel, length scale 1. All Monte Carlo means use the same 100 replications at all ten practical sample sizes T=60,90,120,180,240,360,540,720,1080,1440. Sharpe ratios are per period, not annualized. The exact analytical reference is SR(W*)={d['sr_star']:.12f}, q*={d['q_star']:.12f}.
The only highlighted policy is theory_1: lambda_T=a T^(-0.6), a={d['scale']:.15g}. In the existing protocol the dimensionless theory multiplier is 1 and the frozen scale s_ref equals a. This scale is the leading population eigenvalue in an independent 8,192-group pilot (seed {d['pilot']['population_seed']}, anchor seed {d['pilot']['anchor_seed']}); it is neither estimated from the current training sample nor tuned on test performance. This is a population-calibrated theoretical benchmark, not a fully data-driven tuning procedure.
Learned policies use the fixed rank-512 Nystrom approximation. Population policy performance is numerically integrated with 8,192 independent groups (seed {d['pop']['population_seed']}) and compared with the exact optimum, never a truncated proxy optimum. Intervals are pointwise mean +/- 1.96 Monte Carlo standard errors, conditional on the fixed basis and quadrature; they do not include numerical approximation error. All displayed theory policies pass the existing regret-based rank and quadrature checks. Those checks are not a uniform infinite-dimensional or separate Sharpe-error certificate.

FIGURE 1. SHARPE RATIO LEARNABILITY.
Panel (a) reports the replication-average population Sharpe ratio of the learned Rich6D portfolio under the population-pilot-calibrated theory rule, with pointwise 95% Monte Carlo intervals and the exact tangency Sharpe reference. Panel (b) reports the actual mean Sharpe shortfall on logarithmic axes; the dashed T^(-0.6) rate reference is normalized at T=60, without fitting its scale to the remaining observations. The descriptive log-log OLS slope over the entire ten-point practical grid is {rate.OLS_slope:.6f}, with the existing 1,000-whole-path-bootstrap 95% interval [{rate.bootstrap_low:.6f}, {rate.bootstrap_high:.6f}]; no fitted regression is drawn. The observed approach to the maximum-Sharpe direction illustrates learnability over the available histories but does not prove an asymptotic equality, eliminate approximation error, or validate data-driven tuning. Rich6D's target is smoother than the theorem's minimum regularity condition; the rate upper bound does not require the simulated slope to equal -0.6.

FIGURE 2. EFFECTIVE PORTFOLIO COMPLEXITY.
Panels (a) and (b) use the same fixed 512-dimensional population managed-payoff second-moment operator as the production estimator: C_512(lambda)=sum_j mu_j/(mu_j+lambda). Panel (a) displays the full 96-penalty grid, with markers at the actual, unsnapped theory penalties for T=60,240,720,1440. Panel (b) reports the computed C_512(lambda_T), together with a T^(0.4) rate reference normalized at T=60; the full-grid descriptive complexity slope is {c_rate.OLS_slope:.6f}, not an imposed exponent. Panel (c) uses a separate, fixed full-kernel Rich6D population audit (4,096 groups, 12,288 nodes, seed index 0 / seed 4101162095) and all 800 numerically resolved eigenpairs, on logarithmic axes. For every T, h_j(T)=mu_j/(mu_j+lambda_T) uses exactly this same spectrum; it is neither synthetic nor refitted across sample sizes. The audit's 800 eigenpairs are not substituted into C_512 or presented as the complete infinite-dimensional spectrum. Lower penalties increase the contribution of weaker directions through shrinkage; spectral activation is not evidence that each direction is statistically identified. The finite-operator complexity curve does not establish the asymptotic power-law exponent of the infinite kernel.

FIGURE 3. SHARPE, COMPLEXITY, AND THE LEARNING TRADE-OFF.
Panel (a) traces all 96 diagnostic penalties, from 10^(-2) to 10^(-12), for each of T=60,240,720,1440, using replication-average empirical complexity on the horizontal axis and replication-average population Sharpe on the vertical axis. Diamonds mark the independently calibrated theory policies at their actual penalties, evaluated directly rather than snapped to the grid or selected using performance. Panel (b) fixes T=1440 and plots the exact signed decomposition of population regret: R(W_hat)-R(W*) = B_lambda + ||W_hat-W_lambda^(512)||_S^2 + 2 <W_hat-W_lambda^(512), W_lambda^(512)-W*>_S. B_lambda is the population regularization bias relative to the analytic optimum and includes the rank-512 projection floor ({d['pop']['subspace_regret_floor']:.12g}); the squared estimation error is not labeled as the full variance contribution. The signed cross term is retained without taking absolute values; the vertical scale is symmetric-logarithmic, linear within +/-10^(-5). Diamonds and the vertical dotted line mark the theory policy. Solid segments pass the existing regret-based rank audit; dashed segments touching a failing audit point retain the unresolved observations, and the unresolved high-complexity region in panel (b) is shaded. The audit compares ranks 512 and 1024 on 50 matched paths: both projection-floor/mean-regret and (absolute paired mean difference + 1.96 paired MCSE)/mean-regret must not exceed 5%. These empirical-path curves use 100 replications, not the audit's 50-path complexity averages; flags are matched by T and lambda index. No favorable range is dropped. The figure illustrates the economic trade-off and its algebraic sources; it neither defines a new lambda selector nor certifies the unresolved tail.

REPRODUCTION
python -m simulations.plotting.journal_rich6d
The script verifies saved-input SHA256 identities against the existing independent reconstructions, reaggregates all 100 raw Rich6D paths, reconstructs the theory-policy population Sharpe from saved coefficients, and checks the signed decomposition point by point. Every plotted series is exported in curve_data/; detailed input hashes and panel lineage are in provenance.json. The manuscript is not modified.
"""
    (folder / 'captions_and_provenance.txt').write_text(captions, encoding='utf-8')
    concise = f"""Figure 1. Sharpe ratio learnability.
Rich6D, Matern-3/2, 100 replications at each sample size. Panel (a) reports mean population Sharpe and its exact analytical optimum ({d['sr_star']:.6f}); bands are pointwise 95% Monte Carlo intervals. Panel (b) reports the measured Sharpe gap and a T^(-0.6) rate reference anchored at T=60. The full-grid descriptive slope is {rate.OLS_slope:.3f} (whole-path-bootstrap 95% interval [{rate.bootstrap_low:.3f}, {rate.bootstrap_high:.3f}]). The policy uses lambda_T={d['scale']:.8g} T^(-0.6), calibrated in an independent population pilot; it is a theoretical benchmark, not data-driven tuning. The evidence illustrates convergence over the observed histories, not an exact asymptotic rate equality. Population performance uses the verified rank-512 approximation; the optimum is analytical.

Figure 2. Effective portfolio complexity.
Panel (a) reports population complexity from the fixed rank-512 Rich6D operator, marking the actual theory penalties. Panel (b) reports complexity along the theory rule and a T^(0.4) rate reference anchored at T=60. Panel (c) applies the same penalties to one fixed, independently audited full-kernel spectrum (4,096 groups, seed index 0, 800 resolved eigenpairs), plotting mu_j/(mu_j+lambda_T). The spectrum is numerical, with no power-law substitution or tail extrapolation. Lower penalties activate weaker directions through shrinkage; this does not establish their statistical identification. The finite-operator complexity curve does not prove an infinite-dimensional exponent. All panels use Rich6D and Matern-3/2.

Figure 3. Sharpe, complexity, and the learning trade-off.
Panel (a) plots mean population Sharpe against mean empirical complexity along all 96 penalties for each displayed T; diamonds locate the population-pilot theory policy. Panel (b), at T=1440, preserves the exact regret decomposition into population regularization bias, squared estimation error, and signed cross term. Bias includes the projection floor; squared estimation error is not the entire variance contribution. Solid segments pass the predeclared regret-based rank audit; dashed segments retain unresolved observations, with shading in (b). The audit uses 50 matched paths, while plotted means use 100. These diagnostic curves neither select lambda from test performance nor certify the unresolved tail. The symmetric-log scale is linear within +/-10^(-5).
"""
    (folder / 'captions.txt').write_text(concise, encoding='utf-8')
    d['rates'].to_csv(folder / 'curve_data/descriptive_slopes_all_prespecified_windows.csv', index=False)
    pd.DataFrame({'T': list(unresolved), 'first_failed_audit_point_empirical_complexity_100_paths': list(unresolved.values())}).to_csv(folder / 'curve_data/unresolved_region_starts.csv', index=False)
    provenance = {
        'scope': 'Rich6D only; Matern-3/2; theory_1 only; practical grid; review figures, no manuscript edits',
        'script_sha256': digest(__file__), 'protocol_hash': d['protocol']['protocol_hash'],
        'run_hash': d['pop']['run_hash'], 'parameters': d['pop']['parameters'],
        'theory_rule': {'formula': 'lambda_T = a * T^(-0.6)', 'a': d['scale'],
                        'dimensionless_protocol_multiplier': 1, 'population_pilot': d['pilot'],
                        'implementable_data_driven_selector': False},
        'population_evaluation': d['pop'], 'numerical_verification': d['checks'],
        'prespecified_sample_sizes': d['T'].tolist(), 'replications': 100,
        'rate_references': {'sharpe': 'mean gap at T=60 * (T/60)^(-0.6)',
                            'complexity': 'C_512 at T=60 * (T/60)^(0.4)',
                            'fitted': False},
        'panel_lineage': {
            '1a': 'data/rich6d/methods.csv: method=theory_1, practical T; population_sharpe_mean, ci_low, ci_high. Analytic reference from DGP parameters.',
            '1b': 'Same rows: sharpe_gap_mean, ci_low, ci_high. Descriptive slope and path-bootstrap interval from rates.csv: practical, theory_1, sharpe_gap, full.',
            '2a': 'All 96 frozen penalties; eigenvalues of data/rich6d/population.npz second (512x512); direct sum mu/(mu+lambda). Actual-theory markers from methods.csv.',
            '2b': 'methods.csv population_complexity_mean at all 10 theory_1 practical T, checked against the same population.npz spectrum. No empirical complexity used.',
            '2c': 'audit/spectrum/eigenvalues.csv: operator=rich6d, groups=4096, seed_index=0, indices 1:800, all resolved. Exact h_j(T) with the fixed eigenvalue vector and four actual theory penalties.',
            '3a': 'data/rich6d/curves.csv: all 96 lambda points at T=60,240,720,1440; complexity_mean and population_sharpe_mean; theory markers from methods.csv.',
            '3b': 'curves.csv: T=1440, all 96 lambda points; complexity_mean, population_bias, estimation_norm_mean, cross_term_mean, regret_mean. Actual theory markers from methods.csv.',
            '3_resolution': 'audit/rich6d_rank_resolution.csv: flags aligned by T and lambda_index; 50-path regret-based 512-vs-1024 rank comparison. Plot coordinates use the 100-path means.'},
        'unresolved_region_starts': unresolved,
        'source_sha256': d['hashes'],
        'output_sha256': {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'provenance.json'},
    }
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input / 'figures/journal_rich6d'
    (output / 'curve_data').mkdir(parents=True, exist_ok=True)
    print('Verifying frozen Rich6D inputs and all 100 practical-grid paths...', flush=True)
    data = verified_inputs(args.input)
    print(json.dumps(data['checks'], indent=2), flush=True)
    with plt.rc_context(STYLE):
        figure_one(data, output)
        figure_two(data, output)
        unresolved = figure_three(data, output)
    documentation(data, output, unresolved)
    print(f'Created three PDF/PNG pairs, captions and complete numeric lineage in {output}', flush=True)


if __name__ == '__main__':
    main()
