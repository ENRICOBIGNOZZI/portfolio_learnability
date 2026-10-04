"""Joint chronological bandwidth/ridge selection, isolated from the frozen paper run."""
from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import PanelSequence, digest, load_panels, write_json
from kernels import FeatureBank, cosine, median_distance
from portfolio import annual_splits, complexity_grid, dual_path, sharpe

MULTIPLIERS = (.25, .5, 1., 2., 4.)
BANDWIDTH_BATCH_SIZE = 64


def managed_bandwidths(panel, bank, multipliers):
    """Share X @ frequencies across bandwidths, retaining identical RFF draws."""
    x, r = np.asarray(panel['x'], float), np.asarray(panel['r'], float)
    multipliers = np.asarray(multipliers, float)
    if (x.ndim != 2 or len(x) != len(r) or not len(r)
            or not np.isfinite(x).all() or not np.isfinite(r).all()
            or not np.isfinite(multipliers).all() or (multipliers <= 0).any()):
        raise ValueError('Require finite aligned inputs and positive bandwidths.')
    totals = np.zeros((len(multipliers), bank.maximum))
    scale = np.sqrt(2. / bank.maximum)
    for start in range(0, len(r), BANDWIDTH_BATCH_SIZE):
        stop = start + BANDWIDTH_BATCH_SIZE
        projection = x[start:stop] @ bank.frequencies.T
        for j, multiplier in enumerate(multipliers):
            features = scale * cosine(projection / multiplier + bank.phases)
            totals[j] += features.T @ r[start:stop]
    return totals / len(r)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp.json')
    write_json(temporary, value)
    temporary.replace(path)


def build_cache(panels, bank, multipliers, folder, clean_hash):
    folder.mkdir(parents=True, exist_ok=True)
    path, progress_path = folder / 'managed.npy', folder / 'progress.json'
    identity = {'clean_manifest_sha256': clean_hash,
                'feature_bank': bank.metadata(), 'multipliers': list(multipliers),
                'dates': [str(d.date()) for d in panels.dates],
                'shape': [len(multipliers), len(panels), bank.maximum]}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['identity'] != identity:
            raise ValueError('Cache provenance mismatch: use a new output directory.')
        completed = progress['completed_months']
        if progress.get('status') == 'complete' and digest(path) != progress['sha256']:
            raise ValueError('Managed cache checksum mismatch.')
        g = np.load(path, mmap_mode='r+', allow_pickle=False)
        if list(g.shape) != identity['shape'] or g.dtype != np.float64:
            raise ValueError('Managed cache shape or precision mismatch.')
    else:
        if path.exists():
            raise FileExistsError('Cache has no checkpoint; inspect it before restarting.')
        g = np.lib.format.open_memmap(path, mode='w+', dtype='float64',
                                     shape=tuple(identity['shape']))
        completed = 0
        atomic_json(progress_path, {'identity': identity, 'completed_months': 0,
                                  'status': 'running'})
    start_time = time.monotonic()
    first = completed
    for i in range(completed, len(panels)):
        g[:, i, :] = managed_bandwidths(panels[i], bank, multipliers)
        if (i + 1) % 12 == 0 or i + 1 == len(panels):
            g.flush()
            atomic_json(progress_path, {'identity': identity, 'completed_months': i + 1,
                                      'status': 'running'})
            elapsed = time.monotonic() - start_time
            remaining = elapsed / (i + 1 - first) * (len(panels) - i - 1)
            print(f'{bank.kernel}: managed {i+1}/{len(panels)}, '
                  f'formation {panels.dates[i].date()}, ETA {remaining/60:.1f} min', flush=True)
    if not np.isfinite(g).all():
        raise ValueError('Nonfinite managed payoffs.')
    atomic_json(progress_path, {'identity': identity, 'completed_months': len(panels),
                              'status': 'complete', 'sha256': digest(path)})
    return g


def select_bandwidth(grams, grids, split, multipliers):
    """Read only training and validation blocks; OOS blocks cannot affect selection."""
    _, train, validation, _ = split
    losses, choices = [], []
    for gram, penalties in zip(grams, grids):
        alpha, _, _ = dual_path(gram[np.ix_(train, train)], penalties)
        returns = gram[np.ix_(validation, train)] @ alpha
        loss = np.mean((1. - returns) ** 2, axis=0)
        if not np.isfinite(loss).all():
            raise ValueError('Nonfinite validation losses.')
        losses.append(loss)
        choices.append(int(np.argmin(loss)))
    # Exact bandwidth ties prefer the median heuristic, then nearer multipliers.
    order = sorted(range(len(multipliers)),
                   key=lambda b: (abs(np.log2(multipliers[b])), multipliers[b]))
    chosen = min(order, key=lambda b: losses[b][choices[b]])
    return chosen, choices, losses


def refit_selected(gram, penalty, split):
    _, train, validation, test = split
    history = np.r_[train, validation]
    alpha, _, complexity = dual_path(gram[np.ix_(history, history)], [penalty])
    returns = (gram[np.ix_(test, history)] @ alpha)[:, 0]
    return returns, float(complexity[0])


def fit_experiment(g, dates, multipliers, kernel, ell0, lambda_count=120):
    splits = list(annual_splits(dates))
    initial = splits[0][1]
    grids = [complexity_grid(np.asarray(candidate)[initial], lambda_count) for candidate in g]
    grams = [np.asarray(candidate) @ np.asarray(candidate).T for candidate in g]
    baseline = list(multipliers).index(1.)
    monthly, selections, surfaces = [], [], []
    for split in splits:
        year, train, validation, test = split
        chosen, choices, losses = select_bandwidth(grams, grids, split, multipliers)
        fitted = {}
        for mode, b in [('fixed', baseline), ('tuned', chosen)]:
            j = choices[b]
            if b not in fitted:
                fitted[b] = refit_selected(grams[b], grids[b][j], split)
            returns, complexity = fitted[b]
            selections.append({
                'kernel': kernel, 'mode': mode, 'year': year,
                'bandwidth_multiplier': multipliers[b], 'bandwidth': ell0 * multipliers[b],
                'lambda': grids[b][j], 'lambda_index': j,
                'lambda_boundary': j in (0, len(grids[b])-1),
                'bandwidth_boundary': b in (0, len(multipliers)-1),
                'validation_loss': losses[b][j],
                'effective_complexity': complexity, 'train_months': len(train),
                'validation_months': len(validation), 'refit_months': len(train)+len(validation),
                'last_known_payoff': str((dates[validation[-1]]+pd.offsets.MonthEnd(1)).date()),
                'first_test_payoff': str((dates[test[0]]+pd.offsets.MonthEnd(1)).date())})
            for index, payoff in zip(test, returns):
                monthly.append({'kernel': kernel, 'mode': mode, 'year': year,
                                'formation_date': dates[index],
                                'return_date': dates[index]+pd.offsets.MonthEnd(1),
                                'raw_excess_return': payoff})
        for b, penalties in enumerate(grids):
            for j, penalty in enumerate(penalties):
                surfaces.append({'kernel': kernel, 'year': year,
                                 'bandwidth_multiplier': multipliers[b],
                                 'lambda': penalty, 'validation_loss': losses[b][j],
                                 'best_lambda_for_bandwidth': j == choices[b],
                                 'joint_selected': b == chosen and j == choices[b]})
        print(f'{kernel}: fit {year}, selected ell/median={multipliers[chosen]:g}', flush=True)
    return pd.DataFrame(monthly), pd.DataFrame(selections), pd.DataFrame(surfaces)


def verify_baseline(monthly, source_root, kernel):
    path = source_root / kernel / 'seed_0' / 'p_10000' / 'monthly.csv'
    original = pd.read_csv(path, parse_dates=['formation_date', 'return_date'])
    fixed = monthly[monthly['mode'] == 'fixed']
    if not np.array_equal(fixed.return_date.to_numpy(), original.return_date.to_numpy()):
        raise AssertionError('Frozen baseline dates differ.')
    delta = np.abs(fixed.raw_excess_return.to_numpy() - original.raw_excess_return.to_numpy())
    np.testing.assert_allclose(fixed.raw_excess_return, original.raw_excess_return,
                               rtol=1e-7, atol=1e-9)
    return {'source': str(path), 'sha256': digest(path),
            'maximum_absolute_payoff_difference': float(delta.max()), 'passed': True}


def summarize(monthly, selections):
    rows = []
    for (kernel, mode), frame in monthly.groupby(['kernel', 'mode']):
        selection = selections[(selections.kernel == kernel) & (selections['mode'] == mode)]
        r = frame.raw_excess_return.to_numpy()
        rows.append({'kernel': kernel, 'mode': mode, 'months': len(r),
                     'annualized_sharpe': float(sharpe(r)),
                     'quadratic_loss': float(np.mean((1-r)**2)),
                     'annualized_raw_mean': float(12*r.mean()),
                     'annualized_raw_volatility': float(np.sqrt(12)*r.std(ddof=1)),
                     'bandwidth_boundary_years': int(selection.bandwidth_boundary.sum()),
                     'lambda_boundary_years': int(selection.lambda_boundary.sum()),
                     'nonmedian_bandwidth_years': int((selection.bandwidth_multiplier != 1).sum())})
    return pd.DataFrame(rows)


def render(summary, selections, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    colors = {'gaussian': '#2a6fbb', 'matern32': '#c16b27'}
    labels = {'gaussian': 'Gaussian', 'matern32': 'Matérn-3/2'}
    for k, kernel in enumerate(('gaussian', 'matern32')):
        for j, mode in enumerate(('fixed', 'tuned')):
            row = summary[(summary.kernel == kernel) & (summary['mode'] == mode)].iloc[0]
            x = k + (j-.5)*.34
            for ax, metric, fmt in [(axes[0], 'annualized_sharpe', '.3f'),
                                    (axes[1], 'quadratic_loss', '.4f')]:
                ax.bar(x, row[metric], width=.31, color=colors[kernel],
                       alpha=.45 if mode == 'fixed' else 1.,
                       label=mode.capitalize() if k == 0 else None)
                ax.text(x, row[metric], format(row[metric], fmt), ha='center', va='bottom', fontsize=10)
        s = selections[(selections.kernel == kernel) & (selections['mode'] == 'tuned')]
        axes[2].step(s.year, s.bandwidth_multiplier, where='mid', color=colors[kernel],
                     label=labels[kernel], lw=1.4)
    for ax, title in zip(axes[:2], ['OOS annualized Sharpe ↑', 'OOS quadratic loss ↓']):
        ax.set_title(title, pad=15)
        ax.set_xticks([0, 1], ['Gaussian', 'Matérn-3/2'])
        ax.margins(y=.18)
        ax.legend(frameon=False)
    axes[2].set_title('Bandwidth selected by validation', pad=15)
    axes[2].set_yscale('log', base=2)
    axes[2].set_yticks(list(MULTIPLIERS), [str(x) for x in MULTIPLIERS])
    axes[2].set_ylabel('Multiple of initial median distance')
    axes[2].set_xlabel('Decision year')
    axes[2].legend(frameon=False, loc='upper left')
    fig.suptitle('Joint bandwidth–ridge experiment | 10,000 features, seed 0\n'
                 '47 annual decisions · February 1978–January 2025 OOS payoffs', fontsize=14)
    fig.savefig(output / 'comparison.png', dpi=180)
    fig.savefig(output / 'comparison.pdf')
    plt.close(fig)


def run_kernel(kernel, panels, args, ell0, clean_hash):
    # Each worker owns its yearly panel cache; mutable panel caches are not shared.
    panels = PanelSequence(panels.root, panels.names, panels.records)
    folder = args.out/kernel
    folder.mkdir(parents=True, exist_ok=True)
    bank = FeatureBank(kernel, 130, 10000, ell0, 0)
    atomic_json(folder/'status.json', {'status': 'running', 'stage': 'managed'})
    g = build_cache(panels, bank, MULTIPLIERS, folder/'cache', clean_hash)
    atomic_json(folder/'status.json', {'status': 'running', 'stage': 'fitting'})
    monthly, selections, surface = fit_experiment(g, panels.dates, MULTIPLIERS, kernel, ell0)
    check = verify_baseline(monthly, args.baseline, kernel)
    monthly.to_csv(folder/'monthly.csv', index=False)
    selections.to_csv(folder/'selections.csv', index=False)
    surface.to_csv(folder/'validation_surface.csv', index=False)
    atomic_json(folder/'status.json', {'status': 'complete', 'baseline_check': check})
    return monthly, selections, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', type=Path, default=Path('data/clean'))
    parser.add_argument('--out', type=Path, default=Path('results/bandwidth_tuning'))
    parser.add_argument('--baseline', type=Path, default=Path('results/final/public'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    config = {'multipliers': list(MULTIPLIERS), 'features': 10000, 'seed': 0,
              'kernel_workers': args.workers, 'stock_batch_size': BANDWIDTH_BATCH_SIZE,
              'lambda_count': 120, 'clean_manifest_sha256': digest(args.clean/'manifest.json'),
              'code_sha256': {name: digest(source_dir/name) for name in
                              ('bandwidth_tuning.py', 'kernels.py', 'portfolio.py', 'data_pipeline.py')},
              'baseline_sha256': {kernel: digest(args.baseline/kernel/'seed_0'/'p_10000'/'monthly.csv')
                                   for kernel in ('gaussian', 'matern32')},
              'selection': 'Joint minimum validation mean (1-raw payoff)^2, per kernel and year.',
              'lambda_grid': '120 effective-complexity-spaced penalties, frozen separately for each '
                             'kernel/bandwidth using initial 1963-1972 training.',
              'timing': 'Expanding training, last 60 historical months validation; January-close '
                        'refit on train+validation, February-January OOS payoffs.',
              'interpretation': 'Exploratory historical comparison; same complete-payoff sample '
                                'and execution assumptions as frozen baseline; no trading costs.',
              'scope': 'One RFF seed; bandwidth and lambda selected, feature count fixed. '
                       'Existing paper protocol and results are not modified.'}
    config_path = args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Experiment configuration changed: use a new output directory.')
    atomic_json(config_path, config)
    atomic_json(args.out/'status.json', {'status': 'running', 'stage': 'checking_clean_inputs'})
    print('Validating clean panel files and initial bandwidth sample...', flush=True)
    panels, clean_meta = load_panels(args.clean)
    sample_path = args.clean/'initial_sample.npy'
    if digest(sample_path) != clean_meta['sample_sha256']:
        raise ValueError('Initial sample checksum mismatch.')
    sample = np.load(sample_path, allow_pickle=False)
    if sample.shape != (1000, 130):
        raise ValueError('Require original 1000 x 130 initial sample.')
    ell0 = median_distance(sample)
    months, choices, checks = [], [], {}
    atomic_json(args.out/'status.json', {'status': 'running', 'stage': 'kernel_runs'})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {kernel: pool.submit(run_kernel, kernel, panels, args, ell0,
                                       config['clean_manifest_sha256'])
                   for kernel in ('gaussian', 'matern32')}
        for kernel, future in futures.items():
            monthly, selections, checks[kernel] = future.result()
            months.append(monthly)
            choices.append(selections)
    monthly, selections = pd.concat(months, ignore_index=True), pd.concat(choices, ignore_index=True)
    if monthly.groupby(['kernel', 'mode']).size().ne(564).any():
        raise AssertionError('Incomplete OOS coverage.')
    summary = summarize(monthly, selections)
    summary.to_csv(args.out/'summary.csv', index=False)
    selections.to_csv(args.out/'selections.csv', index=False)
    monthly.to_csv(args.out/'monthly.csv', index=False)
    subperiods = []
    for label, low, high in [('1978-1994', 1978, 1994), ('1995-2009', 1995, 2009), ('2010-2024', 2010, 2024)]:
        frame = summarize(monthly[monthly.year.between(low, high)],
                          selections[selections.year.between(low, high)])
        frame.insert(0, 'decision_years', label)
        subperiods.append(frame)
    pd.concat(subperiods).to_csv(args.out/'subperiods.csv', index=False)
    render(summary, selections, args.out)
    atomic_json(args.out/'verification.json', {'baseline_reproduction': checks,
                                            'full_oos_months_per_strategy': 564})
    atomic_json(args.out/'status.json', {
        'status': 'complete', 'median_bandwidth': ell0,
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'config_sha256': digest(config_path),
        'outputs_sha256': {str(p.relative_to(args.out)): digest(p) for p in
                           sorted(args.out.rglob('*')) if p.suffix in ('.csv', '.png', '.pdf')}})
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
