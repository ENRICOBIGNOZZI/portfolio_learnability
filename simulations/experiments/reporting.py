"""Monte Carlo summaries and whole-replication bootstrap, with fixed rate windows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


def summarize(context, destination):
    design, env = context['design'], context['environment']
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = [Path(context['checkpoint_dir'])/f'{r:04d}.npz' for r in range(env.replications)]
    names = ('complexity', 'population_loss', 'population_sharpe', 'oos_loss', 'oos_sharpe', 'regret',
             'estimation_signed', 'estimation_norm', 'cross_term', 'validation_loss', 'linear_benchmark', 'equal_benchmark')
    values = {name: [] for name in names}
    seeds = []
    for path in checkpoints:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['run_hash']) != context['run_hash']:
                raise ValueError('Saved replication has a different provenance hash.')
            for name in names:
                values[name].append(saved[name])
            seeds.append({'replication': int(saved['replication']), 'train_seed': str(int(saved['train_seed'])),
                          'test_seed': str(int(saved['test_seed'])), 'seed_entropy': saved['seed_entropy'].tolist(),
                          'runtime_seconds': float(saved['runtime_seconds'])})
    values = {name: np.stack(rows) for name, rows in values.items()}
    R, nt, nl = values['regret'].shape
    T, lam = np.asarray(design.T), np.asarray(design.penalties)
    choices = np.argmin(values['validation_loss'], axis=2)
    oracle = np.argmin(values['regret'].mean(axis=0), axis=1)
    rows, main, selection, benchmarks = [], [], [], []
    gather = lambda array, idx: np.take_along_axis(array, idx[..., None], axis=2)[..., 0]
    selected = {name: gather(values[name], choices) for name in names[:9]}
    selected['lambda'] = lam[choices]
    for t, size in enumerate(T):
        for j, penalty in enumerate(lam):
            row = {'environment': env.name, 'T': int(size), 'lambda': penalty,
                   'population_complexity': context['population_complexity'][j],
                   'population_bias': context['population_bias'][j],
                   'population_ridge_loss': context['population_loss'][j], 'replications': R}
            for name in names[:9]:
                sample = values[name][:, t, j]
                mean, se = sample.mean(), sample.std(ddof=1)/np.sqrt(R)
                row.update({name+'_mean': mean, name+'_median': np.median(sample), name+'_mcse': se,
                            name+'_ci_low': mean-1.96*se, name+'_ci_high': mean+1.96*se})
            rows.append(row)
        j = oracle[t]
        row = {'environment': env.name, 'T': int(size), 'oracle_lambda': lam[j],
               'oracle_population_complexity': context['population_complexity'][j],
               'selected_population_complexity': context['population_complexity'][choices[:, t]].mean(),
               'selected_lambda_mean': selected['lambda'][:, t].mean(),
               'selected_lambda_mcse': selected['lambda'][:, t].std(ddof=1)/np.sqrt(R),
               'selected_lambda_median': np.median(selected['lambda'][:, t]),
               'boundary_selection_frequency': np.mean((choices[:, t]==0)|(choices[:, t]==nl-1)),
               'oracle_on_boundary': bool(j in (0, nl-1)), 'replications': R}
        for name in names[:9]:
            for label, data in [('oracle', values[name][:, t, j]), ('selected', selected[name][:, t])]:
                row[f'{label}_{name}'] = data.mean()
                row[f'{label}_{name}_mcse'] = data.std(ddof=1)/np.sqrt(R)
        row['selected_oracle_regret_ratio'] = row['selected_regret']/row['oracle_regret']
        main.append(row)
        for r in range(R):
            selection.append({'environment': env.name, 'T': int(size), 'replication': r,
                              'lambda': lam[choices[r, t]], 'complexity': selected['complexity'][r, t],
                              'regret': selected['regret'][r, t], 'sharpe': selected['population_sharpe'][r, t],
                              'pathwise_oracle_lambda': lam[np.argmin(values['regret'][r, t])],
                              'ensemble_oracle_complexity': values['complexity'][r, t, j]})
        for label, index in [('oracle', j), ('low_complexity', nl-1), ('high_complexity', 0)]:
            benchmarks.append({'environment': env.name, 'T': int(size), 'benchmark': label,
                               'oos_loss': values['oos_loss'][:, t, index].mean(),
                               'oos_sharpe': values['oos_sharpe'][:, t, index].mean(),
                               'regret': values['regret'][:, t, index].mean()})
        benchmarks.extend([
            {'environment': env.name, 'T': int(size), 'benchmark': 'validation',
             'oos_loss': selected['oos_loss'][:, t].mean(), 'oos_sharpe': selected['oos_sharpe'][:, t].mean(),
             'regret': selected['regret'][:, t].mean()},
            {'environment': env.name, 'T': int(size), 'benchmark': 'linear_validation',
             'oos_loss': values['linear_benchmark'][:, t, 0].mean(), 'oos_sharpe': values['linear_benchmark'][:, t, 1].mean(), 'regret': None},
            {'environment': env.name, 'T': int(size), 'benchmark': 'equal_weight',
             'oos_loss': values['equal_benchmark'][:, 0].mean(), 'oos_sharpe': values['equal_benchmark'][:, 1].mean(), 'regret': None}])
    rates = rate_bootstrap(values, context, oracle, choices)
    for name, data in [('curves', rows), ('main_results', main), ('selections', selection),
                       ('benchmarks', benchmarks), ('rates', rates), ('seeds', seeds)]:
        pd.DataFrame(data).to_csv(destination/f'{name}.csv', index=False)
    return pd.DataFrame(rows), pd.DataFrame(main)


def rate_bootstrap(values, context, oracle, choices):
    d, env = context['design'], context['environment']
    T, lam = np.asarray(d.T), np.asarray(d.penalties)
    R, nt, nl = values['regret'].shape
    windows = {'full': np.arange(nt), 'upper_half': np.arange(nt//2, nt),
               'largest_four': np.arange(max(0, nt-4), nt)}
    windows = {name: idx for name, idx in windows.items() if len(idx) >= 3}
    popc = context['population_complexity']
    srstar = context['reference_sharpe']
    rng = np.random.default_rng(np.random.SeedSequence([d.bootstrap_seed, env.code, env.rank]))
    def series(weights):
        mean_regret = np.tensordot(weights, values['regret'], axes=(0, 0))
        idx = np.argmin(mean_regret, axis=1)
        mean_sharpe = np.tensordot(weights, values['population_sharpe'], axes=(0, 0))
        mean_complexity = np.tensordot(weights, values['complexity'], axes=(0, 0))
        # Reapply selection to each path's saved validation losses. Population
        # risk and OOS returns do not enter this operation, even in the bootstrap.
        selection_idx = np.argmin(values['validation_loss'], axis=2)
        selected_regret = np.take_along_axis(values['regret'], selection_idx[..., None], 2)[..., 0]
        selected_sr = np.take_along_axis(values['population_sharpe'], selection_idx[..., None], 2)[..., 0]
        selected_c = np.take_along_axis(values['complexity'], selection_idx[..., None], 2)[..., 0]
        return {'oracle_lambda': lam[idx], 'oracle_complexity': popc[idx],
                'oracle_regret': mean_regret[np.arange(nt), idx],
                'oracle_sharpe_gap': srstar-mean_sharpe[np.arange(nt), idx],
                'selected_lambda': weights@lam[selection_idx],
                'selected_complexity': weights@popc[selection_idx],
                'oracle_empirical_complexity': mean_complexity[np.arange(nt), idx],
                'selected_empirical_complexity': weights@selected_c,
                'selected_regret': weights@selected_regret,
                'selected_sharpe_gap': srstar-weights@selected_sr}
    actual = series(np.ones(R)/R)
    bootstrap = {(key, win): [] for key in actual for win in windows}
    for _ in range(d.bootstrap_replications):
        weights = np.bincount(rng.integers(R, size=R), minlength=R)/R
        sample = series(weights)
        for key, y in sample.items():
            for win, idx in windows.items():
                if np.all(y[idx] > 0):
                    bootstrap[key, win].append(float(np.polyfit(np.log(T[idx]), np.log(y[idx]), 1)[0]))
    result = []
    b = env.theoretical_b
    for key, y in actual.items():
        theoretical = 1/(b+1) if 'complexity' in key else -b/(b+1)
        # Current manuscript, eq:sharpe-level-rate: E6 gives a common positive
        # Sharpe lower bound, hence the full regret exponent also for SR gaps.
        for win, idx in windows.items():
            if np.any(y[idx] <= 0):
                raise ValueError('Nonpositive rate quantity: log-log regression is not defined.')
            fit = linregress(np.log(T[idx]), np.log(y[idx]))
            draws = bootstrap[key, win]
            lo, hi = np.quantile(draws, [.025, .975]) if draws else (np.nan, np.nan)
            result.append({'environment': env.name, 'quantity': key, 'window': win,
                           'T_first': T[idx[0]], 'T_last': T[idx[-1]], 'theoretical_slope': theoretical,
                           'OLS_slope': fit.slope, 'OLS_SE': fit.stderr,
                           'bootstrap_low': lo, 'bootstrap_high': hi, 'bootstrap_draws': len(draws),
                           'complexity_definition': ('empirical trace' if 'empirical_complexity' in key else
                                                     'population trace' if 'complexity' in key else 'not applicable'),
                           'interpretation': ('Theory benchmark/envelope; no equality claimed for a fixed smooth target.' if env.variant=='baseline' else 'Reference slope only: this misspecification does not satisfy all theorem assumptions.')})
    return result
