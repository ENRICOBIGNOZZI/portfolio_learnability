"""Monte Carlo summaries and whole-replication bootstrap, with fixed rate windows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from simulations.provenance import atomic_file, json_write, require


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
                           'OLS_slope': fit.slope, 'OLS_SE': 0. if np.ptp(np.log(y[idx])) == 0 else fit.stderr,
                           'bootstrap_low': lo, 'bootstrap_high': hi, 'bootstrap_draws': len(draws),
                           'complexity_definition': ('empirical trace' if 'empirical_complexity' in key else
                                                     'population trace' if 'complexity' in key else 'not applicable'),
                           'interpretation': ('Theory benchmark/envelope; no equality claimed for a fixed smooth target.' if env.variant=='baseline' else 'Reference slope only: this misspecification does not satisfy all theorem assumptions.')})
    return result


CONFIRMATION_METRICS = ('complexity', 'population_loss', 'population_sharpe', 'oos_loss', 'oos_sharpe',
                        'forward_loss', 'forward_sharpe', 'regret', 'estimation_norm', 'cross_term', 'estimation_signed')
CONFIRMATION_METHODS = ('holdout25', 'rolling3', 'theory_0.25', 'theory_1', 'theory_4',
                        'ensemble_plugin_loss_oracle', 'crossfit_loss_oracle')


def write_frame(path, frame):
    with atomic_file(path, 'w') as handle:
        frame.to_csv(handle, index=False)


def fold_assignment(replications):
    from simulations.diagnostics.confirmation_pilot import seed
    rng = np.random.default_rng(seed('folds'))
    labels = np.empty(replications, dtype=int)
    labels[rng.permutation(replications)] = np.arange(replications)%5
    return labels


def confirmation_choices(values, folds, weights=None):
    """Oracle calibration stays separate from implementable stored selectors."""
    R, nt, nl = values['regret'].shape
    grid_count = nl-3
    weights = np.ones(R)/R if weights is None else weights/np.sum(weights)
    choices = values['choices'].copy()
    oracle = (weights@values['regret'][:, :, :grid_count].reshape(R, -1)).reshape(nt, grid_count).argmin(axis=1)
    crossfit = np.empty((R, nt), dtype=int)
    for fold in range(5):
        calibration = folds != fold
        require(np.any(folds == fold), 'Empty evaluation fold')
        w = weights*calibration
        require(w.sum() > 0, 'Empty oracle calibration fold')
        means = (w@values['regret'][:, :, :grid_count].reshape(R, -1)).reshape(nt, grid_count)/w.sum()
        crossfit[folds == fold] = means.argmin(axis=1)
    return np.concatenate([choices, np.broadcast_to(oracle, (R, nt))[..., None], crossfit[..., None]], axis=2)


def summarize_confirmation(context, destination):
    d, env = context['design'], context['environment']
    destination = Path(destination)
    raw = []
    for r in range(env.replications):
        with np.load(Path(context['checkpoint_dir'])/f'{r:04d}.npz', allow_pickle=False) as saved:
            require(str(saved['run_hash']) == context['run_hash'], 'Confirmation provenance mismatch')
            raw.append(dict(saved))
    folds = fold_assignment(env.replications)
    all_T = sorted({int(T) for item in raw for T in item['T']})
    curves, methods, paths, selection, optimism = [], [], [], [], []
    q, sr = env.parameters().q_star, env.parameters().sr_star
    def stats(x):
        x = np.asarray(x)
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        se = sd/np.sqrt(len(x))
        return {'mean': mean, 'median': float(np.median(x)), 'sd': sd, 'mcse': se,
                'ci_low': mean-1.96*se, 'ci_high': mean+1.96*se,
                'q05': float(np.quantile(x, .05)), 'q25': float(np.quantile(x, .25)),
                'q75': float(np.quantile(x, .75)), 'q95': float(np.quantile(x, .95))}
    for T in all_T:
        indices = [r for r, item in enumerate(raw) if T in item['T']]
        rows = [int(np.flatnonzero(raw[r]['T'] == T)[0]) for r in indices]
        n = len(indices)
        v = {name: np.stack([raw[r][name][j] for r, j in zip(indices, rows)])
             for name in (*CONFIRMATION_METRICS, 'choices', 'penalties', 'population_bias', 'population_complexity')}
        choices = confirmation_choices({k: x[:, None] for k, x in v.items()}, folds[indices])[:, 0]
        for j, lam in enumerate(d.penalties):
            row = {'environment': env.name, 'T': T, 'lambda': lam, 'replications': n,
                   'population_complexity': v['population_complexity'][0, j],
                   'population_bias': v['population_bias'][0, j], 'projection_floor': float(raw[0]['subspace_floor'])}
            for name in CONFIRMATION_METRICS:
                row.update({name+'_'+k: val for k, val in stats(v[name][:, j]).items()})
            curves.append(row)
        for method_index, method in enumerate(CONFIRMATION_METHODS):
            j = choices[:, method_index]
            selected = {name: x[np.arange(n), j] for name, x in v.items() if name != 'choices'}
            selected['lambda'] = selected.pop('penalties')
            selected['normalized_regret'] = selected['regret']/q
            selected['complexity_over_T'] = selected['complexity']/T
            selected['scaled_regret'] = selected['regret']*T**(env.theoretical_b/(env.theoretical_b+1))
            selected['scaled_population_complexity'] = selected['population_complexity']/T**(1/(env.theoretical_b+1))
            selected['sharpe_gap'] = sr-selected['population_sharpe']
            row = {'environment': env.name, 'T': T, 'method': method, 'replications': n,
                   'projection_floor': float(raw[0]['subspace_floor'])}
            for name, x in selected.items():
                row.update({name+'_'+k: val for k, val in stats(x).items()})
            methods.append(row)
            for k, r in enumerate(indices):
                paths.append({'environment': env.name, 'T': T, 'method': method, 'replication': r,
                              'fold': int(folds[r]), 'lambda_index': int(j[k]), **{name: x[k] for name, x in selected.items()}})
            if method_index < 2:
                lam = selected['lambda']
                low, high = j == 0, j == len(d.penalties)-1
                selection.append({'environment': env.name, 'T': T, 'selector': method, 'replications': n,
                                  'lambda_mean': lam.mean(), 'lambda_median': np.median(lam),
                                  'lambda_q25': np.quantile(lam, .25), 'lambda_q75': np.quantile(lam, .75),
                                  'lambda_iqr': np.quantile(lam, .75)-np.quantile(lam, .25),
                                  'lambda_geometric_mean': np.exp(np.log(lam).mean()),
                                  'lower_frequency': low.mean(), 'upper_frequency': high.mean(),
                                  'lower_contribution_to_mean': np.mean(lam*low),
                                  'upper_contribution_to_mean': np.mean(lam*high), 'zero_policy_frequency': 0.,
                                  **{'complexity_'+k: val for k, val in stats(selected['complexity']).items()},
                                  **{'regret_'+k: val for k, val in stats(selected['regret']).items()}})
        gap = v['regret'][np.arange(n), choices[:, -1]]-v['regret'][np.arange(n), choices[:, -2]]
        optimism.append({'environment': env.name, 'T': T, 'replications': n,
                         **{'crossfit_minus_plugin_regret_'+k: val for k, val in stats(gap).items()},
                         'interval_note': 'pointwise paired MCSE conditional on calibrated folds; whole-path oracle recalibration in rates bootstrap'})
        for method in ('zero', 'linear_validation', 'equal_weight'):
            rows_out = []
            for r, t in zip(indices, rows):
                if method == 'zero':
                    values = dict(oos_loss=1., oos_sharpe=0., forward_loss=1., forward_sharpe=0.,
                                  population_loss=1., population_sharpe=0., regret=q, complexity=0.)
                elif method == 'linear_validation':
                    x = raw[r]['linear_benchmark'][t]
                    values = dict(zip(('oos_loss', 'oos_sharpe', 'forward_loss', 'forward_sharpe', 'complexity', 'lambda',
                                       'population_loss', 'population_sharpe', 'regret'), x))
                else:
                    x = raw[r]['equal_benchmark'][t]
                    p = env.parameters()
                    mean = np.sqrt(p.c_beta/3)*p.mu_F[2]
                    second = p.c_beta/3+p.sigma_eps**2/p.N
                    loss = 1-2*mean+second
                    values = dict(zip(('oos_loss', 'oos_sharpe', 'forward_loss', 'forward_sharpe'), x))
                    values.update(population_loss=loss, population_sharpe=mean/np.sqrt(second-mean**2), regret=loss-(1-q))
                rows_out.append(values)
            row = {'environment': env.name, 'T': T, 'method': method, 'replications': n}
            for key in rows_out[0]:
                row.update({key+'_'+k: val for k, val in stats([x[key] for x in rows_out]).items()})
            methods.append(row)
    for name, rows in [('curves', curves), ('methods', methods), ('path_methods', paths),
                       ('selection_diagnostics', selection), ('oracle_optimism', optimism)]:
        write_frame(destination/(name+'.csv'), pd.DataFrame(rows))
    rate_rows, optimism_rows = confirmation_rates(raw, d, env, folds)
    write_frame(destination/'rates.csv', pd.DataFrame(rate_rows))
    write_frame(destination/'oracle_optimism_bootstrap.csv', pd.DataFrame(optimism_rows))
    json_write(destination/'summary_metadata.json', {'schema': 'confirmation-summary/2.0',
               'run_hash': context['run_hash'], 'protocol_hash': d.protocol_hash,
               'folds': folds.tolist(), 'intervals': 'pointwise mean +/-1.96 MCSE, conditional on fixed anchors and quadrature',
               'zero_policy': 'benchmark only; SR convention 0; lambda is not a finite grid value',
               'missing_benchmark_columns': 'not defined for that benchmark, not unrecorded simulation values',
               'rate_cohorts': 'practical grid uses all paths; extended grid uses only the complete extended cohort'})
    return pd.DataFrame(methods)


def confirmation_rates(raw, design, env, folds):
    from simulations.diagnostics.confirmation_pilot import seed
    cohorts = {'practical': (list(design.T), len(raw))}
    if any(len(item['T']) > len(design.T) for item in raw):
        cohorts['extended'] = (list(design.T+design.extra_T), design.extended_replications)
    output, optimism_output = [], []
    for cohort, (sizes, R) in cohorts.items():
        nt = len(sizes)
        metrics = ('regret', 'population_sharpe', 'complexity', 'population_complexity', 'penalties', 'choices')
        v = {name: np.stack([item[name][:nt] for item in raw[:R]]) for name in metrics}
        windows = {'full': np.arange(nt), 'upper_half': np.arange(nt//2, nt), 'largest_four': np.arange(max(0, nt-4), nt)}
        windows = {name: idx for name, idx in windows.items() if len(idx) >= 3}
        times, reps = np.arange(nt)[None, :], np.arange(R)[:, None]
        def series(weights):
            choices = confirmation_choices(v, folds[:R], weights)
            result = {}
            for j, method in enumerate(CONFIRMATION_METHODS):
                index = choices[:, :, j]
                for source, quantity in [('penalties', 'lambda_mean'), ('complexity', 'sample_complexity'),
                                         ('population_complexity', 'population_complexity'), ('regret', 'regret'),
                                         ('population_sharpe', 'sharpe_gap')]:
                    data = v[source][reps, times, index]
                    if quantity == 'sharpe_gap':
                        data = env.parameters().sr_star-data
                    result[method, quantity] = weights@data
            return result
        actual = series(np.ones(R)/R)
        draws = {(key, window): [] for key in actual for window in windows}
        optimism_draws = []
        rng = np.random.default_rng(seed('bootstrap', 0 if cohort == 'practical' else 1))
        def slope(y, idx):
            x = np.log(np.asarray(sizes)[idx]); x -= x.mean()
            return float(x@np.log(y[idx])/(x@x))
        for _ in range(design.bootstrap_replications):
            weights = np.bincount(rng.integers(R, size=R), minlength=R)/R
            sampled = series(weights)
            optimism_draws.append(sampled['crossfit_loss_oracle', 'regret']
                                  -sampled['ensemble_plugin_loss_oracle', 'regret'])
            for key, y in sampled.items():
                for window, idx in windows.items():
                    if np.all(y[idx] > 0):
                        draws[key, window].append(slope(y, idx))
        for (method, quantity), y in actual.items():
            for window, idx in windows.items():
                require(np.all(y[idx] > 0), 'Nonpositive rate quantity: no log slope is defined')
                fit = linregress(np.log(np.asarray(sizes)[idx]), np.log(y[idx]))
                centered_x = np.log(np.asarray(sizes)[idx]); centered_x -= centered_x.mean()
                centered_y = np.log(y[idx]); centered_y -= centered_y.mean()
                residual = centered_y-fit.slope*centered_x
                residual_se = np.sqrt((residual@residual)/((len(idx)-2)*(centered_x@centered_x)))
                samples = draws[(method, quantity), window]
                lo, hi = np.quantile(samples, [.025, .975])
                output.append({'environment': env.name, 'cohort': cohort, 'method': method, 'quantity': quantity,
                               'window': window, 'T_first': sizes[idx[0]], 'T_last': sizes[idx[-1]], 'replications': R,
                               'OLS_slope': fit.slope, 'OLS_SE': residual_se,
                               'bootstrap_low': lo, 'bootstrap_high': hi, 'bootstrap_draws': len(samples),
                               'theoretical_slope': 1/(env.theoretical_b+1) if 'complexity' in quantity else -env.theoretical_b/(env.theoretical_b+1),
                               'interpretation': 'descriptive fit; theory is an envelope, not a required equality'})
        gap = actual['crossfit_loss_oracle', 'regret']-actual['ensemble_plugin_loss_oracle', 'regret']
        low, high = np.quantile(optimism_draws, [.025, .975], axis=0)
        for t, T in enumerate(sizes):
            optimism_output.append({'environment': env.name, 'cohort': cohort, 'T': T, 'replications': R,
                                    'crossfit_minus_plugin_regret': gap[t], 'bootstrap_low': low[t],
                                    'bootstrap_high': high[t], 'bootstrap_draws': len(optimism_draws),
                                    'interval': 'pointwise percentile; whole-path resampling and oracle recalibration; fixed fold labels, anchors and quadrature'})
    return output, optimism_output
