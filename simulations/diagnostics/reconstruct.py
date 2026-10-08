"""Independent, read-only reconstruction of V1 checkpoint aggregates.

Does not import the production summarizer, ridge solver, table builder or design.
Reads the frozen run configuration, not today's default configuration. The full
mode establishes raw-to-CSV consistency; it is not a replay of unrecorded stock
returns. The fast mode establishes cross-table consistency without raw NPZs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.provenance import file_hash, json_write, require, utc_now

METRICS = ('complexity', 'population_loss', 'population_sharpe', 'oos_loss',
           'oos_sharpe', 'regret', 'estimation_signed', 'estimation_norm', 'cross_term')


def equal(actual, expected, label, rtol=3e-8, atol=3e-13):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f'{label}: shape {a.shape} != {b.shape}')
    if b.dtype.kind in 'OUSb':
        require(np.array_equal(a.astype(str), b.astype(str)), f'{label}: values differ')
    else:
        require(np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: nonfinite value')
        require(np.allclose(a, b, rtol=rtol, atol=atol),
                f'{label}: reconstruction differs (max absolute error {np.max(np.abs(a-b))})')


def read_table(path, keys):
    frame = pd.read_csv(path)
    require(not frame.duplicated(keys).any(), f'{path}: duplicate keys')
    require(len(frame) > 0, f'{path}: empty table')
    return frame


def summary(x):
    x = np.asarray(x)
    mean = np.sum(x, axis=0)/len(x)
    # Explicit centered sums independently reconstruct the reported MCSE.
    se = np.sqrt(np.sum((x-mean)**2, axis=0)/(len(x)*(len(x)-1)))
    return mean, np.median(x, axis=0), se


def population(directory, config):
    p = json.loads((directory/'population.json').read_text())
    with np.load(directory/'population.npz', allow_pickle=False) as saved:
        m, S = saved['mean'], saved['second']
        require(str(saved['run_hash']) == p['run_hash'], 'Population identity mismatch')
    require(S.shape == (len(m), len(m)) and np.isfinite(S).all() and np.isfinite(m).all(),
            'Invalid population arrays')
    equal(S, S.T, 'Population symmetry')
    # NumPy/LAPACK divide-and-conquer, independent of the production SciPy solver.
    eigenvalues, vectors = np.linalg.eigh(S)
    require(eigenvalues.min() > 0, 'Population operator is not positive definite')
    a = vectors.T@m
    lam = np.asarray(config['penalties'])
    denominator = eigenvalues[:, None] + lam
    C = (eigenvalues[:, None]/denominator).sum(axis=0)
    # Spectral scalar risk formula avoids the production coefficient evaluator.
    bias = p['q_reference'] - (a[:, None]**2*(eigenvalues[:, None]+2*lam)/denominator**2).sum(axis=0)
    equal(p['subspace_regret_floor'], p['q_reference']-(a*a/eigenvalues).sum(), 'Projection floor')
    equal(p['reference_sharpe'], np.sqrt(p['q_reference']/(1-p['q_reference'])), 'Reference Sharpe')
    return p, C, bias


def load_raw(directory, config, env, pop):
    R, nt, nl = env['replications'], len(config['T']), len(config['penalties'])
    paths = sorted((directory/'replications').glob('*.npz'))
    require([p.name for p in paths] == [f'{r:04d}.npz' for r in range(R)],
            f'{directory}: missing, extra or misnamed checkpoints')
    names = METRICS + ('validation_loss', 'choice', 'linear_benchmark', 'equal_benchmark')
    arrays = {name: [] for name in names}
    seeds = []
    for r, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as raw:
            require(int(raw['replication']) == r, f'{path}: replication identity')
            require(str(raw['run_hash']) == pop['run_hash'], f'{path}: scientific identity')
            entropy = [config['master_seed'], env['code'], r]
            equal(raw['seed_entropy'], entropy, f'{path}: seed entropy', rtol=0, atol=0)
            children = np.random.SeedSequence(entropy).spawn(2)
            expected_seeds = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in children]
            require([int(raw['train_seed']), int(raw['test_seed'])] == expected_seeds, f'{path}: seed identity')
            for name in names:
                x = raw[name]
                shape = (nt, nl) if name in METRICS+('validation_loss',) else (nt,) if name == 'choice' else (nt, 4) if name == 'linear_benchmark' else (2,)
                require(x.shape == shape and np.isfinite(x).all(), f'{path}: invalid {name}')
                arrays[name].append(x)
            equal(raw['choice'], np.argmin(raw['validation_loss'], axis=1), f'{path}: selector', rtol=0, atol=0)
            audit = raw['date_audit']
            require(audit.shape == (5,) and np.isfinite(audit).all(), f'{path}: date audit')
            require(audit[0] == max(config['T'])+config['oos_periods'], f'{path}: date count')
            require(np.max(audit[[1, 2, 4]]) < 1e-11, f'{path}: loading identity')
            if env['variant'] != 'heteroskedastic':
                require(audit[3] < 1e-11, f'{path}: conditional pricing')
            seeds.append((r, *map(str, expected_seeds), str(entropy), float(raw['runtime_seconds'])))
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    require(np.all(arrays['regret'] >= -1e-9), 'Negative population regret')
    require(np.max(np.abs(arrays['population_sharpe'])) <= pop['reference_sharpe']+1e-9, 'Sharpe exceeds optimum')
    require(np.all(np.diff(arrays['complexity'], axis=2) <= 1e-8), 'Nonmonotone complexity')
    require(np.all(arrays['complexity'] <= np.minimum(config['T'], env['rank'])[None, :, None]+1e-6), 'Sample complexity exceeds rank')
    return arrays, seeds


def reconstruct_rates(v, config, env, C, sr):
    R, nt, nl = v['regret'].shape
    T, lam = np.asarray(config['T']), np.asarray(config['penalties'])
    choice = v['validation_loss'].argmin(axis=2)
    rows, times = np.arange(R)[:, None], np.arange(nt)
    selected = {k: v[k][rows, times, choice] for k in ('regret', 'population_sharpe', 'complexity')}
    windows = {'full': np.arange(nt), 'upper_half': np.arange(nt//2, nt), 'largest_four': np.arange(max(0, nt-4), nt)}
    windows = {k: idx for k, idx in windows.items() if len(idx) >= 3}
    def series(weights):
        means = {k: (weights@v[k].reshape(R, -1)).reshape(nt, nl) for k in selected}
        j = means['regret'].argmin(axis=1)
        return {'oracle_lambda': lam[j], 'oracle_complexity': C[j],
                'oracle_regret': means['regret'][times, j],
                'oracle_sharpe_gap': sr-means['population_sharpe'][times, j],
                'selected_lambda': weights@lam[choice], 'selected_complexity': weights@C[choice],
                'oracle_empirical_complexity': means['complexity'][times, j],
                'selected_empirical_complexity': weights@selected['complexity'],
                'selected_regret': weights@selected['regret'],
                'selected_sharpe_gap': sr-weights@selected['population_sharpe']}
    def fit(y, idx):
        x, y = np.log(T[idx]), np.log(y[idx])
        x = x-x.mean()
        slope = x@(y-y.mean())/(x@x)
        error = y-y.mean()-slope*x
        return slope, np.sqrt((error@error)/((len(idx)-2)*(x@x)))
    actual = series(np.full(R, 1/R))
    draws = {(key, win): [] for key in actual for win in windows}
    rng = np.random.default_rng(np.random.SeedSequence([config['bootstrap_seed'], env['code'], env['rank']]))
    for _ in range(config['bootstrap_replications']):
        indices = rng.integers(R, size=R)
        weights = np.zeros(R)
        np.add.at(weights, indices, 1/R)
        for key, y in series(weights).items():
            for win, idx in windows.items():
                if np.all(y[idx] > 0):
                    draws[key, win].append(fit(y, idx)[0])
    results = []
    for key, y in actual.items():
        b = 1+2*env['nu']/6
        for win, idx in windows.items():
            require(np.all(y[idx] > 0), 'Undefined log regression')
            slope, se = fit(y, idx)
            samples = draws[key, win]
            require(len(samples) > 0, 'No valid bootstrap draws')
            lo, hi = np.quantile(samples, [.025, .975])
            results.append({'environment': env['name'], 'quantity': key, 'window': win,
                            'T_first': T[idx[0]], 'T_last': T[idx[-1]],
                            'theoretical_slope': 1/(b+1) if 'complexity' in key else -b/(b+1),
                            'OLS_slope': slope, 'OLS_SE': se, 'bootstrap_low': lo,
                            'bootstrap_high': hi, 'bootstrap_draws': len(samples)})
    return pd.DataFrame(results)


def verify_environment(directory, config, env):
    directory = Path(directory)
    pop, C, bias = population(directory, config)
    v, seeds = load_raw(directory, config, env, pop)
    R, nt, nl = v['regret'].shape
    T, lam = np.asarray(config['T']), np.asarray(config['penalties'])
    choices, oracle = v['validation_loss'].argmin(axis=2), v['regret'].mean(axis=0).argmin(axis=1)
    equal(v['regret'], v['population_loss']-(1-pop['q_reference']), 'Regret identity')
    equal(v['regret'], bias+v['estimation_norm']+v['cross_term'], 'Signed decomposition')
    equal(v['estimation_signed'], v['estimation_norm']+v['cross_term'], 'Signed estimation term')
    curves = read_table(directory/'curves.csv', ['T', 'lambda'])
    equal(curves['T'], np.repeat(T, nl), 'Curve T/order')
    equal(curves['lambda'], np.tile(lam, nt), 'Curve lambda/order')
    equal(curves.environment, np.repeat(env['name'], nt*nl), 'Curve environment')
    equal(curves.replications, np.full(nt*nl, R), 'Curve counts')
    for name, x in [('population_complexity', C), ('population_bias', bias), ('population_ridge_loss', 1-pop['q_reference']+bias)]:
        equal(curves[name], np.tile(x, nt), name)
    for metric in METRICS:
        mean, median, se = summary(v[metric])
        for suffix, x in [('mean', mean), ('median', median), ('mcse', se), ('ci_low', mean-1.96*se), ('ci_high', mean+1.96*se)]:
            equal(curves[metric+'_'+suffix], x.ravel(), metric+'_'+suffix)
    main = read_table(directory/'main_results.csv', ['T'])
    equal(main['T'], T, 'Main T/order')
    expected = {'environment': np.repeat(env['name'], nt), 'replications': np.full(nt, R),
                'oracle_lambda': lam[oracle], 'oracle_population_complexity': C[oracle],
                'selected_population_complexity': C[choices].mean(axis=0),
                'boundary_selection_frequency': ((choices == 0)|(choices == nl-1)).mean(axis=0),
                'oracle_on_boundary': (oracle == 0)|(oracle == nl-1)}
    for key, x in zip(('mean', 'median', 'mcse'), summary(lam[choices])):
        expected['selected_lambda_'+key] = x
    selected = {}
    for name in METRICS:
        selected[name] = v[name][np.arange(R)[:, None], np.arange(nt), choices]
        for label, x in [('selected', selected[name]), ('oracle', v[name][:, np.arange(nt), oracle])]:
            mean, _, se = summary(x)
            expected[label+'_'+name], expected[label+'_'+name+'_mcse'] = mean, se
    expected['selected_oracle_regret_ratio'] = expected['selected_regret']/expected['oracle_regret']
    require(set(main.columns) == {'T', *expected}, 'Unknown or missing main columns')
    for key, x in expected.items():
        equal(main[key], x, 'main/'+key)
    selections = read_table(directory/'selections.csv', ['T', 'replication'])
    check = {'environment': np.repeat(env['name'], nt*R), 'T': np.repeat(T, R), 'replication': np.tile(np.arange(R), nt),
             'lambda': lam[choices].T.ravel(), 'complexity': selected['complexity'].T.ravel(),
             'regret': selected['regret'].T.ravel(), 'sharpe': selected['population_sharpe'].T.ravel(),
             'pathwise_oracle_lambda': lam[v['regret'].argmin(axis=2)].T.ravel(),
             'ensemble_oracle_complexity': v['complexity'][:, np.arange(nt), oracle].T.ravel()}
    require(set(check) == set(selections.columns), 'Selection schema')
    for name, x in check.items():
        equal(selections[name], x, 'selections/'+name)
    benchmarks = read_table(directory/'benchmarks.csv', ['T', 'benchmark'])
    require(len(benchmarks) == nt*6, 'Benchmark count')
    labels = ['oracle', 'low_complexity', 'high_complexity', 'validation', 'linear_validation', 'equal_weight']
    equal(benchmarks.benchmark, np.tile(labels, nt), 'Benchmark order')
    equal(benchmarks['T'], np.repeat(T, 6), 'Benchmark T')
    for t in range(nt):
        for j, label in enumerate(labels):
            row = benchmarks.iloc[t*6+j]
            for name in ('oos_loss', 'oos_sharpe', 'regret'):
                if j < 3:
                    x = v[name][:, t, [oracle[t], nl-1, 0][j]].mean()
                elif j == 3:
                    x = selected[name][:, t].mean()
                elif name == 'regret':
                    require(pd.isna(row[name]), 'Privileged regret must be explicitly absent')
                    continue
                else:
                    k = 0 if name == 'oos_loss' else 1
                    x = v['linear_benchmark'][:, t, k].mean() if j == 4 else v['equal_benchmark'][:, k].mean()
                equal(row[name], x, f'benchmark/{t}/{label}/{name}')
    actual_seeds = pd.read_csv(directory/'seeds.csv', dtype={'train_seed': str, 'test_seed': str})
    require(len(actual_seeds) == R, 'Seed row count')
    for k, name in enumerate(('replication', 'train_seed', 'test_seed', 'seed_entropy', 'runtime_seconds')):
        equal(actual_seeds[name], np.asarray([row[k] for row in seeds]), 'seeds/'+name, rtol=0, atol=1e-10)
    rates = read_table(directory/'rates.csv', ['quantity', 'window'])
    rebuilt = reconstruct_rates(v, config, env, C, pop['reference_sharpe'])
    require(len(rates) == len(rebuilt), 'Rate count')
    for key in rebuilt:
        equal(rates[key], rebuilt[key], 'rates/'+key, atol=2e-10)
    return {'environment': env['name'], 'raw_replications': R, 'passed': True,
            'csv_files_reconstructed': 6, 'bootstrap_draws': config['bootstrap_replications']}


def verify_fast(output, config):
    output = Path(output)
    tables = output/'tables'
    for filename, source in [('all_main_results', 'main_results'), ('all_rate_regressions', 'rates'), ('all_benchmarks', 'benchmarks')]:
        expected = pd.concat([pd.read_csv(output/'data'/e['name']/(source+'.csv')) for e in config['cases']], ignore_index=True)
        actual = pd.read_csv(tables/(filename+'.csv'))
        require(actual.shape == expected.shape and list(actual) == list(expected), filename+' schema')
        for column in actual:
            missing = expected[column].isna()
            require(actual[column].isna().equals(missing), filename+'/'+column+' missing values')
            equal(actual.loc[~missing, column], expected.loc[~missing, column], filename+'/'+column)
    for env in config['cases']:
        directory = output/'data'/env['name']
        curves = read_table(directory/'curves.csv', ['T', 'lambda'])
        main = read_table(directory/'main_results.csv', ['T'])
        selections = read_table(directory/'selections.csv', ['T', 'replication'])
        require(len(curves) == len(config['T'])*len(config['penalties']), 'Curve count')
        require(len(selections) == len(config['T'])*env['replications'], 'Selection count')
        for t, T in enumerate(config['T']):
            c = curves[curves['T'] == T]
            s = selections[selections['T'] == T]
            require(len(c) == len(config['penalties']) and len(s) == env['replications'], 'T group count')
            j = c.regret_mean.argmin()
            equal(main.iloc[t].oracle_lambda, c.iloc[j]['lambda'], 'Aggregate oracle lambda')
            for metric in METRICS:
                equal(main.iloc[t]['oracle_'+metric], c.iloc[j][metric+'_mean'], 'Aggregate oracle '+metric)
                equal(main.iloc[t]['oracle_'+metric+'_mcse'], c.iloc[j][metric+'_mcse'], 'Aggregate oracle MCSE '+metric)
            for source, name in [('complexity', 'complexity'), ('regret', 'regret'), ('sharpe', 'population_sharpe')]:
                mean, _, se = summary(s[source])
                equal(main.iloc[t]['selected_'+name], mean, 'Aggregate selected '+name)
                equal(main.iloc[t]['selected_'+name+'_mcse'], se, 'Aggregate selected MCSE '+name)
    return {'passed': True, 'scope': 'cross-aggregate consistency only; raw checkpoints not accessed'}


def verify_publication_tables(output, config):
    """Reconstruct displayed numeric columns from independently checked CSVs."""
    output = Path(output)
    tables = output/'tables'
    main = pd.read_csv(output/'data/baseline/main_results.csv')
    published = pd.read_csv(tables/'table3_main_results.csv')
    mapping = {'T': 'T', 'Oracle loss': 'oracle_population_loss', 'Selected loss': 'selected_population_loss',
               'Oracle SR': 'oracle_population_sharpe', 'Selected SR': 'selected_population_sharpe',
               'Oracle C': 'oracle_complexity', 'Selected C': 'selected_complexity',
               'Oracle lambda': 'oracle_lambda', 'Selected lambda': 'selected_lambda_mean',
               'Regret ratio': 'selected_oracle_regret_ratio'}
    require(set(published) == set(mapping), 'Publication main table schema')
    for column, source in mapping.items():
        x = main[source]
        if column.endswith('loss'):
            x = x.map(lambda value: float(f'{value:.6f}'))
        elif column.endswith('SR'):
            x = x.map(lambda value: float(f'{value:.5f}'))
        equal(published[column], x, 'Publication main/'+column)
    rates = pd.read_csv(output/'data/baseline/rates.csv')
    table = pd.read_csv(tables/'table4_rates.csv')
    names = {'Oracle lambda': 'oracle_lambda', 'Selected lambda': 'selected_lambda',
             'Oracle C (pop.)': 'oracle_complexity', 'Selected C (pop.)': 'selected_complexity',
             'Oracle C (sample)': 'oracle_empirical_complexity', 'Selected C (sample)': 'selected_empirical_complexity',
             'Oracle regret': 'oracle_regret', 'Selected regret': 'selected_regret',
             'Oracle SR gap': 'oracle_sharpe_gap', 'Selected SR gap': 'selected_sharpe_gap'}
    windows = {'Full': 'full', 'Upper half': 'upper_half', 'Largest four': 'largest_four'}
    numeric = {'Theory': 'theoretical_slope', 'Slope': 'OLS_slope', 'OLS SE': 'OLS_SE',
               'CI low': 'bootstrap_low', 'CI high': 'bootstrap_high'}
    require(len(table) == len(rates), 'Publication rate count')
    for _, row in table.iterrows():
        target = rates[(rates.quantity == names[row.Quantity])&(rates.window == windows[row.Window])]
        require(len(target) == 1, 'Publication rate key')
        for column, source in numeric.items():
            equal(row[column], target.iloc[0][source], 'Publication rate/'+column)
    reference = pd.read_csv(tables/'robustness_reference_status.csv')
    table = pd.read_csv(tables/'table5_robustness.csv')
    require(len(reference) == len(config['cases']) == len(table), 'Robustness table count')
    for i, env in enumerate(config['cases']):
        result = pd.read_csv(output/'data'/env['name']/'main_results.csv').iloc[-1]
        require(reference.iloc[i].Environment == env['name'], 'Robustness environment order')
        for column, source in {'Oracle regret': 'oracle_regret', 'Selected regret': 'selected_regret',
                               'Oracle C': 'oracle_complexity', 'Selected C': 'selected_complexity',
                               'Selected SR': 'selected_population_sharpe'}.items():
            equal(reference.iloc[i][column], result[source], 'Robustness reference/'+column)
            equal(table.iloc[i][column], result[source], 'Robustness publication/'+column)
        equal(reference.iloc[i]['Boundary frequency'], result.boundary_selection_frequency, 'Boundary frequency')
        equal(table.iloc[i]['Boundary (%)'], 100*result.boundary_selection_frequency, 'Boundary percent')
        for key in ('N', 'rho', 'nu'):
            equal(reference.iloc[i][key], env[key], 'Robustness parameter/'+key)
    calibrations = pd.read_csv(tables/'all_calibrations.csv')
    require(len(calibrations) == len(config['cases']), 'Calibration count')
    for i, env in enumerate(config['cases']):
        row = calibrations.iloc[i]
        for key, value in env.items():
            equal(row[key], value, 'Environment calibration/'+key)
        equal(row.theoretical_b, 1+2*env['nu']/6, 'Kernel benchmark exponent')
        p = json.loads((output/'data'/env['name']/'population.json').read_text())
        equal(row.reference_sharpe, p['reference_sharpe'], 'Calibration SR')
    spectral = pd.read_csv(tables/'appendix_spectral_rates.csv')
    for _, row in spectral.iterrows():
        env = next(e for e in config['cases'] if e['name'] == row.environment)
        rate = pd.read_csv(output/'data'/env['name']/'rates.csv')
        b = 1+2*env['nu']/6
        equal(row.theoretical_b, b, 'Spectral table b')
        equal(row.theoretical_regret_exponent, b/(b+1), 'Spectral table regret benchmark')
        equal(row.theoretical_complexity_exponent, 1/(b+1), 'Spectral table complexity benchmark')
        for quantity, column in [('oracle_regret', 'fitted_regret_exponent'), ('oracle_complexity', 'fitted_complexity_exponent')]:
            slope = rate[(rate.quantity == quantity)&(rate.window == 'full')].OLS_slope.iloc[0]
            equal(row[column], -slope if quantity == 'oracle_regret' else slope, 'Spectral learning fit')
        if env['name'] == 'baseline':
            spectral_path = output/'audit/baseline/dgp_spectrum.csv'
            if not spectral_path.exists():
                spectral_path = output.parent/'audit/baseline/dgp_spectrum.csv'
            spectrum = pd.read_csv(spectral_path)
        else:
            spectrum = pd.read_csv(output/'data'/env['name']/'spectrum.csv')
        selected = spectrum[spectrum['index'].between(30, 200)]
        x, y = np.log(selected['index'].to_numpy()), np.log(selected.managed_eigenvalue.to_numpy())
        centered = x-x.mean()
        slope = centered@(y-y.mean())/(centered@centered)
        equal(row.fitted_spectral_b, -slope, 'Descriptive spectral slope')
    return {'passed': True, 'verified': ['table3_main_results', 'table4_rates', 'table5_robustness',
                                        'robustness_reference_status', 'all_calibrations', 'appendix_spectral_rates'],
            'scope': 'Numeric publication CSVs; formatted TeX/prose must be checked separately'}


def verify(output, full=False, report=None, only=None):
    output = Path(output)
    manifest = json.loads((output/'run_manifest.json').read_text())
    config = manifest['configuration']
    records = []
    if full:
        for env in config['cases']:
            if only and env['name'] not in only:
                continue
            print('Reconstructing '+env['name'], flush=True)
            records.append(verify_environment(output/'data'/env['name'], config, env))
    fast = verify_fast(output, config)
    publication = verify_publication_tables(output, config)
    result = {'schema': 'independent-reconstruction/2.0', 'passed': True, 'completed_utc': utc_now(),
              'raw_checkpoint_mode': full, 'scope': only or 'full', 'aggregate_check': fast,
              'publication_table_check': publication,
              'verifier_sha256': file_hash(__file__), 'environments': records,
              'limitations': ['V1 checkpoints store metric surfaces, not stock returns or fitted coefficients; this verifies aggregation, not a new simulation replay.',
                              'Figure pixels, LaTeX prose and formatted publication tables require separate checks.']}
    if report:
        json_write(report, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/paper'))
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.full, args.report, args.only), indent=2))
