"""Independent V2 raw, coefficient, payoff and aggregate reconstruction.

No import of the production summarizer, selector, fitting or metric evaluator.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.diagnostics.reconstruct import equal, summary, read_table
from simulations.provenance import ROOT, file_hash, json_write, require, utc_now

METRICS = ('complexity', 'population_loss', 'population_sharpe', 'oos_loss', 'oos_sharpe',
           'forward_loss', 'forward_sharpe', 'regret', 'estimation_norm', 'cross_term', 'estimation_signed')
METHODS = ('holdout25', 'rolling3', 'theory_0.25', 'theory_1', 'theory_4',
           'ensemble_plugin_loss_oracle', 'crossfit_loss_oracle')


def family_seed(master, family, index=0):
    return int(np.random.SeedSequence([master, family, index]).generate_state(1, dtype=np.uint32)[0])


def raw_check(directory, config, environment, allow_partial=False):
    directory = Path(directory)
    pop = json.loads((directory/'population.json').read_text())
    files = sorted((directory/'replications').glob('*.npz'))
    expected = [f'{r:04d}.npz' for r in range(environment['replications'])]
    if not allow_partial:
        require([p.name for p in files] == expected, 'Missing or extra V2 checkpoints')
    require(bool(files), 'No completed V2 checkpoints')
    with np.load(directory/'population.npz', allow_pickle=False) as saved:
        m, S = saved['mean'], saved['second']
        require(str(saved['run_hash']) == pop['run_hash'], 'Population cache identity')
    require(np.isfinite(m).all() and np.isfinite(S).all(), 'Nonfinite population moment')
    eig, U = np.linalg.eigh(S)
    require(eig[0] > 0, 'Nonpositive population eigenvalue')
    projection = U.T@m
    p = pop['parameters']
    g = p['c_beta']/3
    q = g/(g+p['sigma_eps']**2/p['N'])*np.dot(p['mu_F'], p['mu_F'])
    equal(pop['q_reference'], q, 'Analytic q star')
    equal(pop['reference_sharpe'], np.sqrt(q/(1-q)), 'Analytic SR star')
    floor = q-(projection**2/eig).sum()
    equal(pop['subspace_regret_floor'], floor, 'Population floor')
    R, nt, nl = environment['replications'], len(config['T']), len(config['penalties'])
    raw = []
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            v = dict(saved)
        r = int(path.stem)
        require(r < R and int(v['replication']) == r, 'Replication identity')
        require(str(v['run_hash']) == pop['run_hash'], 'Raw run identity')
        require(str(v['protocol_hash']) == config['protocol_hash'], 'Raw protocol identity')
        sizes = config['T']+(config['extra_T'] if r < config['extended_replications'] and environment['nu'] == 1.5 else [])
        equal(v['T'], sizes, 'Raw T identity', rtol=0, atol=0)
        expected_seeds = [family_seed(config['master_seed'], family, r) for family in (4, 5)]
        require([int(v['train_seed']), int(v['test_seed'])] == expected_seeds, 'Wrong train/test seed')
        equal(v['seed_entropy'], [config['master_seed'], r], 'Seed entropy', rtol=0, atol=0)
        equal(v['loading_map'], environment['loading_map'], 'Loading map')
        equal(v['eta'], environment['eta'], 'Loading eta')
        equal(v['oos_labels'], ['independent_stationary_test', 'contiguous_future_test'], 'OOS identities')
        require(v['date_audit'].shape == (5,) and np.isfinite(v['date_audit']).all(), 'Date audit shape')
        require(v['date_audit'][0] == max(sizes)+2*config['oos_periods'], 'Incorrect observed date count')
        require(np.max(v['date_audit'][1:]) < 1e-10, 'Production loading/pricing invariant')
        for name in (*METRICS, 'population_bias', 'population_complexity', 'penalties'):
            require(v[name].shape == (len(sizes), nl+3) and np.isfinite(v[name]).all(), 'Invalid raw '+name)
        require(v['selected_coefficients'].shape == (len(sizes), environment['rank'], 5), 'Selected coefficient shape')
        require(v['choices'].shape == (len(sizes), 5), 'Choice shape')
        require(v['normal_equation_max_error'].shape == (len(sizes),), 'Normal-equation diagnostic shape')
        require(np.max(v['normal_equation_max_error']) < 1e-8, 'Failed training normal equation')
        for kind in ('independent', 'future'):
            require(v[f'selected_{kind}_payoffs'].shape == (len(sizes), config['oos_periods'], 5), 'Test payoff shape')
        require(np.all(v['complexity'] >= -1e-8), 'Negative complexity')
        require(np.all(v['complexity'] <= np.minimum(sizes, environment['rank'])[:, None]+1e-6), 'Sample complexity rank bound')
        require(np.max(np.abs(v['population_sharpe'])) <= np.sqrt(q/(1-q))+1e-9, 'Population Sharpe bound')
        require(v['regret'].min() >= -1e-9, 'Negative population regret')
        for t, T in enumerate(sizes):
            scale = dict(config['theory_scale'])[environment['name']]
            b = 1+2*environment['nu']/6
            lam = np.r_[config['penalties'], np.asarray([.25, 1, 4])*scale*T**(-b/(b+1))]
            equal(v['penalties'][t], lam, 'Actual penalty formula', rtol=1e-12, atol=1e-18)
            denominator = eig[:, None]+lam
            bias = q-(projection[:, None]**2*(eig[:, None]+2*lam)/denominator**2).sum(axis=0)
            C = (eig[:, None]/denominator).sum(axis=0)
            equal(v['population_bias'][t], bias, 'Population bias')
            equal(v['population_complexity'][t], C, 'Population complexity')
            equal(v['regret'][t], v['population_loss'][t]-(1-q), 'Regret definition')
            equal(v['regret'][t], bias+v['estimation_norm'][t]+v['cross_term'][t], 'Signed decomposition')
            equal(v['estimation_signed'][t], v['estimation_norm'][t]+v['cross_term'][t], 'Signed estimation definition')
            hold = v['holdout25_validation_loss'][t]
            rolling = v['rolling3_validation_loss'][t]
            require(hold.shape == (nl,) and rolling.shape == (nl,) and np.isfinite(hold).all() and np.isfinite(rolling).all(), 'Validation shape/value')
            tied = np.flatnonzero(rolling <= rolling.min()+1e-12)
            selected = [int(hold.argmin()), int(tied[-1]), nl, nl+1, nl+2]
            equal(v['choices'][t], selected, 'Training-only choices', rtol=0, atol=0)
            a = v['selected_coefficients'][t]
            require(np.isfinite(a).all(), 'Nonfinite coefficient')
            expected_payoff = m@a
            second_payoff = np.einsum('ij,ij->j', a, S@a)
            equal(v['population_loss'][t, selected], 1-2*expected_payoff+second_payoff, 'Selected coefficient risk')
            equal(v['population_sharpe'][t, selected], expected_payoff/np.sqrt(second_payoff-expected_payoff**2), 'Selected coefficient Sharpe')
            for kind, loss_name, sr_name in [('independent', 'oos_loss', 'oos_sharpe'), ('future', 'forward_loss', 'forward_sharpe')]:
                payoff = v[f'selected_{kind}_payoffs'][t]
                require(np.isfinite(payoff).all(), 'Nonfinite '+kind+' payoff')
                equal(v[loss_name][t, selected], ((payoff-1)**2).sum(axis=0)/len(payoff), kind+' payoff loss')
                mean = payoff.sum(axis=0)/len(payoff)
                sd = np.sqrt(((payoff-mean)**2).sum(axis=0)/(len(payoff)-1))
                equal(v[sr_name][t, selected], mean/sd, kind+' payoff Sharpe')
        raw.append(v)
    return raw, q


def independent_folds(R, master):
    order = np.random.default_rng(family_seed(master, 7)).permutation(R)
    labels = np.zeros(R, dtype=int)
    for j, r in enumerate(order):
        labels[r] = j%5
    return labels


def statistics(x):
    mean, median, se = summary(x)
    lo, q25, q75, hi = np.quantile(x, [.05, .25, .75, .95])
    return dict(mean=mean, median=median, sd=se*np.sqrt(len(x)), mcse=se,
                ci_low=mean-1.96*se, ci_high=mean+1.96*se, q05=lo, q25=q25, q75=q75, q95=hi)


def check_statistics(row, values, prefix):
    for name, x in statistics(values).items():
        equal(row[prefix+'_'+name], x, prefix+'/'+name)


def verify_aggregates(directory, config, env, raw, q):
    directory = Path(directory)
    curves = read_table(directory/'curves.csv', ['T', 'lambda'])
    methods = read_table(directory/'methods.csv', ['T', 'method'])
    paths = read_table(directory/'path_methods.csv', ['T', 'method', 'replication'])
    diagnostics = read_table(directory/'selection_diagnostics.csv', ['T', 'selector'])
    optimism = read_table(directory/'oracle_optimism.csv', ['T'])
    nl, R = len(config['penalties']), len(raw)
    folds = independent_folds(R, config['master_seed'])
    metadata = json.loads((directory/'summary_metadata.json').read_text())
    equal(metadata['folds'], folds, 'Fold assignment', rtol=0, atol=0)
    sizes = sorted({int(t) for v in raw for t in v['T']})
    require(len(curves) == nl*len(sizes) and len(methods) == len(sizes)*10, 'Published row counts')
    require(len(diagnostics) == len(sizes)*2 and len(optimism) == len(sizes), 'Selection/optimism row counts')
    expected_path_count = 0
    for t, T in enumerate(sizes):
        members = [r for r, v in enumerate(raw) if T in v['T']]
        positions = [int(np.flatnonzero(raw[r]['T'] == T)[0]) for r in members]
        n = len(members)
        expected_path_count += n*len(METHODS)
        values = {key: np.stack([raw[r][key][j] for r, j in zip(members, positions)])
                  for key in (*METRICS, 'population_bias', 'population_complexity', 'penalties', 'choices')}
        c = curves[curves['T'] == T]
        equal(c['lambda'], config['penalties'], 'Published lambda order')
        equal(c.replications, np.repeat(n, nl), 'Curve counts')
        for j in range(nl):
            row = c.iloc[j]
            for key in METRICS:
                check_statistics(row, values[key][:, j], key)
            for key in ('population_complexity', 'population_bias'):
                equal(row[key], values[key][0, j], key)
            equal(row.projection_floor, float(raw[0]['subspace_floor']), 'Projection floor')
        oracle = values['regret'][:, :nl].mean(axis=0).argmin()
        cross = np.empty(n, dtype=int)
        for fold in range(5):
            other = folds[members] != fold
            require(other.any() and (~other).any(), 'Missing calibration/evaluation fold')
            cross[~other] = values['regret'][other, :nl].mean(axis=0).argmin()
        selections = np.c_[values['choices'], np.full(n, oracle), cross]
        for j, method in enumerate(METHODS):
            index = selections[:, j]
            selected = {key: x[np.arange(n), index] for key, x in values.items() if key != 'choices'}
            selected['lambda'] = selected.pop('penalties')
            b = 1+2*env['nu']/6
            selected.update(normalized_regret=selected['regret']/q, complexity_over_T=selected['complexity']/T,
                            scaled_regret=selected['regret']*T**(b/(b+1)),
                            scaled_population_complexity=selected['population_complexity']/T**(1/(b+1)),
                            sharpe_gap=np.sqrt(q/(1-q))-selected['population_sharpe'])
            row = methods[(methods['T'] == T)&(methods.method == method)].iloc[0]
            require(row.replications == n, 'Method replication count')
            actual = paths[(paths['T'] == T)&(paths.method == method)]
            equal(actual.replication, members, 'Method path order', rtol=0, atol=0)
            equal(actual['fold'], folds[members], 'Method fold')
            equal(actual.lambda_index, index, 'Method selected grid index')
            for key, x in selected.items():
                equal(actual[key], x, 'Path method/'+key)
                check_statistics(row, x, key)
            if j < 2:
                diagnostic = diagnostics[(diagnostics['T'] == T)&(diagnostics.selector == method)].iloc[0]
                lam = selected['lambda']
                expected = {'lambda_mean': lam.mean(), 'lambda_median': np.median(lam),
                            'lambda_q25': np.quantile(lam, .25), 'lambda_q75': np.quantile(lam, .75),
                            'lambda_iqr': np.quantile(lam, .75)-np.quantile(lam, .25),
                            'lambda_geometric_mean': np.exp(np.log(lam).mean()),
                            'lower_frequency': np.count_nonzero(index == 0)/n,
                            'upper_frequency': np.count_nonzero(index == nl-1)/n,
                            'lower_contribution_to_mean': lam[index == 0].sum()/n,
                            'upper_contribution_to_mean': lam[index == nl-1].sum()/n, 'zero_policy_frequency': 0.}
                for key, x in expected.items():
                    equal(diagnostic[key], x, 'Selection diagnostic/'+key)
                for key in ('regret', 'complexity'):
                    check_statistics(diagnostic, selected[key], key)
        gap = values['regret'][np.arange(n), cross]-values['regret'][:, oracle]
        check_statistics(optimism[optimism['T'] == T].iloc[0], gap, 'crossfit_minus_plugin_regret')
        p = env
        for benchmark in ('zero', 'linear_validation', 'equal_weight'):
            row = methods[(methods['T'] == T)&(methods.method == benchmark)].iloc[0]
            if benchmark == 'zero':
                expected = {k: np.full(n, v) for k, v in dict(oos_loss=1., oos_sharpe=0., forward_loss=1., forward_sharpe=0.,
                            population_loss=1., population_sharpe=0., regret=q, complexity=0.).items()}
            elif benchmark == 'linear_validation':
                a = np.stack([raw[r]['linear_benchmark'][j] for r, j in zip(members, positions)])
                expected = dict(zip(('oos_loss', 'oos_sharpe', 'forward_loss', 'forward_sharpe', 'complexity', 'lambda', 'population_loss', 'population_sharpe', 'regret'), a.T))
            else:
                a = np.stack([raw[r]['equal_benchmark'][j] for r, j in zip(members, positions)])
                pop = json.loads((directory/'population.json').read_text())['parameters']
                m = np.sqrt(pop['c_beta']/3)*pop['mu_F'][2]
                s = pop['c_beta']/3+pop['sigma_eps']**2/pop['N']
                risk = 1-2*m+s
                expected = dict(zip(('oos_loss', 'oos_sharpe', 'forward_loss', 'forward_sharpe'), a.T))
                expected.update(population_loss=np.full(n, risk), population_sharpe=np.full(n, m/np.sqrt(s-m*m)), regret=np.full(n, risk-(1-q)))
            for key, x in expected.items():
                check_statistics(row, x, key)
    require(len(paths) == expected_path_count, 'Path table count')
    return {'passed': True, 'raw_paths': R, 'curve_rows': len(curves), 'method_rows': len(methods), 'path_rows': len(paths)}


def verify_rates(directory, config, env, raw, q):
    """Resample actual row indices, independently of production weight formulas."""
    table = read_table(Path(directory)/'rates.csv', ['cohort', 'method', 'quantity', 'window'])
    optimism = read_table(Path(directory)/'oracle_optimism_bootstrap.csv', ['cohort', 'T'])
    cohorts = [('practical', config['T'], len(raw))]
    if len(raw[0]['T']) > len(config['T']):
        cohorts.append(('extended', config['T']+config['extra_T'], config['extended_replications']))
    labels = independent_folds(len(raw), config['master_seed'])
    checked, optimism_checked = 0, 0
    for cohort, sizes, R in cohorts:
        nt, nl = len(sizes), len(config['penalties'])
        names = ('regret', 'population_sharpe', 'complexity', 'population_complexity', 'penalties', 'choices')
        v = {name: np.asarray([r[name][:nt] for r in raw[:R]]) for name in names}
        quantities = [('penalties', 'lambda_mean'), ('complexity', 'sample_complexity'),
                      ('population_complexity', 'population_complexity'), ('regret', 'regret'),
                      ('population_sharpe', 'sharpe_gap')]
        windows = {'full': np.arange(nt), 'upper_half': np.arange(nt//2, nt), 'largest_four': np.arange(max(0, nt-4), nt)}
        windows = {key: idx for key, idx in windows.items() if len(idx) >= 3}
        def series(bag):
            oracle = v['regret'][bag, :, :nl].mean(axis=0).argmin(axis=1)
            cross = np.empty((R, nt), dtype=int)
            for fold in range(5):
                training_bag = bag[labels[bag] != fold]
                require(len(training_bag) > 0, 'Undefined crossfit bootstrap calibration')
                selected = v['regret'][training_bag, :, :nl].mean(axis=0).argmin(axis=1)
                cross[labels[bag] == fold] = selected
            indices = [v['choices'][bag, :, j] for j in range(5)]
            indices.extend([np.broadcast_to(oracle, (R, nt)), cross])
            result = {}
            for method, index in zip(METHODS, indices):
                for source, label in quantities:
                    x = v[source][bag[:, None], np.arange(nt)[None, :], index]
                    if label == 'sharpe_gap':
                        x = np.sqrt(q/(1-q))-x
                    result[method, label] = x.sum(axis=0)/R
            return result
        def regression(y, idx):
            x, logy = np.log(np.asarray(sizes)[idx]), np.log(y[idx])
            centered = x-x.mean()
            slope = centered@(logy-logy.mean())/(centered@centered)
            residual = logy-logy.mean()-slope*centered
            se = np.sqrt((residual@residual)/((len(idx)-2)*(centered@centered)))
            return slope, se
        actual = series(np.arange(R))
        draws = {(key, win): [] for key in actual for win in windows}
        gap_draws = []
        rng = np.random.default_rng(family_seed(config['master_seed'], 6, 0 if cohort == 'practical' else 1))
        for _ in range(config['bootstrap_replications']):
            bag = rng.integers(R, size=R)
            sampled = series(bag)
            gap_draws.append(sampled['crossfit_loss_oracle', 'regret']
                             -sampled['ensemble_plugin_loss_oracle', 'regret'])
            for key, y in sampled.items():
                for win, idx in windows.items():
                    if np.all(y[idx] > 0):
                        draws[key, win].append(regression(y, idx)[0])
        for (method, quantity), y in actual.items():
            for win, idx in windows.items():
                rows = table[(table.cohort == cohort)&(table.method == method)&(table.quantity == quantity)&(table.window == win)]
                require(len(rows) == 1, 'Missing/duplicate reported rate')
                row = rows.iloc[0]
                slope, se = regression(y, idx)
                lo, hi = np.quantile(draws[(method, quantity), win], [.025, .975])
                b = 1+2*env['nu']/6
                expected = {'T_first': sizes[idx[0]], 'T_last': sizes[idx[-1]], 'replications': R,
                            'OLS_slope': slope, 'OLS_SE': se, 'bootstrap_low': lo, 'bootstrap_high': hi,
                            'bootstrap_draws': len(draws[(method, quantity), win]),
                            'theoretical_slope': 1/(b+1) if 'complexity' in quantity else -b/(b+1)}
                for key, value in expected.items():
                    equal(row[key], value, f'rate/{cohort}/{method}/{quantity}/{win}/{key}', atol=3e-10)
                checked += 1
        low, high = np.percentile(gap_draws, [2.5, 97.5], axis=0)
        gap = actual['crossfit_loss_oracle', 'regret']-actual['ensemble_plugin_loss_oracle', 'regret']
        for t, T in enumerate(sizes):
            rows = optimism[(optimism.cohort == cohort)&(optimism['T'] == T)]
            require(len(rows) == 1, 'Missing bootstrap optimism cell')
            expected = {'replications': R, 'crossfit_minus_plugin_regret': gap[t],
                        'bootstrap_low': low[t], 'bootstrap_high': high[t],
                        'bootstrap_draws': config['bootstrap_replications']}
            for key, value in expected.items():
                equal(rows.iloc[0][key], value, 'Oracle optimism bootstrap/'+key, atol=3e-12)
            optimism_checked += 1
    require(len(table) == checked, 'Extra unverified rate rows')
    require(len(optimism) == optimism_checked, 'Extra bootstrap optimism rows')
    return {'passed': True, 'rows_reconstructed': checked, 'optimism_rows_reconstructed': optimism_checked,
            'bootstrap_replications': config['bootstrap_replications']}


def verify(output, raw_only=False, allow_partial=False):
    output = Path(output)
    manifest = json.loads((output/'run_manifest.json').read_text())
    config = manifest['configuration']
    reports = []
    for env in config['cases']:
        directory = output/'data'/env['name']
        if allow_partial and not (directory/'replications').exists():
            continue
        raw, q = raw_check(directory, config, env, allow_partial)
        report = {'environment': env['name'], 'raw_paths': len(raw), 'raw_coefficients_and_payoffs_verified': True}
        if not raw_only:
            report['aggregates'] = verify_aggregates(directory, config, env, raw, q)
            report['rate_bootstrap'] = verify_rates(directory, config, env, raw, q)
        reports.append(report)
        print('Verified '+env['name']+f': {len(raw)} raw paths', flush=True)
    result = {'schema': 'confirmation-reconstruction/2.0', 'completed_utc': utc_now(),
              'verifier_sha256': file_hash(__file__), 'partial': allow_partial, 'raw_only': raw_only,
              'verifier_sources': {str(Path(name).relative_to(ROOT)): file_hash(name) for name in
                  (__file__, Path(__file__).with_name('reconstruct.py'), Path(__file__).parents[1]/'provenance.py')},
              'input_hashes': {str(p.relative_to(output)): file_hash(p)
                  for p in sorted((output/'data').rglob('*'))
                  if p.is_file() and p.suffix in ('.npz', '.csv', '.json')},
              'environments': reports, 'passed': True,
              'rate_bootstrap_verification': 'not run' if raw_only else 'independently reconstructed by resampled row indices',
              'limitations': 'Coefficient/payoff reconstruction covers the five stored training-only/theory policies; grid metrics are separately aggregated from all saved surfaces.'}
    json_write(output/'audit'/('partial_raw_verification.json' if allow_partial else 'confirmation_reconstruction.json'), result)
    return result


def verify_published(output):
    """Clone-only relationships; deliberately does not certify missing raw data."""
    output = Path(output)
    config = json.loads((output/'run_manifest.json').read_text())['configuration']
    counts = []
    for env in config['cases']:
        directory = output/'data'/env['name']
        methods = read_table(directory/'methods.csv', ['T', 'method'])
        paths = read_table(directory/'path_methods.csv', ['T', 'method', 'replication'])
        curves = read_table(directory/'curves.csv', ['T', 'lambda'])
        numeric = paths.select_dtypes(include=np.number)
        require(np.isfinite(numeric.to_numpy()).all(), 'Nonfinite path aggregate')
        for (T, method), frame in paths.groupby(['T', 'method']):
            rows = methods[(methods['T'] == T)&(methods.method == method)]
            require(len(rows) == 1, 'Missing method aggregate')
            row = rows.iloc[0]
            require(row.replications == len(frame), 'Method/path count mismatch')
            for key in ('lambda', *METRICS, 'population_complexity', 'population_bias', 'normalized_regret',
                        'complexity_over_T', 'scaled_regret', 'scaled_population_complexity', 'sharpe_gap'):
                check_statistics(row, frame[key].to_numpy(), key)
        for T, frame in curves.groupby('T'):
            equal(frame['lambda'], config['penalties'], 'Published curve penalty grid')
            row = frame.iloc[frame.regret_mean.argmin()]
            oracle = methods[(methods['T'] == T)&(methods.method == 'ensemble_plugin_loss_oracle')].iloc[0]
            equal(oracle.lambda_mean, row['lambda'], 'Published oracle penalty')
            for key in METRICS:
                equal(oracle[key+'_mean'], row[key+'_mean'], 'Published oracle mean/'+key)
                equal(oracle[key+'_mcse'], row[key+'_mcse'], 'Published oracle MCSE/'+key)
        counts.append({'environment': env['name'], 'method_rows': len(methods), 'path_rows': len(paths)})
    return {'passed': True, 'schema': 'confirmation-aggregate-check/2.0', 'environments': counts,
            'scope': 'cross-aggregate relationships only; no raw coefficient or bootstrap reconstruction'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    parser.add_argument('--raw-only', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--aggregates-only', action='store_true')
    args = parser.parse_args()
    result = verify_published(args.output) if args.aggregates_only else verify(args.output, args.raw_only, args.allow_partial)
    print(json.dumps(result, indent=2))
