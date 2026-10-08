"""Read-only reconstruction of rank, quadrature and spectral audit tables.

No calls to their producers or the production risk/summary functions. Rank
curves are reconstructed from stored risk surfaces; quadrature risks are also
recomputed from fixed policy coefficients and each saved moment matrix.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.diagnostics.reconstruct import equal, read_table, summary
from simulations.provenance import ROOT, digest, file_hash, json_write, require, utc_now

POLICIES = ('holdout25', 'rolling3', 'theory_0.25', 'theory_1', 'theory_4',
            'ensemble_plugin_loss_oracle')


def seed(master, family, index):
    return int(np.random.SeedSequence([master, family, index]).generate_state(1, dtype=np.uint32)[0])


def compare_row(row, values, label):
    for key, value in values.items():
        equal(row[key], value, label+'/'+key)


def paired(low, high, floor):
    average, _, error = summary(low-high)
    reference = np.sum(low)/len(low)
    bound = abs(average)+1.96*error
    return {'reference': reference, 'difference': average, 'MCSE': error,
            'lo': average-1.96*error, 'hi': average+1.96*error,
            'bound': bound, 'certified': bool(floor <= .05*reference and bound <= .05*reference)}


def rank(output, config, env):
    audit = output/'audit'
    name = env['name']
    report = json.loads((audit/f'{name}_rank_resolution.json').read_text())
    science = json.loads((output/'run_manifest.json').read_text())['source_hashes']
    expected_identity = digest({'sources': science, 'diagnostic': file_hash(ROOT/'simulations/diagnostics/resolution_v2.py'),
                                'quadrature_implementation': file_hash(ROOT/'simulations/diagnostics/low_memory.py'),
                                'runtime': report['runtime'], 'protocol': config['protocol_hash'],
                                'environment': name, 'groups': 32768, 'subset': 50, 'T': config['T'],
                                'scope': 'practical grid only; extended horizon remains uncertified by this rank test'})
    require(report['identity'] == expected_identity, 'Rank source/runtime/configuration identity')
    directory = audit/'rank_paths_low_memory'/name
    files = sorted(p.name for p in directory.glob('*.npz') if p.name != 'moments.npz')
    require(files == [f'{r:04d}.npz' for r in range(50)], 'Incomplete paired-rank cohort')
    pop = json.loads((output/'data'/name/'population.json').read_text())
    p = pop['parameters']
    g = p['c_beta']/3
    q = g/(g+p['sigma_eps']**2/p['N'])*np.dot(p['mu_F'], p['mu_F'])
    floors = {}
    with np.load(directory/'moments.npz', allow_pickle=False) as saved:
        require(str(saved['identity']) == report['identity'], 'Rank moment identity')
        m, S = saved['mean'], saved['second']
        require(m.shape == (1024,) and S.shape == (1024, 1024), 'Rank moment shape')
        require(np.isfinite(m).all() and np.isfinite(S).all(), 'Rank moment finiteness')
        equal(S, S.T, 'Rank moment symmetry')
        for size in (256, 512, 1024):
            E = saved[f'embedding{size}']
            require(E.shape == (1024, size), 'Embedding shape')
            b, A = E.T@m, E.T@S@E
            floors[size] = float(q-b@np.linalg.solve(A, b))
            equal(report['floors'][str(size)], floors[size], 'Rank projection floor')
    nt, nl = len(config['T']), len(config['penalties'])+3
    rows = []
    for r in range(50):
        with np.load(directory/f'{r:04d}.npz', allow_pickle=False) as saved:
            require(int(saved['replication']) == r, 'Rank replication index')
            require(int(saved['train_seed']) == seed(config['master_seed'], 4, r), 'Rank training seed')
            require(str(saved['identity']) == report['identity'], 'Rank path identity')
            row = {k: saved[k] for k in ('regret', 'complexity', 'payoff_difference', 'penalties')}
        for key, shape in [('regret', (nt, 3, nl)), ('complexity', (nt, 3, nl)),
                           ('payoff_difference', (nt, 2, nl)), ('penalties', (nt, nl))]:
            require(row[key].shape == shape and np.isfinite(row[key]).all(), 'Rank shape/value '+key)
        for t, T in enumerate(config['T']):
            theory = np.array([.25, 1., 4.])*dict(config['theory_scale'])[name]*T**(-.6)
            equal(row['penalties'][t], np.r_[config['penalties'], theory], 'Rank fixed penalty')
        require(np.min(row['payoff_difference']) >= -1e-10, 'Negative squared payoff difference')
        rows.append(row)
    values = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
    frame = read_table(audit/f'{name}_rank_resolution.csv', ['T', 'lambda_index'])
    require(len(frame) == nt*nl, 'Rank curve cell count')
    frame = frame.set_index(['T', 'lambda_index'])
    for t, T in enumerate(config['T']):
        for j in range(nl):
            low, high = values['regret'][:, t, 1, j], values['regret'][:, t, 2, j]
            x = paired(low, high, floors[512])
            compare_row(frame.loc[T, j], {
                'environment': name, 'lambda': values['penalties'][0, t, j], 'replications': 50,
                'regret256': values['regret'][:, t, 0, j].sum()/50,
                'regret512': x['reference'], 'regret1024': high.sum()/50,
                'paired_difference': x['difference'], 'paired_MCSE': x['MCSE'],
                'paired_ci_low': x['lo'], 'paired_ci_high': x['hi'],
                'payoff_squared_difference_512_1024': values['payoff_difference'][:, t, 1, j].sum()/50,
                'complexity512': values['complexity'][:, t, 1, j].sum()/50,
                'complexity1024': values['complexity'][:, t, 2, j].sum()/50,
                'projection_floor512': floors[512], 'projection_floor1024': floors[1024],
                'certified': x['certified']}, f'{name}/{T}/{j}')
    production = []
    for r in range(env['replications']):
        with np.load(output/'data'/name/'replications'/f'{r:04d}.npz', allow_pickle=False) as saved:
            production.append({k: saved[k][:nt] for k in ('regret', 'choices')})
    oracle = np.sum([v['regret'][:, :nl-3] for v in production], axis=0).argmin(axis=1)
    policies = read_table(audit/f'{name}_rank_policy_resolution.csv', ['T', 'method'])
    require(len(policies) == nt*len(POLICIES), 'Rank policy cell count')
    policies = policies.set_index(['T', 'method'])
    for t, T in enumerate(config['T']):
        for j, method in enumerate(POLICIES):
            choices = np.array([v['choices'][t, j] if j < 5 else oracle[t] for v in production[:50]])
            low = values['regret'][np.arange(50), t, 1, choices]
            high = values['regret'][np.arange(50), t, 2, choices]
            x = paired(low, high, floors[512])
            compare_row(policies.loc[T, method], {
                'environment': name, 'paired_paths': 50, 'regret512': x['reference'],
                'regret1024': high.sum()/50, 'paired_difference': x['difference'], 'paired_MCSE': x['MCSE'],
                'ci_low': x['lo'], 'ci_high': x['hi'], 'floor512': floors[512],
                'floor_fraction': floors[512]/x['reference'],
                'difference_confidence_fraction': x['bound']/x['reference'],
                'certified': x['certified']}, f'{name}/{T}/{method}')
    equal(report['certified_cells'], int(frame.certified.sum()), 'Rank report certified count')
    equal(report['total_cells'], len(frame), 'Rank report total count')
    equal(report['policy_cells_certified'], int(policies.certified.sum()), 'Policy certified count')
    equal(report['uniform_curve_certified'], bool(frame.certified.all()), 'Uniform rank certificate')
    return {'curve_cells': len(frame), 'policy_cells': len(policies), 'paths': 50}, oracle, q


def quadrature_summary(frame, result, environment):
    """Independent seed-level uncertainty, not a 200-path independence claim."""
    keys = ['replication', 'T', 'method', 'seed_index', 'groups']
    require(not frame.duplicated(keys).any(), 'Duplicate quadrature path')
    require(np.isfinite(frame[['regret', 'population_sharpe']].to_numpy()).all(), 'Nonfinite quadrature risk')
    require(not result.duplicated(['T', 'method']).any(), 'Duplicate quadrature summary')
    observed = set()
    for (T, method), subset in frame.groupby(['T', 'method']):
        seed_means = {}
        for s in range(4):
            seed_means[s] = {}
            for groups in (8192, 32768):
                part = subset[(subset.seed_index == s)&(subset.groups == groups)].sort_values('replication')
                equal(part.replication.to_numpy(), np.arange(50), 'Quadrature paired cohort')
                seed_means[s][groups] = part.regret.sum()/50
        differences = np.array([seed_means[s][8192]-seed_means[s][32768] for s in range(4)])
        mean, _, se = summary(differences)
        reference = sum(seed_means[s][32768] for s in range(4))/4
        row = result[(result['T'] == T)&(result.method == method)]
        require(len(row) == 1, 'Missing quadrature summary')
        compare_row(row.iloc[0], {'environment': environment, 'paths': 50, 'independent_seeds': 4,
                                 'mean_regret32768': reference, 'difference8192_minus32768': mean,
                                 'quadrature_seed_MCSE': se, 'ci_low': mean-1.96*se, 'ci_high': mean+1.96*se,
                                 'certified': bool(abs(mean)+1.96*se <= .05*reference)}, 'Quadrature summary')
        observed.add((T, method))
    require(set(zip(result['T'], result.method)) == observed, 'Extra quadrature summary')


def quadrature(output, config, env, oracle, q):
    audit, name = output/'audit', env['name']
    report = json.loads((audit/f'{name}_quadrature_resolution.json').read_text())
    science = json.loads((output/'run_manifest.json').read_text())['source_hashes']
    frame = read_table(audit/f'{name}_quadrature_paths.csv', ['replication', 'T', 'method', 'seed_index', 'groups'])
    result = read_table(audit/f'{name}_quadrature_resolution.csv', ['T', 'method'])
    expected_per_pair = 50*(len(config['T'])*6+len(config['extra_T'])*5)
    require(len(frame) == expected_per_pair*8, 'Quadrature raw count')
    indexed = frame.set_index(['seed_index', 'groups', 'replication', 'T', 'method'])
    coefficients = {}
    for r in range(50):
        with np.load(audit/'rank_paths_low_memory'/name/f'{r:04d}.npz', allow_pickle=False) as saved:
            grid_coefficients = saved['coefficients512']
            oracle_coefficients = [grid_coefficients[t, :, j] for t, j in enumerate(oracle)]
        with np.load(output/'data'/name/'replications'/f'{r:04d}.npz', allow_pickle=False) as saved:
            for t, T in enumerate(saved['T']):
                a = saved['selected_coefficients'][t]
                if t < len(config['T']):
                    a = np.column_stack([a, oracle_coefficients[t]])
                coefficients[r, int(T)] = a
    for s in range(4):
        expected_seed = seed(config['master_seed'], 3, s+1)
        for groups in (8192, 32768):
            part = frame[(frame.seed_index == s)&(frame.groups == groups)]
            require(len(part) == expected_per_pair, 'Quadrature seed/resolution count')
            equal(part.seed.to_numpy(), np.full(len(part), expected_seed), 'Quadrature seed family')
            with np.load(audit/'quadrature_moments_low_memory'/name/f'groups{groups}_seed{s}.npz', allow_pickle=False) as saved:
                expected_identity = digest({'sources': science, 'diagnostic': file_hash(ROOT/'simulations/diagnostics/resolution_v2.py'),
                                            'quadrature_implementation': file_hash(ROOT/'simulations/diagnostics/low_memory.py'),
                                            'runtime': report['runtime'], 'protocol': config['protocol_hash'],
                                            'environment': name, 'groups': groups, 'seed': expected_seed})
                require(str(saved['identity']) == expected_identity, 'Quadrature source/runtime/configuration identity')
                equal(int(saved['seed']), expected_seed, 'Quadrature moment seed')
                equal(int(saved['groups']), groups, 'Quadrature moment groups')
                m, S = saved['mean'], saved['second']
            require(m.shape == (env['rank'],) and S.shape == (env['rank'], env['rank']), 'Quadrature moments shape')
            require(np.isfinite(m).all() and np.isfinite(S).all(), 'Quadrature moments values')
            for (r, T), a in coefficients.items():
                means = np.sum(m[:, None]*a, axis=0)
                seconds = np.diag(a.T@S@a)
                risks = 1-2*means+seconds-(1-q)
                sharpe = means/np.sqrt(seconds-means**2)
                for j in range(a.shape[1]):
                    row = indexed.loc[s, groups, r, T, POLICIES[j]]
                    equal(row.regret, risks[j], 'Fixed-coefficient quadrature risk')
                    equal(row.population_sharpe, sharpe[j], 'Fixed-coefficient quadrature Sharpe')
    quadrature_summary(frame, result, name)
    equal(report['certified_cells'], int(result.certified.sum()), 'Quadrature certified count')
    equal(report['total_cells'], len(result), 'Quadrature total count')
    return {'policy_cells': len(result), 'raw_policy_evaluations': len(frame)}


def spectrum(output, config):
    directory = output/'audit/spectrum'
    report = json.loads((directory/'spectrum_resolution.json').read_text())
    science = json.loads((output/'run_manifest.json').read_text())['source_hashes']
    require(report['scientific_source_hashes'] == science, 'Spectrum scientific dependencies')
    require(report['source_sha256'] == file_hash(ROOT/'simulations/diagnostics/spectrum_v2.py'), 'Spectrum diagnostic dependency')
    require(report['kernel_storage_sha256'] == file_hash(ROOT/'simulations/diagnostics/kernel_storage.py'), 'Kernel storage dependency')
    frame = read_table(directory/'eigenvalues.csv', ['groups', 'seed_index', 'operator', 'index'])
    fits = read_table(directory/'descriptive_fits.csv', ['groups', 'seed_index', 'operator', 'first', 'last'])
    require(len(fits) == 3*3*3*3, 'Spectral fit case count')
    expected_rows = 0
    for groups in (1024, 2048, 4096):
        for s in range(3):
            with np.load(directory/f'groups{groups}_seed{s}.npz', allow_pickle=False) as saved:
                expected_identity = digest({'schema': 'spectrum-cache/2.1', 'diagnostic_source': report['source_sha256'],
                                            'kernel_storage_source': report['kernel_storage_sha256'],
                                            'scientific_sources': science, 'groups': groups,
                                            'seed': seed(config['master_seed'], 8, s), 'eigenpairs': 800,
                                            'runtime': report['runtime']})
                require(str(saved['cache_identity']) == expected_identity, 'Spectrum case dependency identity')
                meta = json.loads((directory/f'groups{groups}_seed{s}.json').read_text())
                require(meta['cache_identity'] == expected_identity and meta['groups'] == groups
                        and meta['seed'] == seed(config['master_seed'], 8, s), 'Spectrum case metadata identity')
                for operator in ('kernel', 'baseline_original', 'rich6d'):
                    v, residual = saved[operator+'_values'], saved[operator+'_residuals']
                    require(len(v) > 0 and len(v) == len(residual), 'Spectral eigenpair count')
                    require(np.all(np.diff(v) <= 0), 'Spectral ordering')
                    resolved = (v > v[0]*1e-12)&(residual <= 1e-5*v)
                    rows = frame[(frame.groups == groups)&(frame.seed_index == s)&(frame.operator == operator)].sort_values('index')
                    equal(rows['index'].to_numpy(), np.arange(1, len(v)+1), 'Eigenvalue index')
                    equal(rows.eigenvalue.to_numpy(), v, 'Saved eigenvalue')
                    equal(rows.absolute_residual.to_numpy(), residual, 'Saved eigensolver residual')
                    equal(rows.resolved.to_numpy(), resolved, 'Residual resolution criterion')
                    expected_rows += len(v)
                    for first, last in ((30, 200), (60, 400), (120, 800)):
                        row = fits[(fits.groups == groups)&(fits.seed_index == s)&(fits.operator == operator)&(fits['first'] == first)&(fits['last'] == last)].iloc[0]
                        complete = len(v) >= last and bool(np.all(resolved[first-1:last]))
                        equal(row.resolved, complete, 'Spectral window status')
                        if complete:
                            x, y = np.log(np.arange(first, last+1)), np.log(v[first-1:last])
                            X = np.column_stack([np.ones(len(x)), x])
                            coefficient = np.linalg.lstsq(X, y, rcond=None)[0]
                            error = y-X@coefficient
                            covariance = np.linalg.inv(X.T@X)*(error@error)/(len(x)-2)
                            equal(row.slope, coefficient[1], 'Descriptive spectral slope')
                            equal(row.OLS_SE, np.sqrt(covariance[1, 1]), 'Descriptive spectral SE')
                            equal(row.historical_band_compatible, abs(coefficient[1]+1.5) <= .35, 'Descriptive historical band')
                        else:
                            require(pd.isna(row.slope) and pd.isna(row.OLS_SE), 'Unresolved window must not publish a slope')
    require(len(frame) == expected_rows, 'Extra spectral rows')
    return {'eigenvalue_rows': len(frame), 'window_rows': len(fits),
            'scope': 'cached eigenvalues/residuals to CSV; independent dense operator crosscheck is separate'}


def verify(output, modes=('rank', 'quadrature', 'spectrum')):
    output = Path(output)
    config = json.loads((output/'run_manifest.json').read_text())['configuration']
    details = {}
    if 'rank' in modes or 'quadrature' in modes:
        for env in config['cases']:
            if env['nu'] != 1.5:
                continue
            result, oracle, q = rank(output, config, env)
            details[env['name']] = {'rank': result}
            if 'quadrature' in modes:
                details[env['name']]['quadrature'] = quadrature(output, config, env, oracle, q)
    if 'spectrum' in modes:
        details['spectrum'] = spectrum(output, config)
    report = {'passed': True, 'schema': 'independent-numerical-reconstruction/2.0',
              'completed_utc': utc_now(), 'source_sha256': file_hash(__file__),
              'protocol_hash': config['protocol_hash'], 'details': details,
              'scope': 'raw-to-table reconstruction and fixed-coefficient quadrature; no infinite-kernel certification'}
    paths = list((output/'audit').glob('*_resolution.*'))
    paths += list((output/'audit').glob('*_quadrature_paths.csv'))
    for folder in ('rank_paths_low_memory', 'quadrature_moments_low_memory', 'spectrum'):
        paths += [p for p in (output/'audit'/folder).rglob('*') if p.is_file()]
    report['input_hashes'] = {str(p.relative_to(output)): file_hash(p) for p in sorted(set(paths))}
    report['verifier_sources'] = {str(Path(name).relative_to(ROOT)): file_hash(name) for name in
        (__file__, Path(__file__).with_name('reconstruct.py'), Path(__file__).parents[1]/'provenance.py')}
    json_write(output/'audit/numerical_reconstruction.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    parser.add_argument('--modes', nargs='+', choices=['rank', 'quadrature', 'spectrum'], default=['rank', 'quadrature', 'spectrum'])
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.modes), indent=2))
