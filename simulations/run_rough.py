"""Reproducible rough Rich6D experiment; no historical artifacts are inputs.

Run: VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough --workers 2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh

from simulations.dgp.balanced import DGPParameters, BalancedFactorDGP, beta, w_star
from simulations.dgp.rough import normalization, uniform_tail_bound
from simulations.estimator.kernel import NystromBasis
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.experiments.population import population_ridge
from simulations.diagnostics.low_memory import population_moments_batched
from simulations.provenance import digest, file_hash, json_write, npz_write, utc_now, output_lock

ROOT = Path(__file__).resolve().parents[1]
TIMES = np.array([60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440])
GRID = np.geomspace(1e-12, 1e-2, 96)
METRICS = ('complexity', 'population_loss', 'population_sharpe', 'regret',
           'estimation_norm', 'cross_term', 'population_bias', 'population_complexity')


def seed(family, index=0):
    families = {'anchor': 1, 'pilot': 2, 'quadrature': 3, 'train': 4,
                'bootstrap': 6, 'rank': 7, 'sensitivity': 8, 'truncation': 9}
    return int(np.random.SeedSequence([2026100907, families[family], index]).generate_state(1)[0])


def moments(output, name, p, basis, groups, rng_seed):
    path = output / 'moments' / (name + '.npz')
    if path.exists():
        with np.load(path) as z:
            assert str(z['parameters_hash']) == digest(p.to_dict())
            assert int(z['seed']) == rng_seed and int(z['groups']) == groups
            return z['mean'], z['second']
    m, S = population_moments_batched(p, basis, groups, rng_seed)
    npz_write(path, mean=m, second=S, groups=groups, seed=rng_seed,
              parameters_hash=digest(p.to_dict()))
    return m, S


def truncation_check(output, p, basis):
    report = output / 'fourier_truncation.json'
    if report.exists():
        return json.loads(report.read_text())
    start = time.perf_counter()
    records, policies, payoffs, training = [], {}, {}, {}
    # Common innovations; all streams are separate from production and pilot.
    for M in (128, 512, 2048):
        pm = replace(p, fourier_terms=M)
        policy, payoff, X, balance = [], [], [], 0.
        for date in BalancedFactorDGP(pm, seed('truncation')).simulate(240):
            w = w_star(date.z, pm)
            policy.append(w)
            payoff.append(w @ date.returns / pm.N)
            X.append(basis.managed(date.z, date.returns))
            balance = max(balance, float(np.abs(date.loadings.T @ date.loadings / pm.N - pm.gamma_beta).max()))
        policies[M], payoffs[M], training[M] = np.array(policy), np.array(payoff), np.array(X)
        records.append({'M': M, 'balanced_triplet_max_error': balance,
                        'uniform_psi_tail_bound': uniform_tail_bound(M)})
    ref = 2048
    # Fixed diagnostic penalties, with no performance-based tuning.
    penalties = np.array([1e-7, 1e-6, 1e-5])
    metrics = {}
    for row in records:
        M = row['M']; pm = replace(p, fourier_terms=M)
        m, S = moments(output, f'truncation{M}', pm, basis, 8192, seed('truncation', 1))
        a, _ = ridge_path(training[M], penalties)
        loss, sr = moment_metrics(a, m, S)
        metrics[M] = (loss - (1-p.q_star), sr)
        row.update(max_policy_difference_to_2048=float(np.max(np.abs(policies[M]-policies[ref]))),
                   max_optimal_payoff_difference_to_2048=float(np.max(np.abs(payoffs[M]-payoffs[ref]))))
    for row in records:
        regret, sr = metrics[row['M']]
        row.update(max_sharpe_difference_to_2048=float(np.max(np.abs(sr-metrics[ref][1]))),
                   max_regret_difference_to_2048=float(np.max(np.abs(regret-metrics[ref][0]))))
    assert all(r['balanced_triplet_max_error'] < 1e-14 for r in records)
    assert records[0]['max_policy_difference_to_2048'] < 1e-7
    assert records[0]['max_optimal_payoff_difference_to_2048'] < 1e-9
    assert records[0]['max_sharpe_difference_to_2048'] < 1e-8
    assert records[0]['max_regret_difference_to_2048'] < 1e-8
    result = {'records': records, 'selected_M': 128, 'normalization': normalization(),
              'normalization_rule': 'Parseval: 1/sqrt(sum m^-10/log(m+1)^2); deterministic sum through 65536',
              'normalization_squared_sum_tail_bound': float(65536.**-9/(9*np.log(65537.)**2)),
              'diagnostic_T': 240, 'diagnostic_penalties': penalties.tolist(),
              'selection': 'Smallest candidate passing fixed absolute numerical tolerances; no rate examined.',
              'tolerances': {'policy': 1e-7, 'payoff': 1e-9, 'sharpe': 1e-8, 'regret': 1e-8},
              'limitation': 'Finite M is smooth; no mathematical verification of r=1.',
              'frozen_utc': utc_now(), 'seconds': time.perf_counter()-start}
    pd.DataFrame(records).to_csv(output/'fourier_truncation.csv', index=False)
    json_write(report, result)
    return result


def context(output):
    p = DGPParameters(loading_map='rich6d_rough', eta=.35, fourier_terms=128)
    low = NystromBasis(rank=512, seed=seed('anchor'))
    high = NystromBasis(rank=1024, seed=seed('anchor'))
    truncation = truncation_check(output, p, low)
    print('Fourier truncation frozen:', truncation['selected_M'], flush=True)
    pm, PS = moments(output, 'pilot', p, low, 8192, seed('pilot'))
    s_ref = float(eigh(PS, eigvals_only=True)[-1])
    spec = {'schema': 'rough-rich6d/1', 'parameters': p.to_dict(), 'T': TIMES.tolist(),
            'replications': 100, 'rank_audit_paths': 50, 'rank': 512, 'audit_rank': 1024,
            'grid': GRID.tolist(), 'method': 'theory_1', 'a': s_ref, 'exponent': -.6,
            'theoretical_b': 1.5, 'sharpe_units': 'per period; no annual rescaling',
            'population_groups': 8192, 'rank_groups': 32768,
            'quadrature_groups': [8192,32768], 'quadrature_seeds': 4,
            'rank_and_quadrature_relative_regret_tolerance': .05,
            'bootstrap_replications': 1000, 'master_seed': 2026100907,
            'source_hashes': {n: file_hash(ROOT/n) for n in (
                'simulations/run_rough.py','simulations/dgp/balanced.py','simulations/dgp/rough.py',
                'simulations/estimator/kernel.py','simulations/estimator/ridge.py',
                'simulations/experiments/population.py','simulations/diagnostics/low_memory.py')},
            'truncation_sha256': file_hash(output/'fourier_truncation.json')}
    run_hash = digest(spec)
    protocol = output/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['run_hash'] == run_hash
    else:
        json_write(protocol, {**spec, 'run_hash': run_hash, 'frozen_before_production_utc': utc_now()})
    json_write(output/'calibration.json', {'a': s_ref, 'rule': 'lambda_T=a*T^-0.6',
        'calibration': 'Largest rank-512 population second-moment eigenvalue, existing theory_1 convention',
        'pilot_groups': 8192, 'pilot_seed': seed('pilot'), 'evaluation_seed': seed('quadrature'),
        'independent_population_pilot': True, 'historically_implementable_selector': False,
        'protocol_sha256': file_hash(protocol)})
    m, S = moments(output, 'population', p, low, 8192, seed('quadrature'))
    rm, rS = moments(output, 'rank1024', p, high, 32768, seed('rank'))
    E = np.linalg.solve(high.inverse_root, np.vstack([low.inverse_root, np.zeros((512,512))]))
    np.testing.assert_allclose(high.inverse_root @ E, np.vstack([low.inverse_root, np.zeros((512,512))]), atol=1e-10)
    lm, lS = E.T @ rm, E.T @ rS @ E
    floors = {512: float(p.q_star-lm@np.linalg.solve(lS,lm)),
              1024: float(p.q_star-rm@np.linalg.solve(rS,rm))}
    pop = []
    for T in TIMES:
        lam = np.r_[GRID, s_ref*float(T)**(-.6)]
        a, C, bias, floor = population_ridge(m,S,lam,p.q_star)
        pop.append((lam,a,C,bias))
    metadata = {'parameters': p.to_dict(), 'reference_sharpe': p.sr_star, 'q_reference': p.q_star,
                'subspace_regret_floor': floor, 's_ref': s_ref, 'rank_floors': floors,
                'population_seed': seed('quadrature'), 'basis_seed': seed('anchor'),
                'run_hash': run_hash, 'constants': p.constants()}
    json_write(output/'population.json', metadata)
    npz_write(output/'population.npz', mean=m, second=S, pilot_mean=pm, pilot_second=PS, embedding=E)
    spectrum = eigh(rS, eigvals_only=True)[::-1]
    pd.DataFrame({'rank': np.arange(1,1025), 'eigenvalue': spectrum}).to_csv(output/'spectrum.csv',index=False)
    for groups in (8192,32768):
        for s in range(4):
            moments(output, f'quadrature_{groups}_{s}', p, low, groups, seed('sensitivity',s))
    return dict(p=p, low=low, high=high, mean=m, second=S, E=E,
                rank_mean=rm, rank_second=rS, rank_low_mean=lm, rank_low_second=lS,
                floors=floors, population=pop, spec=spec, run_hash=run_hash,
                population_metadata=metadata, spectrum=spectrum)


def run_replication(index, c, output):
    dest = output / 'replications' / f'{index:04d}.npz'
    if dest.exists():
        with np.load(dest) as z:
            assert str(z['run_hash']) == c['run_hash']
        return
    started = time.perf_counter()
    basis = c['high'] if index < 50 else c['low']
    X = np.empty((1440, basis.rank))
    return_hash = hashlib.sha256()
    for t, date in enumerate(BalancedFactorDGP(c['p'], seed('train', index)).simulate(1440)):
        X[t] = basis.managed(date.z, date.returns)
        return_hash.update(date.returns.tobytes())
        if t == 0 and index < 50:
            np.testing.assert_allclose(X[t] @ c['E'], c['low'].managed(date.z, date.returns), atol=1e-12, rtol=1e-9)
    train = X @ c['E'] if index < 50 else X
    values = {key: [] for key in METRICS}
    theory_coefficients, rank_regrets, rank_complexity = [], [], []
    max_normal_error = max_identity_error = 0.
    for t, T in enumerate(TIMES):
        lam, popcoef, popC, bias = c['population'][t]
        a, C = ridge_path(train[:T], lam)
        loss, sr = moment_metrics(a, c['mean'], c['second'])
        delta = a - popcoef
        norm = np.sum(delta * (c['second'] @ delta), axis=0)
        cross = 2 * np.sum(delta * (c['second'] @ popcoef - c['mean'][:, None]), axis=0)
        regret = loss - (1 - c['p'].q_star)
        np.testing.assert_allclose(regret, bias + norm + cross, atol=1e-10, rtol=1e-8)
        assert np.max(np.abs(sr)) <= c['p'].sr_star + 1e-8
        residual = train[:T].T @ (train[:T] @ a) / T + a * lam - train[:T].mean(axis=0)[:, None]
        max_normal_error = max(max_normal_error, float(np.abs(residual).max()))
        max_identity_error = max(max_identity_error, float(np.abs(regret - bias - norm - cross).max()))
        assert max_normal_error < 1e-8
        current = dict(complexity=C, population_loss=loss, population_sharpe=sr, regret=regret,
                       estimation_norm=norm, cross_term=cross, population_bias=bias, population_complexity=popC)
        for key in METRICS:
            values[key].append(current[key])
        theory_coefficients.append(a[:, -1])
        if index < 50:
            high_a, high_C = ridge_path(X[:T], lam)
            low_loss, _ = moment_metrics(a, c['rank_low_mean'], c['rank_low_second'])
            high_loss, _ = moment_metrics(high_a, c['rank_mean'], c['rank_second'])
            rank_regrets.append(np.stack([low_loss - (1 - c['p'].q_star), high_loss - (1 - c['p'].q_star)]))
            rank_complexity.append(np.stack([C, high_C]))
    npz_write(dest, **{k: np.asarray(v) for k, v in values.items()},
              managed_training=train, managed_rank=512, generated_managed_rank=basis.rank, theory_coefficients=np.asarray(theory_coefficients),
              rank_regrets=np.asarray(rank_regrets), rank_complexity=np.asarray(rank_complexity),
              T=TIMES, penalties=np.asarray([r[0] for r in c['population']]), replication=index,
              train_seed=seed('train', index), stock_return_sha256=return_hash.hexdigest(), run_hash=c['run_hash'],
              maximum_normal_equation_error=max_normal_error, maximum_decomposition_error=max_identity_error,
              runtime_seconds=time.perf_counter() - started)


def stats(x):
    x = np.asarray(x)
    mean, se = x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(len(x))
    return dict(mean=mean, mcse=se, ci_low=mean - 1.96 * se, ci_high=mean + 1.96 * se)


def summarize(c, output):
    raw, raw_hashes = [], {}
    for index in range(100):
        path = output / 'replications' / f'{index:04d}.npz'
        raw_hashes[str(path.relative_to(output))] = file_hash(path)
        with np.load(path) as z:
            assert str(z['run_hash']) == c['run_hash'] and int(z['replication']) == index
            record = {k: z[k] for k in (*METRICS, 'theory_coefficients', 'rank_regrets', 'rank_complexity',
                                        'maximum_normal_equation_error', 'maximum_decomposition_error')}
            # Independently reconstruct every theory policy from saved training data.
            X = z['managed_training']
            if int(z['managed_rank']) == 1024:
                X = X @ c['E']
            for t, T in enumerate(TIMES):
                a = record['theory_coefficients'][t]
                lam = c['population'][t][0][-1]
                residual = X[:T].T @ (X[:T] @ a) / T + lam * a - X[:T].mean(axis=0)
                assert np.abs(residual).max() < 1e-8
                mean = c['mean'] @ a
                second = a @ c['second'] @ a
                np.testing.assert_allclose(mean / np.sqrt(second - mean**2), record['population_sharpe'][t, -1], atol=1e-11)
            np.testing.assert_allclose(record['regret'], record['population_bias'] + record['estimation_norm'] + record['cross_term'], atol=1e-10, rtol=1e-8)
            raw.append(record)
    v = {k: np.stack([r[k] for r in raw]) for k in METRICS}
    summary = {k: stats(x) for k, x in v.items()}
    curves, methods, paths = [], [], []
    for t, T in enumerate(TIMES):
        for j in range(96):
            row = {'environment': 'rich6d_rough', 'T': int(T), 'lambda': GRID[j], 'replications': 100,
                   'projection_floor': c['population_metadata']['subspace_regret_floor']}
            for k in METRICS:
                if k in ('population_bias', 'population_complexity'):
                    row[k] = v[k][0, t, j]
                else:
                    row.update({k + '_' + stat: value[t, j] for stat, value in summary[k].items()})
            curves.append(row)
        row = {'environment': 'rich6d_rough', 'T': int(T), 'method': 'theory_1', 'replications': 100,
               'lambda_mean': c['population'][t][0][-1], 'projection_floor': c['population_metadata']['subspace_regret_floor']}
        for k in METRICS:
            row.update({k + '_' + stat: value[t, -1] for stat, value in summary[k].items()})
        gaps = c['p'].sr_star - v['population_sharpe'][:, t, -1]
        row.update({'sharpe_gap_' + stat: value for stat, value in stats(gaps).items()})
        methods.append(row)
        for r in range(100):
            paths.append({'replication': r, 'T': int(T), 'method': 'theory_1',
                          **{k: v[k][r, t, -1] for k in METRICS}})
    curves, methods = pd.DataFrame(curves), pd.DataFrame(methods)
    curves.to_csv(output / 'curves.csv', index=False)
    methods.to_csv(output / 'methods.csv', index=False)
    pd.DataFrame(paths).to_csv(output / 'path_methods.csv', index=False)
    rr = np.stack([r['rank_regrets'] for r in raw[:50]])
    cc = np.stack([r['rank_complexity'] for r in raw[:50]])
    rank_rows = []
    for t, T in enumerate(TIMES):
        for j, lam in enumerate(c['population'][t][0]):
            low, high = rr[:, t, 0, j], rr[:, t, 1, j]
            delta = low - high
            mean, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(50))
            passed = c['floors'][512] <= .05 * low.mean() and abs(mean) + 1.96 * se <= .05 * low.mean()
            rank_rows.append({'environment': 'rich6d_rough', 'T': int(T), 'lambda_index': j, 'lambda': lam,
                              'replications': 50, 'regret512': low.mean(), 'regret1024': high.mean(),
                              'paired_difference': mean, 'paired_MCSE': se, 'certified': passed,
                              'complexity512': cc[:, t, 0, j].mean(), 'complexity1024': cc[:, t, 1, j].mean(),
                              'projection_floor512': c['floors'][512], 'projection_floor1024': c['floors'][1024]})
    rank = pd.DataFrame(rank_rows)
    rank.to_csv(output / 'rank_resolution.csv', index=False)
    # Same fixed policies, four independent quadrature seeds, both resolutions.
    coefs = np.stack([r['theory_coefficients'] for r in raw[:50]])
    quad = {}
    for groups in (8192, 32768):
        quad[groups] = []
        for s in range(4):
            with np.load(output / f'moments/quadrature_{groups}_{s}.npz') as z:
                m, S = z['mean'], z['second']
            risks = c['p'].q_star - 2 * (coefs @ m) + np.einsum('rti,ij,rtj->rt', coefs, S, coefs, optimize=True)
            quad[groups].append(risks.mean(axis=0))
    quad_rows = []
    differences = np.asarray(quad[8192]) - np.asarray(quad[32768])
    for t, T in enumerate(TIMES):
        avg = differences[:, t].mean(); se = differences[:, t].std(ddof=1) / 2
        reference = np.mean(quad[32768], axis=0)[t]
        quad_rows.append({'T': int(T), 'method': 'theory_1', 'paths': 50, 'independent_seeds': 4,
                          'mean_regret32768': reference, 'difference8192_minus32768': avg,
                          'quadrature_seed_MCSE': se, 'certified': bool(abs(avg) + 1.96 * se <= .05 * reference)})
    quadrature = pd.DataFrame(quad_rows)
    quadrature.to_csv(output / 'quadrature_resolution.csv', index=False)
    gap = c['p'].sr_star - v['population_sharpe'][:, :, -1]
    rng = np.random.default_rng(seed('bootstrap'))
    boot = np.stack([gap[rng.integers(100, size=100)].mean(axis=0) for _ in range(1000)])
    rates = []
    for name, idx in [('full', np.arange(10)), ('upper_half', np.arange(5, 10)), ('largest_four', np.arange(6, 10))]:
        x = np.log(TIMES[idx]); x = x - x.mean()
        slope = float(np.log(gap.mean(axis=0)[idx]) @ x / (x @ x))
        slopes = np.log(boot[:, idx]) @ x / (x @ x)
        rates.append({'quantity': 'sharpe_gap', 'window': name, 'method': 'theory_1', 'cohort': 'practical',
                      'OLS_slope': slope, 'bootstrap_low': np.quantile(slopes, .025),
                      'bootstrap_high': np.quantile(slopes, .975), 'bootstrap_draws': 1000})
    rates = pd.DataFrame(rates)
    rates.to_csv(output / 'rates.csv', index=False)
    verification = {'passed': True, 'fresh_replications': 100, 'paired_rank_paths': 50,
                    'maximum_normal_equation_error': float(max(r['maximum_normal_equation_error'] for r in raw)),
                    'maximum_decomposition_error': float(max(r['maximum_decomposition_error'] for r in raw)),
                    'theory_rank_pass_by_T': rank[rank.lambda_index == 96][['T', 'certified']].to_dict('records'),
                    'theory_quadrature_pass_by_T': quadrature[['T', 'certified']].to_dict('records'),
                    'raw_checkpoint_sha256': raw_hashes, 'completed_utc': utc_now()}
    json_write(output / 'verification.json', verification)
    return curves, methods, rank, rates, verification



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/rich6d_rough_r1_v1'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    with output_lock(args.output):
        c = context(args.output)
        print('Independent pilot a =', c['spec']['a'], flush=True)
        if args.prepare_only:
            return
        if not args.render_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_replication,i,c,args.output): i for i in range(100)}
                for completed, future in enumerate(as_completed(futures),1):
                    future.result()
                    json_write(args.output/'status.json', {'status': 'running', 'completed': completed,
                        'total': 100, 'elapsed_seconds': time.perf_counter()-started, 'updated_utc': utc_now()})
                    print(f'Replications {completed}/100; {time.perf_counter()-started:.1f}s', flush=True)
        tables = summarize(c,args.output)
        from simulations.plotting.rough_final import render
        render(c,args.output,tables)
        json_write(args.output/'status.json', {'status': 'complete', 'replications': 100,
            'run_hash': c['run_hash'], 'completed_utc': utc_now(), 'elapsed_seconds': time.perf_counter()-started})
        manifest = {'run_hash': c['run_hash'], 'python': platform.python_version(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__,
            'platform': platform.platform(), 'git_base': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'command': 'VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough --workers 2',
            'artifact_sha256': {str(f.relative_to(args.output)): file_hash(f) for f in args.output.rglob('*')
                if f.is_file() and f.name not in ('manifest.json','.run.lock')},
            'finished_utc': utc_now()}
        json_write(args.output/'manifest.json',manifest)
        print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
