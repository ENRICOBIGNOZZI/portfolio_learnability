"""Fixed-penalty nested-rank and independent-quadrature confirmation audits.

Uses the canonical economy and ridge solver. Smaller nested spaces are embedded
in the largest space, so every payoff difference uses a common quadrature.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.config.design import design_for
from simulations.estimator.kernel import NystromBasis
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.experiments.monte_carlo import managed_path
from simulations.experiments.population import population_moments
from simulations.diagnostics.confirmation_pilot import seed
from simulations.provenance import (digest, file_hash, json_write, npz_write, require,
                                    source_hashes, utc_now)

_CONTEXT = None


def initialize(context):
    global _CONTEXT
    _CONTEXT = context


def rank_path(index):
    c = _CONTEXT
    path = c['directory']/f'{index:04d}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            require(str(saved['identity']) == c['identity'], 'Rank checkpoint identity mismatch')
        return str(path)
    d, env, basis = c['design'], c['environment'], c['basis']
    X, _, _ = managed_path(env.parameters(), basis, 'baseline', seed('train', index), max(d.T))
    paths = {r: X@E for r, E in c['embeddings'].items()}
    results = {key: [] for key in ('regret', 'complexity', 'payoff_difference', 'coefficients512', 'penalties')}
    for T in d.T:
        theory = np.asarray([.25, 1., 4.])*dict(d.theory_scale)[env.name]*T**(-env.theoretical_b/(env.theoretical_b+1))
        lam = np.r_[d.penalties, theory]
        coefs, regrets, complexities = {}, [], []
        for r, E in c['embeddings'].items():
            a, C = ridge_path(paths[r][:T], lam)
            coefs[r] = E@a
            loss, _ = moment_metrics(coefs[r], c['mean'], c['second'])
            regrets.append(loss-(1-env.parameters().q_star))
            complexities.append(C)
            if r == 512:
                results['coefficients512'].append(a)
        results['regret'].append(regrets)
        results['complexity'].append(complexities)
        paydiff = []
        for r in (256, 512):
            delta = coefs[r]-coefs[1024]
            paydiff.append(np.sum(delta*(c['second']@delta), axis=0))
        results['payoff_difference'].append(paydiff)
        results['penalties'].append(lam)
    npz_write(path, **{k: np.asarray(v) for k, v in results.items()}, replication=index,
              train_seed=np.uint64(seed('train', index)), identity=c['identity'])
    return str(path)


def rank_audit(output, environment, workers=2):
    output = Path(output)
    d = design_for('confirmation_v2')
    manifest = json.loads((output/'run_manifest.json').read_text())
    require(manifest['source_hashes'] == source_hashes(), 'Production scientific sources changed before rank audit')
    env = next(e for e in d.cases if e.name == environment)
    require(env.nu == 1.5, 'Rank audit is predeclared for headline kernels')
    p = env.parameters()
    directory = output/'audit/rank_paths'/env.name
    directory.mkdir(parents=True, exist_ok=True)
    identity = digest({'sources': source_hashes(), 'diagnostic': file_hash(__file__),
                       'protocol': d.protocol_hash, 'environment': env.name, 'groups': 32768,
                       'subset': 50, 'T': d.T, 'scope': 'practical grid only; extended horizon remains uncertified by this rank test'})
    basis = NystromBasis(rank=1024, nu=env.nu, seed=d.basis_seed)
    embeddings = {}
    for rank in (256, 512, 1024):
        small = NystromBasis(rank=rank, nu=env.nu, seed=d.basis_seed)
        require(np.array_equal(basis.anchors[:rank], small.anchors), 'Anchors are not nested')
        padded = np.zeros((1024, rank)); padded[:rank] = small.inverse_root
        embeddings[rank] = np.linalg.solve(basis.inverse_root, padded)
    moments_path = directory/'moments.npz'
    if moments_path.exists():
        with np.load(moments_path, allow_pickle=False) as saved:
            require(str(saved['identity']) == identity, 'Rank moment identity mismatch')
            mean, second = saved['mean'], saved['second']
    else:
        mean, second = population_moments(p, basis, 32768, seed('quadrature', 10))
        npz_write(moments_path, mean=mean, second=second, identity=identity)
    floors = {}
    for rank, E in embeddings.items():
        m, S = E.T@mean, E.T@second@E
        floors[rank] = float(p.q_star-m@np.linalg.solve(S, m))
    c = {'design': d, 'environment': env, 'basis': basis, 'mean': mean, 'second': second,
         'embeddings': embeddings, 'directory': directory, 'identity': identity}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(c,)) as pool:
        for index, _ in enumerate(pool.map(rank_path, range(50))):
            if (index+1)%10 == 0:
                print(f'Rank {env.name}: {index+1}/50', flush=True)
    raw = []
    for r in range(50):
        with np.load(directory/f'{r:04d}.npz', allow_pickle=False) as saved:
            raw.append({k: saved[k] for k in ('regret', 'complexity', 'payoff_difference', 'penalties')})
    values = {k: np.stack([item[k] for item in raw]) for k in raw[0]}
    rows = []
    for t, T in enumerate(d.T):
        for j, lam in enumerate(values['penalties'][0, t]):
            reference = values['regret'][:, t, 1, j]
            delta = values['regret'][:, t, 1, j]-values['regret'][:, t, 2, j]
            mean, se = delta.mean(), delta.std(ddof=1)/np.sqrt(50)
            rows.append({'environment': env.name, 'T': T, 'lambda_index': j, 'lambda': lam,
                         'replications': 50, 'regret256': values['regret'][:, t, 0, j].mean(),
                         'regret512': reference.mean(), 'regret1024': values['regret'][:, t, 2, j].mean(),
                         'paired_difference': mean, 'paired_MCSE': se, 'paired_ci_low': mean-1.96*se,
                         'paired_ci_high': mean+1.96*se,
                         'payoff_squared_difference_512_1024': values['payoff_difference'][:, t, 1, j].mean(),
                         'complexity512': values['complexity'][:, t, 1, j].mean(),
                         'complexity1024': values['complexity'][:, t, 2, j].mean(),
                         'projection_floor512': floors[512], 'projection_floor1024': floors[1024],
                         'certified': bool(floors[512] <= .05*reference.mean() and abs(mean)+1.96*se <= .05*reference.mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'audit'/f'{env.name}_rank_resolution.csv', index=False)
    # Calibrate the plugin oracle once at the production rank, then compare
    # that same penalty at every rank. Training-only choices also stay fixed.
    production = []
    for r in range(env.replications):
        with np.load(output/'data'/env.name/'replications'/f'{r:04d}.npz', allow_pickle=False) as saved:
            production.append({'regret': saved['regret'][:len(d.T)], 'choices': saved['choices'][:len(d.T)]})
    oracle = np.mean([x['regret'][:, :len(d.penalties)] for x in production], axis=0).argmin(axis=1)
    policy_rows = []
    for t, T in enumerate(d.T):
        indices = np.c_[np.stack([x['choices'][t] for x in production[:50]]), np.full(50, oracle[t])]
        for j, method in enumerate(('holdout25', 'rolling3', 'theory_0.25', 'theory_1', 'theory_4', 'ensemble_plugin_loss_oracle')):
            chosen = indices[:, j]
            low = values['regret'][np.arange(50), t, 1, chosen]
            high = values['regret'][np.arange(50), t, 2, chosen]
            delta = low-high
            mean, se = float(delta.mean()), float(delta.std(ddof=1)/np.sqrt(50))
            policy_rows.append({'environment': env.name, 'T': T, 'method': method, 'paired_paths': 50,
                                'regret512': low.mean(), 'regret1024': high.mean(), 'paired_difference': mean,
                                'paired_MCSE': se, 'ci_low': mean-1.96*se, 'ci_high': mean+1.96*se,
                                'floor512': floors[512], 'floor_fraction': floors[512]/low.mean(),
                                'difference_confidence_fraction': (abs(mean)+1.96*se)/low.mean(),
                                'certified': bool(floors[512] <= .05*low.mean() and abs(mean)+1.96*se <= .05*low.mean())})
    policy_frame = pd.DataFrame(policy_rows)
    policy_frame.to_csv(output/'audit'/f'{env.name}_rank_policy_resolution.csv', index=False)
    report = {'schema': 'fixed-lambda-rank-audit/2.0', 'executed': True, 'identity': identity,
              'completed_utc': utc_now(), 'environment': env.name, 'paired_paths': 50,
              'ranks': [256, 512, 1024], 'population_groups': 32768, 'floors': floors,
              'certified_cells': int(frame.certified.sum()), 'total_cells': len(frame),
              'uniform_curve_certified': bool(frame.certified.all()),
              'policy_cells_certified': int(policy_frame.certified.sum()), 'policy_cells': len(policy_frame),
              'scope': 'fixed penalties on practical T grid; extended T not certified by this audit',
              'criterion': 'floor/mean regret <=5%; (abs(mean paired difference)+1.96 MCSE)/mean regret <=5%'}
    json_write(output/'audit'/f'{env.name}_rank_resolution.json', report)
    return report


def quadrature_audit(output, environment):
    output = Path(output)
    d = design_for('confirmation_v2')
    manifest = json.loads((output/'run_manifest.json').read_text())
    require(manifest['source_hashes'] == source_hashes(), 'Production scientific sources changed before quadrature audit')
    env = next(e for e in d.cases if e.name == environment)
    basis = NystromBasis(rank=env.rank, nu=env.nu, seed=d.basis_seed)
    directory = output/'data'/env.name
    production_risk = []
    for r in range(env.replications):
        with np.load(directory/'replications'/f'{r:04d}.npz', allow_pickle=False) as saved:
            production_risk.append(saved['regret'][:len(d.T), :len(d.penalties)])
    oracle = np.mean(production_risk, axis=0).argmin(axis=1)
    coefficients = {}
    for r in range(50):
        with np.load(output/'audit/rank_paths'/env.name/f'{r:04d}.npz', allow_pickle=False) as rank_saved:
            grid_coefficients = rank_saved['coefficients512']
        with np.load(directory/'replications'/f'{r:04d}.npz', allow_pickle=False) as saved:
            for t, T in enumerate(saved['T']):
                a = saved['selected_coefficients'][t]
                labels = ['holdout25', 'rolling3', 'theory_0.25', 'theory_1', 'theory_4']
                if t < len(d.T):
                    require(np.allclose(grid_coefficients[t][:, saved['choices'][t]], a, rtol=1e-6, atol=1e-7),
                            'Paired-rank refit does not reproduce the original rank-512 policies')
                    a = np.column_stack([a, grid_coefficients[t][:, oracle[t]]])
                    labels.append('ensemble_plugin_loss_oracle')
                coefficients[r, int(T)] = a, labels
    records = []
    for s in range(4):
        for groups in (8192, 32768):
            m, S = population_moments(env.parameters(), basis, groups, seed('quadrature', s+1))
            for (r, T), (a, labels) in coefficients.items():
                loss, sr = moment_metrics(a, m, S)
                for j, method in enumerate(labels):
                    records.append({'environment': env.name, 'replication': r, 'T': T, 'method': method,
                                    'seed_index': s, 'seed': seed('quadrature', s+1), 'groups': groups,
                                    'regret': loss[j]-(1-env.parameters().q_star), 'population_sharpe': sr[j]})
            print(f'Quadrature {env.name}: seed {s+1}/4, {groups} groups', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output/'audit'/f'{env.name}_quadrature_paths.csv', index=False)
    grouped = frame.groupby(['T', 'method', 'seed_index', 'groups']).regret.mean().unstack('groups')
    rows = []
    for (T, method), values in grouped.groupby(level=['T', 'method']):
        differences = values[8192]-values[32768]
        mean, se = differences.mean(), differences.std(ddof=1)/np.sqrt(4)
        reference = values[32768].mean()
        rows.append({'environment': env.name, 'T': T, 'method': method, 'paths': 50, 'independent_seeds': 4,
                     'mean_regret32768': reference, 'difference8192_minus32768': mean,
                     'quadrature_seed_MCSE': se, 'ci_low': mean-1.96*se, 'ci_high': mean+1.96*se,
                     'certified': bool(abs(mean)+1.96*se <= .05*reference)})
    result = pd.DataFrame(rows)
    result.to_csv(output/'audit'/f'{env.name}_quadrature_resolution.csv', index=False)
    report = {'schema': 'independent-quadrature-audit/2.0', 'executed': True, 'completed_utc': utc_now(),
              'source_sha256': file_hash(__file__), 'protocol_hash': d.protocol_hash, 'paths': 50,
              'groups': [8192, 32768], 'independent_seeds': [seed('quadrature', s+1) for s in range(4)],
              'certified_cells': int(result.certified.sum()), 'total_cells': len(result),
              'scope': 'fixed holdout, rolling and theory policies at all available T; fixed plugin oracle on practical grid; uncertainty across independent quadrature seeds; no reselection'}
    json_write(output/'audit'/f'{env.name}_quadrature_resolution.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['rank', 'quadrature'])
    parser.add_argument('--environment', required=True)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.mode == 'rank':
        print(rank_audit(args.output, args.environment, args.workers))
    else:
        print(quadrature_audit(args.output, args.environment))
