"""Fresh rough Rich6D economy calibrated to annualized optimal Sharpe three.

Monthly convention: annualized Sharpe = sqrt(12) times marginal monthly
Sharpe, not the Sharpe of compounded annual returns. All returns, fits,
population operators, and numerical audits are regenerated under new means.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh

from simulations.dgp.balanced import DGPParameters, BalancedFactorDGP, w_star
from simulations.dgp.rough import normalization, uniform_tail_bound
from simulations.estimator.kernel import NystromBasis
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.experiments.population import population_ridge
from simulations.provenance import digest, file_hash, json_write, npz_write, utc_now, output_lock
from simulations.run_rough import seed, moments, run_replication, summarize, TIMES, GRID

ROOT = Path(__file__).resolve().parents[1]
ANNUAL_SR = 3.0
PERIODS_PER_YEAR = 12
COMMAND = 'VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough_sr3 --workers 2'


def calibrated_parameters():
    base = DGPParameters(loading_map='rich6d_rough', eta=.35, fourier_terms=128)
    monthly_sr = ANNUAL_SR/np.sqrt(PERIODS_PER_YEAR)
    target_q = monthly_sr**2/(1+monthly_sr**2)
    factor_mean_scale = float(np.sqrt(target_q/base.q_star))
    parameters = replace(base,mu_F=tuple(factor_mean_scale*np.asarray(base.mu_F)))
    np.testing.assert_allclose(parameters.sr_star*np.sqrt(PERIODS_PER_YEAR),ANNUAL_SR,rtol=1e-14)
    np.testing.assert_allclose(parameters.factor_covariance+np.outer(parameters.mu_F,parameters.mu_F),np.eye(3),atol=1e-15)
    return parameters,factor_mean_scale


def truncation_check(output, p, basis, factor_mean_scale):
    report = output / 'fourier_truncation.json'
    if report.exists():
        return json.loads(report.read_text())
    # W* scales linearly with factor means; preserve the original relative
    # policy tolerance using the analytic scale, fixed before simulation.
    policy_tolerance = 1e-7*factor_mean_scale
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
    assert records[0]['max_policy_difference_to_2048'] < policy_tolerance
    assert records[0]['max_optimal_payoff_difference_to_2048'] < 1e-9
    assert records[0]['max_sharpe_difference_to_2048'] < 1e-8
    assert records[0]['max_regret_difference_to_2048'] < 1e-8
    result = {'records': records, 'selected_M': 128, 'normalization': normalization(),
              'normalization_rule': 'Parseval: 1/sqrt(sum m^-10/log(m+1)^2); deterministic sum through 65536',
              'normalization_squared_sum_tail_bound': float(65536.**-9/(9*np.log(65537.)**2)),
              'diagnostic_T': 240, 'diagnostic_penalties': penalties.tolist(),
              'selection': 'Smallest candidate passing fixed absolute numerical tolerances; no rate examined.',
              'policy_tolerance_rule': 'Original absolute tolerance 1e-7 times analytic factor-mean scale; same relative policy precision',
              'parameters': p.to_dict(), 'factor_mean_scale': factor_mean_scale,
              'tolerances': {'policy': policy_tolerance, 'payoff': 1e-9, 'sharpe': 1e-8, 'regret': 1e-8},
              'limitation': 'Finite M is smooth; no mathematical verification of r=1.',
              'frozen_utc': utc_now(), 'seconds': time.perf_counter()-start}
    pd.DataFrame(records).to_csv(output/'fourier_truncation.csv', index=False)
    json_write(report, result)
    return result


def context(output):
    p, factor_mean_scale = calibrated_parameters()
    low = NystromBasis(rank=512, seed=seed('anchor'))
    high = NystromBasis(rank=1024, seed=seed('anchor'))
    truncation = truncation_check(output, p, low, factor_mean_scale)
    print('Fourier truncation frozen:', truncation['selected_M'], flush=True)
    pm, PS = moments(output, 'pilot', p, low, 8192, seed('pilot'))
    s_ref = float(eigh(PS, eigvals_only=True)[-1])
    spec = {'schema': 'rough-rich6d-annual-sr3/1', 'parameters': p.to_dict(), 'T': TIMES.tolist(),
            'replications': 100, 'rank_audit_paths': 50, 'rank': 512, 'audit_rank': 1024,
            'grid': GRID.tolist(), 'method': 'theory_1', 'a': s_ref, 'exponent': -.6,
            'theoretical_b': 1.5, 'sharpe_units': 'raw metrics monthly; figures and explicit annualized columns use sqrt(12)',
            'target_annual_sr': ANNUAL_SR, 'periods_per_year': PERIODS_PER_YEAR,
            'factor_mean_scale': factor_mean_scale, 'reference_monthly_sr': p.sr_star,
            'annualization': 'sqrt(12) times marginal monthly Sharpe; not compounded annual-return Sharpe',
            'training': 'Fresh returns generated and policies refitted under calibrated Gaussian factor means/covariance; no historical outcomes rescaled',
            'seed_pairing': 'Common innovations with the original rough run; independent across the 100 replications and between pilot/evaluation/training families' ,
            'population_groups': 8192, 'rank_groups': 32768,
            'quadrature_groups': [8192,32768], 'quadrature_seeds': 4,
            'rank_and_quadrature_relative_regret_tolerance': .05,
            'bootstrap_replications': 1000, 'master_seed': 2026100907,
            'source_hashes': {n: file_hash(ROOT/n) for n in (
                'simulations/run_rough_sr3.py','simulations/run_rough.py','simulations/dgp/balanced.py','simulations/dgp/rough.py',
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
        'protocol_sha256': file_hash(protocol), 'target_annual_sr': ANNUAL_SR,
        'periods_per_year': PERIODS_PER_YEAR, 'monthly_sr': p.sr_star,
        'factor_mean_scale': factor_mean_scale, 'factor_means': list(p.mu_F),
        'factor_covariance': p.factor_covariance.tolist()})
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
    metadata = {'parameters': p.to_dict(), 'reference_sharpe': p.sr_star, 'reference_sharpe_annualized': ANNUAL_SR,
                'periods_per_year': PERIODS_PER_YEAR, 'q_reference': p.q_star,
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



_WORKER_CONTEXT = None
_WORKER_OUTPUT = None


def initialize_worker(context, output):
    global _WORKER_CONTEXT, _WORKER_OUTPUT
    _WORKER_CONTEXT, _WORKER_OUTPUT = context, output


def execute_replication(index):
    run_replication(index, _WORKER_CONTEXT, _WORKER_OUTPUT)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/rich6d_rough_sr3_annual_monthly_v1'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    with output_lock(args.output):
        c = context(args.output)
        print('Calibration: monthly SR*', c['p'].sr_star, 'annual SR*', ANNUAL_SR,
              'pilot a', c['spec']['a'], flush=True)
        if args.prepare_only:
            return
        if not args.render_only:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                     initargs=(c,args.output)) as pool:
                futures = [pool.submit(execute_replication,i) for i in range(100)]
                for completed,future in enumerate(as_completed(futures),1):
                    future.result()
                    status = {'status':'running','completed':completed,'total':100,
                        'elapsed_seconds':time.perf_counter()-start,'updated_utc':utc_now()}
                    json_write(args.output/'status.json',status)
                    print(f'Replications {completed}/100; {status["elapsed_seconds"]:.1f}s',flush=True)
        tables = summarize(c,args.output)
        from simulations.plotting.rough_final import render
        render(c,args.output,tables)
        json_write(args.output/'status.json', {'status':'complete','replications':100,
            'run_hash':c['run_hash'],'completed_utc':utc_now(),'elapsed_seconds':time.perf_counter()-start})
        manifest = {'run_hash':c['run_hash'],'python':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
            'platform':platform.platform(),'git_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'command':COMMAND,'target_annual_sr':ANNUAL_SR,'periods_per_year':PERIODS_PER_YEAR,
            'artifact_sha256':{str(f.relative_to(args.output)):file_hash(f) for f in args.output.rglob('*')
                if f.is_file() and f.name not in ('manifest.json','.run.lock')},'finished_utc':utc_now()}
        json_write(args.output/'manifest.json',manifest)
        print('COMPLETE',args.output,flush=True)


if __name__ == '__main__':
    main()
