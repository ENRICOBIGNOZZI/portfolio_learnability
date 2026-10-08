"""One-command, resumable simulation pipeline: python -m simulations.run --profile paper."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from simulations.config.design import design_for
from simulations.diagnostics.baseline import AuditSettings, run_audit, audit_dates, audit_monte_carlo, audit_spectrum
from simulations.dgp.balanced import DGPParameters
from simulations.dgp.robustness import condition_status
from simulations.estimator.kernel import NystromBasis
from simulations.experiments.population import population_moments, population_ridge
from simulations.diagnostics.path_integrity import initialize_observed_worker, observed_replication
from simulations.experiments.reporting import summarize
from simulations.plotting.figures import render_environment


ROOT = Path(__file__).resolve().parents[1]
SCIENCE_SOURCES = ['simulations/config/design.py', 'simulations/dgp/balanced.py', 'simulations/dgp/robustness.py',
                   'simulations/estimator/kernel.py', 'simulations/estimator/ridge.py',
                   'simulations/experiments/population.py', 'simulations/experiments/monte_carlo.py']


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def ensure_baseline():
    destination = ROOT/'simulations/outputs/audit/baseline'
    path = destination/'dgp_audit.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    hashes = report.get('provenance', {}).get('source_sha256', {})
    current = bool(hashes) and all((ROOT/name).exists() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name, h in hashes.items())
    if not report.get('passed') or not current:
        report = run_audit(DGPParameters(), AuditSettings(), destination)
    if not report['passed']:
        raise RuntimeError('Baseline acceptance failed; downstream production is blocked.')
    return report


def environment_audit(env, destination):
    path = destination/'environment_audit.json'
    fingerprint = hashlib.sha256(json.dumps(asdict(env), sort_keys=True).encode()).hexdigest()
    if path.exists():
        report = json.loads(path.read_text())
        kernel_audit = report.get('actual_kernel_conditional_audit', {})
        kernel_audit_present = env.nu==1.5 or (
            kernel_audit.get('passed') and len(kernel_audit.get('states', []))>=24)
        if report.get('environment_hash')==fingerprint and report.get('passed') and kernel_audit_present:
            return report
    p = env.parameters()
    if env.variant != 'baseline':
        from simulations.diagnostics.variants import audit_variant
        report = audit_variant(env)
    else:
        settings = replace(AuditSettings(), date_pool=48, audited_dates=6, mc_periods=20000)
        dates, mc = audit_dates(p, settings), audit_monte_carlo(p, settings)
        report = {'passed': dates['passed'] and mc['passed'], 'conditional': dates, 'monte_carlo': mc,
                  'conditions': condition_status('baseline')}
        if env.nu != 1.5:
            spectral, rows = audit_spectrum(p, replace(settings, slope_tolerance=.5), nu=env.nu)
            import pandas as pd
            pd.DataFrame(rows).to_csv(destination/'spectrum.csv', index=False)
            report['spectrum'] = spectral
            report['passed'] = report['passed'] and spectral['passed']
            from simulations.diagnostics.variants import audit_variant
            report['actual_kernel_conditional_audit'] = audit_variant(env)
            report['passed'] = report['passed'] and report['actual_kernel_conditional_audit']['passed']
    report['environment_hash'] = fingerprint
    json_write(path, report)
    if not report['passed']:
        raise RuntimeError(f'{env.name} failed acceptance; its production is blocked.')
    return report


def context_for(design, environment, output, source_hash):
    p = environment.parameters()
    basis = NystromBasis(rank=environment.rank, nu=environment.nu, seed=design.basis_seed)
    key = hashlib.sha256(json.dumps({'design': design.to_dict(), 'environment': asdict(environment),
                                   'source_hash': source_hash}, sort_keys=True).encode()).hexdigest()
    destination = output/'data'/environment.name
    destination.mkdir(parents=True, exist_ok=True)
    moments_path = destination/'population.npz'
    if moments_path.exists():
        with np.load(moments_path, allow_pickle=False) as saved:
            if str(saved['run_hash']) != key:
                raise ValueError('Population cache differs from this configuration/source. Use a new output directory.')
            mean, second = saved['mean'], saved['second']
    else:
        mean, second = population_moments(p, basis, design.population_groups, design.population_seed, environment.variant)
        np.savez_compressed(moments_path, mean=mean, second=second, run_hash=key)
    q = p.q_star if environment.variant!='heteroskedastic' else float(mean@np.linalg.solve(second, mean))
    coefficients, complexity, bias, floor = population_ridge(mean, second, np.asarray(design.penalties), q)
    sr = np.sqrt(q/(1-q))
    context = {'design': design, 'environment': environment, 'basis': basis,
               'population_mean': mean, 'population_second': second, 'population_coefficients': coefficients,
               'population_complexity': complexity, 'population_bias': bias,
               'population_loss': 1-q+bias, 'optimal_loss': 1-q, 'reference_sharpe': sr,
               'checkpoint_dir': str(destination/'replications'), 'run_hash': key}
    json_write(destination/'population.json', {'q_reference': q, 'reference_sharpe': sr, 'subspace_regret_floor': floor,
                'theoretical_b': environment.theoretical_b, 'parameters': p.to_dict(), 'environment': asdict(environment),
                'reference_kind': 'exact conditional optimum' if environment.variant!='heteroskedastic' else 'finite Nyström-subspace optimum only',
                'population_groups': design.population_groups, 'population_seed': design.population_seed,
                'basis_seed': design.basis_seed, 'run_hash': key})
    return context, destination


def run(profile='smoke', output=None, only=None, render_only=False, workers=None):
    design = design_for(profile)
    worker_count = design.workers if workers is None else int(workers)
    if worker_count < 1:
        raise ValueError('workers must be positive.')
    output = Path(output) if output else ROOT/'simulations/outputs'/profile
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest = output/'run_manifest.json'
    numerical_versions = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
    if previous_manifest.exists() and next((output/'data').rglob('*.npz'), None) is not None:
        previous = json.loads(previous_manifest.read_text())
        if any(previous.get(key)!=value for key,value in numerical_versions.items()):
            raise ValueError('Cached arrays come from a different numerical runtime. Use the pinned environment or a fresh --output directory.')
    started = time.perf_counter()
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SCIENCE_SOURCES}
    source_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {'status': 'running', 'profile': profile, 'configuration': design.to_dict(), 'config_hash': design.digest,
                'source_hashes': source_hashes, 'git_sha': git, 'working_tree_sources_hashed': True,
                'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                'hardware': platform.platform(), 'workers': worker_count, 'completed_environments': [],
                'logical_cpus': os.cpu_count(),
                'BLAS_thread_environment': {name: os.environ.get(name) for name in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
                'package_versions': {name: importlib.metadata.version(name) for name in ('numpy','scipy','pandas','matplotlib','jinja2')},
                'derived_output_source_hashes': {
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((ROOT/'simulations').rglob('*.py'))
                    if str(path.relative_to(ROOT)) not in SCIENCE_SOURCES},
                'scope': 'full' if not only else list(only)}
    json_write(output/'run_manifest.json', manifest)
    ensure_baseline()
    for env in design.cases:
        if only and env.name not in only:
            continue
        print(f'[{profile}] {env.name}: {env.replications} replications, rank {env.rank}', flush=True)
        context, destination = context_for(design, env, output, source_hash)
        if env.name!='baseline':
            environment_audit(env, destination)
        if not render_only:
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize_observed_worker, initargs=(context,)) as executor:
                futures = [executor.submit(observed_replication, r) for r in range(env.replications)]
                completed = 0
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed%25==0 or completed==env.replications:
                        print(f'  {env.name}: {completed}/{env.replications}', flush=True)
                        manifest['current_environment'] = env.name
                        manifest['current_replications'] = completed
                        manifest['elapsed_seconds'] = time.perf_counter()-started
                        json_write(output/'run_manifest.json', manifest)
        summarize(context, destination)
        render_environment(destination, output/'figures', env.name)
        manifest['completed_environments'].append(env.name)
        json_write(output/'run_manifest.json', manifest)
    if not only:
        from simulations.plotting.tables import build_tables
        from simulations.plotting.robustness import render_comparisons
        build_tables(output, profile)
        render_comparisons(output, profile)
        if profile == 'paper':
            from simulations.diagnostics.approximation import rank_audit
            manifest['approximation_audit'] = rank_audit(output)
            if not manifest['approximation_audit']['passed']:
                manifest['status'] = 'blocked_by_approximation_audit'
                json_write(output/'run_manifest.json', manifest)
                raise RuntimeError('Approximation-rank acceptance failed; headline results are not accepted.')
            from simulations.diagnostics.quadrature import quadrature_audit
            manifest['quadrature_audit'] = quadrature_audit(output, workers=worker_count)
            if not manifest['quadrature_audit']['passed']:
                manifest['status'] = 'blocked_by_quadrature_audit'
                json_write(output/'run_manifest.json', manifest)
                raise RuntimeError('Population-quadrature acceptance failed; headline results are not accepted.')
    manifest['status'] = 'computation_complete_pending_final_audit'
    manifest['elapsed_seconds'] = time.perf_counter()-started
    json_write(output/'run_manifest.json', manifest)
    if not only:
        from simulations.diagnostics.verify import verify
        manifest['numerical_verification'] = verify(output, profile)
        manifest['status'] = 'numerically_verified_pending_paper_integration'
        json_write(output/'run_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'paper'], default='smoke')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--only', nargs='+', help='Explicit subset; never marked as a complete paper run')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--workers', type=int, help='Execution concurrency only; does not alter scientific configuration or seeds.')
    args = parser.parse_args()
    run(args.profile, args.output, args.only, args.render_only, args.workers)


if __name__=='__main__':
    main()
