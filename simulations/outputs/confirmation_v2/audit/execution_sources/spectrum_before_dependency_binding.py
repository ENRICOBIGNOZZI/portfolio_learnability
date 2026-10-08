"""Group-quadrature spectrum at fixed resolutions/seeds, without slope gates.

The balanced triplet rows are orthogonal: B_g B_g'=c_beta I_3. Consequently
the quadrature coefficient operator is A=a I+gamma B B', with
a=(c_beta+sigma_eps²)/(N*3M), gamma=(1-1/G)/(3M)². Nonzero managed eigenvalues
are those of A^(1/2) K A^(1/2). This avoids the dense RKHS square root and the
three large intermediates used by the original full-spectrum audit.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

from simulations.dgp.balanced import DGPParameters, balanced_triplets, beta
from simulations.estimator.kernel import matern
from simulations.diagnostics.confirmation_pilot import seed
from simulations.provenance import file_hash, json_write, npz_write, require, utc_now


def managed_operator(K, loadings, p):
    n = len(K)
    a = (p.c_beta+p.sigma_eps**2)/(p.N*n)
    gamma = (1-1/p.G)/n**2
    norm = n*p.c_beta/3
    equal = loadings.T@loadings
    require(np.allclose(equal, norm*np.eye(3), rtol=1e-12, atol=1e-12), 'Quadrature loading balance')
    scale = (np.sqrt(a+gamma*norm)-np.sqrt(a))/norm
    def root(x):
        return np.sqrt(a)*x+scale*(loadings@(loadings.T@x))
    def multiply(x):
        return root(K@root(x))
    return LinearOperator(K.shape, matvec=multiply, matmat=multiply, dtype=np.float64)


def dense_crosscheck():
    from simulations.diagnostics.baseline import population_operators
    rows = []
    for eta in (0., .35):
        p = DGPParameters(loading_map='rich6d' if eta else 'baseline_original', eta=eta)
        groups, s = 128, seed('spectrum', 20)
        kernel_values, original, z = population_operators(p, groups, s)
        K = matern(z, z)
        operator = managed_operator(K, beta(z, p), p)
        explicit = operator@np.eye(len(K))
        require(np.max(np.abs(explicit-explicit.T)) < 1e-12, 'Managed operator symmetry')
        reference = np.linalg.eigvalsh(original)
        current = np.linalg.eigvalsh(explicit)
        require(np.allclose(reference, current, rtol=1e-9, atol=1e-13), 'Independent group operator mismatch')
        partial, _ = eigsh(operator, k=100, which='LA', tol=1e-9,
                            v0=np.random.default_rng(771).normal(size=len(K)))
        error = float(np.max(np.abs(np.sort(partial)-current[-100:])))
        require(error < 1e-11, 'Partial eigensolver does not reproduce dense reference')
        rows.append({'loading_map': p.loading_map, 'groups': groups,
                     'independent_operator_eigenvalue_max_error': float(np.max(np.abs(current-reference))),
                     'partial_dense_max_error': error, 'passed': True})
    return rows


def eigenpairs(operator, n, count, random_seed, dense=None):
    if dense is not None:
        values, vectors = eigh(dense, subset_by_index=[n-count, n-1], check_finite=False)
        converged = True
    else:
        try:
            values, vectors = eigsh(operator, k=count, which='LA', tol=1e-8,
                                     ncv=min(n, 2*count+1), maxiter=1000,
                                     v0=np.random.default_rng(random_seed).normal(size=n))
            converged = True
        except ArpackNoConvergence as exc:
            values, vectors = exc.eigenvalues, exc.eigenvectors
            converged = False
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    residuals = np.empty(len(values))
    # Blocked multiplication bounds peak memory and verifies actual eigenpairs.
    for first in range(0, len(values), 32):
        v = vectors[:, first:first+32]
        residuals[first:first+32] = np.linalg.norm(operator@v-v*values[first:first+32], axis=0)
    return values, residuals, converged


def run(output, groups_grid=(1024, 2048, 4096), seed_indices=(0, 1, 2)):
    output = Path(output)
    destination = output/'audit/spectrum'
    destination.mkdir(parents=True, exist_ok=True)
    check = dense_crosscheck()
    identity = file_hash(__file__)
    json_write(destination/'dense_crosscheck.json', {'passed': True, 'source_sha256': identity, 'cases': check})
    diagnostics, spectra, fits = [], [], []
    for groups in groups_grid:
        for seed_index in seed_indices:
            cached = destination/f'groups{groups}_seed{seed_index}.npz'
            meta = destination/f'groups{groups}_seed{seed_index}.json'
            if cached.exists() and meta.exists():
                with np.load(cached, allow_pickle=False) as saved:
                    require(str(saved['source_sha256']) == identity, 'Spectral cache identity mismatch')
                    values = dict(saved)
                record = json.loads(meta.read_text())
            else:
                started = time.perf_counter()
                rng = np.random.default_rng(seed('spectrum', seed_index))
                ranks = np.clip(rng.random((groups, 6)), np.finfo(float).eps, 1-np.finfo(float).eps)
                z = balanced_triplets(ranks)
                n = len(z)
                K = np.empty((n, n))
                for first in range(0, n, 256):
                    K[first:first+256] = matern(z[first:first+256], z)
                kernel = LinearOperator(K.shape, matvec=lambda x: K@x/n, matmat=lambda x: K@x/n, dtype=np.float64)
                v, residual, success = eigenpairs(kernel, n, 800, seed('spectrum', seed_index), dense=K/n if n <= 6144 else None)
                values = {'kernel_values': v, 'kernel_residuals': residual, 'source_sha256': identity}
                convergence = {'kernel': success}
                for loading, eta in [('baseline_original', 0.), ('rich6d', .35)]:
                    p = DGPParameters(loading_map=loading, eta=eta)
                    operator = managed_operator(K, beta(z, p), p)
                    explicit = operator@np.eye(n) if n <= 3072 else None
                    v, residual, success = eigenpairs(operator, n, 800, seed('spectrum', seed_index), dense=explicit)
                    values[loading+'_values'], values[loading+'_residuals'] = v, residual
                    convergence[loading] = success
                    del operator, explicit
                record = {'groups': groups, 'nodes': n, 'seed_index': seed_index, 'seed': seed('spectrum', seed_index),
                          'source_sha256': identity, 'completed_utc': utc_now(), 'convergence': convergence,
                          'elapsed_seconds': time.perf_counter()-started,
                          'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          'estimated_kernel_bytes': n*n*8,
                          'algorithm': 'dense partial solve at smaller sizes; full-kernel matrix operator and partial Lanczos at larger sizes'}
                npz_write(cached, **values)
                json_write(meta, record)
                del K
                print(f'Spectrum: groups={groups}, seed={seed_index}, {record["elapsed_seconds"]:.1f}s', flush=True)
            diagnostics.append(record)
            for name in ('kernel', 'baseline_original', 'rich6d'):
                v, residual = values[name+'_values'], values[name+'_residuals']
                resolved = (v > v[0]*1e-12)&(residual <= 1e-5*v)
                for j in range(len(v)):
                    spectra.append({'groups': groups, 'seed_index': seed_index, 'operator': name, 'index': j+1,
                                    'eigenvalue': v[j], 'absolute_residual': residual[j], 'resolved': bool(resolved[j])})
                for first, last in ((30, 200), (60, 400), (120, 800)):
                    complete = len(v) >= last and bool(resolved[first-1:last].all())
                    fit = {'groups': groups, 'seed_index': seed_index, 'operator': name, 'first': first, 'last': last,
                           'resolved': complete, 'slope': None, 'OLS_SE': None, 'historical_band_compatible': None}
                    if complete:
                        x, y = np.log(np.arange(first, last+1)), np.log(v[first-1:last])
                        centered = x-x.mean()
                        slope = centered@(y-y.mean())/(centered@centered)
                        error = y-y.mean()-slope*centered
                        fit.update(slope=float(slope), OLS_SE=float(np.sqrt((error@error)/((len(x)-2)*(centered@centered)))),
                                   historical_band_compatible=bool(abs(slope+1.5) <= .35))
                    fits.append(fit)
    pd.DataFrame(spectra).to_csv(destination/'eigenvalues.csv', index=False)
    pd.DataFrame(fits).to_csv(destination/'descriptive_fits.csv', index=False)
    report = {'schema': 'spectrum-resolution/2.0', 'executed': True, 'source_sha256': identity,
              'completed_utc': utc_now(), 'analytic_E5_status': 'operator sandwich and analytic Matern asymptotics; fixed N',
              'operator_identity_status': 'independent dense group-formula and partial-solver crosschecks passed',
              'quadrature_resolution_status': 'multiple independent group quadratures; compare reported eigenvalues and slopes across resolutions',
              'spectral_fit_descriptive_statistics': 'descriptive_fits.csv; no slope-closeness gate',
              'all_requested_eigenpairs_resolved': all(row['resolved'] for row in fits),
              'cases': diagnostics, 'dense_crosscheck': check,
              'scope': 'headline nu=1.5, both maps; 800 largest eigenpairs, not a complete infinite-dimensional spectrum'}
    json_write(destination/'spectrum_resolution.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    parser.add_argument('--groups', nargs='+', type=int, default=[1024, 2048, 4096])
    parser.add_argument('--seed-indices', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()
    run(args.output, args.groups, args.seed_indices)
