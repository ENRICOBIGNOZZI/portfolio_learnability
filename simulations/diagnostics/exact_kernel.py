"""Small full-kernel temporal Gram reference, independent of Nyström fitting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.dgp.balanced import BalancedFactorDGP, DGPParameters
from simulations.estimator.kernel import NystromBasis, matern
from simulations.estimator.ridge import ridge_path
from simulations.provenance import file_hash, json_write, require, utc_now, source_hashes


def temporal_gram(dates, ell=1.):
    N, T = len(dates[0].returns), len(dates)
    G = np.empty((T, T))
    for t, left in enumerate(dates):
        for s in range(t+1):
            right = dates[s]
            G[t, s] = G[s, t] = left.returns@matern(left.z, right.z, 1.5, ell)@right.returns/N**2
    return G


def independent_gram(dates, ell=1.):
    """Explicit Euclidean distances and trigonometric-stock contractions."""
    N, T = len(dates[0].returns), len(dates)
    z = np.stack([d.z for d in dates])
    returns = np.stack([d.returns for d in dates])
    G = np.empty((T, T))
    for t in range(T):
        # One date against all dates; memory O(T*N²*D), not O((T*N)²).
        radius = np.sqrt(3*np.sum((z[t][None, :, None, :]-z[:, None, :, :])**2, axis=-1))/ell
        K = (1+radius)*np.exp(-radius)
        G[t] = np.einsum('i,sij,sj->s', returns[t], K, returns)/N**2
    return G


def exact_fit(G, penalties):
    T = len(G)
    eig, U = np.linalg.eigh(G)
    require(eig[0] >= -1e-12, 'Full temporal Gram is not positive semidefinite')
    alpha = U@((U.T@np.ones(T))[:, None]/(eig[:, None]+T*np.asarray(penalties)))
    complexity = (eig[:, None]/(eig[:, None]+T*np.asarray(penalties))).sum(axis=0)
    return alpha, complexity


def run(destination, sizes=(30, 60), lengths=(60, 120), ranks=(256, 512, 1024)):
    destination = Path(destination)
    penalties = np.geomspace(1e-10, 1e-3, 12)
    rows = []
    max_gram_error = max_equation_error = max_complexity_error = 0.
    for eta in (0., .35):
        for N in sizes:
            p = DGPParameters(N=N, loading_map='rich6d' if eta else 'baseline_original', eta=eta)
            all_dates = list(BalancedFactorDGP(p, 202610080101).simulate(max(lengths)))
            for T in lengths:
                dates = all_dates[:T]
                G, G2 = temporal_gram(dates), independent_gram(dates)
                gram_error = np.max(np.abs(G-G2))
                require(gram_error < 1e-12, 'Independent kernel Grams differ')
                max_gram_error = max(max_gram_error, float(gram_error))
                alpha, C = exact_fit(G, penalties)
                payoff = G@alpha
                loss = ((1-payoff)**2).mean(axis=0)
                norm = (alpha*payoff).sum(axis=0)
                for j, lam in enumerate(penalties):
                    system = G2+T*lam*np.eye(T)
                    solved = np.linalg.solve(system, np.ones(T))
                    residual = np.max(np.abs(system@alpha[:, j]-1))
                    c2 = np.trace(np.linalg.solve(system, G2))
                    max_equation_error = max(max_equation_error, float(residual))
                    max_complexity_error = max(max_complexity_error, float(abs(C[j]-c2)))
                    require(residual < 1e-8 and abs(C[j]-c2) < 1e-7, 'Exact normal equation/trace mismatch')
                    require(np.allclose(alpha[:, j], solved, rtol=1e-7, atol=1e-7), 'Independent dual coefficients mismatch')
                    require(abs(loss[j]+lam*norm[j]-T*lam*np.mean(alpha[:, j])) < 1e-9, 'Penalized objective identity mismatch')
                for rank in ranks:
                    basis = NystromBasis(rank=rank, seed=202610080102)
                    X = np.asarray([basis.managed(d.z, d.returns) for d in dates])
                    coef, complexity = ridge_path(X, penalties)
                    approx_payoff = X@coef
                    for j, lam in enumerate(penalties):
                        rows.append({'loading_map': p.loading_map, 'N': N, 'T': T, 'rank': rank,
                                     'lambda': lam, 'exact_complexity': C[j], 'nystrom_complexity': complexity[j],
                                     'exact_empirical_loss': loss[j], 'nystrom_empirical_loss': np.mean((1-approx_payoff[:, j])**2),
                                     'training_payoff_squared_difference': np.mean((payoff[:, j]-approx_payoff[:, j])**2),
                                     'exact_rkhs_norm_squared': norm[j], 'nystrom_rkhs_norm_squared': coef[:, j]@coef[:, j]})
                print(f'Exact kernel: {p.loading_map} N={N} T={T}', flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination/'exact_kernel_reference.csv', index=False)
    report = {'schema': 'exact-kernel-reference/2.0', 'passed': True, 'completed_utc': utc_now(),
              'scientific_source_hashes': source_hashes(),
              'source_sha256': file_hash(__file__), 'economic_seed': 202610080101, 'anchor_seed': 202610080102,
              'max_independent_gram_error': max_gram_error, 'max_normal_equation_error': max_equation_error,
              'max_complexity_error': max_complexity_error, 'sizes': list(sizes), 'lengths': list(lengths),
              'ranks': list(ranks), 'penalties': penalties.tolist(), 'rows': len(rows),
              'interpretation': 'Exact-solver identities verified. Nyström differences are measured, not assumed zero; these small samples do not certify headline resolution.'}
    json_write(destination/'exact_kernel_reference.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2/audit'))
    print(run(parser.parse_args().output))
