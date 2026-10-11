"""Targeted higher-rank audit in raw kernel coordinates, with bounded storage.

Cholesky whitening differs from symmetric whitening by an orthogonal rotation;
the Matérn RKHS penalty and fitted policy are unchanged. Population evaluation
streams balanced groups and integrates factors/noise analytically, rather than
using a short realized-return test history.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.stats import qmc

from simulations.dgp.balanced import balanced_triplets, beta
from simulations.estimator.kernel import matern
from simulations.estimator.ridge import ridge_path
from simulations.extended.design import OUTPUT, BASIS_SEED, BASE_A, parameters, seed
from simulations.extended.histories import raw_managed_history
from simulations.extended.population import raw_integrals
from simulations.extended.followup_diagnostics import DEST
from simulations.provenance import json_write, npz_write, file_hash, utc_now

TIMES = np.array([1440, 2160, 3240, 4860])
CACHE = Path('/tmp/rich6d_rank_followup_cache')


class RawBasis:
    def __init__(self, rank):
        if rank < 4 or rank & (rank-1):
            raise ValueError('Rank must be a power of two.')
        self.rank = rank
        self.anchors = 2*qmc.Sobol(6, scramble=True, seed=BASIS_SEED).random_base2(rank.bit_length()-1)-1

    def raw(self, z):
        return matern(z, self.anchors)


def evaluate(basis, coefficients, groups, quadrature_seed, p=None):
    """Exact factor/noise moments with independently sampled characteristic groups."""
    p = parameters() if p is None else p
    rng = np.random.default_rng(quadrature_seed)
    coefficients = np.atleast_2d(coefficients)
    count_policies = coefficients.shape[1]
    hs = np.zeros((count_policies, 3))
    h2 = np.zeros(count_policies)
    w2 = np.zeros(count_policies)
    for start in range(0, groups, 256):
        count = min(256, groups-start)
        ranks = np.clip(rng.random((count, 6)), np.finfo(float).eps, 1-np.finfo(float).eps)
        z = balanced_triplets(ranks)
        w = (basis.raw(z)@coefficients).reshape(count, 3, count_policies)
        b = beta(z, p).reshape(count, 3, 3)
        h = np.einsum('grm,grk->gmk', w, b)/3
        hs += h.sum(axis=0)
        h2 += np.sum(h*h, axis=(0, 2))
        w2 += np.sum(w*w, axis=(0, 1))
    mean_h = hs/groups
    mean = mean_h@np.asarray(p.mu_F)
    second = h2/(groups*p.G)+(1-1/p.G)*np.sum(mean_h**2, axis=1)+p.sigma_eps**2/p.N*w2/(3*groups)
    variance = second-mean**2
    if np.any(variance <= 0):
        raise ValueError('Nonpositive population variance.')
    return dict(regret=p.q_star-2*mean+second, annual_SR=np.sqrt(12)*mean/np.sqrt(variance),
                mean=mean, second=second)


def freeze():
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST/'rank_protocol.json'
    plan = dict(schema='targeted-rank-followup/1', ranks=[4096, 8192],
        environment='baseline', times=TIMES.tolist(), paired_pilot_indices=list(range(12)),
        population_groups=32768, population_seed=seed('population'),
        independent_evaluation_groups=131072, independent_evaluation_seed=seed('quadrature', 1),
        basis_seed=BASIS_SEED, a=BASE_A, tolerance=.05,
        scope='Targeted numerical audit, not replacement 300-path production. No new final-Monte-Carlo replication indices.',
        rule='Compare unchanged theory policies at nested ranks; retain 5% floor and abs(paired mean)+1.96 MCSE criteria. No certification above the largest investigated rank.',
        original_protocol_sha256=file_hash(OUTPUT/'protocol.json'))
    if path.exists():
        if json.loads(path.read_text())['design'] != plan:
            raise ValueError('Frozen follow-up design mismatch.')
    else:
        json_write(path, dict(frozen_utc=utc_now(), design=plan))
    CACHE.mkdir(parents=True, exist_ok=True)


def population(rank):
    freeze()
    dest = DEST/f'population_P{rank}.npz'
    if dest.exists() and (CACHE/f'chol_P{rank}.npy').exists():
        return
    start = time.perf_counter()
    p = parameters()
    basis = RawBasis(rank)
    gram = basis.raw(basis.anchors)
    L = cholesky(gram, lower=True, check_finite=False)
    np.save(CACHE/f'chol_P{rank}.npy', L)
    del L
    print(f'P{rank}: anchor Cholesky ready', flush=True)
    stats = raw_integrals(p, basis, 32768, seed('population'))
    m = stats['h_mean']@np.asarray(p.mu_F)
    S = stats.pop('hh')
    S /= p.G
    S += (1-1/p.G)*(stats['h_mean']@stats['h_mean'].T)
    S += (p.sigma_eps**2/p.N)*stats.pop('kk')
    del stats
    gc.collect()
    print(f'P{rank}: raw population moments ready ({time.perf_counter()-start:.1f}s)', flush=True)
    penalties = np.r_[0., BASE_A*TIMES.astype(float)**(-.6)]
    coefficients, residuals = [], []
    for lam in penalties:
        system = S+lam*gram
        factor = cholesky(system, lower=True, overwrite_a=False, check_finite=False)
        c = cho_solve((factor, True), m, check_finite=False)
        residuals.append(float(np.max(np.abs(system@c-m))))
        if residuals[-1] > 1e-8:
            raise ValueError('Raw population normal equations failed.')
        coefficients.append(c)
        del system, factor
    coefficients = np.asarray(coefficients).T
    mean = m@coefficients
    second = np.sum(coefficients*(S@coefficients), axis=0)
    regret = p.q_star-2*mean+second
    sr = np.sqrt(12)*mean/np.sqrt(second-mean**2)
    del S, gram
    gc.collect()
    independent = evaluate(basis, coefficients, 131072, seed('quadrature', 1))
    npz_write(dest, rank=rank, times=TIMES, penalties=penalties, raw_coefficients=coefficients,
              regret=regret, annual_SR=sr, independent_regret=independent['regret'],
              independent_annual_SR=independent['annual_SR'], normal_residual=residuals)
    json_write(DEST/f'population_resources_P{rank}.json', dict(completed_utc=utc_now(),
        elapsed_seconds=time.perf_counter()-start, rank=rank,
        fitted_quadrature_floor=float(regret[0]), independent_optimum_policy_regret=float(independent['regret'][0]),
        note='Independent regret evaluates the quadrature-fitted subspace optimum; it is not a certified exact minimum.'))
    print(f'P{rank}: population complete; floor={regret[0]:.8g}, independent={independent["regret"][0]:.8g}', flush=True)


def pilot(index, rank=8192, fit_only=False):
    freeze()
    dest = DEST/('fits' if fit_only else 'pilots')/f'P{rank}_rep_{index:03d}.npz'
    if dest.exists():
        return
    start = time.perf_counter()
    basis = RawBasis(rank)
    if not (CACHE/f'chol_P{rank}.npy').exists():
        np.save(CACHE/f'chol_P{rank}.npy', cholesky(basis.raw(basis.anchors), lower=True))
    L = np.load(CACHE/f'chol_P{rank}.npy')
    raw, hashes = raw_managed_history(basis, seed('rank_training', index), 100000+index,
        int(TIMES[-1]), 0, progress=lambda t: print(f'P{rank} pilot {index}: T={t}', flush=True))
    X = solve_triangular(L, raw['baseline'].T, lower=True, check_finite=False).T
    del raw
    coefficients = []
    residuals = []
    for T in TIMES:
        lam = BASE_A*float(T)**(-.6)
        a, _ = ridge_path(X[:T], np.array([lam]))
        residual = X[:T].T@(X[:T]@a)/T+lam*a-X[:T].mean(axis=0)[:, None]
        residuals.append(float(np.max(np.abs(residual))))
        if residuals[-1] > 1e-8:
            raise ValueError('Pilot normal equations failed.')
        coefficients.append(solve_triangular(L.T, a, lower=False, check_finite=False)[:, 0])
    coefficients = np.asarray(coefficients).T
    del X, L
    gc.collect()
    # Evaluate both ranks in ONE kernel pass at each quadrature, preserving pairing.
    from simulations.extended.population import basis_cached
    low_basis = basis_cached(OUTPUT/'pilot', 4096, BASIS_SEED)
    with np.load(OUTPUT/'rank_pilot'/'P4096'/f'rep_{index:03d}.npz') as old:
        old_times = old['baseline_T']
        positions = [int(np.flatnonzero(old_times == t)[0]) for t in TIMES]
        low_c = low_basis.inverse_root@old['baseline_theory_coefficients'][positions].T
        old_hashes = json.loads(str(old['returns_hashes']))
        # Original pilot checkpoints predate the T3240 prefix hash. Verify
        # their saved T1440 prefix and independently replay the full original
        # economic stream (without any kernel work) to its terminal horizon.
        if hashes['baseline_T1440'] != old_hashes['baseline_T1440']:
            raise ValueError('Paired return histories differ.')
        from simulations.extended.histories import PairedHistory
        replay = PairedHistory(seed('rank_training', index), 100000+index)
        for _ in range(int(old_times[-1])):
            replay.step(False)
        if replay.hashes()['baseline'] != old_hashes['baseline']:
            raise ValueError('Complete original pilot history differs.')
        reference_regret = old['baseline_loss'][positions, -1]-(1-parameters().q_star)
        reference_sr = old['baseline_sr'][positions, -1]*np.sqrt(12)
    del low_basis
    if fit_only:
        npz_write(dest, index=index, rank=rank, times=TIMES, raw_coefficients=coefficients,
            lower_raw_coefficients=low_c, reference_regret=reference_regret,
            reference_annual_SR=reference_sr, normal_residual=residuals,
            returns_hashes=json.dumps(hashes), source_sha256=file_hash(Path(__file__)),
            frozen_design_sha256=file_hash(DEST/'rank_protocol.json'),
            elapsed_seconds=time.perf_counter()-start)
        print(f'P{rank} pilot {index} fit complete in {time.perf_counter()-start:.1f}s', flush=True)
        return
    combined = np.column_stack([np.pad(low_c, ((0, rank-4096), (0, 0))), coefficients])
    primary = evaluate(basis, combined, 32768, seed('population'))
    independent = evaluate(basis, combined, 131072, seed('quadrature', 1))
    np.testing.assert_allclose(primary['regret'][:len(TIMES)], reference_regret, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(primary['annual_SR'][:len(TIMES)], reference_sr, atol=1e-8, rtol=1e-7)
    npz_write(dest, index=index, rank=rank, times=TIMES, raw_coefficients=coefficients,
        primary_regret=primary['regret'].reshape(2, -1),
        independent_regret=independent['regret'].reshape(2, -1),
        primary_annual_SR=primary['annual_SR'].reshape(2, -1),
        independent_annual_SR=independent['annual_SR'].reshape(2, -1),
        normal_residual=residuals, returns_hashes=json.dumps(hashes),
        elapsed_seconds=time.perf_counter()-start)
    print(f'P{rank} pilot {index} complete in {time.perf_counter()-start:.1f}s', flush=True)


def evaluate_fits(rank=8192):
    freeze()
    combined, records = [], []
    for index in range(12):
        with np.load(DEST/'fits'/f'P{rank}_rep_{index:03d}.npz') as z:
            if int(z['index']) != index or int(z['rank']) != rank:
                raise ValueError('Invalid fit identity.')
            if str(z['frozen_design_sha256']) != file_hash(DEST/'rank_protocol.json'):
                raise ValueError('Fit design mismatch.')
            records.append({k: z[k] for k in z.files})
            combined.extend([np.pad(z['lower_raw_coefficients'], ((0, rank-4096), (0, 0))), z['raw_coefficients']])
    combined = np.column_stack(combined)
    basis = RawBasis(rank)
    primary = evaluate(basis, combined, 32768, seed('population'))
    print('All paired fits: primary evaluation complete', flush=True)
    independent = evaluate(basis, combined, 131072, seed('quadrature', 1))
    for index, record in enumerate(records):
        part = slice(index*8, (index+1)*8)
        loss = primary['regret'][part].reshape(2, 4)
        sr = primary['annual_SR'][part].reshape(2, 4)
        np.testing.assert_allclose(loss[0], record['reference_regret'], atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(sr[0], record['reference_annual_SR'], atol=1e-8, rtol=1e-7)
        npz_write(DEST/'pilots'/f'P{rank}_rep_{index:03d}.npz',
            **record, primary_regret=loss, primary_annual_SR=sr,
            independent_regret=independent['regret'][part].reshape(2, 4),
            independent_annual_SR=independent['annual_SR'][part].reshape(2, 4))
    print('All 12 paired fits evaluated and original rank-4096 results reproduced', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'population', 'pilot', 'fit', 'evaluate'])
    parser.add_argument('--rank', type=int, default=8192)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'population':
        population(args.rank)
    elif args.stage == 'evaluate':
        evaluate_fits(args.rank)
    else:
        pilot(args.index, args.rank, fit_only=args.stage == 'fit')


if __name__ == '__main__':
    main()
