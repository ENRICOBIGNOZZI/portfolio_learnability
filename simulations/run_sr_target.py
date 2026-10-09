"""Fresh Rich6D simulation with a specified annualized analytic Sharpe target.

Only the factor means and the covariance I-mu*mu' change. All stock returns and
policies are regenerated/refitted; no historical policy outcomes are rescaled.
The population second-moment operator is invariant because E[FF']=I remains
fixed. Verified spectral/quadrature operators may therefore be reused, with
explicit hashes and the corresponding linear transformation of their means.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from simulations.dgp.balanced import DGPParameters, BalancedFactorDGP
from simulations.estimator.kernel import NystromBasis
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.experiments.population import population_ridge
from simulations.diagnostics.low_memory import population_moments_batched
from simulations.diagnostics.confirmation_pilot import seed
from simulations.provenance import digest, file_hash, json_write, npz_write, source_hashes, utc_now, output_lock

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'simulations/outputs/confirmation_v2'
TIMES = np.array([60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440])
GRID = np.geomspace(1e-12, 1e-2, 96)
METRICS = ('complexity', 'population_loss', 'population_sharpe', 'regret',
           'estimation_norm', 'cross_term', 'population_bias', 'population_complexity')


def target_parameters(annual_sr, periods):
    if annual_sr <= 0 or periods < 1:
        raise ValueError('Positive target and annual period count required.')
    base = DGPParameters(loading_map='rich6d', eta=.35)
    sr = annual_sr / np.sqrt(periods)
    q = sr**2 / (1 + sr**2)
    scale = np.sqrt(q / base.q_star)
    p = replace(base, mu_F=tuple(scale * np.asarray(base.mu_F)))
    np.testing.assert_allclose(p.sr_star * np.sqrt(periods), annual_sr, rtol=1e-13)
    np.testing.assert_allclose(p.factor_covariance + np.outer(p.mu_F, p.mu_F), np.eye(3), atol=1e-15)
    return p, float(scale)


def context(output, annual_sr, periods):
    p, scale = target_parameters(annual_sr, periods)
    sources = source_hashes()
    sources[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    historical = {}
    for name in ('confirmation_reconstruction.json', 'numerical_reconstruction.json'):
        report = json.loads((OLD / 'audit' / name).read_text())
        assert report['passed']
        historical.update(report['input_hashes'])
    rels = ['data/rich6d/population.npz', 'data/rich6d/population.json',
            'audit/rank_paths_low_memory/rich6d/moments.npz',
            'audit/spectrum/eigenvalues.csv', 'audit/spectrum/groups4096_seed0.json']
    rels += [f'audit/quadrature_moments_low_memory/rich6d/groups{groups}_seed{s}.npz'
             for groups in (8192, 32768) for s in range(4)]
    old_hashes = {rel: file_hash(OLD / rel) for rel in rels}
    for rel, value in old_hashes.items():
        assert historical.get(rel) == value, 'Historical operator verification: ' + rel
    spec = {'schema': 'rich6d-annual-sharpe-target/1', 'target_annual_sr': annual_sr,
            'periods_per_year': periods, 'annualization': 'sqrt(periods_per_year) times marginal per-period Sharpe',
            'parameters': p.to_dict(), 'factor_mean_scale': scale, 'T': TIMES.tolist(),
            'replications': 100, 'rank_audit_paths': 50, 'rank': 512, 'audit_rank': 1024,
            'grid': GRID.tolist(), 'method': 'theory_1', 'bootstrap_replications': 1000,
            'source_hashes': sources, 'invariant_operator_source_sha256': old_hashes,
            'seed_design': 'Existing seed families, fresh returns under the NEW parameters; common innovations enable comparisons.',
            'scope': 'Population-evaluated practical-grid simulation; no historical selectors, OOS panels or extended cohort are requested.'}
    run_hash = digest(spec)
    freeze = output / 'protocol.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())['run_hash'] == run_hash, 'New scientific sources/configuration require a fresh output directory.'
    else:
        json_write(freeze, {**spec, 'run_hash': run_hash, 'frozen_before_production_utc': utc_now()})
    low = NystromBasis(rank=512, seed=seed('anchor'))
    high = NystromBasis(rank=1024, seed=seed('anchor'))
    assert np.array_equal(low.anchors, high.anchors[:512])
    m, S = population_moments_batched(p, low, 8192, seed('quadrature'))
    pm, PS = population_moments_batched(p, low, 8192, seed('pilot', 1))
    with np.load(OLD / 'data/rich6d/population.npz') as old:
        np.testing.assert_allclose(S, old['second'], atol=2e-17, rtol=1e-10)
        np.testing.assert_allclose(m, scale * old['mean'], atol=2e-16, rtol=1e-10)
    s_ref = float(np.linalg.eigvalsh(PS)[-1])
    old_pilot = json.loads((OLD / 'audit/cost_accuracy_pilot.json').read_text())
    expected = next(r['s_ref'] for r in old_pilot['records'] if r['loading_map'] == 'rich6d' and r['nu'] == 1.5 and r['rank'] == 512)
    np.testing.assert_allclose(s_ref, expected, atol=1e-16, rtol=1e-10)
    with np.load(OLD / 'audit/rank_paths_low_memory/rich6d/moments.npz') as saved:
        rank_m, rank_S, E = scale * saved['mean'], saved['second'], saved['embedding512']
    np.testing.assert_allclose(high.inverse_root @ E, np.vstack([low.inverse_root, np.zeros((512, 512))]), atol=1e-10)
    rank_low_m, rank_low_S = E.T @ rank_m, E.T @ rank_S @ E
    floors = {512: float(p.q_star - rank_low_m @ np.linalg.solve(rank_low_S, rank_low_m)),
              1024: float(p.q_star - rank_m @ np.linalg.solve(rank_S, rank_m))}
    pop = []
    for T in TIMES:
        lam = np.r_[GRID, s_ref * float(T)**(-.6)]
        coef, C, bias, floor = population_ridge(m, S, lam, p.q_star)
        pop.append((lam, coef, C, bias))
    npz_write(output / 'population.npz', mean=m, second=S, pilot_mean=pm, pilot_second=PS, run_hash=run_hash)
    population = {'parameters': p.to_dict(), 'reference_sharpe': p.sr_star, 'reference_sharpe_annual': annual_sr,
                  'q_reference': p.q_star, 'subspace_regret_floor': float(floor), 's_ref': s_ref,
                  'population_groups': 8192, 'population_seed': seed('quadrature'), 'basis_seed': seed('anchor'),
                  'pilot_seed': seed('pilot', 1), 'periods_per_year': periods, 'run_hash': run_hash,
                  'spectrum_note': "The SAME second-moment spectrum is valid: factor means change, covariance becomes I-mu*mu', and E[FF']=I is invariant. This is verified on recomputed production and pilot moments."}
    json_write(output / 'population.json', population)
    return dict(p=p, scale=scale, low=low, high=high, mean=m, second=S, E=E,
                rank_mean=rank_m, rank_second=rank_S, rank_low_mean=rank_low_m,
                rank_low_second=rank_low_S, floors=floors, population=pop,
                spec=spec, run_hash=run_hash, population_metadata=population)


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
              managed_training=X, managed_rank=basis.rank, theory_coefficients=np.asarray(theory_coefficients),
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
            row = {'environment': 'rich6d', 'T': int(T), 'lambda': GRID[j], 'replications': 100,
                   'projection_floor': c['population_metadata']['subspace_regret_floor']}
            for k in METRICS:
                if k in ('population_bias', 'population_complexity'):
                    row[k] = v[k][0, t, j]
                else:
                    row.update({k + '_' + stat: value[t, j] for stat, value in summary[k].items()})
            curves.append(row)
        row = {'environment': 'rich6d', 'T': int(T), 'method': 'theory_1', 'replications': 100,
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
            rank_rows.append({'environment': 'rich6d', 'T': int(T), 'lambda_index': j, 'lambda': lam,
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
            with np.load(OLD / f'audit/quadrature_moments_low_memory/rich6d/groups{groups}_seed{s}.npz') as z:
                m, S = c['scale'] * z['mean'], z['second']
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


def render(c, output, tables):
    import matplotlib.pyplot as plt
    from simulations.plotting import journal_rich6d_final as figures
    curves, methods, rank, rates, verification = tables
    spectrum = pd.read_csv(OLD / 'audit/spectrum/eigenvalues.csv')
    spectrum = spectrum[(spectrum.operator == 'rich6d') & (spectrum.groups == 4096) & (spectrum.seed_index == 0)].sort_values('index')
    assert len(spectrum) == 800 and spectrum.resolved.all()
    periods = c['spec']['periods_per_year']
    display_scale = np.sqrt(periods)
    d = figures.prepare({'curves': curves, 'theory': methods, 'rank': rank, 'rates': rates,
                         'spectrum': spectrum, 'sr_star': c['p'].sr_star,
                         'sharpe_display_scale': display_scale, 'sharpe_ylim': (0, c['spec']['target_annual_sr'] * 1.10),
                         'sharpe_axis_label': 'Annualized population Sharpe ratio',
                         'sharpe_gap_axis_label': 'Annualized population Sharpe gap'})
    folder = output / 'figures';folder.mkdir(exist_ok=True)
    with plt.rc_context(figures.STYLE):
        for fn in (figures.figure_zero, figures.figure_one, figures.figure_two, figures.figure_three, figures.figure_four):
            fn(d, folder)
    data = folder / 'curve_data';data.mkdir(exist_ok=True)
    theory = d['final_theory'].copy()
    theory['population_sharpe_mean_annualized'] = theory.population_sharpe_mean * display_scale
    theory['sharpe_gap_mean_annualized'] = theory.sharpe_gap_mean * display_scale
    theory.to_csv(data / 'theory_sequences.csv', index=False)
    plotted = d['final_paths'].copy()
    plotted['population_sharpe_mean_annualized'] = plotted.population_sharpe_mean * display_scale
    plotted.to_csv(data / 'full_penalty_paths.csv', index=False)
    d['final_spectrum'].to_csv(data / 'spectrum_filters_cumulative.csv', index=False)
    failed = rank[(rank.lambda_index == 96) & ~rank.certified]['T'].tolist()
    captions = f"""Common design. Fresh Rich6D simulation: 100 replications at each of ten T=60,...,1440; Matern-3/2; only theory_1. The annual target SR*={c['spec']['target_annual_sr']:g} is defined as sqrt({periods}) times the marginal per-period Sharpe ({c['p'].sr_star:.12g}); this is a reporting convention, not the Sharpe of compounded annual returns. Factor means are {list(c['p'].mu_F)} and factor covariance is I-mu*mu'. Training stock returns and all fitted policies were newly generated/refitted under these parameters. No historical policy outcomes were scaled to create new results.
The theory penalty uses lambda_T={c['population_metadata']['s_ref']:.15g} T^(-0.6). Its independent population pilot was recomputed: this is a population-informed theoretical benchmark, not a fully feasible historical selector. The second-moment operator is unchanged analytically because E[FF']=I, and recomputed production/pilot moments verify this identity. Therefore the verified 4096-group, seed-4101162095 Rich6D spectrum is explicitly reused; its 800 resolved eigenvalues define the common partial population trace C_800. It is not the complete infinite-dimensional trace or the production rank-512 trace.

Figure 0. The economic spectrum. The three panels show the same 800 verified population eigenvalues, the filters mu/(mu+lambda_T) for T=60,240,720,1440, and their cumulative sums. They describe strength and shrinkage activation of directions, not statistical identification or a fitted power-law spectrum. Spectrum and filters are unchanged under this factor-mean recalibration by the second-moment invariance above.

Figure 1. Annualized Sharpe learnability. Means and pointwise 95% Monte Carlo intervals come from the 100 NEW per-period population evaluations, all multiplied by sqrt({periods}); the horizontal analytic optimum is {c['spec']['target_annual_sr']:g}. The gap benchmark T^(-0.6) is anchored at T=60, not fitted. It illustrates observed recovery, not an exact asymptotic rate equality. The full-grid slope is {rates[rates.window=='full'].OLS_slope.iloc[0]:.6f}, with 1000-whole-path-bootstrap interval [{rates[rates.window=='full'].bootstrap_low.iloc[0]:.6f}, {rates[rates.window=='full'].bootstrap_high.iloc[0]:.6f}]. Numerical rank and quadrature checks are reported separately; bands condition on fixed numerical approximations.

Figure 2. Effective complexity over time. Recomputed pilot penalties, the common resolved C_800, and C_800/T are shown with prescribed theory guides anchored at T=60. These population-spectral quantities are unchanged because the second-moment spectrum is invariant. They do not establish an infinite-spectrum exponent or count statistically identified directions.

Figure 3. Annualized Sharpe versus population complexity. All 96 diagnostic penalties and all four sample sizes use fresh fitted policies; diamonds use the actual theory penalties. Both panels retain the full paths. The horizontal coordinate is C_800, not empirical or rank-512 complexity. The rank audit was rerun on 50 matched NEW paths; dashed segments touch a point failing the predeclared 5% regret-based rank test. This is not a test-performance selector or a claim that the theory policy is optimal.

Figure 4. Exact regret decomposition at T=1440. New population bias, squared estimation error, signed cross term and total regret are plotted without annualizing loss. Bias includes the new approximation floor {c['population_metadata']['subspace_regret_floor']:.12g}. Squared estimation error is not the whole variance contribution. Diamonds show the theory policy; shading and dashed segments retain unresolved rank-sensitive regions. The signed identity is checked in every new replication and penalty cell.

Numerical resolution. Theory-rule T values failing the NEW rank test: {failed}. Four independent quadrature seeds compare 8192 and 32768 groups on the same new fixed policies, using exactly transformed invariant operators; full results are in quadrature_resolution.csv. No old certification flags or Monte Carlo outcomes are reused.

Numerical provenance. curves.csv and methods.csv aggregate the 100 checkpoints in replications/; each checkpoint includes regenerated training managed returns, stock-return hash, seeds, fitted theory coefficients and every diagnostic path metric. verification.json reconstructs theory normal equations, population Sharpe and signed decomposition. rank_resolution.csv uses the first 50 new rank-1024 paths with the identical nested rank-512 histories; rates.csv bootstraps whole new paths. spectrum_filters_cumulative.csv and theory_sequences.csv contain the exact displayed spectral series. protocol.json binds parameters, seeds, source hashes and reused invariant-operator hashes before production. The manuscript is unchanged.
"""
    (folder / 'captions_and_provenance.txt').write_text(captions, encoding='utf-8')
    json_write(folder / 'provenance.json', {'run_hash': c['run_hash'], 'annualization_factor': display_scale,
                'target_annual_sr': c['spec']['target_annual_sr'], 'periods_per_year': periods,
                'parameters': c['p'].to_dict(), 'policy_outcomes': '100 freshly simulated and refitted paths',
                'unchanged_operator_reuse': c['population_metadata']['spectrum_note'],
                'verification': verification, 'figure_source_sha256': file_hash(figures.__file__),
                'output_sha256': {str(f.relative_to(folder)): file_hash(f) for f in folder.rglob('*') if f.is_file() and f.name != 'provenance.json'}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annual-sr', type=float, default=3.)
    parser.add_argument('--periods-per-year', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output):
        c = context(args.output, args.annual_sr, args.periods_per_year)
        print('Calibration:', json.dumps(c['population_metadata']), flush=True)
        if args.prepare_only:
            return
        start = time.perf_counter()
        if not args.render_only:
            for index in range(100):
                run_replication(index, c, args.output)
                status = {'status': 'simulation_running', 'completed_replications': index + 1,
                          'total_replications': 100, 'elapsed_seconds': time.perf_counter() - start,
                          'run_hash': c['run_hash'], 'updated_utc': utc_now()}
                json_write(args.output / 'status.json', status)
                print(f'Replication {index+1}/100; {status["elapsed_seconds"]:.1f}s elapsed', flush=True)
        tables = summarize(c, args.output)
        render(c, args.output, tables)
        json_write(args.output / 'status.json', {'status': 'complete', 'replications': 100,
                   'run_hash': c['run_hash'], 'completed_utc': utc_now(), 'elapsed_seconds': time.perf_counter() - start})
        print('COMPLETE:', args.output, flush=True)


if __name__ == '__main__':
    main()
