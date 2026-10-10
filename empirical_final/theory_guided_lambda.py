"""One historical anchor, past-only Gaussian spectra, unchanged ridge estimator.

Reproduce with OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m empirical_final.theory_guided_lambda. No licensed rows are exported.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits, threadpool_info

from data_pipeline import digest
from kernels import FeatureBank, array_hash
from portfolio import annual_splits, complexity_grid, dual_path, fit_windows, managed_matrix, sharpe
from empirical_final.core import accounting_step, performance
from empirical_final.run import ROOT, INPUTS, bind, clean_manifest, load_managed, public, read_json
from empirical_final.three_experiments import block_counts, boot_metrics

DEFAULT_OUTPUT = ROOT / 'outputs/theory_guided_lambda_20261010'
LEGACY = ROOT / 'outputs/empirical_three_experiments_20261009'
SEED = 20261010
REPLICATES = 5000
POLICIES = ('cv', 'spectral', 'spectral_grid')
SCENARIOS = ('gross', 'target_trade25', 'borrow30', 'target_trade25_borrow30', 'trade25_borrow30')
SOURCES = ('portfolio.py', 'kernels.py', 'data_pipeline.py', 'empirical_final/run.py',
           'empirical_final/core.py', 'empirical_final/three_experiments.py')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def save(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, float_format='%.17g')
    return frame


def spectrum(gram):
    """Match dual_path PSD tolerance; retain small positive and exact zero values."""
    gram = np.asarray(gram, float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not len(gram) or not np.isfinite(gram).all():
        raise ValueError('Finite nonempty square Gram required')
    values = np.linalg.eigvalsh((gram+gram.T)/2)
    tol = max(float(np.abs(values).max()), 1e-30)*1e-10
    if values.min() < -tol:
        raise ValueError('Meaningfully negative Gram eigenvalue')
    return np.maximum(values[::-1], 0)/len(gram), dict(
        minimum_unclipped_gram_eigenvalue=float(values.min()), psd_tolerance=tol,
        clipped_eigenvalues=int((values < 0).sum()),
        active_rank=int((values > max(float(values.max()), 1e-30)*1e-12).sum()))


def check_mu(mu):
    mu = np.asarray(mu, float)
    if mu.ndim != 1 or not len(mu) or not np.isfinite(mu).all() or np.any(mu < 0):
        raise ValueError('Finite nonnegative spectrum required')
    return mu


def complexity(mu, penalty):
    mu = check_mu(mu)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Positive finite penalty required')
    return float(np.sum(mu/(mu+penalty)))


def sensitivity(mu, penalty):
    mu = check_mu(mu)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Positive finite penalty required')
    return float(np.sum(mu/(mu+penalty)**2))


def calibrate_rho(mu, n_months, anchor, bounds):
    """Interior anchor only: a boundary identifies an interval, not a constant."""
    lower, upper = bounds
    if not np.isfinite([n_months, anchor, lower, upper]).all() or n_months <= 0 or not 0 < lower < anchor < upper:
        raise ValueError('Boundary anchor does not point-identify rho; interior anchor required')
    s0 = sensitivity(mu, anchor)
    if s0 <= 0 or not np.isfinite(s0):
        raise ValueError('Degenerate spectral sensitivity')
    return float(n_months/s0)


def choose_lambda(mu, n_months, rho, lower, upper):
    mu = check_mu(mu)
    if not np.isfinite([n_months, rho, lower, upper]).all() or n_months <= 0 or rho <= 0 or not 0 < lower < upper:
        raise ValueError('Positive finite T, rho and ordered bounds required')
    gradient = lambda lam: 1-rho/n_months*sensitivity(mu, lam)
    gl, gu = gradient(lower), gradient(upper)
    if gl >= 0:
        lam, boundary, calls = lower, 'lower', 0
    elif gu <= 0:
        lam, boundary, calls = upper, 'upper', 0
    else:
        # Root in log(lambda) keeps relative accuracy for raw penalties near 1e-7.
        root, info = brentq(lambda z: gradient(np.exp(z)), np.log(lower), np.log(upper),
                           xtol=1e-13, rtol=1e-14, full_output=True)
        if not info.converged:
            raise ValueError('Spectral scalar root did not converge')
        lam, boundary, calls = float(np.exp(root)), 'interior', info.function_calls
    grad = float(gradient(lam))
    kkt = abs(grad) if boundary == 'interior' else max(0., -grad if boundary == 'lower' else grad)
    return dict(penalty=float(lam), C=complexity(mu, lam), boundary=boundary,
                derivative_residual=grad, kkt_residual=float(kkt),
                derivative_lower=float(gl), derivative_upper=float(gu), root_calls=calls,
                objective=float(lam+rho/n_months*complexity(mu, lam)))


def anchor_from_history(g, dates, grid):
    """Calibration function cannot read 1978 or later managed observations."""
    year, train, validation, test = next(annual_splits(dates))
    h = np.r_[train, validation]
    past = np.asarray(g[h], float)
    gram = past@past.T
    alpha, _, _ = dual_path(gram[np.ix_(train, train)], grid)
    losses = np.mean((1-gram[np.ix_(validation, train)]@alpha)**2, axis=0)
    choice = int(np.argmin(losses))
    lam0 = float(grid[choice])
    mu0, psd = spectrum(gram)
    rho = calibrate_rho(mu0, len(h), lam0, (float(min(grid)), float(max(grid))))
    root = choose_lambda(mu0, len(h), rho, float(min(grid)), float(max(grid)))
    np.testing.assert_allclose(root['penalty'], lam0, rtol=1e-11, atol=0)
    return dict(year=year, lam0=lam0, T0=len(h), rho0=rho, S0=sensitivity(mu0, lam0),
        cv_choice=choice, C0=complexity(mu0, lam0), relative_anchor_error=abs(root['penalty']/lam0-1),
        train_first=str(dates[train[0]].date()), train_last=str(dates[train[-1]].date()),
        validation_first=str(dates[validation[0]].date()), validation_last=str(dates[validation[-1]].date()),
        first_refit_formation=str(dates[h[0]].date()), last_refit_formation=str(dates[h[-1]].date()),
        first_refit_payoff=str((dates[h[0]]+pd.offsets.MonthEnd(1)).date()),
        last_known_payoff=str((dates[h[-1]]+pd.offsets.MonthEnd(1)).date()),
        decision_date=str(dates[test[0]].date()), first_test_payoff=str((dates[test[0]]+pd.offsets.MonthEnd(1)).date()),
        calibration_includes_test_payoffs=False, interior=True, **psd)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def freeze(out, replicates):
    if (out/'audit/config.json').exists():
        raise ValueError('Existing frozen run: select --phase costs or report, or a fresh --output')
    for part in ('audit', 'tables', 'figures', 'publication'):
        (out/part).mkdir(parents=True, exist_ok=True)
    clean = clean_manifest()
    folder, manifest = public('gaussian')
    g, dates = load_managed('gaussian', clean)
    source = read_json(ROOT/'results/final/public/gaussian/seed_0/source.json')
    grid = np.asarray(manifest['lambda_grid'])
    if len(grid) != 120 or not np.all(np.diff(grid) > 0) or min(grid) <= 0:
        raise ValueError('Expected original positive 120-candidate grid')
    np.testing.assert_allclose(complexity_grid(np.asarray(g[:120])), grid, rtol=2e-7, atol=0)
    kappa = float(read_json(ROOT/'results/final/public/linear/seed_0/calibration.json')['scale'])
    # Cache construction batches 64 stocks; the original managed_matrix uses 256.
    # These different reduction orders preserve input identity, not bitwise arrays.
    compatibility = dict(clean_manifest_and_feature_bank_match=True,
        cache_array_sha256=array_hash(g), original_array_sha256=source['managed_matrix_sha256'],
        managed_array_bitwise_match=array_hash(g)==source['managed_matrix_sha256'],
        explanation='The bandwidth cache uses 64-stock reductions; the original managed_matrix uses 256. '
                    'Different float64 accumulation orders can change final bits. Data/feature identity is verified; '
                    'acceptance additionally requires every original selected payoff to agree at rtol=1e-7, atol=1e-9.')
    first_file = clean['files'][0]
    data = pd.read_parquet(bind(ROOT/'data/clean'/first_file['name'], first_file['sha256']))
    data = data[data.eom.eq(dates[0])].sort_values('id')
    b = source['feature_bank']
    bank = FeatureBank('gaussian', b['input_dimension'], 10000, b['ell'], b['seed'])
    original_order = managed_matrix([dict(x=data[clean['characteristics']].to_numpy(), r=data.r.to_numpy())],bank)[0]
    np.testing.assert_allclose(original_order,g[0],rtol=1e-10,atol=1e-15)
    compatibility['first_month_original_reduction_maximum_error'] = float(np.max(np.abs(original_order-g[0])))
    write_json(out/'audit/baseline_compatibility.json', compatibility)
    print('Baseline data and bank identities verified; array-bit reconciliation recorded before comparison',flush=True)
    # Explicitly compare both the original fixed-bandwidth and three-experiment snapshot.
    ref = pd.read_csv(bind(folder/'monthly.csv'), parse_dates=['formation_date', 'return_date'])
    legacy = pd.read_csv(bind(LEGACY/'tables/e1_selected_monthly.csv'), parse_dates=['return_date'])
    annual_ref = pd.read_csv(bind(LEGACY/'tables/e1_annual_paths.csv'))
    bind(ROOT/'outputs/tables/formation_timing_audit.json')
    for name in SOURCES:
        bind(ROOT/name)
    config = dict(experiment='One-Constant Theory-Guided Spectral Shrinkage',
        starting_commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
        starting_unrelated_status=git('status', '--short'), feature_bank=source['feature_bank'],
        managed_matrix_sha256=array_hash(g), kappa=kappa, spectrum_units='raw uncentered managed payoffs; Gram/T',
        managed_array_bitwise_match=compatibility['managed_array_bitwise_match'],
        lambda_grid=grid.tolist(), lambda_min=float(grid.min()), lambda_max=float(grid.max()),
        lambda_grid_source='results/final/public/gaussian/seed_0/p_10000/manifest.json',
        grid_initial_formation=['1963-01-31', '1972-12-31'], rho_calibration_decision=1978,
        decisions=[1978, 2024], primary_decisions=[1979, 2024], seed=SEED, bootstrap_replicates=replicates,
        bootstrap_blocks=[12, 6, 24], inference='paired circular blocks conditional on fitted paths',
        baseline_tolerance=dict(rtol=1e-7, atol=1e-9), psd_relative_tolerance=1e-10,
        active_rank_relative_threshold=1e-12, root='Brent in log(lambda), fixed original bounds',
        costs='Actual reconstructed signed weights; target-change and exact self-financing conventions',
        initial_cost_state='Cash at first February 1978 payoff; primary window carries existing state',
        terminal_cost_state='Mark terminal holdings; no forced terminal liquidation',
        source_hashes={n:digest(ROOT/n) for n in SOURCES}, python=platform.python_version(),
        module_sha256=digest(Path(__file__)), threadpools=threadpool_info())
    write_json(out/'audit/config.json', config)
    write_json(out/'audit/input_hashes.json', dict(INPUTS))
    return clean, g, dates, grid, ref, legacy, annual_ref, config


def selection(out, private, replicates):
    started = time.monotonic()
    clean, g, dates, grid, ref, legacy, old_annual, config = freeze(out, replicates)
    anchor = anchor_from_history(g, dates, grid)
    write_json(out/'audit/anchor.json', anchor)
    rho = anchor['rho0']  # Assigned once, before reading any OOS payoff.
    gram = np.asarray(g)@np.asarray(g).T
    rf = pd.read_csv(bind(ROOT/'data/risk_free.csv'), parse_dates=['return_date']).set_index('return_date').rf
    annual, monthly, betas, eigenvalues = [], [], [], []
    maxima = dict(baseline_payoff_error=0., legacy_payoff_error=0., complexity_error=0.,
                  root_kkt_residual=0., dual_relative_residual=0., direct_payoff_error=0., primal_payoff_error=0.)
    for result in fit_windows(g, dates, grid):
        year, j = result['year'], result['choice']
        h, o = np.r_[result['train'], result['validation']], result['test']
        t = len(h)
        mu, psd = spectrum(gram[np.ix_(h, h)])
        eigenvalues.extend(dict(year=year, T=t, rank=k+1, mu=float(value)) for k,value in enumerate(mu))
        np.testing.assert_allclose(mu, result['mu'], rtol=1e-6, atol=max(mu)*1e-12)
        rule = choose_lambda(mu, t, rho, grid.min(), grid.max())
        objectives = grid+rho/t*np.sum(mu[:, None]/(mu[:, None]+grid[None, :]), axis=0)
        grid_j = int(np.argmin(objectives))
        penalties = np.array([grid[j], rule['penalty'], grid[grid_j]])
        alpha, _, cs = dual_path(gram[np.ix_(h, h)], penalties)
        beta = np.asarray(g[h]).T@alpha
        raw = gram[np.ix_(o, h)]@alpha
        rdates = dates[o]+pd.offsets.MonthEnd(1)
        old = ref.set_index('return_date').loc[rdates]
        old2 = legacy.set_index('return_date').loc[rdates]
        selected = old_annual[old_annual.decision_year.eq(year)&old_annual.selected].iloc[0]
        if j != int(selected.candidate) or not np.isclose(grid[j], selected.penalty, rtol=1e-12, atol=0):
            raise ValueError(f'Canonical CV selection differs in {year}')
        np.testing.assert_allclose(raw[:, 0], result['test_returns'][:, j], rtol=1e-7, atol=1e-9)
        for values, key in [(old.raw_excess_return, 'baseline_payoff_error'), (old2.raw_return, 'legacy_payoff_error')]:
            np.testing.assert_allclose(raw[:, 0], values, rtol=1e-7, atol=1e-9)
            maxima[key] = max(maxima[key], float(np.max(np.abs(raw[:, 0]-values))))
        np.testing.assert_array_equal(dates[o], old.formation_date)
        cerror = abs(rule['C']-cs[1])
        if cerror > 1e-7 or rule['kkt_residual'] > 1e-10:
            raise ValueError('Complexity or root numerical check failed')
        residual = np.linalg.norm(gram[np.ix_(h, h)]@alpha+t*alpha*penalties-1)/np.sqrt(t*len(penalties))
        direct_error = 0.
        if year in (1978, 2000, 2024):
            direct = np.linalg.solve(gram[np.ix_(h, h)]+t*penalties[1]*np.eye(t), np.ones(t))
            direct_error = float(np.max(np.abs(gram[np.ix_(o, h)]@direct-raw[:, 1])))
            np.testing.assert_allclose(gram[np.ix_(o, h)]@direct, raw[:, 1], rtol=1e-7, atol=1e-9)
        primal_error = float(np.max(np.abs(np.asarray(g[o])@beta-raw)))
        if residual > 1e-7 or primal_error > 1e-8:
            raise ValueError('Ridge normal equation or payoff aggregation failed')
        maxima['complexity_error'] = max(maxima['complexity_error'], cerror)
        maxima['root_kkt_residual'] = max(maxima['root_kkt_residual'], rule['kkt_residual'])
        maxima['dual_relative_residual'] = max(maxima['dual_relative_residual'], float(residual))
        maxima['direct_payoff_error'] = max(maxima['direct_payoff_error'], direct_error)
        maxima['primal_payoff_error'] = max(maxima['primal_payoff_error'], primal_error)
        row = dict(year=year, T=t, lambda_spectral=rule['penalty'], lambda_cv=float(grid[j]),
            lambda_spectral_grid=float(grid[grid_j]), C_spectral=cs[1], C_cv=cs[0], C_spectral_grid=cs[2],
            C_spectral_over_T=cs[1]/t, C_cv_over_T=cs[0]/t,
            rho0=rho, cv_choice=j, spectral_grid_choice=grid_j, boundary=rule['boundary'],
            cv_boundary=j in (0, len(grid)-1), derivative_residual=rule['derivative_residual'],
            kkt_residual=rule['kkt_residual'], derivative_lower=rule['derivative_lower'],
            derivative_upper=rule['derivative_upper'], proxy_objective=rule['objective'],
            proxy_grid_objective=objectives[grid_j], dual_relative_residual=residual,
            primal_payoff_error=primal_error, direct_payoff_error=direct_error,
            complexity_error=cerror, first_formation=dates[h[0]], last_formation=dates[h[-1]],
            first_observation_date=dates[h[0]]+pd.offsets.MonthEnd(1),
            last_observation_date=dates[h[-1]]+pd.offsets.MonthEnd(1), decision_date=dates[o[0]],
            first_oos_return=rdates[0], last_oos_return=rdates[-1],
            train_first=dates[result['train'][0]], train_last=dates[result['train'][-1]],
            validation_first=dates[result['validation'][0]], validation_last=dates[result['validation'][-1]], **psd)
        for k, policy in enumerate(POLICIES):
            excess = config['kappa']*raw[:, k]
            row.update({f'{policy}_{key}':value for key,value in performance(excess, excess+rf.loc[rdates].to_numpy()).items()})
        annual.append(row)
        for i, date in enumerate(rdates):
            month = dict(decision_year=year, formation_date=dates[o[i]], return_date=date,
                         rf=float(rf.loc[date]), kappa=config['kappa'])
            for k, policy in enumerate(POLICIES):
                month[f'{policy}_raw'] = raw[i, k]
                month[f'{policy}_excess'] = config['kappa']*raw[i, k]
                month[f'{policy}_total'] = month[f'{policy}_excess']+month['rf']
            month['spectral_minus_cv_excess'] = month['spectral_excess']-month['cv_excess']
            monthly.append(month)
        betas.append(beta)
        save(out/'tables/annual_selection.csv', annual)
        save(out/'tables/monthly_oos.csv', monthly)
        print(f'Selection {year}: T={t}, C(CV/spec)={cs[0]:.3f}/{cs[1]:.3f}, lambda={rule["penalty"]:.9g}', flush=True)
    if len(annual) != 47 or len(monthly) != 564:
        raise ValueError('Expected 47 decisions / 564 months')
    private.mkdir(parents=True, exist_ok=True)
    coefficient_path = private/'coefficients.npz'
    np.savez(coefficient_path, beta=np.stack(betas))
    save(out/'tables/empirical_eigenvalues.csv', eigenvalues)
    write_json(out/'audit/selection_verification.json', dict(passed=True, decisions=47, months=564,
        baseline_reproduced=True, clean_manifest_and_feature_bank_match=True,
        managed_array_bitwise_match=config['managed_array_bitwise_match'], coefficients_sha256=digest(coefficient_path),
        elapsed_seconds=time.monotonic()-started, **maxima))
    write_json(out/'audit/input_hashes.json', dict(INPUTS))


def verify_inputs(out):
    for name, expected in json.loads((out/'audit/input_hashes.json').read_text()).items():
        bind(ROOT/name, expected)
    config = json.loads((out/'audit/config.json').read_text())
    for name, expected in config['source_hashes'].items():
        bind(ROOT/name, expected)
    return config


def account_targets(target, returns, rf, state, previous):
    """Public aggregates only; reuse exact repository self-financing accounting."""
    ids = target.index.union(state.index)
    step = accounting_step(target.reindex(ids, fill_value=0).to_numpy(), returns.reindex(ids).to_numpy(),
        rf, state.reindex(ids, fill_value=0).to_numpy(), .0025, .003)
    drift = pd.Series(step.pop('next_drift'), index=ids)
    drift = drift[target.reindex(ids, fill_value=0).to_numpy() != 0]
    target_turnover = float(target.subtract(previous, fill_value=0).abs().sum())
    gross = float(target@returns.reindex(target.index))
    short = float(np.maximum(-target, 0).sum())
    common = dict(gross_exposure=float(target.abs().sum()), net_exposure=float(target.sum()),
                  short_notional=short, target_turnover=target_turnover, stocks=len(target))
    rows = []
    for scenario in SCENARIOS:
        if scenario == 'trade25_borrow30':
            rows.append(dict(scenario=scenario, **(common | step)))
        else:
            trade = .0025*target_turnover if scenario.startswith('target_trade25') else 0.
            borrow = .003/12*short if scenario in ('borrow30', 'target_trade25_borrow30') else 0.
            excess = gross-trade-borrow
            rows.append(dict(scenario=scenario, **common, total_return=rf+excess, excess_return=excess,
                gross_excess_return=gross, turnover=target_turnover, trading_fee=trade, borrowing_fee=borrow,
                trading_return_drag=trade, accounting_residual=0.))
    return rows, drift


def costs(out, private):
    started = time.monotonic()
    config = verify_inputs(out)
    clean = clean_manifest()
    cp = private/'coefficients.npz'
    expected = json.loads((out/'audit/selection_verification.json').read_text())['coefficients_sha256']
    betas = np.load(bind(cp, expected), allow_pickle=False)['beta']
    bmeta = config['feature_bank']
    bank = FeatureBank('gaussian', bmeta['input_dimension'], 10000, bmeta['ell'], bmeta['seed'])
    if bank.metadata() != bmeta:
        raise ValueError('Gaussian feature bank differs')
    ref = pd.read_csv(out/'tables/monthly_oos.csv', parse_dates=['formation_date']).set_index('formation_date')
    state = {p:pd.Series(dtype=float) for p in POLICIES[:2]}
    previous = {p:pd.Series(dtype=float) for p in POLICIES[:2]}
    rows, checks = [], []
    for yi, year in enumerate(range(1978, 2025)):
        item = next(i for i in clean['files'] if i['name'] == f'jkp_{year}.parquet')
        path = bind(ROOT/'data/clean'/item['name'], item['sha256'])
        frame = pd.read_parquet(path, columns=['id', 'eom', 'return_date', 'r', *clean['characteristics']])
        old_weights = pd.read_parquet(bind(ROOT/f'results/final/private/gaussian/seed_0/p_10000/weights_{year}.parquet'))
        if frame.duplicated(['id', 'eom']).any() or old_weights.duplicated(['id', 'formation_date']).any():
            raise ValueError('Duplicate stock observations')
        max_payoff, max_weight, max_account = 0., 0., 0.
        for date, data in frame.groupby('eom', sort=True):
            data = data.sort_values('id').set_index('id')
            retdate = date+pd.offsets.MonthEnd(1)
            if not data.return_date.eq(retdate).all():
                raise ValueError('Stock payoff timing differs')
            x = data[clean['characteristics']].to_numpy(float)
            if not np.isfinite(x).all() or not np.isfinite(data.r).all() or np.max(np.abs(x)) > .5+1e-12:
                raise ValueError('Invalid canonical characteristics or returns')
            raw_weights = bank.scores(x, betas[yi, :, :2])/len(data)
            old = old_weights[old_weights.formation_date.eq(date)].set_index('id').reindex(data.index)
            np.testing.assert_allclose(raw_weights[:, 0], old.raw_weight, rtol=2e-6, atol=1e-9)
            max_weight = max(max_weight, float(np.max(np.abs(raw_weights[:, 0]-old.raw_weight))))
            for k, policy in enumerate(POLICIES[:2]):
                target = pd.Series(config['kappa']*raw_weights[:, k], index=data.index)
                raw = float(raw_weights[:, k]@data.r)
                err = abs(raw-ref.loc[date, f'{policy}_raw'])
                max_payoff = max(max_payoff, err)
                np.testing.assert_allclose(raw, ref.loc[date, f'{policy}_raw'], rtol=1e-7, atol=1e-9)
                accounts, state[policy] = account_targets(target, data.r, float(ref.loc[date, 'rf']), state[policy], previous[policy])
                for account in accounts:
                    max_account = max(max_account, account['accounting_residual'])
                    rows.append(dict(policy=policy, decision_year=year, formation_date=date,
                                     return_date=retdate, rf=float(ref.loc[date, 'rf']), **account))
                previous[policy] = target
        checks.append(dict(year=year, maximum_raw_payoff_error=max_payoff,
                           maximum_cv_raw_weight_error=max_weight, maximum_accounting_residual=max_account))
        save(out/'tables/costs.csv', rows)
        save(out/'tables/stock_reconstruction_checks.csv', checks)
        write_json(out/'audit/cost_checkpoint.json', dict(completed_year=year, elapsed_seconds=time.monotonic()-started))
        print(f'Stock accounting {year}/2024: payoff error {max_payoff:.2g}, CV weight error {max_weight:.2g}', flush=True)
    # Reconcile the unchanged exact drift-cost CV account with its existing run.
    frame = pd.DataFrame(rows)
    old_path = ROOT/'outputs/final_empirical_20261008/tables/monthly_accounting.csv'
    old = pd.read_csv(bind(old_path), parse_dates=['return_date'])
    old = old[old.kernel.eq('gaussian')&old.scenario.eq('trade25_borrow30')].sort_values('return_date')
    actual = frame[frame.policy.eq('cv')&frame.scenario.eq('trade25_borrow30')].sort_values('return_date')
    np.testing.assert_array_equal(actual.return_date, old.return_date)
    np.testing.assert_allclose(actual[['excess_return', 'turnover', 'trading_fee', 'borrowing_fee']],
                               old[['excess_return', 'turnover', 'trading_fee', 'borrowing_fee']], rtol=2e-6, atol=1e-9)
    write_json(out/'audit/cost_verification.json', dict(passed=True, months_per_policy_scenario=564,
        policies=2, scenarios=SCENARIOS, actual_signed_weights=True, cv_stock_weights_reproduced=True,
        original_drift_cost_account_reproduced=True, elapsed_seconds=time.monotonic()-started,
        maximum_raw_payoff_error=max(c['maximum_raw_payoff_error'] for c in checks),
        maximum_cv_raw_weight_error=max(c['maximum_cv_raw_weight_error'] for c in checks),
        maximum_accounting_residual=max(c['maximum_accounting_residual'] for c in checks)))
    write_json(out/'audit/input_hashes.json', dict(INPUTS))


def summarize(out):
    config = verify_inputs(out)
    monthly = pd.read_csv(out/'tables/monthly_oos.csv', parse_dates=['return_date'])
    annual = pd.read_csv(out/'tables/annual_selection.csv')
    accounts_path = out/'tables/costs.csv'
    cost_verified = (out/'audit/cost_verification.json').exists()
    accounts = pd.read_csv(accounts_path, parse_dates=['return_date']) if cost_verified else None
    summaries, bootstraps = [], []
    periods = [('1978-2024',1978,2024), ('1979-2024',1979,2024), ('1979-1989',1979,1989),
               ('1990-1999',1990,1999), ('2000-2009',2000,2009), ('2010-2019',2010,2019), ('2020-2024',2020,2024)]
    for label, lo, hi in periods:
        m = monthly[monthly.decision_year.between(lo, hi)].sort_values('return_date')
        a = annual[annual.year.between(lo, hi)]
        scenario_returns = {}
        for policy in POLICIES:
            summaries.append(dict(period=label, policy=policy, scenario='gross', months=len(m),
                first_return=m.return_date.min(), last_return=m.return_date.max(),
                mean_C=float(a['C_'+('cv' if policy == 'cv' else policy)].mean()),
                **performance(m[f'{policy}_excess'], m[f'{policy}_total'])))
        scenario_returns['gross'] = m[['cv_excess','spectral_excess']].to_numpy()
        if accounts is not None:
            sub = accounts[accounts.decision_year.between(lo, hi)]
            for (policy, scenario), f in sub.groupby(['policy', 'scenario'], sort=False):
                f = f.sort_values('return_date')
                if scenario != 'gross':
                    summaries.append(dict(period=label, policy=policy, scenario=scenario, months=len(f),
                        first_return=f.return_date.min(), last_return=f.return_date.max(),
                        mean_C=float(a['C_'+policy].mean()), **performance(f.excess_return, f.total_return)))
                row = summaries[-1] if scenario != 'gross' else next(r for r in summaries
                    if r['period']==label and r['policy']==policy and r['scenario']=='gross')
                row.update(
                    mean_gross_exposure=float(f.gross_exposure.mean()), mean_net_exposure=float(f.net_exposure.mean()),
                    mean_short_notional=float(f.short_notional.mean()), annual_turnover=float(12*f.turnover.mean()),
                    annual_target_turnover=float(12*f.target_turnover.mean()), annual_trading_fees=float(12*f.trading_fee.mean()),
                    annual_borrowing_fees=float(12*f.borrowing_fee.mean()))
            for scenario in SCENARIOS[1:]:
                wide = sub[sub.scenario.eq(scenario)].pivot(index='return_date',columns='policy',values='excess_return').sort_index()
                if not wide.index.equals(pd.DatetimeIndex(m.return_date)) or wide.isna().any().any():
                    raise ValueError('Cost comparison month alignment failed')
                scenario_returns[scenario] = wide[['cv','spectral']].to_numpy()
        for block in (12,6,24):
            counts = block_counts(len(m), block, config['seed'], config['bootstrap_replicates'])
            for scenario, r in scenario_returns.items():
                b = boot_metrics(r, counts)
                for metric in ('sharpe', 'mean', 'volatility'):
                    observed = sharpe(r) if metric == 'sharpe' else 12*r.mean(axis=0) if metric == 'mean' else np.sqrt(12)*r.std(axis=0, ddof=1)
                    delta = b[metric][:,1]-b[metric][:,0]
                    low, high = np.quantile(delta, [.025,.975])
                    bootstraps.append(dict(period=label, scenario=scenario, metric=metric,
                        spectral_minus_cv=float(observed[1]-observed[0]), ci_low=low, ci_high=high,
                        bootstrap_se=float(delta.std(ddof=1)), fraction_positive=float(np.mean(delta>0)),
                        block_months=block, replicates=config['bootstrap_replicates'], seed=config['seed'],
                        months=len(m), inference='conditional on fitted paths; percentile interval'))
            # Grid robustness uses the same paired draws, not a favorable new inference choice.
            grid_r = m[['cv_excess','spectral_grid_excess']].to_numpy()
            delta = np.diff(boot_metrics(grid_r, counts)['sharpe'],axis=1)[:,0]
            low, high = np.quantile(delta, [.025,.975])
            bootstraps.append(dict(period=label, scenario='spectral_grid_gross', metric='sharpe',
                spectral_minus_cv=float(np.diff(sharpe(grid_r))[0]), ci_low=low, ci_high=high,
                bootstrap_se=float(delta.std(ddof=1)), fraction_positive=float(np.mean(delta>0)),
                block_months=block, replicates=config['bootstrap_replicates'], seed=config['seed'],
                months=len(m), inference='conditional on fitted paths; percentile interval'))
    summary = save(out/'tables/summary.csv', summaries)
    paired = save(out/'tables/paired_bootstrap.csv', bootstraps)
    make_figures(out, monthly, annual, accounts)
    make_report(out, config, annual, summary, paired, cost_verified)
    verification = dict(passed=True, real_data=True, baseline_reproduced=True, years=47, months=564,
        primary_months=552, primary_first_return='1979-02-28', primary_last_return='2025-01-31',
        rho_frozen=bool(annual.rho0.nunique()==1), continuous_boundary_counts=annual.boundary.value_counts().to_dict(),
        maximum_kkt_residual=float(annual.kkt_residual.max()), net_performance_reconstructed=cost_verified,
        all_numeric_annual_finite=bool(np.isfinite(annual.select_dtypes('number').to_numpy()).all()),
        all_numeric_monthly_finite=bool(np.isfinite(monthly.select_dtypes('number').to_numpy()).all()),
        main_manuscript_modified=False, licensed_rows_exported=False,
        sources={name:digest(ROOT/name) for name in (*SOURCES,'empirical_final/theory_guided_lambda.py')})
    if not all(verification[k] for k in ('rho_frozen','all_numeric_annual_finite','all_numeric_monthly_finite')):
        raise ValueError('Final finite/frozen checks failed')
    write_json(out/'audit/verification.json', verification)
    write_json(out/'audit/output_hashes.json', {str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
                                             if p.is_file() and p.name!='output_hashes.json'})


def make_figures(out, monthly, annual, accounts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42})
    colors = {'cv':'#284f75','spectral':'#b16a30'}
    labels = {'cv':'Annual CV','spectral':'One-anchor spectral'}
    def finish(fig, name):
        fig.savefig(out/'figures'/f'{name}.pdf', bbox_inches='tight')
        fig.savefig(out/'figures'/f'{name}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(8,6.5),sharex=True,layout='constrained')
    for p in colors:
        axes[0].plot(annual.year, annual[f'C_{p}'],label=labels[p],color=colors[p])
        axes[1].semilogy(annual.year,annual[f'lambda_{p}'],color=colors[p])
    axes[0].set(ylabel='Effective complexity C',title='Fixed Gaussian bank; rho calibrated in 1978 only')
    axes[1].set(xlabel='January decision year',ylabel='Ridge penalty (raw units, log)')
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.axvspan(1977.8,1978.2,color='grey',alpha=.15)
        ax.grid(alpha=.2)
    finish(fig,'complexity_paths')
    scenarios = ['gross','target_trade25_borrow30','trade25_borrow30'] if accounts is not None else ['gross']
    fig, axes = plt.subplots(len(scenarios),1,figsize=(8,3*len(scenarios)),sharex=True,layout='constrained',squeeze=False)
    titles = dict(gross='Gross',target_trade25_borrow30='25 bp target turnover + 30 bp/year short fee',
                  trade25_borrow30='25 bp drift-adjusted trades + 30 bp/year short fee')
    for ax, sc in zip(axes[:,0],scenarios):
        for p in colors:
            if sc == 'gross':
                dates, r = monthly.return_date, monthly[f'{p}_total'].to_numpy()
            else:
                f = accounts[accounts.policy.eq(p)&accounts.scenario.eq(sc)].sort_values('return_date')
                dates, r = f.return_date, f.total_return.to_numpy()
            # Normalize at the primary evaluation start, carrying preexisting trading state.
            keep = monthly.decision_year.ge(1979).to_numpy()
            ax.semilogy(dates[keep],np.cumprod(1+r[keep]),label=labels[p],color=colors[p])
        ax.set(title=titles[sc],ylabel='Wealth (log scale)')
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    axes[-1,0].set_xlabel('Realized return month; February 1979-January 2025')
    finish(fig,'wealth_comparison')


def make_report(out, config, annual, summary, paired, has_costs):
    anchor = json.loads((out/'audit/anchor.json').read_text())
    primary = summary[summary.period.eq('1979-2024')]
    gross = primary[primary.scenario.eq('gross')].set_index('policy')
    delta = paired[paired.period.eq('1979-2024')&paired.scenario.eq('gross')&paired.metric.eq('sharpe')&paired.block_months.eq(12)].iloc[0]
    # Convincing value requires a positive primary gross paired interval and cost confirmation.
    convincing = delta.ci_low > 0
    if has_costs:
        net_delta = paired[paired.period.eq('1979-2024')&paired.scenario.eq('target_trade25_borrow30')&paired.metric.eq('sharpe')&paired.block_months.eq(12)].iloc[0]
        convincing &= net_delta.ci_low > 0
    verdict = ('Consider this a candidate empirical addition, with the caveats below.' if convincing else
               'Recommend excluding this method from the paper: it does not add convincing financial value over annual CV.')
    table = ['| Scenario | CV SR | Spectral SR | Difference | 12-month paired 95% interval |',
             '|---|---:|---:|---:|---:|']
    for sc in SCENARIOS if has_costs else ('gross',):
        s = primary[primary.scenario.eq(sc)].set_index('policy')
        d = paired[paired.period.eq('1979-2024')&paired.scenario.eq(sc)&paired.metric.eq('sharpe')&paired.block_months.eq(12)].iloc[0]
        table.append(f'| {sc} | {s.loc["cv","sharpe"]:.4f} | {s.loc["spectral","sharpe"]:.4f} | {d.spectral_minus_cv:+.4f} | [{d.ci_low:+.4f}, {d.ci_high:+.4f}] |')
    sensitivity_rows = paired[paired.period.eq('1979-2024')&paired.scenario.eq('gross')&paired.metric.eq('sharpe')]
    blocks = '\n'.join(f'- {int(r.block_months)} months: [{r.ci_low:+.4f}, {r.ci_high:+.4f}], bootstrap SE {r.bootstrap_se:.4f}.' for r in sensitivity_rows.itertuples())
    sub = summary[summary.policy.isin(['cv','spectral'])&summary.scenario.eq('gross')&~summary.period.isin(['1978-2024','1979-2024'])]
    subtext = '\n'.join(f'- {label}: CV {f.set_index("policy").loc["cv","sharpe"]:.3f}, spectral {f.set_index("policy").loc["spectral","sharpe"]:.3f}.' for label,f in sub.groupby('period',sort=False))
    costs_text = ('Both policies were reconstructed from the licensed stock panels, using the identical frozen feature bank. '
        'Monthly weights, identifiers, features and model arrays remain private under results/. CV weights and the original drift-adjusted net account were independently reconciled. '
        'Stock-level check maxima are in audit/cost_verification.json and tables/stock_reconstruction_checks.csv. '
        'Target turnover is the two-sided sum of absolute changes in signed target weights, including entry, exits, and annual refits. '
        'The requested illustration subtracts 0.0025 times target turnover and 0.003/12 times short notional from gross excess returns. '
        'The repository self-financing sensitivity separately solves c=0.0025*||(1-c)w-d||_1 and deducts the short fee on post-cost NAV using core.accounting_step. '
        'Each policy has its own weights and fee path; costs are never inferred from baseline turnover. '
        'Entry from cash is charged in February 1978; the primary 1979-2024 comparison carries that account state. '
        'Terminal positions are marked without forced liquidation. These are illustrative fees, not observed transaction or lending costs; financing uses the frozen cash rate without impact or an extra funding spread.'
        if has_costs else 'Gross-only: stock reconstruction has not completed. No net-performance claim is made; see audit/cost_unavailable.json for the missing input.')
    numerical = json.loads((out/'audit/selection_verification.json').read_text())
    text = f'''# One-Constant Theory-Guided Spectral Shrinkage

{verdict}

## Frozen design and chronology

Starting commit: `{config['starting_commit']}` on `{config['branch']}`. This isolated run reuses the real JKP cache, 130 characteristics, the fixed 10,000-feature Gaussian bank (seed 0, bandwidth {config['feature_bank']['ell']:.15g}), and portfolio.py's annual_splits, fit_windows and dual_path. The original annual CV minimizes validation response-one squared loss, not validation Sharpe. No stock preprocessing, feature selection, bandwidth tuning or manuscript change is included. The inherited complete-payoff stock sample can itself depend on payoff availability; this experiment preserves that limitation rather than claiming point-in-time universe purity (see the hashed formation-timing audit).

The 120 positive penalties are fixed from January 1963-December 1972 formation observations. Bounds remain [{config['lambda_min']:.16g}, {config['lambda_max']:.16g}]. Training for year y ends in y-6, validation uses formation years y-5 through y-1, and refitting uses all past formation observations from January 1963 through December y-1. The last refit label is realized at January close y, when the policy is formed; the first OOS payoff is February y. Each annual test ends in January y+1. Thus decision years 1978-2024 contain 564 unique OOS months (February 1978-January 2025). The primary 1979-2024 comparison contains 552 months (February 1979-January 2025), excluding the anchored decision year. Annual dates are in annual_selection.csv and every formation/return pair is in monthly_oos.csv. Performance annualizes concatenated excess returns using sample standard deviations; annual Sharpes are descriptive only.

## Algebra and calibration

Use unscaled, uncentered managed payoffs, mu=eig(gg'/T), C(lambda)=sum mu/(mu+lambda), and minimize F(lambda)=lambda+(rho/T)C(lambda). The spectrum and penalty share raw units; the frozen display scale kappa={config['kappa']:.17g} multiplies both policies only after fitting. Zero eigenvalues contribute zero. Meaningfully negative Gram eigenvalues are rejected; only roundoff negatives within 1e-10 times the largest absolute Gram eigenvalue are clipped. Active rank uses a 1e-12 relative threshold for diagnostics only, without truncating the proxy spectrum.

For a nonzero spectrum, F''(lambda)=2(rho/T)sum mu/(mu+lambda)^3 is strictly positive. Derivative signs identify the constrained boundary optimum; otherwise Brent solves the unique root in log(lambda), with fixed bounds. This is an algebraic result about the proxy, not a theorem about financial optimality.

The only calibration uses the 1978 CV decision: train 1963-1972, validate 1973-1977, refit T0={anchor['T0']} through December 1977 formation (last known return January 31, 1978). Its interior raw penalty is lambda0={anchor['lam0']:.17g}, C0={anchor['C0']:.9f}, S0={anchor['S0']:.17g}; rho0=T0/S0={anchor['rho0']:.17g}. The constant is frozen for all future years. The root matches lambda0 to relative error {anchor['relative_anchor_error']:.3g}. Calibration has no 1978 test-payoff argument and a future-perturbation test verifies isolation. A boundary anchor would fail with an explicit identification limitation rather than silently pick rho.

## Primary common-month evidence

{chr(10).join(table)}

Gross annual mean/volatility: CV {100*gross.loc['cv','annual_excess_return']:.3f}%/{100*gross.loc['cv','annual_volatility']:.3f}%; spectral {100*gross.loc['spectral','annual_excess_return']:.3f}%/{100*gross.loc['spectral','annual_volatility']:.3f}%. Gross maximum drawdown: CV {100*gross.loc['cv','maximum_drawdown']:.3f}%, spectral {100*gross.loc['spectral','maximum_drawdown']:.3f}%. All means, volatilities, drawdowns, exposures and fees by scenario and subperiod are in summary.csv. Wealth uses total returns including the same cash rate and is normalized at the primary-window start.

Paired circular-block uncertainty uses {config['bootstrap_replicates']} shared month-index draws per period, seed {config['seed']}; differences are spectral minus CV. Percentile intervals are conditional on the fitted paths, not whole-pipeline sampling inference; there is no retraining or recalibration in the bootstrap. Arbitrary nonstationarity limits a blanket bootstrap interpretation, and subperiod comparisons are descriptive without multiple-comparison adjustment.

Gross block sensitivity:

{blocks}

Gross subperiod Sharpe:

{subtext}

The discrete-grid proxy uses exactly the same 120 frozen candidates and rho. Its primary gross Sharpe is {gross.loc['spectral_grid','sharpe']:.4f}, versus {gross.loc['spectral','sharpe']:.4f} for the continuous rule. Its annual choices and paired intervals are retained as a small grid robustness check, without any OOS grid-hunting.

## Complexity, numerical checks and cost accounting

Primary mean C: CV {gross.loc['cv','mean_C']:.3f}; spectral {gross.loc['spectral','mean_C']:.3f}. Spectral complexity ranges from {annual.C_spectral.min():.3f} to {annual.C_spectral.max():.3f}; lambda ranges from {annual.lambda_spectral.min():.9g} to {annual.lambda_spectral.max():.9g}. Boundary counts: {annual.boundary.value_counts().to_dict()}. The annual table contains C, lambda, active rank, PSD clipping, derivative/KKT residuals, ridge residuals, chronology, and each method's annual realized metrics. Increasing rho under a fixed spectrum increases the selected penalty and cannot increase active complexity; this is covered by deterministic tests.

All 47 annual CV selections and all 564 payoffs reproduce the original fixed-bandwidth Gaussian snapshot and the three-experiment snapshot, at rtol=1e-7, atol=1e-9. The clean-manifest and feature-bank identities match exactly. The cache's managed-array hash differs from the original array hash: the cache uses 64-stock reductions while original managed_matrix uses 256, so float64 accumulation orders differ. The first real managed month is independently recomputed with the original reduction routine, and the full selected payoff paths reconcile. Both hashes and the first-month check are recorded in baseline_compatibility.json before comparison; this byte mismatch is disclosed rather than treating the arrays as bitwise identical. Maximum original/legacy raw payoff errors are {numerical['baseline_payoff_error']:.3g}/{numerical['legacy_payoff_error']:.3g}; maximum root KKT residual {numerical['root_kkt_residual']:.3g}; maximum complexity discrepancy {numerical['complexity_error']:.3g}; maximum ridge residual {numerical['dual_relative_residual']:.3g}. Independent direct ridge solves at 1978, 2000 and 2024 and feature-space payoff aggregation agree. The entire calculation uses deterministic float64 CPU routines and explicitly limited BLAS threads, with no GPU dependency.

{costs_text}

## Limits of the theoretical interpretation

- The starting A*lambda+B*C(lambda)/T is a theoretical upper envelope / fixed-lambda bound under the paper's assumptions, not an identity for exact Sharpe regret. Calibration does not estimate A or B independently; rho is one historically calibrated tuning ratio and need not be structural under nonstationarity.
- A data-dependent empirical-complexity argmin does not automatically inherit a theorem proved only for fixed deterministic lambda. A uniform-in-lambda concentration result would be required.
- Gaussian managed spectra need not be polynomial. This rule uses the full empirical spectrum and never estimates b or extrapolates the sample tail.
- A constrained numerical minimum need not be an unconstrained oracle optimum. Finite sample rank can alter behavior near zero regularization. Bounds must not be expanded after observing returns; frequent boundary solutions would be a limitation.
- Changing complexity does not prove changing population spectral exponents or establish optimality for nonstationary returns.
- The 1978 agreement is guaranteed by construction and supplies no independent evidence of predictive superiority. It is excluded from the primary comparison.

## Reproduction and output scope

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m empirical_final.theory_guided_lambda --output outputs/theory_guided_lambda_20261010_reproduction
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_theory_guided_lambda.py tests/test_final_empirical.py tests/test_audit.py tests/test_bandwidth_tuning.py tests/test_three_empirical_experiments.py
```

The default bootstrap count is 5,000. A new output directory is required for a fresh full run. Selection writes annual checkpoints, then stores private coefficients under results/<output-name>/. `--phase costs` reruns stock reconstruction from verified coefficients; `--phase report` rebuilds summaries and figures after hash verification. Public cost checkpoints are refreshed after every stock year. Input hashes, original source hashes, software/thread details, the anchor and numerical checks are under audit/. Hashes publish no licensed observations or holdings. Public output is restricted to portfolio-level aggregates, figures, this report, tests, and a separate candidate subsection. The main manuscript remains unchanged. The candidate is intentionally at most one page and is not inserted automatically.
'''
    (out/'REPORT.md').write_text(text)
    tex = r'''% Candidate only; do not insert automatically into the main manuscript.
\subsection{One-Constant Theory-Guided Spectral Shrinkage}
\label{app:one_constant_spectral}
Using past-only raw managed eigenvalues $\widehat\mu_{j,y}$ of
$g_yg_y^\top/T_y$, consider
\begin{equation}
 \widehat C_y(\lambda)=\sum_j\frac{\widehat\mu_{j,y}}{\widehat\mu_{j,y}+\lambda},\quad
 \widehat\lambda_y=\underset{\lambda\in[\lambda_{\min},\lambda_{\max}]}{\arg\min}
 \left\{\lambda+\frac{\rho_0}{T_y}\widehat C_y(\lambda)\right\}.
\end{equation}
The original 120-point grid fixes the bounds. Strict convexity permits
Brent root finding, with boundary selection by derivative signs:
\begin{equation}
 \rho_0=\frac{T_0}{\sum_j\widehat\mu_{j,0}/(\widehat\mu_{j,0}+\lambda_0)^2},\quad
 1=\frac{\rho_0}{T_y}\sum_j\frac{\widehat\mu_{j,y}}{(\widehat\mu_{j,y}+\widehat\lambda_y)^2}.
\end{equation}
Calibration uses only the 1978 decision (training 1963--1972, validation
1973--1977, refit $T_0=180$): $\lambda_0=ANCHOR$, $\rho_0=RHO$.
The ratio is frozen thereafter; first-year agreement holds by construction.

With identical 10,000 Gaussian features, bandwidth, annual splits, ridge
solver and scale, the 552 common returns for decisions 1979--2024 give
CV/spectral gross Sharpe CVSR/SPSR. The difference DELTA has paired
95\% interval [LOW,HIGH] (12-month circular blocks, 5,000 draws,
conditional on fitted paths). COSTTEXT
There are BOUNDARY boundary solutions; grid-rule Sharpe is GRIDSR. VERDICT

The fixed-penalty upper bound motivates this proposal without guaranteeing
oracle optimality. Data-dependent selection requires uniform-in-penalty
concentration; historical calibration need not be structural. No polynomial
Gaussian tail is assumed. Finite sample rank and nonstationarity remain limitations.
'''
    nettext = ''
    if has_costs:
        n = primary[primary.scenario.eq('target_trade25_borrow30')].set_index('policy')
        nettext = f'Reconstructed 25-bp target-turnover plus 30-bp/year short-fee Sharpes are {n.loc["cv","sharpe"]:.3f}/{n.loc["spectral","sharpe"]:.3f}.'
    else:
        nettext = 'Stock accounting is unavailable; these results are gross only.'
    scientific = lambda x: (lambda m,e: m+r'\times10^{'+str(int(e))+'}')(*f'{x:.3e}'.split('e'))
    replacements = dict(ANCHOR=scientific(anchor['lam0']), RHO=scientific(anchor['rho0']),
        CVSR=f'{gross.loc["cv","sharpe"]:.3f}', SPSR=f'{gross.loc["spectral","sharpe"]:.3f}',
        DELTA=f'{delta.spectral_minus_cv:+.3f}', LOW=f'{delta.ci_low:+.3f}', HIGH=f'{delta.ci_high:+.3f}',
        COSTTEXT=nettext, BOUNDARY=str(int(annual.boundary.ne('interior').sum())),
        GRIDSR=f'{gross.loc["spectral_grid","sharpe"]:.3f}',
        VERDICT='We recommend excluding this method: financial improvement is not convincing.' if not convincing else 'The positive paired evidence merits consideration with these limitations.')
    for old, new in replacements.items():
        tex = tex.replace(old,new)
    (out/'publication/one_page_algorithm_appendix.tex').write_text(tex)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--phase', choices=['all','selection','costs','report'], default='all')
    parser.add_argument('--bootstrap-replicates', type=int, default=REPLICATES)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 2:
        parser.error('At least two bootstrap replicates required')
    out = args.output.resolve()
    private = ROOT/'results'/out.name
    with threadpool_limits(limits=1):
        if args.phase in ('all','selection'):
            selection(out, private, args.bootstrap_replicates)
        if args.phase in ('all','costs'):
            costs(out, private)
        if args.phase in ('all','report'):
            summarize(out)
    print(f'Finished {args.phase}: {out}', flush=True)


if __name__ == '__main__':
    main()
