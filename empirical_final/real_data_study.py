"""Five exploratory, real-stock JKP experiments, with bounded stock blocks.

Run through empirical_final.memory_guard. No simulation or legacy fitted cache
enters this study. Unknown future payoffs never determine ranks or membership.
"""
from __future__ import annotations

import argparse
from collections import deque
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline import digest
from empirical_final.core import accounting_step
from empirical_final.real_data_core import chronological_fit, match_groups, nested_universe, past_stock_betas
from kernels import FeatureBank, median_distance
from portfolio import sharpe

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
SCENARIOS = {'missing_total_minus100': -1., 'missing_total_zero': 0., 'missing_total_plus100': 1.}
HISTORIES = [36, 60, 120, 240, 360]
CAPS = [100, 500, 1000, 3000]
RIDGE = np.logspace(-6, 2, 17)
COMPLEXITIES = [1, 2, 4, 8, 16, 24, 32, 64, 128]
PCR = [.05, .1, .25, .5, .75, 1.]
FAMILIES = {
    'value': ['be_me', 'ni_me', 'fcf_me', 'sale_me', 'ocf_me'],
    'profitability': ['gp_at', 'cop_at', 'op_at', 'ni_be'],
    'momentum': ['ret_12_1', 'ret_6_1', 'resff3_12_1'],
}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str)+'\n')


def months(source, files, names, last_year=2024):
    """One monthly panel plus one 8,192-row Arrow batch; never a yearly frame."""
    columns = ['id', 'eom', 'r', 'return_known', *names]
    for record in files:
        year = int(Path(record['name']).stem.split('_')[-1])
        if year > last_year:
            continue
        pending, last = [], None
        for batch in pq.ParquetFile(source/record['name']).iter_batches(
                batch_size=8192, columns=columns, use_threads=False):
            frame = batch.to_pandas()
            frame['eom'] = pd.to_datetime(frame.eom)
            if not frame.eom.is_monotonic_increasing:
                raise ValueError('Formation panels must be sorted by month')
            for date, part in frame.groupby('eom', sort=False):
                if last is not None and date != last:
                    yield last, pd.concat(pending, ignore_index=True)
                    pending = []
                pending.append(part.copy())
                last = date
        if pending:
            yield last, pd.concat(pending, ignore_index=True)
        pa.default_memory_pool().release_unused()


def definition(names, features, seed):
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(names))
    variants = {}
    def add(label, selected, kernel='gaussian', cap=None, neutral=None):
        variants[label] = dict(names=sorted(selected), kernel=kernel, cap=cap, neutral=neutral,
                               dimension=len(selected)+1 if kernel == 'linear' else features)
    for count in [8, 16, 32, 64, len(names)]:
        for prefix, kernel in [('L', 'linear'), ('G', 'gaussian')]:
            add(f'{prefix}{count}', order[:count], kernel)
    for j in range(24):
        add(f'A{j:02}', rng.choice(names, 32, replace=False).tolist(), 'linear')
    for family, removed in FAMILIES.items():
        removed = sorted(set(removed) & set(names))
        if not removed:
            raise ValueError('Missing economic-family names')
        add('drop_'+family, sorted(set(names)-set(removed)))
    for j in range(3):
        removed = rng.choice(names, 32, replace=False)
        add(f'drop_random{j}', sorted(set(names)-set(removed)))
    for cap in CAPS:
        add(f'N{cap}', names, cap=cap)
    add('neutral_market', names, neutral=1)
    add('neutral_six', names, neutral=6)
    return variants


def initial_sample(source, files, names, seed):
    rng = np.random.default_rng(seed)
    sample = []
    for _, frame in months(source, files, names, 1972):
        x = frame[names].to_numpy(float)
        sample.append(x[rng.choice(len(x), min(8, len(x)), replace=False)])
    return np.vstack(sample)


def banks_for(variants, names, sample, features, seed):
    banks = {}
    for label, spec in variants.items():
        if spec['cap'] or spec['neutral']:
            continue
        coords = [names.index(name) for name in spec['names']]
        ell = median_distance(sample[:, coords]) if spec['kernel'] != 'linear' else 1.
        banks[label] = FeatureBank(spec['kernel'], len(coords), maximum=features, ell=ell, seed=seed)
    return banks


def factor_history(history, factors):
    if not history:
        return np.empty((0, len(FACTORS)))
    return factors.reindex([item[0]+pd.offsets.MonthEnd(1) for item in history])[FACTORS].to_numpy(float)


def panel_inputs(date, frame, names, history, factors):
    ids = frame.id.to_numpy(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate security at formation')
    x, r = frame[names].to_numpy(float), frame.r.to_numpy(float)
    if not np.isfinite(x).all() or not np.array_equal(frame.return_known.to_numpy(bool), np.isfinite(r)):
        raise ValueError('Invalid formation characteristics or payoff-known flag')
    observations = [(item[1], item[2]) for item in history]
    exposures = past_stock_betas(ids, observations, factor_history(history, factors))
    history.append((date, ids, r))  # Only used at subsequent formation dates.
    return ids, x, r, exposures


def build_cache(args, source, raw_manifest, names, variants, banks, dates, factors, identity):
    cache = args.private/'managed'
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache/'manifest.json'
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if meta['identity'] != identity or meta['status'] != 'complete':
            raise ValueError('Cache identity changed; use a new private output directory')
        for label, expected in meta['files'].items():
            if digest(cache/f'{label}.npy') != expected:
                raise ValueError('Managed cache checksum failed')
        print('Verified this revision\'s managed cache', flush=True)
        return cache
    arrays = {label: np.lib.format.open_memmap(cache/f'{label}.npy', mode='w+', dtype='float64',
              shape=(len(dates), 2, spec['dimension'])) for label, spec in variants.items()}
    history, counts = deque(maxlen=60), []
    main = 'G'+str(len(names))
    j = 0
    for date, frame in months(source, raw_manifest['files'], names):
        if j >= len(dates) or date != dates[j]:
            raise ValueError('Incomplete or unexpected formation calendar')
        ids, x, r, exposures = panel_inputs(date, frame, names, history, factors)
        known = np.where(np.isfinite(r), r, 0.)
        missing = (~np.isfinite(r)).astype(float)
        cap_masks = {}
        for cap in CAPS:
            chosen = nested_universe(ids, cap, args.seed)
            mask = np.zeros(len(ids))
            mask[chosen] = 1
            cap_masks[cap] = mask
        for label, bank in banks.items():
            spec = variants[label]
            coords = [names.index(name) for name in spec['names']]
            total = np.zeros((2, spec['dimension']))
            cap_totals = {cap: np.zeros_like(total) for cap in CAPS} if label == main else {}
            cross = np.zeros((spec['dimension'], 6)) if label == main else None
            for start in range(0, len(ids), 128):
                stop = start+128
                phi = bank.features(x[start:stop, coords])
                payoff_basis = np.column_stack([known[start:stop], missing[start:stop]])
                total += (phi.T @ payoff_basis).T
                if label == main:
                    cross += phi.T @ exposures[start:stop]
                    for cap, mask in cap_masks.items():
                        cap_totals[cap] += (phi.T @ (payoff_basis*mask[start:stop, None])).T
            arrays[label][j] = total/len(ids)
            if label == main:
                for cap, value in cap_totals.items():
                    arrays[f'N{cap}'][j] = value/cap_masks[cap].sum()
                for label_neutral, width in [('neutral_market', 1), ('neutral_six', 6)]:
                    b = exposures[:, :width]
                    adjustment = cross[:, :width] @ np.linalg.pinv(b.T@b) @ (b.T @ np.column_stack([known, missing]))
                    arrays[label_neutral][j] = (total-adjustment.T)/len(ids)
        counts.append(dict(formation_date=date, stocks=len(ids), unknown_returns=int(missing.sum()),
                           **{f'actual_N{cap}': int(mask.sum()) for cap, mask in cap_masks.items()}))
        j += 1
        if date.month == 12:
            for array in arrays.values():
                array.flush()
            print(f'Managed JKP {date.year}: {len(ids)} stocks, {int(missing.sum())} unresolved payoffs', flush=True)
        del x, frame
    if j != len(dates):
        raise ValueError('Missing formation months')
    for array in arrays.values():
        array.flush()
    del arrays
    gc.collect()
    pd.DataFrame(counts).to_csv(args.out/'tables/universe.csv', index=False)
    write_json(manifest_path, dict(status='complete', identity=identity,
               files={label: digest(cache/f'{label}.npy') for label in variants}))
    return cache


def load_g(cache, label, imposed_total, rf):
    basis = np.load(cache/f'{label}.npy', mmap_mode='r')
    return basis[:, 0]+basis[:, 1]*(imposed_total-rf)[:, None]


def select_pair(cache, dates, rf, args):
    decision = int(np.flatnonzero(dates.year == args.start_year)[0])
    rows = []
    for j in range(24):
        label = f'A{j:02}'
        g = load_g(cache, label, 0., rf)
        fit = chronological_fit(g, decision, 360, 12, RIDGE)
        rows.append(dict(variant=label, validation_sharpe=float(sharpe(fit['validation_returns'][:, fit['choice']])),
                         **fit['diagnostics']))
    pd.DataFrame(rows).to_csv(args.out/'tables/A_candidates.csv', index=False)
    pair = match_groups(rows)
    write_json(args.out/'audit/A_match.json', dict(pair=pair, tolerance_annualized_sharpe=.15,
               reference_scenario='missing_total_zero', selected_before_first_test=str(dates[decision]),
               rule='Maximize combined concentration and signal-alignment spread among validation-Sharpe matches; same pair for all payoff scenarios.',
               status='matched' if pair else 'No admissible pair; experiment A unavailable at the fixed tolerance.'))
    return pair


def policy_specs(names, pair):
    main, linear = 'G'+str(len(names)), 'L'+str(len(names))
    policies = []
    def add(experiment, label, variant, history=240, kind='ridge', fixed=None):
        policies.append(dict(experiment=experiment, policy=label, variant=variant, history=history, kind=kind, fixed=fixed))
    if pair:
        for label in pair:
            add('A', 'matched_'+label, label)
    for cap in [*CAPS, 'all']:
        for history in HISTORIES:
            add('B', f'N{cap}_T{history}', main if cap == 'all' else f'N{cap}', history)
    for count in [8, 16, 32, 64, len(names)]:
        for prefix in ['L', 'G']:
            add('C', f'{prefix}{count}', f'{prefix}{count}')
    for history in HISTORIES:
        for j, complexity in enumerate(COMPLEXITIES):
            add('D', f'T{history}_C{complexity}', main, history, 'complexity', j)
        add('D', f'T{history}_validated_C', main, history, 'complexity')
    add('E', 'ridge_all', main)
    for label in ['drop_'+f for f in FAMILIES]+[f'drop_random{j}' for j in range(3)]+['neutral_market', 'neutral_six']:
        add('E', label, label)
    for j, fraction in enumerate(PCR):
        add('E', f'PCR_{fraction:g}', main, kind='pcr', fixed=j)
    add('E', 'PCR_validated', main, kind='pcr')
    return policies


def fit_policies(args, cache, variants, names, dates, factors, rf, pair):
    policies = policy_specs(names, pair)
    annual, spectral, unavailable = [], [], []
    coefficients = {}
    calibration = {}
    for scenario, imposed in SCENARIOS.items():
        for year in range(args.start_year, args.end_year+1):
            decision = int(np.flatnonzero(dates.year == year)[0])
            fitted, data = {}, {}
            for spec in policies:
                variant, history, kind = spec['variant'], spec['history'], spec['kind']
                if variant not in data:
                    data[variant] = load_g(cache, variant, imposed, rf)
                key = variant, history, kind
                if key not in fitted:
                    values = RIDGE if kind == 'ridge' else COMPLEXITIES if kind == 'complexity' else PCR
                    fitted[key] = chronological_fit(data[variant], decision, history, 12, values, kind)
                fit = fitted[key]
                choice = fit['choice'] if spec['fixed'] is None else spec['fixed']
                if not fit['eligible'][choice]:
                    unavailable.append(dict(scenario=scenario, decision_year=year, **spec,
                                            reason='Requested C is at least the numerical rank in inner training or refit'))
                    continue
                coeff_key = scenario, year, spec['experiment'], spec['policy']
                coefficients[coeff_key] = fit['beta'][:, choice].copy()
                annual.append(dict(scenario=scenario, decision_year=year, **spec,
                    selected_candidate=choice, penalty=float(fit['penalty'][choice]), C=float(fit['complexity'][choice]),
                    validation_loss=float(fit['validation_loss'][choice]),
                    last_train_payoff=dates[fit['refit'][-1]]+pd.offsets.MonthEnd(1),
                    first_test_payoff=dates[decision]+pd.offsets.MonthEnd(1), **fit['diagnostics']))
                if scenario == 'missing_total_zero' and year == args.start_year and spec['experiment'] == 'C' and spec['variant'] == 'L'+str(len(names)):
                    calibration['beta'] = fit['calibration_beta']
                    calibration['variant'] = spec['variant']
                    calibration['dates'] = dates[fit['validation']]
                if spec['experiment'] == 'E' and spec['policy'] == 'ridge_all':
                    for rank, (mu, signal) in enumerate(zip(fit['eigenvalues'], fit['signal'], strict=True), 1):
                        spectral.append(dict(scenario=scenario, decision_year=year, rank=rank, eigenvalue=mu, historical_signal=signal))
            del fitted, data
            print(f'Past-only fits {scenario} / {year}', flush=True)
    annual = pd.DataFrame(annual)
    if not (pd.to_datetime(annual.last_train_payoff) < pd.to_datetime(annual.first_test_payoff)).all():
        raise ValueError('Temporal leakage in fitting')
    annual.to_csv(args.out/'tables/annual_policies.csv', index=False)
    pd.DataFrame(spectral).to_csv(args.out/'tables/economic_spectrum.csv', index=False)
    pd.DataFrame(unavailable, columns=['scenario', 'decision_year', *policies[0].keys(), 'reason']).to_csv(
        args.out/'tables/unavailable_complexities.csv', index=False)
    # Private aggregate coefficients facilitate accounting, never publish stock positions.
    return policies, coefficients, annual, calibration


def account(args, source, manifest, names, variants, banks, dates, factors, cash, policies, coefficients, annual, calibration):
    rows, states, insolvent, calibration_gross = [], {}, set(), []
    monthly_path = args.out/'tables/monthly_policies.csv'
    wrote_monthly = False
    history = deque(maxlen=60)
    main = 'G'+str(len(names))
    scale = None
    annual_lookup = annual.set_index(['scenario', 'decision_year', 'experiment', 'policy'])
    first_calibration = calibration['dates'].min()
    for date, frame in months(source, manifest['files'], names, args.end_year):
        ids, x, r, exposures = panel_inputs(date, frame, names, history, factors)
        if date < first_calibration:
            continue
        if date in calibration['dates']:
            spec = variants[calibration['variant']]
            coords = [names.index(name) for name in spec['names']]
            score = banks[calibration['variant']].scores(x[:, coords], calibration['beta'])
            calibration_gross.append(float(np.mean(np.abs(score))))
        if date.year < args.start_year:
            continue
        if scale is None:
            if len(calibration_gross) != 12 or np.median(calibration_gross) <= 0:
                raise ValueError('Incomplete training-only common-scale calibration')
            scale = float(1/np.median(calibration_gross))
            write_json(args.out/'audit/scale.json', dict(scale=scale, target_median_gross=1.,
                       calibration_dates=list(calibration['dates']), scenario='missing_total_zero',
                       rule='Common affine-linear scale, calibrated on the last 12 pretest months; frozen for every experiment and payoff scenario.'))
        retdate = date+pd.offsets.MonthEnd(1)
        rf = float(cash.loc[retdate])
        grouped = {}
        for spec in policies:
            for scenario in SCENARIOS:
                key = scenario, date.year, spec['experiment'], spec['policy']
                if key in coefficients:
                    grouped.setdefault(spec['variant'], []).append((key, spec))
        for variant, entries in grouped.items():
            spec = variants[variant]
            bank = banks[main] if spec['neutral'] or spec['cap'] else banks[variant]
            coords = [names.index(name) for name in spec['names']]
            beta = np.column_stack([coefficients[key] for key, _ in entries])
            scores = np.empty((len(ids), len(entries)))
            for start in range(0, len(ids), 128):
                scores[start:start+128] = bank.features(x[start:start+128, coords]) @ beta
            if spec['neutral']:
                b = exposures[:, :spec['neutral']]
                scores -= b @ (np.linalg.pinv(b) @ scores)
                error = np.max(np.abs(b.T@scores))/max(np.max(np.abs(scores)), 1.)/len(ids)
                if error > 1e-10:
                    raise ValueError('Stock positions fail factor-neutrality identity')
            for j, (key, policy) in enumerate(entries):
                scenario = key[0]
                private_key = scenario, policy['experiment'], policy['policy']
                # B: training cap changes information; deployment always uses ALL formation stocks.
                raw_weights = scores[:, j]/len(ids)
                returns = np.where(np.isfinite(r), r, SCENARIOS[scenario]-rf)
                raw = float(raw_weights@returns)
                target = pd.Series(scale*raw_weights, index=ids)
                old = states.get(private_key, pd.Series(dtype=float))
                union = target.index.union(old.index)
                net, turnover, fees, borrowing = np.nan, np.nan, np.nan, np.nan
                if private_key not in insolvent:
                    try:
                        aligned_returns = pd.Series(returns, index=ids).reindex(union).to_numpy()
                        step = accounting_step(target.reindex(union, fill_value=0.).to_numpy(), aligned_returns, rf,
                            old.reindex(union, fill_value=0.).to_numpy(), args.trade_bps/1e4, args.borrow_bps/1e4)
                        net = step['excess_return']
                        turnover, fees, borrowing = step['turnover'], step['trading_fee'], step['borrowing_fee']
                        states[private_key] = pd.Series(step['next_drift'], index=union).reindex(ids)
                    except ValueError as error:
                        if 'insolvency' not in str(error).lower() and 'exhaust' not in str(error).lower() and 'noncontractive' not in str(error).lower():
                            raise
                        insolvent.add(private_key)
                fitted = annual_lookup.loc[key]
                rows.append(dict(scenario=scenario, experiment=policy['experiment'], policy=policy['policy'],
                    formation_date=date, return_date=retdate, history=policy['history'], variant=variant,
                    raw_return=raw, Q=(1-raw)**2, scaled_gross_excess=scale*raw,
                    net_excess=net, turnover=turnover, trading_fee=fees, borrowing_fee=borrowing,
                    C=fitted.C, gross_exposure=float(target.abs().sum()),
                    unknown_abs_weight=float(np.abs(target.to_numpy()[~np.isfinite(r)]).sum()),
                    net_insolvent=private_key in insolvent))
        if date.month == 12:
            pd.DataFrame(rows).to_csv(monthly_path, mode='a' if wrote_monthly else 'w',
                                     header=not wrote_monthly, index=False, float_format='%.14g')
            wrote_monthly = True
            rows.clear()
            print(f'Stock accounting {date.year}; common scale {scale:.6g}', flush=True)
    monthly = pd.read_csv(monthly_path, parse_dates=['formation_date', 'return_date'],
                          dtype={name: 'category' for name in ['scenario', 'experiment', 'policy', 'variant']})
    # Independent stock-score versus monthly managed-return equality, including neutralization and imposed payoffs.
    for key, group in monthly.groupby(['scenario', 'experiment', 'policy'], sort=False):
        spec = next(p for p in policies if p['experiment'] == key[1] and p['policy'] == key[2])
        deployed_variant = main if variants[spec['variant']]['cap'] else spec['variant']
        g = load_g(args.private/'managed', deployed_variant, SCENARIOS[key[0]], cash.reindex(dates+pd.offsets.MonthEnd(1)).to_numpy())
        for year, block in group.groupby(group.formation_date.dt.year):
            ix = dates.get_indexer(block.formation_date)
            expected = g[ix] @ coefficients[(key[0], year, key[1], key[2])]
            np.testing.assert_allclose(block.raw_return, expected, rtol=2e-7, atol=1e-9)
    summaries = []
    expected_months = (args.end_year-args.start_year+1)*12
    for key, group in monthly.groupby(['scenario', 'experiment', 'policy'], sort=False):
        complete = len(group) == expected_months
        valid_net = complete and np.isfinite(group.net_excess).all()
        summaries.append(dict(scenario=key[0], experiment=key[1], policy=key[2], months=len(group),
            complete_common_period=complete, raw_Q=float(group.Q.mean()),
            gross_sharpe=float(sharpe(group.scaled_gross_excess)) if complete else np.nan,
            net_sharpe=float(sharpe(group.net_excess)) if valid_net else np.nan,
            net_status='complete' if valid_net else 'insolvent' if group.net_insolvent.any() else 'incomplete',
            mean_C=float(group.C.mean()), mean_gross=float(group.gross_exposure.mean()),
            annual_turnover=float(12*group.turnover.mean()) if valid_net else np.nan,
            mean_unknown_abs_weight=float(group.unknown_abs_weight.mean())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out/'tables/summary.csv', index=False, float_format='%.11g')
    # Nested PCR increments are evaluated on identical subsequent months, including their OOS cross term.
    increments = []
    for scenario in SCENARIOS:
        p = monthly[(monthly.scenario == scenario) & (monthly.experiment == 'E')]
        previous = None
        for fraction in PCR:
            label = f'PCR_{fraction:g}'
            group = p[p.policy == label].set_index('return_date')
            if previous is not None:
                joined = previous[['raw_return']].join(group[['raw_return']], lsuffix='_previous', rsuffix='_next', how='inner')
                added = joined.raw_return_next-joined.raw_return_previous
                increments.append(dict(scenario=scenario, through_fraction=fraction, months=len(joined),
                    delta_OOS_loss=float(np.mean((1-joined.raw_return_next)**2-(1-joined.raw_return_previous)**2)),
                    OOS_cross_term=float(-2*np.mean((1-joined.raw_return_previous)*added)),
                    OOS_added_square=float(np.mean(added**2))))
            previous = group
    pd.DataFrame(increments).to_csv(args.out/'tables/E_incremental_directions.csv', index=False)
    return summary


def report(args, summary, pair):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    central = summary[summary.scenario == 'missing_total_zero']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    b = central[central.experiment == 'B'].copy()
    for cap in [*CAPS, 'all']:
        part = b[b.policy.str.startswith(f'N{cap}_')].copy()
        part['T'] = part.policy.str.extract(r'_T(\d+)').astype(int)
        part = part.sort_values('T')
        axes[0].plot(part['T'], part.raw_Q, marker='o', label=f'N={cap}')
    axes[0].set(xlabel='Available months T', ylabel='OOS quadratic loss', title='B: training N; common deployment universe')
    axes[0].legend(fontsize=7)
    for prefix, label in [('L', 'Linear'), ('G', 'Gaussian RFF')]:
        c = central[(central.experiment == 'C') & central.policy.str.startswith(prefix)].copy()
        c['count'] = c.policy.str[1:].astype(int)
        c = c.sort_values('count')
        axes[1].plot(c['count'], c.raw_Q, marker='o', label=label)
    axes[1].set(xlabel='Characteristics', ylabel='OOS quadratic loss', title='C: nested, predeclared features')
    axes[1].legend(fontsize=7)
    d = central[(central.experiment == 'D') & ~central.policy.str.endswith('validated_C') & central.complete_common_period]
    for history in HISTORIES:
        part = d[d.policy.str.startswith(f'T{history}_')].sort_values('mean_C')
        axes[2].plot(part.mean_C, part.raw_Q, marker='.', label=f'T={history}')
    axes[2].set(xlabel='Effective complexity C', ylabel='OOS quadratic loss', title='D: descriptive OOS surface slices')
    axes[2].legend(fontsize=7)
    fig.suptitle('Exploratory JKP sensitivity: missing TOTAL payoff imposed at zero; not identified performance', fontsize=10)
    fig.savefig(args.out/'figures/BCD_loss.png', dpi=160)
    plt.close(fig)
    lines = [
        'FIVE REAL-DATA JKP EXPERIMENTS — EXPLORATORY RESULTS',
        f'Common formation test: {args.start_year}–{args.end_year}; payoffs February {args.start_year}–January {args.end_year+1}.',
        f'Gaussian approximation: {args.features} fixed random Fourier features; not the legacy 10,000-feature specification.',
        'All stock observations are real JKP observations. No simulation or old fitted policy is used.',
        '',
        'Identification: the source contains 7,489 unresolved subsequent returns. Formation ranks and N include these stocks.',
        'Three separate imposed TOTAL-return scenarios (-100%, 0%, +100%) are fitted and evaluated. These are sensitivity assumptions,',
        'not recovered data, confidence bounds, or a certified real-time trading record. No scenario is promoted to a primary estimate.',
        'The dictionary and characteristic snapshot are retrospective; historical data vintages are not certified.',
        '',
        'A: 24 fixed random groups of 32 characteristics; compare prior-validation Sharpe within 0.15, then maximize spectral spread.',
        f'Frozen matched pair: {pair}. Matching is observational, population Sharpe equivalence is not established.',
        'B: nested stable identifier samples at N=100,500,1000,3000,all; T=36,60,120,240,360. All deploy on the full formation universe.',
        'Monthly actual N is min(requested N, available stocks), explicitly reported in universe.csv.',
        'C: nested fixed subsets of 8,16,32,64,130 characteristics; affine linear and Gaussian; past-only lengthscales and penalty tuning.',
        'D: exact effective-complexity targets versus T, plus validation-selected C. Infeasible inner-training complexities are reported, not capped.',
        'E: economic-family and random ablations; ridge versus PCR; market and six-factor position neutrality using trailing 60 observed months.',
        'New-listing exposures use a fixed market-beta-one / other-beta-zero prior (12 standardized observations), without excluding securities.',
        'PCR directions use the economic managed-return second-moment operator. Direction increments are measured by subsequent loss.',
        '',
        'T includes the final 12 chronological validation months; selected parameters are refitted on all T months.',
        'January-close decisions can use December-formation payoffs realized in January, before the first February test payoff.',
        'A single scale targets pretest linear median gross 1. It is frozen across policies and payoff scenarios.',
        f'Net Sharpe assumes {args.trade_bps:g} bps per traded dollar and {args.borrow_bps:g} annual bps on shorts; illustrative costs, not observed execution data.',
        'Costs use exact self-financing accounting, drifted positions, and liquidation of departed identifiers at the next formation mark.',
        'A nonpositive NAV or noncontractive execution cost invalidates the entire net path; no post-insolvency returns enter net Sharpe.',
        'Rows without the complete common period are not ranked. OOS surfaces are descriptive; their maxima do not select policies.',
        '',
        'See tables/summary.csv for every experiment under all three payoff assumptions; figures show only the labelled zero-total sensitivity.',
        'See audit/ for source hashes, protocol, common-scale calibration, temporal checks and separately generated memory logs.',
    ]
    (args.out/'REPORT.txt').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'data/formation_only')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/real_data_ABCDE_20261021')
    parser.add_argument('--private', type=Path, default=ROOT/'results/real_data_ABCDE_20261021')
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--start-year', type=int, default=1994)
    parser.add_argument('--end-year', type=int, default=2024)
    parser.add_argument('--trade-bps', type=float, default=10.)
    parser.add_argument('--borrow-bps', type=float, default=30.)
    args = parser.parse_args()
    if args.start_year < 1994 or not args.start_year <= args.end_year <= 2024 or args.features < 16 or min(args.trade_bps, args.borrow_bps) < 0:
        parser.error('Need 1994 <= start <= end <= 2024, at least 16 RFF and nonnegative costs')
    for folder in ['tables', 'audit', 'figures']:
        (args.out/folder).mkdir(parents=True, exist_ok=True)
    args.private.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source/'manifest.json').read_text())
    if manifest['status'] != 'formation_only_partial_payoffs' or manifest['audit']['future_availability_perturbation_rank_error'] != 0:
        raise ValueError('Require the audited formation-only source, never complete-payoff ranks')
    # The audited panels already bind the initial-history selection. Reading
    # their schema avoids depending on a second, possibly offloaded sidecar.
    metadata = {'id', 'eom', 'return_date', 'me', 'r', 'return_known'}
    names = sorted(set(pq.ParquetFile(args.source/manifest['files'][0]['name']).schema_arrow.names)-metadata)
    if len(names) != 130:
        raise ValueError('Expected the initial-history-frozen 130-characteristic dictionary')
    factors_path = ROOT/'outputs/empirical_three_experiments_20261009/tables/french_factors.csv'
    factors = pd.read_csv(factors_path, parse_dates=['return_date']).set_index('return_date')
    cash_path = ROOT/'data/risk_free.csv'
    cash_frame = pd.read_csv(cash_path, parse_dates=['return_date']).set_index('return_date')
    cash = cash_frame['rf']
    dates = pd.date_range('1963-01-31', '2024-12-31', freq='ME')
    rf = cash.reindex(dates+pd.offsets.MonthEnd(1)).to_numpy(float)
    if not np.isfinite(rf).all():
        raise ValueError('Incomplete source cash rates')
    variants = definition(names, args.features, args.seed)
    identity = dict(features=args.features, seed=args.seed, variants=variants,
                    source_manifest_sha256=digest(args.source/'manifest.json'),
                    factors_sha256=digest(factors_path), cash_sha256=digest(cash_path),
                    code={name: digest(ROOT/'empirical_final'/name) for name in ['real_data_study.py', 'real_data_core.py']},
                    dependencies={name: digest(ROOT/name) for name in ['kernels.py', 'portfolio.py', 'empirical_final/core.py']})
    frozen = args.out/'audit/protocol.json'
    protocol = dict(identity=identity, start_year=args.start_year, end_year=args.end_year,
                    histories=HISTORIES, validation_months=12, ridge_ratios=RIDGE.tolist(), complexity_targets=COMPLEXITIES,
                    pcr_fractions=PCR, imposed_missing_total_returns=SCENARIOS, trade_bps=args.trade_bps, borrow_bps=args.borrow_bps)
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError('Frozen run identity changed: choose new output directories')
    write_json(frozen, protocol)
    # Verify every licensed formation file before accepting the pre-existing ranks.
    for record in manifest['files']:
        if digest(args.source/record['name']) != record['sha256']:
            raise ValueError('Formation-source checksum mismatch: '+record['name'])
    sample = initial_sample(args.source, manifest['files'], names, args.seed)
    banks = banks_for(variants, names, sample, args.features, args.seed)
    write_json(args.out/'audit/feature_banks.json', {label: bank.metadata() for label, bank in banks.items()})
    cache = build_cache(args, args.source, manifest, names, variants, banks, dates, factors, identity)
    pair = select_pair(cache, dates, rf, args)
    policies, coefficients, annual, calibration = fit_policies(args, cache, variants, names, dates, factors, rf, pair)
    summary = account(args, args.source, manifest, names, variants, banks, dates, factors, cash, policies, coefficients, annual, calibration)
    report(args, summary, pair)
    write_json(args.out/'audit/execution.json', dict(status='complete_exploratory_sensitivities',
               identity=identity, matched_pair=pair, stock_vs_managed_identity_passed=True,
               temporal_checks_passed=True, observations='Real JKP only',
               unresolved_source_payoffs=manifest['audit']['unresolved_payoffs'],
               identified_performance=False,
               artifact_sha256={str(path.relative_to(args.out)): digest(path) for path in sorted(args.out.rglob('*')) if path.is_file() and 'memory_' not in path.name and path.name != 'execution.json'}))
    print(f'Finished five real-data sensitivities: {args.out}', flush=True)


if __name__ == '__main__':
    main()
