"""Independently audit every reported penalty distribution and slope window."""
import json

import numpy as np
import pandas as pd
from scipy.special import ndtri

from simulations.extended.design import OUTPUT, ENVIRONMENTS, WINDOWS
from simulations.provenance import json_write, utc_now


METRICS = ('annual_SR', 'annual_gap', 'regret', 'empirical_complexity',
           'relative_complexity', 'coefficient_norm_squared')
DECOMPOSITION = ('population_bias', 'estimation_norm', 'cross_term')


def check_statistics(rows, values):
    """Rows follow the flattened trailing dimensions of replication-first values."""
    values = np.asarray(values)
    if len(rows) != int(np.prod(values.shape[1:])) or not np.isfinite(values).all():
        raise ValueError('Summary rows do not cover the complete finite distribution.')
    n = len(values)
    average = np.sum(values, axis=0) / n
    sd = np.sqrt(np.sum((values - average) ** 2, axis=0) / (n - 1))
    quantiles = np.quantile(values, [.025, .5, .975], axis=0, method='linear')
    expected = dict(mean=average, median=quantiles[1], sd=sd,
                    p025=quantiles[0], p975=quantiles[2], mcse=sd / np.sqrt(n))
    for key, reference in expected.items():
        np.testing.assert_allclose(rows[key], reference.reshape(-1), rtol=1e-10, atol=1e-12,
                                   err_msg=f'Incorrect {key} in replication summary.')


def check_slope_rows(rows, times, values, windows=WINDOWS):
    """Use OLS solves and replication influence values, independently of gap_slopes."""
    if len(rows) != len(windows) or set(rows.window) != set(windows):
        raise ValueError('Slope table must contain every predeclared window exactly once.')
    times, values = np.asarray(times), np.asarray(values)
    deterministic = values.ndim == 1
    for window, minimum in windows.items():
        row = rows[rows.window == window].iloc[0]
        keep = times >= minimum
        count = int(keep.sum())
        if int(row.observations) != count or float(row.T_min) != minimum:
            raise ValueError('Slope window has incorrect horizon membership.')
        if 'T_max' in rows and int(row.T_max) != int(times[-1]):
            raise ValueError('Slope window has incorrect maximum horizon.')
        if count < 3:
            if bool(row.available):
                raise ValueError('An unavailable slope window was reported as available.')
            continue
        if not deterministic and not bool(row.available):
            raise ValueError('A required slope window is missing.')
        x = np.column_stack([np.ones(count), np.log(times[keep])])
        observations = values[keep] if deterministic else values[:, keep]
        average = observations if deterministic else observations.mean(axis=0)
        slope = np.linalg.lstsq(x, np.log(average), rcond=None)[0][1]
        np.testing.assert_allclose(row.slope, slope, rtol=1e-10, atol=1e-12)
        if deterministic:
            continue
        n = len(observations)
        weights = np.linalg.pinv(x)[1]
        influences = (observations - average) @ (weights / average)
        delta = np.linalg.norm(influences) / np.sqrt(n * (n - 1))
        leave = (observations.sum(axis=0) - observations) / (n - 1)
        leave_slopes = np.linalg.lstsq(x, np.log(leave).T, rcond=None)[0][1]
        jackknife = np.sqrt((n - 1) / n * np.sum((leave_slopes - leave_slopes.mean()) ** 2))
        for key, reference in dict(MCSE_delta=delta, MCSE_jackknife=jackknife,
                CI_low=slope - ndtri(.975) * delta, CI_high=slope + ndtri(.975) * delta).items():
            np.testing.assert_allclose(row[key], reference, rtol=1e-9, atol=1e-12,
                                       err_msg=f'Incorrect {key} for window {window}.')


def check_penalty_rows(rows, values, times, penalties):
    rows = rows.sort_values(['T', 'penalty_index'])
    nt, candidates = len(times), values.shape[2]
    if rows.duplicated(['T', 'penalty_index']).any():
        raise ValueError('Duplicated horizon/penalty summary row.')
    np.testing.assert_array_equal(rows['T'], np.repeat(times, candidates))
    np.testing.assert_array_equal(rows.penalty_index, np.tile(np.arange(candidates), nt))
    np.testing.assert_array_equal(rows.replications, len(values))
    np.testing.assert_array_equal(rows.exact_theory_choice, rows.penalty_index == candidates - 1)
    np.testing.assert_allclose(rows.penalty, penalties.reshape(-1), rtol=1e-12, atol=0)
    check_statistics(rows, values)


def verify():
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    full = pd.read_csv(OUTPUT / 'full_penalty_distributions.csv')
    theory = pd.read_csv(OUTPUT / 'theory_distributions.csv')
    slopes = pd.read_csv(OUTPUT / 'slopes_all_windows.csv')
    common_slopes = pd.read_csv(OUTPUT / 'slopes_robustness_common_grid.csv')
    paired = pd.read_csv(OUTPUT / 'paired_robustness.csv')
    if (set(full.environment) != set(ENVIRONMENTS) or set(full.metric) != set(METRICS)
            or set(theory.environment) != set(ENVIRONMENTS)
            or set(theory.metric) != set(METRICS + DECOMPOSITION)):
        raise ValueError('Distribution tables contain missing or unexpected environments/metrics.')
    expected_quantities = {'annual_gap', 'empirical_complexity', 'relative_complexity',
                           'population_complexity', 'population_relative_complexity'}
    for table, grid in [(slopes, 'environment_full'), (common_slopes, 'robustness_common')]:
        if (set(table.environment) != set(ENVIRONMENTS) or set(table.quantity) != expected_quantities
                or set(table.grid) != {grid}):
            raise ValueError('Slope table has missing or unexpected environments, quantities or grids.')
    datasets, horizons = {}, {}
    for name in ENVIRONMENTS:
        with np.load(OUTPUT / 'distributions' / f'{name}.npz') as z:
            data = {key: z[key] for key in z.files}
        times = data['T']
        if int(data['replications']) != 300 or str(data['run_hash']) != protocol['run_hash']:
            raise ValueError('Summary dataset does not belong to complete frozen production.')
        penalties = np.column_stack([np.tile(protocol['penalties'], (len(times), 1)),
                                     protocol['a'][name] * times.astype(float) ** (-.6)])
        datasets[name], horizons[name] = data, times
        for metric in METRICS:
            values = data[metric]
            if values.shape != (300, len(times), 97):
                raise ValueError('Missing full-penalty replication outcomes.')
            rows = full[(full.environment == name) & (full.metric == metric)]
            check_penalty_rows(rows, values, times, penalties)
            selected = theory[(theory.environment == name) & (theory.metric == metric)].sort_values('T')
            np.testing.assert_array_equal(selected['T'], times)
            np.testing.assert_array_equal(selected.penalty_index, 96)
            np.testing.assert_array_equal(selected.replications, 300)
            np.testing.assert_array_equal(selected.exact_theory_choice, True)
            np.testing.assert_allclose(selected.penalty, penalties[:, -1], rtol=1e-12, atol=0)
            check_statistics(selected, values[:, :, -1])
        for metric in DECOMPOSITION:
            selected = theory[(theory.environment == name) & (theory.metric == metric)].sort_values('T')
            np.testing.assert_array_equal(selected['T'], times)
            np.testing.assert_array_equal(selected.replications, 300)
            np.testing.assert_array_equal(selected.penalty_index, 96)
            np.testing.assert_array_equal(selected.exact_theory_choice, True)
            np.testing.assert_allclose(selected.penalty, penalties[:, -1], rtol=1e-12, atol=0)
            check_statistics(selected, data['theory_' + metric])
        for metric in ('annual_gap', 'empirical_complexity', 'relative_complexity'):
            rows = slopes[(slopes.environment == name) & (slopes.quantity == metric)]
            check_slope_rows(rows, times, data[metric][:, :, -1])
            keep = np.isin(times, protocol['robustness_T'])
            np.testing.assert_array_equal(times[keep], protocol['robustness_T'])
            rows = common_slopes[(common_slopes.environment == name) & (common_slopes.quantity == metric)]
            check_slope_rows(rows, times[keep], data[metric][:, keep, -1])
        with np.load(OUTPUT / 'population' / f'{name}_theory.npz') as pop:
            for metric, values in [('complexity', pop['complexity']),
                                   ('relative_complexity', pop['complexity'] / times)]:
                rows = slopes[(slopes.environment == name) & (slopes.quantity == 'population_' + metric)]
                check_slope_rows(rows, times, values)
                rows = common_slopes[(common_slopes.environment == name) &
                                     (common_slopes.quantity == 'population_' + metric)]
                check_slope_rows(rows, times[keep], values[keep])
    others = set(ENVIRONMENTS) - {'baseline'}
    if (set(paired.environment) != others or set(paired.reference) != {'baseline'}
            or set(paired.metric) != {'annual_SR', 'annual_gap', 'empirical_complexity'}):
        raise ValueError('Paired table has missing or unexpected comparisons.')
    for name in others:
        times = horizons[name]
        positions = np.searchsorted(horizons['baseline'], times)
        np.testing.assert_array_equal(horizons['baseline'][positions], times)
        for metric in ('annual_SR', 'annual_gap', 'empirical_complexity'):
            rows = paired[(paired.environment == name) & (paired.metric == metric)].sort_values('T')
            np.testing.assert_array_equal(rows['T'], times)
            difference = datasets[name][metric][:, :, -1] - datasets['baseline'][metric][:, positions, -1]
            check_statistics(rows, difference)
    result = dict(passed=True, run_hash=protocol['run_hash'], verified_utc=utc_now(),
        full_penalty_rows=len(full), theory_rows=len(theory), paired_rows=len(paired), slope_rows=len(slopes),
        common_grid_slope_rows=len(common_slopes),
        statistics='All means, medians, sample SDs, linear 2.5/97.5 percentiles and mean MCSEs independently recomputed.',
        slopes='All predeclared windows, OLS slopes, full-cross-T influence MCSEs, leave-one-path jackknife MCSEs and Monte Carlo intervals independently recomputed.')
    json_write(OUTPUT / 'summary_integrity.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
