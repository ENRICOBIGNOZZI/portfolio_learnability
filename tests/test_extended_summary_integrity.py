"""Audit tests use synthetic statistics in temporary directories, never study outputs."""
import json

import numpy as np
import pandas as pd
import pytest

from simulations.extended import verify_summaries as audit
from simulations.extended.design import ENVIRONMENTS, T_REQUIRED, T_ROBUSTNESS, PENALTIES
from simulations.extended.statistics import distribution, gap_slopes, deterministic_slopes


def penalty_table(values, times, penalties):
    candidates = values.shape[2]
    return pd.DataFrame(dict(T=np.repeat(times, candidates),
        penalty_index=np.tile(np.arange(candidates), len(times)),
        penalty=penalties.reshape(-1), replications=len(values),
        exact_theory_choice=np.tile(np.arange(candidates) == candidates - 1, len(times)),
        **{key: value.reshape(-1) for key, value in distribution(values).items()}))


def test_diagnostic_percentile_corruption_is_rejected():
    times = np.array([60, 240, 1440])
    values = np.random.default_rng(42).uniform(.1, 2., (12, 3, 4))
    penalties = np.tile([.01, .1, 1., .04], (3, 1))
    rows = penalty_table(values, times, penalties)
    audit.check_penalty_rows(rows, values, times, penalties)
    rows.loc[0, 'p025'] += .1  # Diagnostic index zero, not the theory choice.
    with pytest.raises(AssertionError):
        audit.check_penalty_rows(rows, values, times, penalties)


def test_diagonal_only_slope_mcse_is_rejected():
    times = np.array([60, 240, 720, 1440, 2160, 3240])
    rng = np.random.default_rng(12)
    values = (times / 60.) ** (-.6) * np.exp(
        .1 * rng.normal(size=(20, 1)) + .05 * rng.normal(size=(20, 1)) * np.log(times / 60.))
    rows = pd.DataFrame(gap_slopes(times, values))
    audit.check_slope_rows(rows, times, values)
    x = np.log(times)
    gradient = (x - x.mean()) / np.sum((x - x.mean()) ** 2) / values.mean(axis=0)
    wrong = np.sqrt(np.sum(gradient ** 2 * values.var(axis=0, ddof=1) / len(values)))
    assert not np.isclose(wrong, rows.loc[0, 'MCSE_delta'])
    rows.loc[0, 'MCSE_delta'] = wrong
    with pytest.raises(AssertionError):
        audit.check_slope_rows(rows, times, values)


def test_all_complete_table_shapes_and_paired_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'OUTPUT', tmp_path)
    (tmp_path / 'distributions').mkdir()
    (tmp_path / 'population').mkdir()
    constants = {name: .0001 for name in ENVIRONMENTS}
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(run_hash='synthetic-audit-fixture',
        penalties=PENALTIES.tolist(), a=constants)))
    full, theory, slopes, datasets = [], [], [], {}
    rng = np.random.default_rng(7)
    base = rng.uniform(.8, 1.2, (300, len(T_REQUIRED), 97))
    for number, name in enumerate(ENVIRONMENTS):
        times = np.array(T_REQUIRED if name == 'baseline' else T_ROBUSTNESS)
        values = base[:, :len(times)] * (1 + number * .01)
        data = {metric: values * (i + 1) for i, metric in enumerate(audit.METRICS)}
        data.update({'theory_' + metric: values[:, :, -1] * (i + 1)
                     for i, metric in enumerate(audit.DECOMPOSITION)})
        np.savez(tmp_path / 'distributions' / f'{name}.npz', T=times, replications=300,
                 run_hash='synthetic-audit-fixture', **data)
        penalties = np.column_stack([np.tile(PENALTIES, (len(times), 1)),
                                     constants[name] * times.astype(float) ** (-.6)])
        for metric in audit.METRICS:
            table = penalty_table(data[metric], times, penalties).assign(environment=name, metric=metric)
            full.append(table)
            theory.append(table[table.exact_theory_choice])
        for metric in audit.DECOMPOSITION:
            theory.append(pd.DataFrame(dict(environment=name, metric=metric, T=times,
                penalty_index=96, penalty=penalties[:, -1], replications=300, exact_theory_choice=True,
                **distribution(data['theory_' + metric]))))
        for metric in ('annual_gap', 'empirical_complexity', 'relative_complexity'):
            slopes.extend(dict(environment=name, quantity=metric, **row)
                          for row in gap_slopes(times, data[metric][:, :, -1]))
        complexity = (times / 60.) ** .38
        np.savez(tmp_path / 'population' / f'{name}_theory.npz', complexity=complexity)
        for metric, y in [('complexity', complexity), ('relative_complexity', complexity / times)]:
            slopes.extend(dict(environment=name, quantity='population_' + metric, **row)
                          for row in deterministic_slopes(times, y))
        datasets[name] = data
    paired = []
    for name in set(ENVIRONMENTS) - {'baseline'}:
        for metric in ('annual_SR', 'annual_gap', 'empirical_complexity'):
            difference = datasets[name][metric][:, :, -1] - datasets['baseline'][metric][:, :len(T_ROBUSTNESS), -1]
            paired.append(pd.DataFrame(dict(environment=name, reference='baseline', metric=metric,
                                            T=T_ROBUSTNESS, **distribution(difference))))
    pd.concat(full).to_csv(tmp_path / 'full_penalty_distributions.csv', index=False)
    pd.concat(theory).to_csv(tmp_path / 'theory_distributions.csv', index=False)
    pd.DataFrame(slopes).to_csv(tmp_path / 'slopes_all_windows.csv', index=False)
    pd.concat(paired).to_csv(tmp_path / 'paired_robustness.csv', index=False)
    result = audit.verify()
    assert result['full_penalty_rows'] == 35502
    assert result['theory_rows'] == 549
    assert result['paired_rows'] == 144
    assert result['slope_rows'] == 100
    # A plausible but wrong paired difference must fail even when all marginal
    # distributions and every slope still pass.
    table = pd.read_csv(tmp_path / 'paired_robustness.csv')
    table.loc[0, 'mean'] += .01
    table.to_csv(tmp_path / 'paired_robustness.csv', index=False)
    with pytest.raises(AssertionError):
        audit.verify()
