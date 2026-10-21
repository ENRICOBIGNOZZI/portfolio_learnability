"""Read and visualize the aggregate A–E sensitivities; no portfolio is retuned."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import digest


def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    tables, figures = out/'tables', out/'figures'
    summary = pd.read_csv(tables/'summary.csv')
    spectrum = pd.read_csv(tables/'economic_spectrum.csv')
    increments = pd.read_csv(tables/'E_incremental_directions.csv')
    candidates = pd.read_csv(tables/'A_candidates.csv')
    match = json.loads((out/'audit/A_match.json').read_text())
    central = summary[summary.scenario == 'missing_total_zero']
    scenario_labels = {'missing_total_minus100': 'Missing total −100%',
                       'missing_total_zero': 'Missing total 0%', 'missing_total_plus100': 'Missing total +100%'}
    colors = ['#167d8d', '#b85f2b', '#6556a5']
    caption = 'Exploratory JKP sensitivities; imposed missing payoffs, not identified population performance'
    generated = []

    def finish(fig, name):
        fig.suptitle(caption, fontsize=9)
        path = figures/(name+'.png')
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(path)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    scatter = ax[0].scatter(candidates.validation_sharpe, candidates.top_decile_signal_share,
                            c=candidates.top_decile_mass, cmap='viridis', s=42)
    fig.colorbar(scatter, ax=ax[0], label='Top-decile spectral mass', shrink=.8)
    ax[0].set(xlabel='Prior 12-month validation Sharpe', ylabel='Prior top-decile signal share',
              title='A: past-only matching candidates')
    pair = match['pair']
    if pair:
        for name in pair:
            row = candidates.set_index('variant').loc[name]
            ax[0].annotate(name, (row.validation_sharpe, row.top_decile_signal_share), xytext=(4, 5), textcoords='offset points')
        for color, scenario in zip(colors, scenario_labels, strict=True):
            a = summary[(summary.experiment == 'A') & (summary.scenario == scenario)].set_index('policy')
            ax[1].plot(pair, [a.loc['matched_'+name, 'raw_Q'] for name in pair], marker='o', color=color,
                       label=scenario_labels[scenario])
        ax[1].legend(fontsize=8)
    else:
        ax[1].text(.5, .5, 'No match at the fixed tolerance', ha='center', transform=ax[1].transAxes)
    ax[1].set(title='A: subsequent learning loss', ylabel='Common-period OOS quadratic loss')
    finish(fig, 'A_matching_and_learning')

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    first, last = int(spectrum.decision_year.min()), int(spectrum.decision_year.max())
    for year, style in [(first, '-'), (last, '--')]:
        s = spectrum[(spectrum.scenario == 'missing_total_zero') & (spectrum.decision_year == year)].sort_values('rank')
        ranks = s['rank']/s['rank'].max()
        ax[0].plot(ranks, s.eigenvalue.cumsum()/s.eigenvalue.sum(), color=colors[0], linestyle=style, label=f'{year}: spectral mass')
        ax[0].plot(ranks, s.historical_signal.cumsum()/s.historical_signal.sum(), color=colors[1], linestyle=style, label=f'{year}: historical signal')
    ax[0].set(xlabel='Fraction of past economic directions', ylabel='Cumulative share',
              title='E: variance concentration versus signal alignment', ylim=(0, 1.03))
    ax[0].legend(fontsize=7)
    for color, scenario in zip(colors, scenario_labels, strict=True):
        e = increments[increments.scenario == scenario].sort_values('through_fraction')
        ax[1].plot(e.through_fraction, e.delta_OOS_loss, marker='o', color=color, label=scenario_labels[scenario])
    ax[1].axhline(0, color='.4', linewidth=.8)
    ax[1].set(xlabel='Retained fraction of economic directions', ylabel='Incremental OOS loss (negative improves)',
              title='E: economic value of adding directions')
    ax[1].legend(fontsize=7)
    finish(fig, 'E_spectrum_and_incremental_value')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    e = summary[(summary.experiment == 'E') & summary.complete_common_period].copy()
    order = ['ridge_all', 'drop_value', 'drop_profitability', 'drop_momentum',
             'drop_random0', 'drop_random1', 'drop_random2', 'neutral_market', 'neutral_six', 'PCR_validated']
    for color, scenario in zip(colors, scenario_labels, strict=True):
        s = e[e.scenario == scenario].set_index('policy').reindex(order)
        baseline = s.loc['ridge_all', 'raw_Q']
        ax[0].plot(s.raw_Q-baseline, np.arange(len(order)), marker='o', color=color, label=scenario_labels[scenario])
        ax[1].plot(s.net_sharpe, np.arange(len(order)), marker='o', color=color)
    ax[0].axvline(0, color='.4', linewidth=.8)
    for axis in ax:
        axis.set_yticks(np.arange(len(order)), [name.replace('_', ' ') for name in order])
        axis.invert_yaxis()
    ax[0].set(xlabel='OOS loss minus full ridge loss', title='E: characteristic ablations and neutralization')
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Net annualized Sharpe', title='Illustrative trading and short-borrow costs')
    finish(fig, 'E_ablations_and_net_returns')

    # The full cross-scenario table is the evidence; no best surface cell is promoted to a selector.
    selected = summary[(summary.experiment.isin(['A', 'C', 'E'])) & summary.complete_common_period]
    selected.to_csv(tables/'selected_comparisons.csv', index=False, float_format='%.11g')
    records = []
    for scenario in scenario_labels:
        b = summary[(summary.scenario == scenario) & (summary.experiment == 'B')].set_index('policy')
        e = summary[(summary.scenario == scenario) & (summary.experiment == 'E')].set_index('policy')
        c = summary[(summary.scenario == scenario) & (summary.experiment == 'C')].set_index('policy')
        records.append(dict(scenario=scenario, B_full_T360_minus_T36_Q=b.loc['Nall_T360', 'raw_Q']-b.loc['Nall_T36', 'raw_Q'],
            B_N3000_minus_N100_T240_Q=b.loc['N3000_T240', 'raw_Q']-b.loc['N100_T240', 'raw_Q'],
            C_G130_minus_L130_Q=c.loc['G130', 'raw_Q']-c.loc['L130', 'raw_Q'],
            E_neutral_market_minus_ridge_Q=e.loc['neutral_market', 'raw_Q']-e.loc['ridge_all', 'raw_Q'],
            E_neutral_six_minus_ridge_Q=e.loc['neutral_six', 'raw_Q']-e.loc['ridge_all', 'raw_Q'],
            E_PCR_selected_minus_ridge_Q=e.loc['PCR_validated', 'raw_Q']-e.loc['ridge_all', 'raw_Q']))
    contrasts = pd.DataFrame(records)
    contrasts.to_csv(tables/'predefined_contrasts.csv', index=False, float_format='%.11g')
    lines = ['EXPLORATORY A–E EVIDENCE', '', caption,
             'Every contrast is the first named policy loss minus the policy after "minus" in the CSV column name; negative favors the first named policy.',
             'These are descriptive comparisons, without a sampling-uncertainty or population-optimality claim.', '',
             'Characteristic ablations change available policies, not the underlying financial economy.', '',
             'Random E ablations remove 32 characteristics, and are not cardinality-matched controls for the smaller economic families.', '',
             contrasts.to_string(index=False, float_format=lambda x: f'{x:.5g}'), '',
             'Complete-period net path statuses:', summary.groupby(['scenario', 'net_status'], observed=True).size().to_string(), '',
             'Common-period status counts:', summary.groupby(['scenario', 'complete_common_period'], observed=True).size().to_string(), '',
             'A prior matching audit:', json.dumps(match, indent=2), '',
             'REPORT.txt contains source identification limits, chronology, accounting, and the experiment definitions.']
    evidence = out/'EVIDENCE.txt'
    evidence.write_text('\n'.join(lines)+'\n')
    generated += [tables/'selected_comparisons.csv', tables/'predefined_contrasts.csv', evidence]
    write = dict(code_sha256=digest(Path(__file__)), source_execution_sha256=digest(out/'audit/execution.json'),
                 inputs={path.name: digest(path) for path in [tables/'summary.csv', tables/'economic_spectrum.csv', tables/'E_incremental_directions.csv', tables/'A_candidates.csv']},
                 outputs={str(path.relative_to(out)): digest(path) for path in generated})
    (out/'audit/analysis.json').write_text(json.dumps(write, indent=2)+'\n')
    print(contrasts.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/real_data_ABCDE_20261021')
    analyze(parser.parse_args().out)
