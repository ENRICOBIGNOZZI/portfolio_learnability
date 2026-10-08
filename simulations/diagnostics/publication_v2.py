"""Verify numeric cell lineage, formatted TeX tables and manuscript macros."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.provenance import file_hash, json_write, require, utc_now
from simulations.diagnostics.reconstruct import equal, read_table, summary


def verify_paired_comparisons(output):
    output = Path(output)
    actual = read_table(output/'tables/paired_comparisons.csv', ['comparison', 'method', 'T', 'metric'])
    frames = {name: read_table(output/'data'/name/'path_methods.csv', ['T', 'method', 'replication'])
              for name in ('baseline_original', 'rich6d')}
    comparisons = []
    for method in ('holdout25', 'rolling3', 'theory_1'):
        comparisons.append(('rich6d_minus_baseline', method,
                            frames['baseline_original'][frames['baseline_original'].method == method],
                            frames['rich6d'][frames['rich6d'].method == method]))
    for name, frame in frames.items():
        comparisons.append((name+'_rolling_minus_holdout', 'rolling3',
                            frame[frame.method == 'holdout25'], frame[frame.method == 'rolling3']))
    checked = 0
    for comparison, method, left, right in comparisons:
        keys = ['T', 'replication']
        left, right = left.set_index(keys).sort_index(), right.set_index(keys).sort_index()
        require(left.index.equals(right.index), 'Unpaired comparison paths')
        for T in left.index.get_level_values('T').unique():
            for metric in ('regret', 'oos_loss', 'forward_loss', 'population_sharpe'):
                delta = right.loc[T, metric].to_numpy()-left.loc[T, metric].to_numpy()
                mean, _, se = summary(delta)
                rows = actual[(actual.comparison == comparison)&(actual.method == method)
                              &(actual['T'] == T)&(actual.metric == metric)]
                require(len(rows) == 1, 'Missing paired comparison')
                for key, value in {'replications': len(delta), 'mean': mean, 'mcse': se,
                                   'ci_low': mean-1.96*se, 'ci_high': mean+1.96*se}.items():
                    equal(rows.iloc[0][key], value, 'Paired comparison/'+key)
                checked += 1
    require(len(actual) == checked, 'Extra unverified paired comparison')
    return checked


def verify(output):
    output = Path(output)
    paired_rows = verify_paired_comparisons(output)
    lineage = json.loads((output/'paper/number_lineage.json').read_text())
    for name, expected in lineage['source_hashes'].items():
        require(file_hash(output/name) == expected, 'Changed publication input: '+name)
    for name, expected in lineage['generated_hashes'].items():
        require(file_hash(output/name) == expected, 'Changed manuscript source: '+name)
    macros = (output/'paper/simulation_numbers.tex').read_text()
    count = 0
    tables = {}
    for entry in lineage['cells']:
        source = pd.read_csv(output/entry['file'])
        for key, value in entry['filters'].items():
            source = source[source[key] == value]
        require(len(source) == 1, 'Ambiguous source of reported number')
        expected = source.iloc[0][entry['column']]
        if entry['kind'] == 'prose':
            formatted = format(expected, entry['format'])
            require(formatted == entry['rendered'], 'Changed numeric prose value')
            require('\\newcommand{\\'+entry['macro']+'}{'+formatted+'}' in macros, 'Numeric prose macro mismatch')
        else:
            if entry['table'] not in tables:
                tables[entry['table']] = pd.read_csv(output/entry['table'])
            actual = tables[entry['table']].iloc[entry['row']][entry['target_column']]
            equal(actual, expected, 'Publication table cell')
        count += 1
    # Verify every tabular cell at the declared four-significant-digit precision.
    # This does not invoke the production emitter.
    for name, frame in tables.items():
        tex = (output/Path(name).with_suffix('.tex')).read_text()
        body = tex.split('\\begin{tabular}', 1)[1].split('\\end{tabular}', 1)[0]
        data = body.split('\\midrule', 1)[1].split('\\bottomrule', 1)[0]
        actual_rows = [line.strip() for line in data.splitlines() if '&' in line and line.strip().endswith('\\\\')]
        require(len(actual_rows) == len(frame), 'Formatted table row count: '+name)
        def escape(value):
            return str(value).replace('\\', '\\textbackslash ').replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')
        for (_, row), actual in zip(frame.iterrows(), actual_rows):
            cells = [x.strip() for x in actual[:-2].split('&')]
            require(len(cells) == len(row), 'Formatted table column count')
            for raw, formatted in zip(row, cells):
                if isinstance(raw, (float, np.floating)):
                    require(formatted == format(raw, '.4g'), 'Formatted numeric table value')
                else:
                    require(formatted == escape(raw), 'Formatted table label/value')
    # The unresolved-region table must include every failed practical-grid
    # policy cell, not only faithfully reproduce an arbitrary subset of them.
    expected_unresolved = set()
    for environment, label in [('baseline_original', 'Baseline'), ('rich6d', 'Rich6D')]:
        rank = read_table(output/'audit'/f'{environment}_rank_policy_resolution.csv', ['T', 'method'])
        quad = read_table(output/'audit'/f'{environment}_quadrature_resolution.csv', ['T', 'method'])
        for _, row in rank.iterrows():
            other = quad[(quad['T'] == row['T'])&(quad.method == row.method)]
            require(len(other) == 1, 'Missing numerical counterpart in publication scope')
            if not bool(row.certified) or not bool(other.iloc[0].certified):
                expected_unresolved.add((label, row.method, int(row['T'])))
    reported_unresolved = set()
    for name, frame in tables.items():
        if Path(name).stem.startswith('confirmation_unresolved_'):
            for _, row in frame.iterrows():
                key = (row.Map, row.Method, int(row['T']))
                require(key not in reported_unresolved, 'Duplicate unresolved publication cell')
                reported_unresolved.add(key)
    require(reported_unresolved == expected_unresolved, 'Unresolved numerical regions omitted or mislabeled')
    maxima_checked = 0
    for environment in ('baseline_original', 'rich6d'):
        name = f'tables/confirmation_curve_maximum_{environment}.csv'
        require(name in tables, 'Missing map in maximum-loss sensitivity tables')
        frame = tables[name]
        rank = read_table(output/'audit'/f'{environment}_rank_resolution.csv', ['T', 'lambda_index'])
        require(set(frame['T']) == set(rank['T']) and not frame.duplicated(['T']).any(), 'Maximum-loss training grid')
        for _, row in frame.iterrows():
            curve = rank[(rank['T'] == row['T'])&(rank.lambda_index < 96)].sort_values('lambda_index')
            chosen = curve.iloc[int(curve.regret512.to_numpy().argmax())]
            equal(row['lambda'], chosen['lambda'], 'Maximum-loss fixed penalty')
            equal(row['Max regret512'], chosen.regret512, 'Maximum-loss diagnostic')
            maxima_checked += 1
    report = {'schema': 'publication-reconstruction/2.0', 'passed': True, 'completed_utc': utc_now(),
              'source_sha256': file_hash(__file__), 'numeric_lineage_cells': count, 'tables_checked': list(tables),
              'paired_comparison_rows': paired_rows,
              'unresolved_policy_cells_reported': len(reported_unresolved),
              'maximum_loss_cells_reported': maxima_checked,
              'source_hashes': lineage['source_hashes'], 'generated_hashes': lineage['generated_hashes'],
              'scope': 'Every dynamic manuscript number and published table cell; scientific constants are fixed by the frozen protocol.'}
    json_write(output/'audit/publication_reconstruction.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    print(json.dumps(verify(parser.parse_args().output), indent=2))
