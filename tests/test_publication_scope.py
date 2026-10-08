"""An accurate publication must not silently omit failed numerical regions."""
import json

import pandas as pd
import pytest

from simulations.diagnostics import publication_v2
from simulations.plotting.tables import emit
from simulations.provenance import file_hash


def test_unresolved_region_cannot_be_omitted_even_with_consistent_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(publication_v2, 'verify_paired_comparisons', lambda output: 0)
    (tmp_path/'audit').mkdir()
    (tmp_path/'paper').mkdir()
    (tmp_path/'paper/simulation_numbers.tex').write_text('')
    for environment, passed in [('baseline_original', False), ('rich6d', True)]:
        pd.DataFrame([{'T': 60, 'method': 'holdout25', 'floor_fraction': .01,
                       'difference_confidence_fraction': .1 if not passed else .01,
                       'certified': passed}]).to_csv(tmp_path/'audit'/f'{environment}_rank_policy_resolution.csv', index=False)
        pd.DataFrame([{'T': 60, 'method': 'holdout25', 'certified': True}]).to_csv(
            tmp_path/'audit'/f'{environment}_quadrature_resolution.csv', index=False)
    filename = 'tables/confirmation_unresolved_1.csv'
    emit(pd.DataFrame([{'Map': 'Baseline', 'Method': 'holdout25', 'T': 60,
                       'Floor / risk': .01, 'Rank bound / risk': .1,
                       'Rank pass': False, 'Quadrature pass': True}]),
         tmp_path/'tables', 'confirmation_unresolved_1', 'Fixture only', 'tab:fixture')
    cells = []
    for column, source, field in [
            ('Method', 'rank_policy', 'method'), ('T', 'rank_policy', 'T'),
            ('Floor / risk', 'rank_policy', 'floor_fraction'),
            ('Rank bound / risk', 'rank_policy', 'difference_confidence_fraction'),
            ('Rank pass', 'rank_policy', 'certified'), ('Quadrature pass', 'quadrature', 'certified')]:
        cells.append({'kind': 'table', 'table': filename, 'row': 0, 'target_column': column,
                      'file': f'audit/baseline_original_{source}_resolution.csv',
                      'filters': {'T': 60, 'method': 'holdout25'}, 'column': field})
    for environment in ('baseline_original', 'rich6d'):
        file = f'audit/{environment}_rank_resolution.csv'
        curve = pd.DataFrame([
            {'T': 60, 'lambda_index': 0, 'lambda': .001, 'regret512': .1, 'regret1024': .2,
             'complexity512': 59., 'complexity1024': 59.5, 'payoff_squared_difference_512_1024': .03, 'certified': False},
            {'T': 60, 'lambda_index': 1, 'lambda': .1, 'regret512': .01, 'regret1024': .01,
             'complexity512': 2., 'complexity1024': 2., 'payoff_squared_difference_512_1024': .0001, 'certified': True}])
        curve.to_csv(tmp_path/file, index=False)
        columns = {'T': 'T', 'lambda': 'lambda', 'Max regret512': 'regret512',
                   'Same lambda regret1024': 'regret1024', 'C512': 'complexity512', 'C1024': 'complexity1024',
                   'Payoff error': 'payoff_squared_difference_512_1024', 'Rank pass': 'certified'}
        name = 'confirmation_curve_maximum_'+environment
        emit(pd.DataFrame([{target: curve.iloc[0][field] for target, field in columns.items()}]),
             tmp_path/'tables', name, 'Fixture only', 'tab:'+name)
        for target, field in columns.items():
            cells.append({'kind': 'table', 'table': f'tables/{name}.csv', 'row': 0,
                          'target_column': target, 'file': file, 'filters': {'T': 60, 'lambda_index': 0},
                          'column': field})
    lineage = {'cells': cells,
               'source_hashes': {p['file']: file_hash(tmp_path/p['file']) for p in cells},
               'generated_hashes': {str(p.relative_to(tmp_path)): file_hash(p)
                                    for p in (tmp_path/'tables').glob('*')}}
    path = tmp_path/'paper/number_lineage.json'
    path.write_text(json.dumps(lineage))
    result = publication_v2.verify(tmp_path)
    assert result['unresolved_policy_cells_reported'] == 1
    assert result['maximum_loss_cells_reported'] == 2
    lineage['cells'] = [cell for cell in cells if 'curve_maximum' not in cell['table']]
    path.write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match='Missing map in maximum-loss'):
        publication_v2.verify(tmp_path)
    # Simulate a faulty producer omitting the entire table and its lineage.
    # Its remaining hashes remain consistent, so only the independent scope
    # check can detect this omission.
    lineage['cells'] = []
    path.write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match='Unresolved numerical regions omitted'):
        publication_v2.verify(tmp_path)
