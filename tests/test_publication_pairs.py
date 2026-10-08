"""Publication comparisons must preserve path pairing and reject stale numbers."""
import numpy as np
import pandas as pd
import pytest

from simulations.diagnostics.publication_v2 import verify_paired_comparisons
from simulations.plotting.confirmation import paired_comparisons


def test_generated_paired_figures_and_table_are_independently_checked(tmp_path):
    (tmp_path/'figures').mkdir()
    for name, map_shift in [('baseline_original', 0.), ('rich6d', .002)]:
        directory = tmp_path/'data'/name
        directory.mkdir(parents=True)
        rows = []
        for T in (60, 120):
            for r in range(10):
                for method, shift in [('holdout25', 0.), ('rolling3', .001), ('theory_1', -.001)]:
                    value = .01+(r+1)*(shift+map_shift)
                    rows.append({'T': T, 'replication': r, 'method': method,
                                 'regret': value, 'oos_loss': 1+value, 'forward_loss': 1+2*value,
                                 'population_sharpe': .1-value})
        pd.DataFrame(rows).to_csv(directory/'path_methods.csv', index=False)
    paired_comparisons(tmp_path)
    assert verify_paired_comparisons(tmp_path) == 40
    path = tmp_path/'tables/paired_comparisons.csv'
    frame = pd.read_csv(path)
    row = frame[(frame.comparison == 'rich6d_minus_baseline')&(frame.method == 'holdout25')
                &(frame['T'] == 60)&(frame.metric == 'regret')].iloc[0]
    np.testing.assert_allclose(row['mean'], .011)
    np.testing.assert_allclose(row.mcse, .002*np.sqrt(82.5/90))
    frame.loc[0, 'mcse'] *= 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Paired comparison/mcse'):
        verify_paired_comparisons(tmp_path)
