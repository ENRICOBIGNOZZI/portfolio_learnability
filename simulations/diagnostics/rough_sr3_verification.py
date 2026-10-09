"""Audit the economic SR3 calibration and the presentation-unit conversion."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.provenance import file_hash, json_write


def verify(output, baseline=None):
    output = Path(output)
    protocol = json.loads((output/'protocol.json').read_text())
    population = json.loads((output/'population.json').read_text())
    assert protocol['target_annual_sr'] == 3 and protocol['periods_per_year'] == 12
    np.testing.assert_allclose(population['reference_sharpe']*np.sqrt(12),3,rtol=1e-14)
    for name,sha in protocol['source_hashes'].items():
        assert file_hash(name)==sha, name
    checked_columns = []
    for name in ('methods_annualized.csv','curves_annualized.csv'):
        table = pd.read_csv(output/name)
        for column in table:
            if column.endswith('_annualized'):
                raw = column.removesuffix('_annualized')
                np.testing.assert_allclose(table[column],np.sqrt(12)*table[raw],atol=1e-13,rtol=1e-12)
                checked_columns.append(name+':'+column)
    paths = pd.read_csv(output/'path_methods_annualized.csv')
    assert len(paths)==1000 and paths.replication.nunique()==100
    np.testing.assert_allclose(paths.population_sharpe_annualized,np.sqrt(12)*paths.population_sharpe,atol=1e-13,rtol=1e-12)
    np.testing.assert_allclose(paths.sharpe_gap_annualized,3-paths.population_sharpe_annualized,atol=1e-13,rtol=1e-12)
    saved = json.loads((output/'verification.json').read_text())
    assert len(saved['raw_checkpoint_sha256'])==100
    for name,sha in saved['raw_checkpoint_sha256'].items():
        assert file_hash(output/name)==sha, name
    result = {'annual_target':3.,'monthly_reference':population['reference_sharpe'],
              'periods_per_year':12,'verified_checkpoint_hashes':100,
              'annualized_columns_verified':checked_columns,
              'source_hashes_match_frozen_protocol':True,'validator_sha256':file_hash(__file__)}
    if baseline is not None:
        baseline = Path(baseline)
        changed = 0
        for index in range(100):
            with np.load(output/f'replications/{index:04d}.npz') as new, np.load(baseline/f'replications/{index:04d}.npz') as old:
                assert int(new['train_seed'])==int(old['train_seed'])
                changed += int(str(new['stock_return_sha256'])!=str(old['stock_return_sha256']))
        assert changed==100
        with np.load(output/'population.npz') as new, np.load(baseline/'population.npz') as old:
            second_error = float(np.max(np.abs(new['second']-old['second'])))
            mean_error = float(np.max(np.abs(new['mean']-protocol['factor_mean_scale']*old['mean'])))
            assert second_error < 1e-16 and mean_error < 1e-14
        result.update(changed_stock_return_hashes=changed, baseline_protocol_sha256=file_hash(baseline/'protocol.json'),
            population_second_moment_difference=second_error,
            population_mean_scaling_error=mean_error,
            explanation='Same innovations, different economic returns; population second moments invariant because E[FF\u2032]=I. Old operators are read only for this post-run audit, not for production or tuning.')
    json_write(output/'annualization_verification.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('simulations/outputs/rich6d_rough_sr3_annual_monthly_v1'))
    parser.add_argument('--baseline',type=Path,default=Path('simulations/outputs/rich6d_rough_r1_v1'))
    args=parser.parse_args()
    print(json.dumps(verify(args.output,args.baseline),indent=2))
