"""A seed-level quadrature interval must reject table and cohort mutations."""
import numpy as np
import pandas as pd
import pytest

from simulations.diagnostics.reconstruct_resolution import quadrature_summary


def fixture_tables():
    paths = pd.DataFrame([
        {'replication': r, 'T': 60, 'method': 'theory_1', 'seed_index': s,
         'groups': groups, 'regret': 1+r*.0001+(.001*(s+1) if groups == 8192 else 0),
         'population_sharpe': .03}
        for r in range(50) for s in range(4) for groups in (8192, 32768)])
    # Four independent differences .001, .002, .003, .004; 50 common paths
    # do not multiply the number of independent quadrature realizations.
    se = np.sqrt(5e-6/12)
    result = pd.DataFrame([{'environment': 'test', 'T': 60, 'method': 'theory_1',
                           'paths': 50, 'independent_seeds': 4,
                           'mean_regret32768': 1.00245,
                           'difference8192_minus32768': .0025,
                           'quadrature_seed_MCSE': se, 'ci_low': .0025-1.96*se,
                           'ci_high': .0025+1.96*se, 'certified': True}])
    return paths, result


def test_independent_quadrature_interval():
    paths, result = fixture_tables()
    quadrature_summary(paths, result, 'test')


@pytest.mark.parametrize('mutation', ['mcse', 'confidence', 'certificate', 'missing', 'duplicate', 'nonfinite'])
def test_mutated_quadrature_evidence_is_rejected(mutation):
    paths, result = fixture_tables()
    if mutation == 'mcse':
        result.loc[0, 'quadrature_seed_MCSE'] /= np.sqrt(50)
    elif mutation == 'confidence':
        result.loc[0, 'ci_high'] += .01
    elif mutation == 'certificate':
        result.loc[0, 'certified'] = False
    elif mutation == 'missing':
        paths = paths.iloc[1:]
    elif mutation == 'duplicate':
        paths = pd.concat([paths, paths.iloc[:1]], ignore_index=True)
    elif mutation == 'nonfinite':
        paths.loc[0, 'regret'] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        quadrature_summary(paths, result, 'test')
