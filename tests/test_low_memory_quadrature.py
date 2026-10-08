"""Changing evaluation memory must preserve draws and population moments."""
import numpy as np
import pytest

from simulations.dgp.balanced import DGPParameters
from simulations.estimator.kernel import NystromBasis
from simulations.experiments.population import population_moments
from simulations.diagnostics.low_memory import population_moments_batched


@pytest.mark.parametrize('loading,eta', [('baseline_original', 0.), ('rich6d', .35)])
@pytest.mark.parametrize('variant', ['baseline', 'heteroskedastic'])
def test_batched_matches_canonical_moments(loading, eta, variant):
    p = DGPParameters(loading_map=loading, eta=eta)
    basis = NystromBasis(rank=32, seed=71)
    expected = population_moments(p, basis, groups=257, seed=971, variant=variant)
    # Non-divisors exercise the final partial batch and sequential RNG draws.
    for batch in (1, 17, 256):
        actual = population_moments_batched(p, basis, groups=257, seed=971,
                                            variant=variant, batch_groups=batch)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-15)


def test_cross_sectional_ranks_cannot_use_independent_group_formula():
    with pytest.raises(ValueError, match='independent population groups'):
        population_moments_batched(DGPParameters(), NystromBasis(rank=4),
                                   groups=10, variant='empirical_ranks')
