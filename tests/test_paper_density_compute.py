import numpy as np
import pytest

from simulations.model import fit_path
from simulations.paper_density_compute import exact_ridge_complexity


@pytest.mark.parametrize('shape',[(40,12),(12,40),(12,12)])
@pytest.mark.parametrize('penalty',[1e-10,1e-7])
def test_exact_cholesky_ridge_and_trace_match_full_spectral_fit(shape,penalty):
    rng=np.random.default_rng(551021)
    G=(rng.normal(size=(shape[0],3))@rng.normal(size=(3,shape[1])))*1e-3
    G+=rng.normal(size=shape)*1e-6
    expected,C,_=fit_path(G,[penalty]);actual,trace=exact_ridge_complexity(G,penalty)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(trace,C,rtol=1e-8,atol=1e-9)


def test_exact_cholesky_rejects_nonpositive_penalty():
    with pytest.raises(ValueError):exact_ridge_complexity(np.eye(3),0.)
