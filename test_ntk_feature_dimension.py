import numpy as np
import pytest

from Kernels.kernel_function import PortfolioKernel


def test_ntk_1000_means_1000_final_coordinates():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(9, 7))
    kernel = PortfolioKernel(
        kernel="ntk",
        n_random_features=1000,
        random_state=0,
    )
    features = kernel.features(X)

    assert kernel.ntk_hidden_directions == 500
    assert features.shape == (9, 1000)
    assert np.isfinite(features).all()


def test_stationary_and_ntk_nominal_dimensions_match():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(5, 4))

    gaussian = PortfolioKernel(
        kernel="gaussian",
        n_random_features=1000,
        random_state=0,
    )
    ntk = PortfolioKernel(
        kernel="ntk",
        n_random_features=1000,
        random_state=0,
    )

    assert gaussian.features(X).shape[1] == 1000
    assert ntk.features(X).shape[1] == 1000


def test_ntk_requires_even_final_dimension():
    X = np.ones((3, 2))
    kernel = PortfolioKernel(
        kernel="ntk",
        n_random_features=999,
    )
    with pytest.raises(ValueError):
        kernel.features(X)
