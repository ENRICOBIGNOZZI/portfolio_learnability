import numpy as np
import pandas as pd
import pytest

from Utils.empirical_analysis import (
    maximum_drawdown, paired_block_summary, response_one_decomposition,
)
from Utils.paper_figures import realization_dates


def test_response_one_decomposition_and_optimal_scaling():
    rng = np.random.default_rng(23)
    for returns in (rng.normal(.2, .5, 50), np.zeros(50), -np.ones(50)):
        d = response_one_decomposition(returns)
        assert d["native_loss"] == pytest.approx(d["ex_post_scale_minimum_loss"] + d["scale_gap"])
        assert np.mean((1-d["ex_post_scale"]*returns)**2) == pytest.approx(d["ex_post_scale_minimum_loss"])
        assert d["scale_gap"] >= 0


def test_drawdown_counts_first_month_loss_from_initial_capital():
    assert maximum_drawdown([-.2, .1]) == pytest.approx(-.2)
    assert maximum_drawdown([.1, .1]) == pytest.approx(0)


def test_paired_bootstrap_preserves_identical_paths():
    returns = np.random.default_rng(19).normal(.01, .1, 100)
    d = paired_block_summary(returns, returns, n_bootstrap=30)
    np.testing.assert_allclose(d[["estimate", "lower", "upper"]], 0, atol=1e-14)


def test_jkp_formation_dates_map_to_next_month_end():
    dates = realization_dates(pd.DataFrame({"date": ["2020-01-31", "2024-12-31"]}))
    assert dates.tolist() == [pd.Timestamp("2020-02-29"), pd.Timestamp("2025-01-31")]
