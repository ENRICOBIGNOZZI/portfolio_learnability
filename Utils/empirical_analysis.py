"""Shared analysis utilities for the paper's empirical figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.utils import _annualized_sharpe, _response_one_loss


def load_result_bundle(result_dir, *, load_models=True):
    """Load the standard files written by ``train_model``."""
    result_dir = Path(result_dir)
    bundle = {
        "result_dir": result_dir,
        "diagnostics": pd.read_parquet(result_dir / "lambda_diagnostics.parquet"),
        "monthly": pd.read_parquet(result_dir / "monthly_oos_portfolios.parquet"),
        "selected": pd.read_parquet(result_dir / "selected_oos_portfolio.parquet"),
    }
    if load_models:
        bundle["models"] = np.load(result_dir / "models.npz", allow_pickle=False)
    with (result_dir / "metadata.json").open(encoding="utf-8") as stream:
        bundle["metadata"] = json.load(stream)
    return bundle


def fixed_lambda_curve(bundle):
    """Aggregate the non-overlapping OOS path and rolling in-sample results."""
    diagnostics = bundle["diagnostics"]
    monthly = bundle["monthly"]
    grouped = diagnostics.groupby("lambda", sort=True)
    curve = grouped[
        [
            "in_sample_loss",
            "validation_loss",
            "in_sample_sharpe",
            "validation_sharpe",
            "effective_dimension",
            "train_effective_dimension",
        ]
    ].mean()
    curve["oos_loss_native"] = monthly.groupby("lambda")[
        "portfolio_return_native"
    ].apply(_response_one_loss)
    curve["oos_sharpe_native"] = monthly.groupby("lambda")[
        "portfolio_return_native"
    ].apply(_annualized_sharpe)
    curve["oos_sharpe_capped"] = monthly.groupby("lambda")[
        "portfolio_return"
    ].apply(_annualized_sharpe)
    curve["mean_gross_exposure"] = monthly.groupby("lambda")[
        "sum_absolute_weights"
    ].mean()
    return curve.reset_index()


def selected_point(bundle):
    """Return the chronologically validation-selected portfolio summary."""
    diagnostics = bundle["diagnostics"]
    selected_diagnostics = diagnostics.loc[diagnostics["selected_by_validation"]]
    selected = bundle["selected"].sort_values("date")
    return {
        "effective_dimension": float(selected_diagnostics["effective_dimension"].mean()),
        "oos_loss_native": _response_one_loss(selected["portfolio_return_native"]),
        "oos_sharpe_native": _annualized_sharpe(selected["portfolio_return_native"]),
        "oos_sharpe_capped": _annualized_sharpe(selected["portfolio_return"]),
        "mean_gross_exposure": float(selected["sum_absolute_weights"].mean()),
    }

def maximum_drawdown(returns):
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns).cumprod()
    running_maximum = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_maximum - 1.0
    return float(drawdown.min())


def performance_metrics(returns, *, gross_exposure=None):
    returns = pd.Series(returns, dtype=float).dropna()
    metrics = {
        "annualized_mean": float(12.0 * returns.mean()),
        "annualized_volatility": float(np.sqrt(12.0) * returns.std(ddof=1)),
        "annualized_sharpe": _annualized_sharpe(returns),
        "maximum_drawdown": maximum_drawdown(returns),
        "months": int(len(returns)),
    }
    if gross_exposure is not None:
        metrics["mean_gross_exposure"] = float(pd.Series(gross_exposure).mean())
        metrics["maximum_gross_exposure"] = float(pd.Series(gross_exposure).max())
    return metrics


def rolling_sharpe(returns, window=60):
    returns = pd.Series(returns, dtype=float)
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    rolling_volatility = returns.rolling(window, min_periods=window).std(ddof=1)
    return np.sqrt(12.0) * rolling_mean / rolling_volatility
