"""Shared analysis utilities for the paper's empirical figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.utils import _annualized_sharpe, _response_one_loss
from Utils.utils import train_model


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


def fixed_rho_curve(bundle):
    """Aggregate the non-overlapping OOS path and rolling in-sample results."""
    diagnostics = bundle["diagnostics"]
    monthly = bundle["monthly"]
    grouped = diagnostics.groupby("rho", sort=True)
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
    curve["oos_loss_native"] = monthly.groupby("rho")[
        "portfolio_return_native"
    ].apply(_response_one_loss)
    curve["oos_sharpe_native"] = monthly.groupby("rho")[
        "portfolio_return_native"
    ].apply(_annualized_sharpe)
    curve["oos_sharpe_capped"] = monthly.groupby("rho")[
        "portfolio_return"
    ].apply(_annualized_sharpe)
    curve["mean_turnover"] = monthly.groupby("rho")["turnover"].mean()
    curve["mean_gross_exposure"] = monthly.groupby("rho")[
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
        "mean_turnover": float(selected["turnover"].mean()),
        "mean_gross_exposure": float(selected["sum_absolute_weights"].mean()),
    }


def _circular_block_indices(n_observations, block_length, rng):
    n_blocks = int(np.ceil(n_observations / block_length))
    starts = rng.integers(0, n_observations, size=n_blocks)
    indices = np.concatenate(
        [
            (start + np.arange(block_length, dtype=int)) % n_observations
            for start in starts
        ]
    )
    return indices[:n_observations]


def block_bootstrap_performance(
    monthly,
    *,
    n_bootstrap=500,
    block_length=12,
    random_state=0,
):
    """Circular block-bootstrap bands for the fixed-rho OOS curves."""
    native = monthly.pivot(
        index="date", columns="rho", values="portfolio_return_native"
    ).sort_index(axis=1)
    capped = monthly.pivot(
        index="date", columns="rho", values="portfolio_return"
    ).reindex(columns=native.columns)
    if native.isna().any().any() or capped.isna().any().any():
        raise ValueError("The OOS lambda path must form a complete date-by-rho panel.")

    native_values = native.to_numpy(dtype=float)
    capped_values = capped.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    losses = np.empty((n_bootstrap, native_values.shape[1]), dtype=float)
    sharpes = np.empty_like(losses)
    for bootstrap_index in range(n_bootstrap):
        indices = _circular_block_indices(
            len(native_values), block_length, rng
        )
        sampled_native = native_values[indices]
        sampled_capped = capped_values[indices]
        losses[bootstrap_index] = np.mean((1.0 - sampled_native) ** 2, axis=0)
        mean = sampled_capped.mean(axis=0)
        volatility = sampled_capped.std(axis=0, ddof=1)
        sharpes[bootstrap_index] = np.divide(
            np.sqrt(12.0) * mean,
            volatility,
            out=np.full_like(mean, np.nan),
            where=volatility > 1e-12,
        )

    return pd.DataFrame(
        {
            "rho": native.columns.to_numpy(dtype=float),
            "loss_lower": np.nanquantile(losses, 0.025, axis=0),
            "loss_upper": np.nanquantile(losses, 0.975, axis=0),
            "sharpe_lower": np.nanquantile(sharpes, 0.025, axis=0),
            "sharpe_upper": np.nanquantile(sharpes, 0.975, axis=0),
        }
    )


def maximum_drawdown(returns):
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return np.nan
    wealth = (1.0 + returns).cumprod()
    running_maximum = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_maximum - 1.0
    return float(drawdown.min())


def performance_metrics(returns, *, turnover=None, gross_exposure=None):
    returns = pd.Series(returns, dtype=float).dropna()
    metrics = {
        "annualized_mean": float(12.0 * returns.mean()),
        "annualized_volatility": float(np.sqrt(12.0) * returns.std(ddof=1)),
        "annualized_sharpe": _annualized_sharpe(returns),
        "maximum_drawdown": maximum_drawdown(returns),
        "months": int(len(returns)),
    }
    if turnover is not None:
        metrics["mean_monthly_turnover"] = float(pd.Series(turnover).mean())
    if gross_exposure is not None:
        metrics["mean_gross_exposure"] = float(pd.Series(gross_exposure).mean())
        metrics["maximum_gross_exposure"] = float(pd.Series(gross_exposure).max())
    return metrics


def rolling_sharpe(returns, window=60):
    returns = pd.Series(returns, dtype=float)
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    rolling_volatility = returns.rolling(window, min_periods=window).std(ddof=1)
    return np.sqrt(12.0) * rolling_mean / rolling_volatility


def run_history_length_experiment(
    data_dir,
    output_root,
    *,
    ell,
    history_months=(60, 120, 180, 240, 360, 480),
    evaluation_year_start=2020,
    evaluation_year_end=2024,
    n_random_features=1024,
    random_state=0,
    force=False,
):
    """Run fixed-evaluation Matérn experiments for several history lengths."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for months in history_months:
        if months % 12 != 0:
            raise ValueError("Every history length must be a whole number of years.")
        result_dir = output_root / f"matern32_T{int(months)}"
        expected = result_dir / "metadata.json"
        if force or not expected.exists():
            train_model(
                data_dir=data_dir,
                characteristics="all",
                kernel="matern32",
                kernel_parameters={
                    "ell": float(ell),
                    "n_random_features": int(n_random_features),
                    "random_state": int(random_state),
                    "batch_size": 4096,
                },
                train_years=int(months // 12),
                validation_years=5,
                test_years=1,
                test_year_start=evaluation_year_start,
                test_year_end=evaluation_year_end,
                target_volatility=0.10,
                max_gross_exposure=10.0,
                native_oos_only=True,
                output_dir=result_dir,
            )
        bundle = load_result_bundle(result_dir, load_models=False)
        diagnostics = bundle["diagnostics"]
        monthly = bundle["monthly"]
        native_loss = (
            monthly.groupby("rho")["portfolio_return_native"]
            .apply(_response_one_loss)
            .rename("oos_loss_native")
            .reset_index()
        )
        curve = (
            diagnostics.groupby("rho", as_index=False)["effective_dimension"]
            .mean()
            .merge(native_loss, on="rho", how="inner", validate="one_to_one")
        )
        selected_diagnostics = diagnostics.loc[
            diagnostics["selected_by_validation"]
        ]
        selected_monthly = monthly.loc[monthly["selected_by_validation"]]
        validation_complexity = float(
            selected_diagnostics["effective_dimension"].mean()
        )
        validation_oos_loss = _response_one_loss(
            selected_monthly["portfolio_return_native"]
        )
        ex_post = curve.loc[curve["oos_loss_native"].idxmin()]
        records.append(
            {
                "history_months": int(months),
                "validation_selected_complexity": validation_complexity,
                "ex_post_optimal_complexity": float(ex_post["effective_dimension"]),
                "validation_selected_oos_loss": validation_oos_loss,
                "ex_post_minimum_oos_loss": float(ex_post["oos_loss_native"]),
                "result_dir": str(result_dir.resolve()),
            }
        )
    summary = pd.DataFrame(records).sort_values("history_months")
    summary.to_parquet(output_root / "history_length_summary.parquet", index=False)
    return summary


def paired_block_summary(left, right, *, n_bootstrap=1000, block_length=12,
                         random_state=0):
    """Paired inference for two frozen OOS return streams (no model refitting).

    Identical calendar blocks are sampled for both strategies. Intervals are
    pointwise, conditional on saved fits; they do not account for model search.
    """
    values = np.column_stack([left, right]).astype(float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Paired returns must be finite and contain at least two dates.")
    def statistics(x):
        sr = np.sqrt(12) * x.mean(axis=0) / x.std(axis=0, ddof=1)
        return np.array([sr[0] - sr[1], 12 * (x[:, 0] - x[:, 1]).mean()])
    rng = np.random.default_rng(random_state)
    draws = np.empty((n_bootstrap, 2))
    for i in range(n_bootstrap):
        draws[i] = statistics(values[_circular_block_indices(len(values), block_length, rng)])
    estimate = statistics(values)
    return pd.DataFrame({
        "metric": ["annualized_sharpe_difference", "annualized_mean_difference"],
        "estimate": estimate, "lower": np.nanquantile(draws, .025, axis=0),
        "upper": np.nanquantile(draws, .975, axis=0),
        "months": len(values), "block_length": block_length,
        "bootstrap_repetitions": n_bootstrap,
    })


def response_one_decomposition(returns):
    """Exact empirical identity Q(R) = min_a Q(aR) + E[R²](1-a*)².

    a* = E[R]/E[R²] is an ex-post diagnostic, never an investment rule.
    Its scale-free loss uses the squared monthly Sharpe (ddof=0); sign is lost.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0 or not np.isfinite(r).all():
        raise ValueError("Returns must be finite and nonempty.")
    mean, second = float(r.mean()), float(np.mean(r*r))
    a = mean / second if second > 0 else 0.0
    minimum = 1 - mean * a
    scale_gap = second * (1-a)**2
    return {"native_loss": float(np.mean((1-r)**2)),
            "ex_post_scale_minimum_loss": minimum, "scale_gap": scale_gap,
            "ex_post_scale": a, "mean_native_return": mean}


def history_uncertainty(result_dir, *, n_bootstrap=500, random_state=0):
    """Frozen-path, block-bootstrap uncertainty including re-minimizing rho."""
    bundle = load_result_bundle(result_dir, load_models=False)
    monthly, d = bundle["monthly"], bundle["diagnostics"]
    path = monthly.pivot(index="date", columns="rho", values="portfolio_return_native").sort_index()
    selected = bundle["selected"].set_index("date").reindex(path.index).portfolio_return_native.to_numpy()
    if path.isna().any().any() or not np.isfinite(selected).all():
        raise ValueError("History paths must have matching, complete dates.")
    complexity = d.groupby("rho").effective_dimension.mean().reindex(path.columns).to_numpy()
    losses = (1-path.to_numpy())**2
    selected_losses = (1-selected)**2
    rng = np.random.default_rng(random_state)
    boot = np.empty((n_bootstrap, 3))
    for i in range(n_bootstrap):
        ix = _circular_block_indices(len(path), 12, rng)
        avg = losses[ix].mean(axis=0)
        winner = int(avg.argmin())
        boot[i] = complexity[winner], selected_losses[ix].mean(), avg[winner]
    out = {}
    for i, key in enumerate(["ex_post_complexity", "selected_loss", "ex_post_loss"]):
        out[key+"_lower"], out[key+"_upper"] = np.quantile(boot[:, i], [.025, .975])
    chosen = d.loc[d.selected_by_validation]
    out["selected_complexity_q25"], out["selected_complexity_q75"] = chosen.effective_dimension.quantile([.25, .75])
    out["refit_months"] = int(d.n_refit_months.max())
    out["oos_months"] = len(path)
    return out
