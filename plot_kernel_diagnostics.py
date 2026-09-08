"""Plot paper-consistent Gaussian portfolio diagnostics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utils.utils import _annualized_sharpe


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "gaussian_rff_all_characteristics"


def _mean_and_standard_error(frame, value):
    grouped = frame.groupby("rho")[value]
    summary = grouped.agg(["mean", "std", "count"])
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    return summary


def _stitched_oos_sharpe(monthly_oos, column="portfolio_return"):
    return monthly_oos.groupby("rho")[column].apply(
        _annualized_sharpe
    )


def plot_diagnostics(results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    diagnostics = pd.read_parquet(results_dir / "lambda_diagnostics.parquet")
    monthly_oos = pd.read_parquet(results_dir / "monthly_oos_portfolios.parquet")
    selected_oos = pd.read_parquet(results_dir / "selected_oos_portfolio.parquet")
    cap_values = selected_oos["max_gross_exposure"].dropna()
    gross_cap = float(cap_values.iloc[0]) if not cap_values.empty else None
    capped_label = (
        f"Stitched OOS (gross cap={gross_cap:g})"
        if gross_cap is not None
        else "Stitched out-of-sample"
    )

    in_sample = _mean_and_standard_error(diagnostics, "in_sample_sharpe")
    oos_windows = _mean_and_standard_error(diagnostics, "oos_sharpe")
    summary = in_sample.add_prefix("is_").join(
        oos_windows.add_prefix("oos_window_")
    )
    summary["oos_sharpe"] = _stitched_oos_sharpe(monthly_oos)
    summary["oos_sharpe_raw"] = _stitched_oos_sharpe(
        monthly_oos, "portfolio_return_raw"
    )
    summary["effective_dimension"] = diagnostics.groupby("rho")[
        "effective_dimension"
    ].mean()
    summary = summary.sort_index()

    finite_oos = summary["oos_sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_oos.empty:
        raise ValueError("All stitched OOS Sharpes are undefined.")
    optimal_rho = float(finite_oos.idxmax())
    optimal_complexity = float(summary.loc[optimal_rho, "effective_dimension"])
    optimal_sharpe = float(summary.loc[optimal_rho, "oos_sharpe"])
    boundary_optimum = optimal_rho in {
        float(summary.index.min()),
        float(summary.index.max()),
    }

    figure, axis = plt.subplots(figsize=(8, 5))
    is_mean = summary["is_mean"].to_numpy()
    is_se = summary["is_se"].fillna(0.0).to_numpy()
    axis.plot(
        summary.index,
        is_mean,
        marker="o",
        label="In-sample",
        color="#1f77b4",
    )
    axis.fill_between(
        summary.index,
        is_mean - 1.96 * is_se,
        is_mean + 1.96 * is_se,
        alpha=0.15,
        color="#1f77b4",
    )
    axis.plot(
        summary.index,
        summary["oos_sharpe"],
        marker="o",
        label=capped_label,
        color="#d62728",
    )
    axis.plot(
        summary.index,
        summary["oos_sharpe_raw"],
        linestyle="--",
        label="OOS before leverage cap",
        color="#7f7f7f",
        alpha=0.8,
    )
    axis.axvline(optimal_rho, color="black", linestyle="--", alpha=0.6)
    annotation = f"Empirical OOS maximum rho={optimal_rho:.2g}"
    if boundary_optimum:
        annotation += " (grid boundary)"
    axis.text(
        0.03,
        0.05,
        annotation,
        transform=axis.transAxes,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xlabel(r"Relative ridge penalty $\rho=\lambda/\mu_{max}$")
    axis.set_ylabel("Annualized Sharpe ratio (symlog scale)")
    axis.set_title("Gaussian RFF response-one portfolio: Sharpe vs penalty")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    lambda_path = results_dir / "sharpe_vs_lambda.png"
    figure.savefig(lambda_path, dpi=180)
    plt.close(figure)

    ordered = summary.sort_values("effective_dimension")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        ordered["effective_dimension"],
        ordered["is_mean"],
        marker="o",
        label="In-sample",
        color="#1f77b4",
    )
    axis.plot(
        ordered["effective_dimension"],
        ordered["oos_sharpe"],
        marker="o",
        label=capped_label,
        color="#d62728",
    )
    axis.plot(
        ordered["effective_dimension"],
        ordered["oos_sharpe_raw"],
        linestyle="--",
        label="OOS before leverage cap",
        color="#7f7f7f",
        alpha=0.8,
    )
    axis.scatter([optimal_complexity], [optimal_sharpe], color="black", zorder=5)
    axis.text(
        0.48,
        0.05,
        f"Empirical maximum: C={optimal_complexity:.1f}",
        transform=axis.transAxes,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xlabel(r"Effective dimension $\sum_j \mu_j/(\mu_j+\lambda)$")
    axis.set_ylabel("Annualized Sharpe ratio (symlog scale)")
    axis.set_title("Gaussian RFF portfolio: Sharpe vs managed-payoff complexity")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    complexity_path = results_dir / "sharpe_vs_effective_dimension.png"
    figure.savefig(complexity_path, dpi=180)
    plt.close(figure)

    selected_oos = selected_oos.sort_values("date")
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(selected_oos["date"], selected_oos["equity"], color="#2ca02c")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("Compounded equity\n(symlog scale)")
    axes[0].set_title(
        "Validation-selected Gaussian RFF portfolio"
        + (f" (gross cap={gross_cap:g})" if gross_cap is not None else "")
    )
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        selected_oos["date"],
        selected_oos["sum_weights"],
        label="Sum of weights (net)",
    )
    axes[1].plot(
        selected_oos["date"],
        selected_oos["sum_absolute_weights"],
        label="Sum of absolute weights (gross)",
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Exposure")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    portfolio_path = results_dir / "selected_oos_equity_and_exposures.png"
    figure.savefig(portfolio_path, dpi=180)
    plt.close(figure)

    summary_path = results_dir / "diagnostic_curve.parquet"
    summary.reset_index().to_parquet(summary_path, index=False)
    print(f"Empirical OOS maximum rho: {optimal_rho:.6g}")
    print(f"Mean effective dimension there: {optimal_complexity:.3f}")
    if boundary_optimum:
        print("Warning: the empirical maximum is on the grid boundary.")
    print(f"Saved: {lambda_path}")
    print(f"Saved: {complexity_path}")
    print(f"Saved: {portfolio_path}")
    print(f"Saved: {summary_path}")
    return {
        "lambda_plot": lambda_path,
        "complexity_plot": complexity_path,
        "portfolio_plot": portfolio_path,
        "summary": summary_path,
    }


if __name__ == "__main__":
    plot_diagnostics()
