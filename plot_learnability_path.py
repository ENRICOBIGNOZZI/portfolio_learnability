"""Create the paper figure for the empirical learnability path.

The figure combines, for one representation:
  (A) lambda over time,
  (B) effective complexity over time,
  (C) rolling out-of-sample Sharpe over time,
  (D) lambda against effective complexity.

It compares annual cross-validation with the spectrum-guided rule

    lambda_t = c T_t^{-b_hat_t/(b_hat_t+1)},

where c is calibrated once in the initial validation period.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rolling_sharpe(frame, window=60):
    frame = frame.sort_values("return_date").copy()
    r = frame["raw_portfolio_return"].astype(float)
    mean = r.rolling(window, min_periods=window).mean()
    vol = r.rolling(window, min_periods=window).std(ddof=1)
    frame["rolling_sharpe"] = np.sqrt(12.0) * mean / vol
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Directory containing annual_lambda_paths.csv and monthly_returns.parquet.",
    )
    parser.add_argument("--kernel", default="gaussian")
    parser.add_argument("--rolling-months", type=int, default=60)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("learnability_path"),
    )
    args = parser.parse_args()

    annual = pd.read_csv(args.analysis_root / "annual_lambda_paths.csv")
    monthly = pd.read_parquet(args.analysis_root / "monthly_returns.parquet")

    annual = annual.loc[
        annual["kernel"].eq(args.kernel)
    ].sort_values("test_year").copy()

    monthly = monthly.loc[
        monthly["kernel"].eq(args.kernel)
    ].copy()
    monthly["return_date"] = pd.to_datetime(monthly["return_date"])

    cv_monthly = rolling_sharpe(
        monthly.loc[monthly["rule"].eq("annual_cross_validation")],
        args.rolling_months,
    )
    spectral_monthly = rolling_sharpe(
        monthly.loc[monthly["rule"].eq("spectral_b_single_constant")],
        args.rolling_months,
    )

    if annual.empty:
        raise ValueError(f"No annual data for kernel={args.kernel!r}.")
    if cv_monthly.empty or spectral_monthly.empty:
        raise ValueError("Missing monthly returns for one of the two lambda rules.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.2),
        gridspec_kw={"hspace": 0.30, "wspace": 0.28},
    )
    ax_lambda, ax_complexity, ax_sharpe, ax_relation = axes.ravel()

    # A. Lambda over time
    ax_lambda.plot(
        annual["test_year"],
        annual["lambda_cv"],
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        label="Annual cross-validation",
    )
    ax_lambda.plot(
        annual["test_year"],
        annual["lambda_spectral_grid"],
        marker="s",
        markersize=3.5,
        linewidth=1.2,
        label="Spectrum-guided rule",
    )
    ax_lambda.set_yscale("log")
    ax_lambda.set_xlabel("Test year")
    ax_lambda.set_ylabel(r"$\lambda_t$")
    ax_lambda.text(
        0.02, 0.96, "A. Regularization",
        transform=ax_lambda.transAxes,
        ha="left", va="top",
    )

    # B. Effective complexity over time
    ax_complexity.plot(
        annual["test_year"],
        annual["complexity_cv"],
        marker="o",
        markersize=3.5,
        linewidth=1.2,
    )
    ax_complexity.plot(
        annual["test_year"],
        annual["complexity_spectral"],
        marker="s",
        markersize=3.5,
        linewidth=1.2,
    )
    ax_complexity.set_xlabel("Test year")
    ax_complexity.set_ylabel("Effective complexity")
    ax_complexity.text(
        0.02, 0.96, "B. Effective complexity",
        transform=ax_complexity.transAxes,
        ha="left", va="top",
    )

    # C. Rolling OOS Sharpe
    ax_sharpe.plot(
        cv_monthly["return_date"],
        cv_monthly["rolling_sharpe"],
        linewidth=1.2,
    )
    ax_sharpe.plot(
        spectral_monthly["return_date"],
        spectral_monthly["rolling_sharpe"],
        linewidth=1.2,
    )
    ax_sharpe.set_xlabel("Return date")
    ax_sharpe.set_ylabel(
        f"{args.rolling_months}-month OOS Sharpe"
    )
    ax_sharpe.text(
        0.02, 0.96, "C. Out-of-sample performance",
        transform=ax_sharpe.transAxes,
        ha="left", va="top",
    )

    # D. Lambda-complexity path
    ax_relation.plot(
        annual["complexity_cv"],
        annual["lambda_cv"],
        marker="o",
        markersize=3.5,
        linewidth=1.1,
    )
    ax_relation.plot(
        annual["complexity_spectral"],
        annual["lambda_spectral_grid"],
        marker="s",
        markersize=3.5,
        linewidth=1.1,
    )
    ax_relation.set_yscale("log")
    ax_relation.set_xlabel("Effective complexity")
    ax_relation.set_ylabel(r"$\lambda_t$")
    ax_relation.text(
        0.02, 0.96, r"D. $\lambda$ and effective complexity",
        transform=ax_relation.transAxes,
        ha="left", va="top",
    )

    handles, labels = ax_lambda.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.08,
        top=0.92,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)


if __name__ == "__main__":
    main()
