"""Spectral rule for the regularization path.

For each kernel/year, estimate the managed-spectrum exponent b from a fixed
rank window, use lambda_T = c T^{-b/(b+1)}, calibrate c once from the first
validation window, and then keep c fixed out of sample.

The script only uses already-generated chronological training outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


def annualized_sharpe(x):
    x = np.asarray(x, dtype=float)
    return np.sqrt(12.0) * x.mean() / x.std(ddof=1)


def fit_b(eigenvalues, rank_min, rank_max):
    g = eigenvalues.loc[
        eigenvalues["eigenvalue_number"].between(rank_min, rank_max)
        & eigenvalues["eigenvalue"].gt(0)
    ].copy()
    if len(g) < 10:
        return np.nan, np.nan
    fit = linregress(
        np.log(g["eigenvalue_number"].to_numpy(float)),
        np.log(g["eigenvalue"].to_numpy(float)),
    )
    return -float(fit.slope), float(fit.rvalue**2)


def nearest_lambda(grid, target):
    grid = np.asarray(grid, dtype=float)
    return float(grid[np.argmin(np.abs(np.log(grid) - np.log(target)))])


def run_one(folder, kernel, rank_min, rank_max):
    eig = pd.read_parquet(folder / "kernel_eigenvalues.parquet")
    diag = pd.read_parquet(folder / "lambda_diagnostics.parquet")
    lret = pd.read_parquet(folder / "lambda_portfolio_returns.parquet")

    years = sorted(diag["test_year"].unique())
    rows = []

    for year in years:
        dy = diag.loc[diag["test_year"].eq(year)].copy()
        ey = eig.loc[eig["test_year"].eq(year)].copy()
        b, r2 = fit_b(ey, rank_min, rank_max)

        meta = dy.iloc[0]
        T = 12 * (int(meta["validation_end"]) - int(meta["train_start"]) + 1)
        cv = dy.loc[dy["selected"]].iloc[0]

        rows.append(
            {
                "kernel": kernel,
                "test_year": int(year),
                "T": int(T),
                "b_hat": b,
                "b_r2": r2,
                "lambda_cv": float(cv["lambda"]),
                "complexity_cv": float(cv["effective_dimension"]),
                "validation_loss_cv": float(cv["validation_loss_raw"]),
                "oos_sharpe_cv_year": float(cv["out_of_sample_sharpe_raw"]),
            }
        )

    annual = pd.DataFrame(rows)
    if annual["b_hat"].isna().any():
        raise RuntimeError(f"Insufficient spectrum to estimate b for {kernel}")

    first = annual.iloc[0]
    exponent0 = first["b_hat"] / (first["b_hat"] + 1.0)

    # Calibrate only once: the first validation-optimal lambda determines c.
    c_hat = first["lambda_cv"] * first["T"] ** exponent0

    monthly_rule = []
    monthly_cv = []

    for i, row in annual.iterrows():
        exponent = row["b_hat"] / (row["b_hat"] + 1.0)
        target = c_hat * row["T"] ** (-exponent)

        dy = diag.loc[diag["test_year"].eq(row["test_year"])].copy()
        chosen = nearest_lambda(dy["lambda"].to_numpy(float), target)
        rule_diag = dy.loc[np.isclose(dy["lambda"], chosen, rtol=1e-12, atol=0)].iloc[0]

        annual.loc[i, "lambda_spectral_target"] = target
        annual.loc[i, "lambda_spectral_grid"] = chosen
        annual.loc[i, "complexity_spectral"] = float(rule_diag["effective_dimension"])
        annual.loc[i, "oos_sharpe_spectral_year"] = float(rule_diag["out_of_sample_sharpe_raw"])

        ry = lret.loc[lret["formation_date"].astype(str).str[:4].astype(int).eq(row["test_year"])]
        rule_month = ry.loc[np.isclose(ry["lambda"], chosen, rtol=1e-12, atol=0)].copy()
        cv_month = ry.loc[np.isclose(ry["lambda"], row["lambda_cv"], rtol=1e-12, atol=0)].copy()
        monthly_rule.append(rule_month)
        monthly_cv.append(cv_month)

    rule = pd.concat(monthly_rule, ignore_index=True)
    cv = pd.concat(monthly_cv, ignore_index=True)

    response_target = float(rule["response_target"].iloc[0]) if "response_target" in rule else 1.0
    rr = rule["raw_portfolio_return"].to_numpy(float)
    rc = cv["raw_portfolio_return"].to_numpy(float)

    summary = pd.DataFrame(
        [
            {
                "kernel": kernel,
                "rule": "annual_cross_validation",
                "months": len(rc),
                "sharpe": annualized_sharpe(rc),
                "criterion": np.mean((response_target - rc) ** 2),
            },
            {
                "kernel": kernel,
                "rule": "spectral_b_single_constant",
                "months": len(rr),
                "sharpe": annualized_sharpe(rr),
                "criterion": np.mean((response_target - rr) ** 2),
            },
        ]
    )

    annual["c_hat_initial_validation"] = c_hat
    annual["rank_min"] = rank_min
    annual["rank_max"] = rank_max
    return annual, summary, rule, cv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["gaussian", "matern12", "matern32", "matern52"],
    )
    parser.add_argument("--rank-min", type=int, default=5)
    parser.add_argument("--rank-max", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out = args.output or (args.results_root / "spectral_lambda_experiment")
    out.mkdir(parents=True, exist_ok=True)

    annual_all = []
    summary_all = []
    rule_all = []
    cv_all = []

    for kernel in args.kernels:
        folder = args.results_root / kernel / "all"
        if not folder.exists():
            continue
        annual, summary, rule, cv = run_one(
            folder, kernel, args.rank_min, args.rank_max
        )
        annual_all.append(annual)
        summary_all.append(summary)
        rule["kernel"] = kernel
        rule["rule"] = "spectral_b_single_constant"
        cv["kernel"] = kernel
        cv["rule"] = "annual_cross_validation"
        rule_all.append(rule)
        cv_all.append(cv)

    pd.concat(annual_all, ignore_index=True).to_csv(
        out / "annual_lambda_paths.csv", index=False
    )
    pd.concat(summary_all, ignore_index=True).to_csv(
        out / "performance_summary.csv", index=False
    )
    pd.concat(rule_all + cv_all, ignore_index=True).to_parquet(
        out / "monthly_returns.parquet", index=False
    )

    print(pd.concat(summary_all, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
