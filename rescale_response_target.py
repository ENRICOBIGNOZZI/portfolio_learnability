"""Exact post-fit rescaling from response target c=1 to c=0.1.

For ridge portfolio learning,
    argmin_f ||c - Af||^2 + lambda ||f||_H^2
equals c times the c=1 solution at the same lambda. Therefore lambda choices,
effective complexity, spectra, and Sharpe ratios are unchanged. Portfolio
returns, weights, gross exposure, and fitted coefficients scale linearly with c;
quadratic losses scale with c^2.

This script transforms a completed c=1 uncapped result tree into the exact
c=0.1 result tree without retraining.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SOURCE_TARGET = 1.0
RESPONSE_TARGET = 0.1
SCALE = RESPONSE_TARGET / SOURCE_TARGET

KERNELS = ("linear", "gaussian", "ntk", "matern12", "matern32", "matern52")
RANDOM_KERNELS = ("gaussian", "ntk", "matern12", "matern32", "matern52")


def scale_table(path: Path, linear_cols=(), quadratic_cols=(), add_target=True):
    if not path.exists():
        return
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(path)

    for column in linear_cols:
        if column in frame:
            frame[column] = frame[column].astype(float) * SCALE
    for column in quadratic_cols:
        if column in frame:
            frame[column] = frame[column].astype(float) * SCALE**2
    if add_target:
        frame["response_target"] = RESPONSE_TARGET

    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def kernel_folders():
    for kernel in KERNELS:
        base = RESULTS / kernel / "all"
        if base.exists():
            yield kernel, 0, base
        if kernel in RANDOM_KERNELS:
            seed = base / "seed_1"
            if seed.exists():
                yield kernel, 1, seed


def scale_kernel_outputs():
    return_cols = (
        "raw_portfolio_return",
        "portfolio_return",
    )
    exposure_cols = (
        "raw_gross_exposure",
        "gross_exposure",
        "net_exposure",
        "max_abs_weight",
        "p99_abs_weight",
    )
    lambda_exposure_cols = (
        "raw_gross_exposure",
        "gross_exposure",
    )

    for kernel, seed, folder in kernel_folders():
        scale_table(
            folder / "portfolio_returns.parquet",
            linear_cols=return_cols + exposure_cols,
        )
        scale_table(
            folder / "lambda_portfolio_returns.parquet",
            linear_cols=return_cols + lambda_exposure_cols,
        )
        scale_table(
            folder / "lambda_diagnostics.parquet",
            linear_cols=(
                "validation_average_raw_gross_exposure",
                "test_average_raw_gross_exposure",
            ),
            quadratic_cols=(
                "validation_loss",
                "validation_loss_raw",
            ),
        )
        scale_table(
            folder / "lengthscale_diagnostics.parquet",
            quadratic_cols=("validation_loss",),
        )

        for path in folder.glob("weights/**/*.parquet"):
            scale_table(
                path,
                linear_cols=("raw_weight", "weight"),
            )

        print(f"Scaled {kernel} seed {seed}", flush=True)


def scale_expanding_gaussian():
    folder = RESULTS / "expanding_gaussian" / "all"
    if not folder.exists():
        return

    scale_table(
        folder / "annual_diagnostics.parquet",
        quadratic_cols=("annual_oos_loss", "annual_oos_loss_ma5"),
    )
    scale_table(
        folder / "annual_diagnostics.csv",
        quadratic_cols=("annual_oos_loss", "annual_oos_loss_ma5"),
    )
    scale_table(
        folder / "monthly_returns.parquet",
        linear_cols=("raw_portfolio_return",),
    )
    scale_table(
        folder / "calibration.csv",
        quadratic_cols=("validation_loss",),
    )

    annual = pd.read_parquet(folder / "annual_diagnostics.parquet")
    monthly = pd.read_parquet(folder / "monthly_returns.parquet")
    rows = []
    for rule, group in monthly.groupby("rule"):
        r = group["raw_portfolio_return"].to_numpy(dtype=float)
        diag = annual.loc[annual["rule"].eq(rule)]
        rows.append({
            "rule": rule,
            "months": len(group),
            "sharpe": np.sqrt(12.0) * r.mean() / r.std(ddof=1),
            "quadratic_criterion": np.mean((RESPONSE_TARGET - r) ** 2),
            "mean_C": diag["C"].mean(),
            "mean_C_over_T": diag["C_over_T"].mean(),
            "response_target": RESPONSE_TARGET,
        })
    pd.DataFrame(rows).to_csv(folder / "performance_summary.csv", index=False)


def scale_expanding_matern():
    folder = RESULTS / "expanding_matern" / "all"
    if not folder.exists():
        return

    scale_table(
        folder / "annual_diagnostics.parquet",
        quadratic_cols=("annual_oos_loss", "annual_oos_loss_ma5"),
    )
    scale_table(
        folder / "annual_diagnostics.csv",
        quadratic_cols=("annual_oos_loss", "annual_oos_loss_ma5"),
    )
    scale_table(
        folder / "monthly_returns.parquet",
        linear_cols=("raw_portfolio_return",),
    )
    scale_table(
        folder / "calibration_and_rates.csv",
        quadratic_cols=("validation_loss",),
    )

    for path in folder.glob("*_calibration_profile.parquet"):
        scale_table(path, quadratic_cols=("validation_loss",))

    for path in folder.glob("*_betas.npz"):
        saved = np.load(path)
        scaled = {key: saved[key] * SCALE for key in saved.files}
        np.savez_compressed(path, **scaled)

    annual = pd.read_parquet(folder / "annual_diagnostics.parquet")
    monthly = pd.read_parquet(folder / "monthly_returns.parquet")
    rows = []
    for (kernel, rule), group in monthly.groupby(["kernel", "rule"]):
        r = group["raw_portfolio_return"].to_numpy(dtype=float)
        diag = annual.loc[
            annual["kernel"].eq(kernel) & annual["rule"].eq(rule)
        ]
        first = diag.iloc[0]
        rows.append({
            "kernel": kernel,
            "rule": rule,
            "months": len(group),
            "sharpe": np.sqrt(12.0) * r.mean() / r.std(ddof=1),
            "quadratic_criterion": np.mean((RESPONSE_TARGET - r) ** 2),
            "mean_C": diag["C"].mean(),
            "mean_C_over_T": diag["C_over_T"].mean(),
            "b_hat": first["b_hat"],
            "alpha_b": first["alpha_b"],
            "alpha_s_benchmark": first["alpha_s"],
            "s": first["s"],
            "b_theory_benchmark": first["b_theory"],
            "response_target": RESPONSE_TARGET,
        })
    pd.DataFrame(rows).to_csv(folder / "performance_summary.csv", index=False)


def clear_derived_outputs():
    for folder in (
        RESULTS / "complexity_analysis",
        RESULTS / "seed_stability",
        RESULTS / "learnability_diagnostics",
        RESULTS / "final_spectra",
    ):
        if folder.exists():
            shutil.rmtree(folder)

    for path in RESULTS.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg", ".html"}:
            path.unlink()

    for name in (
        "final_uncapped_performance_summary.csv",
        "final_performance_summary.csv",
    ):
        path = RESULTS / name
        if path.exists():
            path.unlink()


def write_manifest():
    payload = {
        "source_response_target": SOURCE_TARGET,
        "response_target": RESPONSE_TARGET,
        "scale": SCALE,
        "identity": "f_c(lambda) = c * f_1(lambda)",
        "unchanged": [
            "selected lambda",
            "lengthscale",
            "effective complexity",
            "managed-payoff spectrum",
            "Sharpe ratio",
        ],
        "linear_scaling": [
            "portfolio returns",
            "portfolio weights",
            "gross exposure",
            "net exposure",
            "fitted coefficients",
        ],
        "quadratic_scaling": [
            "response-target quadratic loss",
        ],
        "note": (
            "Exact algebraic transformation of the completed uncapped c=1 run; "
            "no OOS information is used and no hyperparameter is re-selected."
        ),
    }
    (RESULTS / "response_target_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main():
    clear_derived_outputs()
    scale_kernel_outputs()
    scale_expanding_gaussian()
    scale_expanding_matern()
    write_manifest()
    print(
        f"Exact response-target rescaling complete: c={SOURCE_TARGET} -> "
        f"c={RESPONSE_TARGET}",
        flush=True,
    )


if __name__ == "__main__":
    main()
