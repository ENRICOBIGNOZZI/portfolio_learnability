"""Final Gaussian learnability experiment: one calibration, expanding history.

The purpose of this file is deliberately narrow. It produces the empirical
objects that correspond directly to the Gaussian limit of the paper:

    lambda_T = lambda_0 * T_0 / T,

with lambda_0 calibrated once on the initial train/validation split. Every
subsequent operator estimate, coefficient refit, and complexity measure uses
the full expanding history available before the test year.

No r parameter enters this experiment.

Outputs are written only to results/expanding_gaussian. Existing baseline
results are read but never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Kernels.kernel_function import PortfolioKernel, fit_lambda_grid
from Utils.utils import build_monthly_panels, compute_sharpe_ratio
from download_JKP.read_dataset import (
    DEFAULT_DATA_DIR,
    available_jkp_characteristics,
    available_jkp_years,
    load_dataset,
)
from plot_style import save_figure, use_plot_style


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "results" / "expanding_gaussian"
KERNEL = "gaussian"
N_RANDOM_FEATURES = 2000
RANDOM_STATE = 0
MOVING_AVERAGE_YEARS = 5

RULE_LABELS = {
    "fixed": r"Fixed initial $\lambda_0$",
    "inverse_T": r"Initial calibration + $1/T$",
    "annual_validation": "Annual validation benchmark",
}


def baseline_spec(characteristic_name: str) -> dict:
    """Read the original Gaussian design without changing it."""
    folder = ROOT / "results" / KERNEL / characteristic_name
    diagnostics = pd.read_parquet(folder / "lambda_diagnostics.parquet")
    first_year = int(diagnostics["test_year"].min())
    first = diagnostics.loc[diagnostics["test_year"].eq(first_year)]
    return {
        "lengthscale": float(first["lengthscale"].iloc[0]),
        "lambda_grid": np.sort(first["lambda"].unique()),
        "first_test_year": first_year,
        "train_start": int(first["train_start"].iloc[0]),
        "train_end": int(first["train_end"].iloc[0]),
        "validation_end": int(first["validation_end"].iloc[0]),
    }


def original_selected_lambdas(characteristic_name: str) -> dict[int, float]:
    """Annual-validation choices from the saved chronological baseline."""
    path = ROOT / "results" / KERNEL / characteristic_name / "lambda_diagnostics.parquet"
    diagnostics = pd.read_parquet(path)
    selected = diagnostics.loc[diagnostics["selected"]].copy()
    if selected["test_year"].duplicated().any():
        raise ValueError("Expected exactly one selected lambda per test year.")
    return {
        int(row.test_year): float(row.lambda_value)
        for row in selected.rename(columns={"lambda": "lambda_value"}).itertuples()
    }


def matrix_cache_path(characteristic_name: str) -> Path:
    return OUTPUT_ROOT / characteristic_name / "gaussian_managed_payoffs.npz"


def build_managed_matrix(characteristics, characteristic_name: str, spec: dict):
    """Build one Gaussian managed-payoff row per month, streaming annual panels."""
    folder = OUTPUT_ROOT / characteristic_name
    folder.mkdir(parents=True, exist_ok=True)
    cache = matrix_cache_path(characteristic_name)
    if cache.exists():
        saved = np.load(cache)
        return saved["matrix"], pd.DatetimeIndex(saved["dates"])

    kernel = PortfolioKernel(
        kernel=KERNEL,
        ell=spec["lengthscale"],
        n_random_features=N_RANDOM_FEATURES,
        random_state=RANDOM_STATE,
    )
    rows = []
    dates = []
    years = available_jkp_years(DEFAULT_DATA_DIR)

    for year in years:
        frame = load_dataset(characteristics=characteristics, years=[year])
        panels = build_monthly_panels(frame, characteristics=characteristics)
        for month in panels:
            features = kernel.features(month["x"])
            rows.append((features.T @ month["r"]) / month["n_assets"])
            dates.append(month["formation_date"].to_datetime64())
        if year % 5 == 0 or year == years[-1]:
            print(f"Gaussian managed payoffs through {year}", flush=True)

    matrix = np.vstack(rows)
    dates = pd.DatetimeIndex(np.asarray(dates))
    np.savez_compressed(cache, matrix=matrix, dates=dates.to_numpy())
    return matrix, dates


def calibrate_once(matrix: np.ndarray, dates: pd.DatetimeIndex, spec: dict):
    """Initial lambda calibration: 1963-72 train, 1973-77 validation."""
    train_mask = (
        (dates.year >= spec["train_start"])
        & (dates.year <= spec["train_end"])
    )
    validation_mask = (
        (dates.year > spec["train_end"])
        & (dates.year <= spec["validation_end"])
    )
    train = matrix[train_mask]
    validation = matrix[validation_mask]

    betas, _ = fit_lambda_grid(train, spec["lambda_grid"])
    ordered = np.asarray(spec["lambda_grid"])
    beta_matrix = np.column_stack([betas[value] for value in ordered])
    validation_returns = validation @ beta_matrix
    losses = np.mean((1.0 - validation_returns) ** 2, axis=0)
    index = int(np.argmin(losses))

    return {
        "lambda0": float(ordered[index]),
        "T0": int(len(train)),
        "validation_loss": float(losses[index]),
        "at_boundary": bool(index in (0, len(ordered) - 1)),
    }


def complexity(eigenvalues: np.ndarray, lambda_value: float) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return float(np.sum(eigenvalues / (eigenvalues + lambda_value)))


def evaluate_expanding(
    matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    spec: dict,
    calibration: dict,
    annual_validation_lambdas: dict[int, float],
):
    """Refit on every observation before each test year."""
    annual_rows = []
    monthly_rows = []

    years = sorted(
        year for year in set(dates.year)
        if year >= spec["first_test_year"]
    )

    for year in years:
        before = dates.year < year
        test = dates.year == year
        refit = matrix[before]
        test_matrix = matrix[test]
        T = int(len(refit))

        if T == 0 or len(test_matrix) == 0:
            continue

        lambda_fixed = calibration["lambda0"]
        lambda_inverse = calibration["lambda0"] * calibration["T0"] / T
        lambda_validation = annual_validation_lambdas[year]

        rule_lambdas = {
            "fixed": float(lambda_fixed),
            "inverse_T": float(lambda_inverse),
            "annual_validation": float(lambda_validation),
        }
        all_lambdas = np.unique(list(rule_lambdas.values()))
        fitted, eigenvalues = fit_lambda_grid(refit, all_lambdas)

        for rule, lambda_value in rule_lambdas.items():
            raw = test_matrix @ fitted[lambda_value]
            C = complexity(eigenvalues, lambda_value)

            annual_rows.append({
                "test_year": year,
                "T": T,
                "rule": rule,
                "lambda": lambda_value,
                "C": C,
                "C_over_T": C / T,
                "annual_oos_loss": float(np.mean((1.0 - raw) ** 2)),
                "annual_oos_sharpe": float(compute_sharpe_ratio(raw)),
            })

            for formation_date, value in zip(dates[test], raw):
                monthly_rows.append({
                    "test_year": year,
                    "formation_date": formation_date,
                    "return_date": formation_date + pd.offsets.MonthEnd(1),
                    "T": T,
                    "rule": rule,
                    "lambda": lambda_value,
                    "C": C,
                    "C_over_T": C / T,
                    "raw_portfolio_return": float(value),
                })

        if year % 10 == 0 or year == years[-1]:
            print(f"Expanding Gaussian refits through {year}", flush=True)

    annual = pd.DataFrame(annual_rows).sort_values(["rule", "test_year"])
    monthly = pd.DataFrame(monthly_rows).sort_values(["rule", "return_date"])
    return annual, monthly


def add_display_smoothers(annual: pd.DataFrame) -> pd.DataFrame:
    """Five-year centered moving averages for display only."""
    output = annual.copy()
    for field in ("C", "C_over_T", "annual_oos_loss", "annual_oos_sharpe"):
        output[f"{field}_ma5"] = (
            output.groupby("rule", group_keys=False)[field]
            .transform(
                lambda series: series.rolling(
                    MOVING_AVERAGE_YEARS,
                    center=True,
                    min_periods=3,
                ).mean()
            )
        )
    return output


def performance_summary(annual: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, group in monthly.groupby("rule"):
        returns = group["raw_portfolio_return"].to_numpy()
        rows.append({
            "rule": rule,
            "label": RULE_LABELS[rule],
            "months": len(group),
            "sharpe": float(compute_sharpe_ratio(returns)),
            "quadratic_criterion": float(np.mean((1.0 - returns) ** 2)),
            "mean_C": float(
                annual.loc[annual["rule"].eq(rule), "C"].mean()
            ),
            "mean_C_over_T": float(
                annual.loc[annual["rule"].eq(rule), "C_over_T"].mean()
            ),
        })
    return pd.DataFrame(rows)


def plot_main(annual: pd.DataFrame, folder: Path):
    """Main-text figure: fixed lambda versus the Gaussian 1/T rule."""
    use_plot_style()
    keep = annual.loc[annual["rule"].isin(["fixed", "inverse_T"])].copy()

    figure, axes = plt.subplots(3, 1, figsize=(7.3, 9.0), sharex=True)

    styles = {
        "fixed": {"linestyle": "--", "linewidth": 1.8},
        "inverse_T": {"linestyle": "-", "linewidth": 2.0},
    }

    for rule in ("fixed", "inverse_T"):
        data = keep.loc[keep["rule"].eq(rule)].sort_values("T")
        axes[0].plot(
            data["T"], data["lambda"],
            label=RULE_LABELS[rule],
            **styles[rule],
        )
        axes[1].plot(
            data["T"], data["C_ma5"],
            label=RULE_LABELS[rule],
            **styles[rule],
        )
        axes[2].plot(
            data["T"], data["C_over_T_ma5"],
            label=RULE_LABELS[rule],
            **styles[rule],
        )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Penalty $\lambda_T$")
    axes[0].set_title("Shrinkage declines as market history expands")

    axes[1].set_ylabel(r"$\mathcal{C}_T$")
    axes[1].set_title("Learnable effective complexity")

    axes[2].set_ylabel(r"$\mathcal{C}_T/T$")
    axes[2].set_xlabel(r"Expanding estimation history $T$ (months)")
    axes[2].set_title("Complexity relative to available history")

    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)

    axes[0].legend(loc="best")
    figure.text(
        0.5, 0.015,
        "Gaussian representation. C and C/T are five-year centered moving averages "
        "of annual expanding-window estimates; the lambda schedules are exact.",
        ha="center", fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(figure, folder / "gaussian_expanding_main.png")
    plt.close(figure)


def plot_validation_benchmark(annual: pd.DataFrame, folder: Path):
    """Appendix diagnostic including annual validation."""
    use_plot_style()
    figure, axes = plt.subplots(2, 1, figsize=(7.3, 6.3), sharex=True)

    for rule in ("annual_validation", "fixed", "inverse_T"):
        data = annual.loc[annual["rule"].eq(rule)].sort_values("T")
        axes[0].plot(
            data["T"], data["C_over_T_ma5"],
            label=RULE_LABELS[rule],
            linewidth=1.8,
            linestyle=":" if rule == "annual_validation" else "-",
        )
        axes[1].plot(
            data["T"], data["annual_oos_loss_ma5"],
            label=RULE_LABELS[rule],
            linewidth=1.8,
            linestyle=":" if rule == "annual_validation" else "-",
        )

    axes[0].set_ylabel(r"$\mathcal{C}_T/T$")
    axes[0].set_title("Relative complexity")
    axes[1].set_ylabel("Response-one loss")
    axes[1].set_xlabel(r"Expanding estimation history $T$ (months)")
    axes[1].set_title("Out-of-sample quadratic criterion")

    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)

    axes[0].legend(loc="best")
    figure.text(
        0.5, 0.012,
        "Five-year centered moving averages shown for readability. "
        "All underlying annual values are saved separately.",
        ha="center", fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, folder / "gaussian_expanding_validation_benchmark.png")
    plt.close(figure)


def plot_raw_appendix(annual: pd.DataFrame, folder: Path):
    """Raw annual expanding-window values: no smoothing."""
    use_plot_style()
    figure, axes = plt.subplots(3, 1, figsize=(7.3, 9.0), sharex=True)

    for rule in ("annual_validation", "fixed", "inverse_T"):
        data = annual.loc[annual["rule"].eq(rule)].sort_values("T")
        label = RULE_LABELS[rule]
        axes[0].plot(data["T"], data["lambda"], marker="o", markersize=2, label=label)
        axes[1].plot(data["T"], data["C"], marker="o", markersize=2, label=label)
        axes[2].plot(data["T"], data["C_over_T"], marker="o", markersize=2, label=label)

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\lambda_T$")
    axes[1].set_ylabel(r"$\mathcal{C}_T$")
    axes[2].set_ylabel(r"$\mathcal{C}_T/T$")
    axes[2].set_xlabel(r"Expanding estimation history $T$ (months)")
    axes[0].legend(loc="best")

    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)

    figure.suptitle("Gaussian expanding-window diagnostics: raw annual values")
    figure.tight_layout()
    save_figure(figure, folder / "gaussian_expanding_raw_appendix.png")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characteristics", nargs="+", default=["all"])
    args = parser.parse_args()

    characteristic_name = "_".join(args.characteristics)
    characteristics = (
        available_jkp_characteristics(DEFAULT_DATA_DIR)
        if characteristic_name == "all"
        else args.characteristics
    )

    folder = OUTPUT_ROOT / characteristic_name
    folder.mkdir(parents=True, exist_ok=True)

    spec = baseline_spec(characteristic_name)
    matrix, dates = build_managed_matrix(
        characteristics,
        characteristic_name,
        spec,
    )
    calibration = calibrate_once(matrix, dates, spec)
    validation_lambdas = original_selected_lambdas(characteristic_name)

    annual, monthly = evaluate_expanding(
        matrix,
        dates,
        spec,
        calibration,
        validation_lambdas,
    )
    annual = add_display_smoothers(annual)
    summary = performance_summary(annual, monthly)

    annual.to_parquet(folder / "annual_diagnostics.parquet", index=False)
    annual.to_csv(folder / "annual_diagnostics.csv", index=False)
    monthly.to_parquet(folder / "monthly_returns.parquet", index=False)
    summary.to_csv(folder / "performance_summary.csv", index=False)
    pd.DataFrame([{
        **calibration,
        "lengthscale": spec["lengthscale"],
        "characteristic_name": characteristic_name,
        "rule": "lambda_T = lambda0 * T0 / T",
    }]).to_csv(folder / "calibration.csv", index=False)

    plot_main(annual, folder)
    plot_validation_benchmark(annual, folder)
    plot_raw_appendix(annual, folder)

    print(summary.to_string(index=False))
    print(f"Saved final Gaussian expanding-window artifacts in {folder}", flush=True)


if __name__ == "__main__":
    main()
