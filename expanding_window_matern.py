"""Expanding-window Matérn learnability experiment.

For each Matérn representation, calibrate lambda_0 once on the initial
1963-1972 training block and 1973-1977 validation block. Estimate the
economically relevant spectral exponent b from the pre-OOS managed-payoff
spectrum (all information through 1977), then freeze that estimate before the
first test year. Under the current r=1 theorem the headline schedule is

    alpha_b = b_hat / (b_hat + 1),
    lambda_T = lambda_0 (T / T_0)^(-alpha_b).

The Sobolev/Matérn relation b = 2s/D is retained only as a transparent
theoretical benchmark. It is not imposed on the empirical portfolio spectrum.
Annual expanding-window b estimates are saved as diagnostics but never used to
choose the operational OOS penalty, avoiding a new layer of annual tuning.

All numerical objects required for later analysis are retained:
annual diagnostics, annual raw spectra, monthly returns, initial calibration
profiles, fitted coefficients, calibration metadata, and performance
summaries. Main-text figures use five-year centered moving averages only for
C_T and C_T/T; raw annual series are saved separately.
"""

from __future__ import annotations

import argparse
import json
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
from plot_style import KERNEL_COLORS, save_figure, use_plot_style


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "results" / "expanding_matern"
N_RANDOM_FEATURES = 2000
RANDOM_STATE = 0
MOVING_AVERAGE_YEARS = 5

MATERN_NU = {
    "matern12": 0.5,
    "matern32": 1.5,
    "matern52": 2.5,
}
KERNEL_LABELS = {
    "matern12": "Matérn 1/2",
    "matern32": "Matérn 3/2",
    "matern52": "Matérn 5/2",
}
RULE_LABELS = {
    "fixed": r"Fixed initial $\lambda_0$",
    "empirical_b": r"Pre-OOS spectrum $\widehat b$",
    "theory_s": r"Matérn $s$ benchmark",
    "annual_validation": "Annual validation benchmark",
}


def baseline_spec(kernel: str, characteristic_name: str) -> dict:
    folder = ROOT / "results" / kernel / characteristic_name
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


def original_selected_lambdas(kernel: str, characteristic_name: str) -> dict[int, float]:
    path = ROOT / "results" / kernel / characteristic_name / "lambda_diagnostics.parquet"
    diagnostics = pd.read_parquet(path)
    selected = diagnostics.loc[diagnostics["selected"]].copy()
    if selected["test_year"].duplicated().any():
        raise ValueError(f"{kernel}: expected exactly one selected lambda per test year.")
    return {
        int(row.test_year): float(row.lambda_value)
        for row in selected.rename(columns={"lambda": "lambda_value"}).itertuples()
    }


def build_managed_matrix(
    kernel_name: str,
    characteristics,
    characteristic_name: str,
    spec: dict,
):
    folder = OUTPUT_ROOT / characteristic_name / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / f"{kernel_name}_managed_payoffs.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["matrix"], pd.DatetimeIndex(saved["dates"])

    kernel = PortfolioKernel(
        kernel=kernel_name,
        ell=spec["lengthscale"],
        n_random_features=N_RANDOM_FEATURES,
        random_state=RANDOM_STATE,
    )
    rows, dates = [], []
    years = available_jkp_years(DEFAULT_DATA_DIR)

    for year in years:
        frame = load_dataset(characteristics=characteristics, years=[year])
        panels = build_monthly_panels(frame, characteristics=characteristics)
        for month in panels:
            features = kernel.features(month["x"])
            rows.append((features.T @ month["r"]) / month["n_assets"])
            dates.append(month["formation_date"].to_datetime64())
        if year % 5 == 0 or year == years[-1]:
            print(f"{kernel_name}: managed payoffs through {year}", flush=True)

    matrix = np.vstack(rows)
    dates = pd.DatetimeIndex(np.asarray(dates))
    np.savez_compressed(cache, matrix=matrix, dates=dates.to_numpy())
    return matrix, dates


def calibration_profile(
    matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    spec: dict,
) -> tuple[dict, pd.DataFrame]:
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
    sharpes = np.asarray(
        [compute_sharpe_ratio(validation_returns[:, i]) for i in range(len(ordered))]
    )
    index = int(np.argmin(losses))

    profile = pd.DataFrame({
        "lambda": ordered,
        "validation_loss": losses,
        "validation_sharpe_raw": sharpes,
        "selected": np.arange(len(ordered)) == index,
    })
    calibration = {
        "lambda0": float(ordered[index]),
        "T0": int(len(train)),
        "validation_loss": float(losses[index]),
        "validation_sharpe_raw": float(sharpes[index]),
        "at_boundary": bool(index in (0, len(ordered) - 1)),
    }
    return calibration, profile


def theoretical_matern_rate(input_dimension: int, nu: float) -> dict:
    D = float(input_dimension)
    s = D / 2.0 + float(nu)
    b = 2.0 * s / D
    alpha = b / (b + 1.0)
    return {
        "D": int(input_dimension),
        "nu": float(nu),
        "s": float(s),
        "b_theory": float(b),
        "alpha_s": float(alpha),
    }


def estimate_spectral_b(
    eigenvalues: np.ndarray,
    fit_start: int = 5,
    fit_fraction: float = 0.80,
    fit_end_cap: int = 450,
) -> dict:
    """Estimate polynomial decay on a pre-specified interior spectrum window."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError("No positive eigenvalues available for spectral fit.")

    values = np.sort(values)[::-1]
    values = values[values > values[0] * 1e-10]
    ranks = np.arange(1, len(values) + 1)

    fit_end = min(
        fit_end_cap,
        len(values),
        max(fit_start + 9, int(np.floor(fit_fraction * len(values)))),
    )
    take = (ranks >= fit_start) & (ranks <= fit_end)
    if take.sum() < 10:
        raise ValueError("Too few interior eigenvalues for a stable b estimate.")

    x = np.log(ranks[take].astype(float))
    y = np.log(values[take])
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    denominator = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / denominator

    b_hat = float(-slope)
    if not np.isfinite(b_hat) or b_hat <= 1:
        raise ValueError(
            f"Estimated b={b_hat:.4g} does not satisfy the paper's b>1 condition."
        )

    return {
        "b_hat": b_hat,
        "alpha_b": b_hat / (b_hat + 1.0),
        "b_fit_start": int(fit_start),
        "b_fit_end": int(fit_end),
        "b_fit_fraction": float(fit_fraction),
        "b_fit_r2": float(r_squared),
        "b_active_rank": int(len(values)),
    }


def pre_oos_spectral_rate(
    matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    first_test_year: int,
) -> tuple[dict, pd.DataFrame]:
    """Estimate b once using all managed payoffs available before OOS begins."""
    pre_oos = matrix[dates.year < first_test_year]
    _, eigenvalues = fit_lambda_grid(pre_oos, [1.0])
    fit = estimate_spectral_b(eigenvalues)
    spectrum = pd.DataFrame({
        "rank": np.arange(1, len(eigenvalues) + 1),
        "eigenvalue": eigenvalues,
    })
    return fit, spectrum


def complexity(eigenvalues: np.ndarray, lambda_value: float) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return float(np.sum(eigenvalues / (eigenvalues + lambda_value)))


def evaluate_kernel(
    kernel_name: str,
    matrix: np.ndarray,
    dates: pd.DatetimeIndex,
    spec: dict,
    calibration: dict,
    rate: dict,
    spectral_rate: dict,
    annual_validation_lambdas: dict[int, float],
):
    annual_rows = []
    monthly_rows = []
    spectrum_rows = []
    spectral_fit_rows = []
    beta_save = {}

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
        lambda_empirical = calibration["lambda0"] * (
            T / calibration["T0"]
        ) ** (-spectral_rate["alpha_b"])
        lambda_theory = calibration["lambda0"] * (
            T / calibration["T0"]
        ) ** (-rate["alpha_s"])
        lambda_validation = annual_validation_lambdas[year]

        rule_lambdas = {
            "fixed": float(lambda_fixed),
            "empirical_b": float(lambda_empirical),
            "theory_s": float(lambda_theory),
            "annual_validation": float(lambda_validation),
        }

        all_lambdas = np.unique(list(rule_lambdas.values()))
        fitted, eigenvalues = fit_lambda_grid(refit, all_lambdas)

        annual_fit = estimate_spectral_b(eigenvalues)
        spectral_fit_rows.append({
            "kernel": kernel_name,
            "test_year": year,
            "T": T,
            **annual_fit,
        })

        for rank, value in enumerate(eigenvalues, start=1):
            spectrum_rows.append({
                "kernel": kernel_name,
                "test_year": year,
                "T": T,
                "rank": rank,
                "eigenvalue": float(value),
            })

        for rule, lambda_value in rule_lambdas.items():
            beta_save[f"{year}_{rule}"] = fitted[lambda_value]
            raw = test_matrix @ fitted[lambda_value]
            C = complexity(eigenvalues, lambda_value)

            annual_rows.append({
                "kernel": kernel_name,
                "test_year": year,
                "T": T,
                "rule": rule,
                "lambda": lambda_value,
                "C": C,
                "C_over_T": C / T,
                "annual_oos_loss": float(np.mean((1.0 - raw) ** 2)),
                "annual_oos_sharpe": float(compute_sharpe_ratio(raw)),
                **rate,
                **spectral_rate,
            })

            for formation_date, value in zip(dates[test], raw):
                monthly_rows.append({
                    "kernel": kernel_name,
                    "test_year": year,
                    "formation_date": formation_date,
                    "return_date": formation_date + pd.offsets.MonthEnd(1),
                    "T": T,
                    "rule": rule,
                    "lambda": lambda_value,
                    "C": C,
                    "C_over_T": C / T,
                    "raw_portfolio_return": float(value),
                    **rate,
                    **spectral_rate,
                })

        if year % 10 == 0 or year == years[-1]:
            print(f"{kernel_name}: expanding refits through {year}", flush=True)

    return (
        pd.DataFrame(annual_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(spectrum_rows),
        pd.DataFrame(spectral_fit_rows),
        beta_save,
    )


def add_display_smoothers(annual: pd.DataFrame) -> pd.DataFrame:
    output = annual.sort_values(["kernel", "rule", "test_year"]).copy()
    for field in ("C", "C_over_T", "annual_oos_loss", "annual_oos_sharpe"):
        output[f"{field}_ma5"] = (
            output.groupby(["kernel", "rule"], group_keys=False)[field]
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
    for (kernel, rule), group in monthly.groupby(["kernel", "rule"]):
        returns = group["raw_portfolio_return"].to_numpy()
        diag = annual.loc[
            annual["kernel"].eq(kernel) & annual["rule"].eq(rule)
        ]
        rows.append({
            "kernel": kernel,
            "label": KERNEL_LABELS[kernel],
            "rule": rule,
            "rule_label": RULE_LABELS[rule],
            "months": len(group),
            "sharpe": float(compute_sharpe_ratio(returns)),
            "quadratic_criterion": float(np.mean((1.0 - returns) ** 2)),
            "mean_C": float(diag["C"].mean()),
            "mean_C_over_T": float(diag["C_over_T"].mean()),
            "b_hat": float(diag["b_hat"].iloc[0]),
            "alpha_b": float(diag["alpha_b"].iloc[0]),
            "alpha_s_benchmark": float(diag["alpha_s"].iloc[0]),
            "s": float(diag["s"].iloc[0]),
            "b_theory_benchmark": float(diag["b_theory"].iloc[0]),
        })
    return pd.DataFrame(rows)


def plot_kernel_main(kernel: str, annual: pd.DataFrame, folder: Path):
    keep = annual.loc[
        annual["kernel"].eq(kernel)
        & annual["rule"].isin(["fixed", "empirical_b"])
    ].copy()

    fig, axes = plt.subplots(3, 1, figsize=(7.3, 9.0), sharex=True)
    styles = {
        "fixed": {"linestyle": "--", "linewidth": 1.8},
        "empirical_b": {"linestyle": "-", "linewidth": 2.0},
    }
    for rule in ("fixed", "empirical_b"):
        data = keep.loc[keep["rule"].eq(rule)].sort_values("T")
        axes[0].plot(data["T"], data["lambda"], label=RULE_LABELS[rule], **styles[rule])
        axes[1].plot(data["T"], data["C_ma5"], label=RULE_LABELS[rule], **styles[rule])
        axes[2].plot(data["T"], data["C_over_T_ma5"], label=RULE_LABELS[rule], **styles[rule])

    rate = keep.iloc[0]
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Penalty $\lambda_T$")
    axes[0].set_title(
        rf"{KERNEL_LABELS[kernel]} shrinkage: "
        rf"$\lambda_T\propto T^{{-{rate.alpha_b:.3f}}}$"
    )
    axes[1].set_ylabel(r"$\mathcal{{C}}_T$")
    axes[1].set_title("Learnable effective complexity")
    axes[2].set_ylabel(r"$\mathcal{{C}}_T/T$")
    axes[2].set_xlabel(r"Expanding estimation history $T$ (months)")
    axes[2].set_title("Complexity relative to available history")
    axes[0].legend(loc="best")

    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)

    fig.text(
        0.5, 0.012,
        rf"Pre-OOS managed-payoff spectrum: $\widehat b={rate.b_hat:.2f}$, "
        rf"fit ranks {int(rate.b_fit_start)}--{int(rate.b_fit_end)}, "
        rf"$R^2={rate.b_fit_r2:.2f}$. "
        r"$\mathcal{C}_T$ and $\mathcal{C}_T/T$ use five-year centered moving averages; "
        "raw annual values are retained.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, folder / f"{kernel}_expanding_main.png")
    plt.close(fig)


def plot_all_matern(annual: pd.DataFrame, folder: Path):
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.2), sharex=True)

    for kernel in MATERN_NU:
        data = annual.loc[
            annual["kernel"].eq(kernel)
            & annual["rule"].eq("empirical_b")
        ].sort_values("T")
        label = (
            f"{KERNEL_LABELS[kernel]} "
            + rf"($\widehat b={data['b_hat'].iloc[0]:.2f}$, "
            + rf"$\alpha={data['alpha_b'].iloc[0]:.3f}$)"
        )
        axes[0].plot(data["T"], data["lambda"], label=label, color=KERNEL_COLORS[kernel])
        axes[1].plot(data["T"], data["C_ma5"], label=label, color=KERNEL_COLORS[kernel])
        axes[2].plot(data["T"], data["C_over_T_ma5"], label=label, color=KERNEL_COLORS[kernel])

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\lambda_T$")
    axes[0].set_title("Matérn penalty schedules from the pre-OOS economic spectrum")
    axes[1].set_ylabel(r"$\mathcal{C}_T$")
    axes[1].set_title("Effective complexity")
    axes[2].set_ylabel(r"$\mathcal{C}_T/T$")
    axes[2].set_xlabel(r"Expanding estimation history $T$ (months)")
    axes[2].set_title("Relative effective complexity")
    axes[0].legend(loc="best")

    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)

    fig.tight_layout()
    save_figure(fig, folder / "matern_expanding_comparison.png")
    plt.close(fig)


def plot_raw_appendix(kernel: str, annual: pd.DataFrame, folder: Path):
    data = annual.loc[annual["kernel"].eq(kernel)].copy()
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 9.0), sharex=True)

    for rule in ("annual_validation", "fixed", "empirical_b", "theory_s"):
        part = data.loc[data["rule"].eq(rule)].sort_values("T")
        linestyle = ":" if rule == "annual_validation" else "-"
        axes[0].plot(part["T"], part["lambda"], marker="o", markersize=2,
                     linestyle=linestyle, label=RULE_LABELS[rule])
        axes[1].plot(part["T"], part["C"], marker="o", markersize=2,
                     linestyle=linestyle, label=RULE_LABELS[rule])
        axes[2].plot(part["T"], part["C_over_T"], marker="o", markersize=2,
                     linestyle=linestyle, label=RULE_LABELS[rule])

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\lambda_T$")
    axes[1].set_ylabel(r"$\mathcal{C}_T$")
    axes[2].set_ylabel(r"$\mathcal{C}_T/T$")
    axes[2].set_xlabel(r"Expanding estimation history $T$ (months)")
    axes[0].legend(loc="best")
    fig.suptitle(f"{KERNEL_LABELS[kernel]}: raw annual expanding-window diagnostics")

    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)

    fig.tight_layout()
    save_figure(fig, folder / f"{kernel}_raw_annual_appendix.png")
    plt.close(fig)


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
    input_dimension = len(characteristics)

    folder = OUTPUT_ROOT / characteristic_name
    folder.mkdir(parents=True, exist_ok=True)
    use_plot_style()

    all_annual = []
    all_monthly = []
    all_spectra = []
    metadata_rows = []

    for kernel, nu in MATERN_NU.items():
        spec = baseline_spec(kernel, characteristic_name)
        matrix, dates = build_managed_matrix(
            kernel, characteristics, characteristic_name, spec
        )
        calibration, profile = calibration_profile(matrix, dates, spec)
        profile.insert(0, "kernel", kernel)
        profile.to_parquet(folder / f"{kernel}_calibration_profile.parquet", index=False)

        rate = theoretical_matern_rate(input_dimension, nu)
        spectral_rate, pre_oos_spectrum = pre_oos_spectral_rate(
            matrix, dates, spec["first_test_year"]
        )
        pre_oos_spectrum.insert(0, "kernel", kernel)
        pre_oos_spectrum.to_parquet(
            folder / f"{kernel}_pre_oos_spectrum.parquet", index=False
        )

        validation_lambdas = original_selected_lambdas(kernel, characteristic_name)
        annual, monthly, spectrum, annual_spectral_fits, betas = evaluate_kernel(
            kernel,
            matrix,
            dates,
            spec,
            calibration,
            rate,
            spectral_rate,
            validation_lambdas,
        )

        all_annual.append(annual)
        all_monthly.append(monthly)
        all_spectra.append(spectrum)
        annual_spectral_fits.to_parquet(
            folder / f"{kernel}_annual_spectral_fits.parquet", index=False
        )
        np.savez_compressed(folder / f"{kernel}_betas.npz", **betas)

        metadata_rows.append({
            "kernel": kernel,
            "label": KERNEL_LABELS[kernel],
            "lengthscale": spec["lengthscale"],
            "n_random_features": N_RANDOM_FEATURES,
            **calibration,
            **rate,
            **spectral_rate,
            "rule": "lambda_T = lambda0 * (T/T0)^(-b_hat/(b_hat+1))",
            "theory_rule_benchmark": "lambda_T = lambda0 * (T/T0)^(-2s/(2s+D))",
        })

    annual = add_display_smoothers(pd.concat(all_annual, ignore_index=True))
    monthly = pd.concat(all_monthly, ignore_index=True).sort_values(
        ["kernel", "rule", "return_date"]
    )
    spectra = pd.concat(all_spectra, ignore_index=True).sort_values(
        ["kernel", "test_year", "rank"]
    )
    summary = performance_summary(annual, monthly)
    metadata = pd.DataFrame(metadata_rows)

    annual.to_parquet(folder / "annual_diagnostics.parquet", index=False)
    annual.to_csv(folder / "annual_diagnostics.csv", index=False)
    monthly.to_parquet(folder / "monthly_returns.parquet", index=False)
    spectra.to_parquet(folder / "annual_spectra.parquet", index=False)
    summary.to_csv(folder / "performance_summary.csv", index=False)
    metadata.to_csv(folder / "calibration_and_rates.csv", index=False)
    (folder / "experiment_manifest.json").write_text(
        json.dumps({
            "characteristic_name": characteristic_name,
            "input_dimension": input_dimension,
            "n_random_features": N_RANDOM_FEATURES,
            "kernels": MATERN_NU,
            "moving_average_years": MOVING_AVERAGE_YEARS,
            "headline_rule": "lambda_T=lambda0*(T/T0)^(-b_hat/(b_hat+1))",
            "b_estimation": "frozen pre-OOS managed-payoff spectrum; ranks 5 to min(450, 80% active rank)",
            "theory_s_rule": "appendix benchmark only",
            "raw_data_retained": [
                "annual_diagnostics.parquet",
                "monthly_returns.parquet",
                "annual_spectra.parquet",
                "*_calibration_profile.parquet",
                "*_pre_oos_spectrum.parquet",
                "*_annual_spectral_fits.parquet",
                "*_betas.npz",
                "performance_summary.csv",
                "calibration_and_rates.csv",
            ],
        }, indent=2),
        encoding="utf-8",
    )

    for kernel in MATERN_NU:
        plot_kernel_main(kernel, annual, folder)
        plot_raw_appendix(kernel, annual, folder)
    plot_all_matern(annual, folder)

    print("\nMatérn calibration and frozen pre-OOS empirical spectral rates")
    print(metadata.to_string(index=False))
    print("\nMatérn schedule performance")
    print(summary.to_string(index=False))
    print(f"\nSaved all Matérn expanding-window outputs in {folder}", flush=True)


if __name__ == "__main__":
    main()
