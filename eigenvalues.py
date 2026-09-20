"""Plot raw managed-portfolio spectra and estimate Matérn tail exponents.

Main figures use raw eigenvalues. Power-law slopes are reported only for the
Matérn kernels and are fitted on a pre-specified interior tail: the first four
directions and the final 10% of numerically nonzero eigenvalues are excluded.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import KERNEL_COLORS, save_figure, use_plot_style

use_plot_style()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--characteristics", nargs="+", default=["all"])
args = parser.parse_args()

characteristics = "all" if args.characteristics == ["all"] else args.characteristics
characteristic_name = "all" if characteristics == "all" else "_".join(characteristics)

kernel_names = ["linear", "gaussian", "ntk", "matern12", "matern32", "matern52"]
kernel_labels = {
    "linear": "Linear",
    "gaussian": "Gaussian",
    "ntk": "NTK",
    "matern12": "Matérn 1/2",
    "matern32": "Matérn 3/2",
    "matern52": "Matérn 5/2",
}
matern_kernels = {"matern12", "matern32", "matern52"}

project_folder = Path(__file__).resolve().parent


def interior_matern_fit(ranks, values, first_rank=5, upper_fraction=0.90):
    """Fit log(mu_j) = a - b log(j) on a fixed interior spectral tail."""
    ranks = np.asarray(ranks, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 12:
        return np.nan, np.nan, np.full(n, False)

    last_position = max(first_rank + 4, int(np.floor(upper_fraction * n)))
    last_position = min(last_position, n)

    mask = (ranks >= first_rank)
    sorted_positions = np.arange(1, n + 1)
    mask &= sorted_positions <= last_position

    if mask.sum() < 5:
        return np.nan, np.nan, mask

    slope, intercept = np.polyfit(np.log(ranks[mask]), np.log(values[mask]), 1)
    return float(-slope), float(intercept), mask


def robust_matern_windows(ranks, values):
    """Pre-specified sensitivity windows; no window is chosen using fit quality."""
    rows = []
    for first_rank in (3, 5, 10):
        for upper_fraction in (0.80, 0.90, 0.95):
            b, intercept, mask = interior_matern_fit(
                ranks, values, first_rank=first_rank, upper_fraction=upper_fraction
            )
            rows.append({
                "first_rank": first_rank,
                "upper_fraction": upper_fraction,
                "estimated_b": b,
                "number_fit_eigenvalues": int(mask.sum()),
            })
    return pd.DataFrame(rows)


all_estimates = []

for kernel_name in kernel_names:
    results_folder = project_folder / "results" / kernel_name / characteristic_name
    eigenvalues = pd.read_parquet(results_folder / "kernel_eigenvalues.parquet")

    last_test_year = int(eigenvalues["test_year"].max())
    spectrum = (
        eigenvalues.loc[eigenvalues["test_year"].eq(last_test_year)]
        .sort_values("eigenvalue_number")
        .copy()
    )
    lengthscale = float(spectrum["lengthscale"].iloc[0])

    largest = float(spectrum["eigenvalue"].max())
    spectrum = spectrum.loc[
        np.isfinite(spectrum["eigenvalue"]) &
        (spectrum["eigenvalue"] > largest * 1e-10)
    ].copy()

    ranks = spectrum["eigenvalue_number"].to_numpy(dtype=float)
    values = spectrum["eigenvalue"].to_numpy(dtype=float)

    # Main paper object: raw spectrum, no normalization.
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(ranks, values, color=KERNEL_COLORS[kernel_name], linewidth=1.6)
    ax.set_xlabel("Ordered managed-portfolio direction")
    ax.set_ylabel(r"Raw eigenvalue $\hat{\mu}_j$")
    ax.set_title(f"{kernel_labels[kernel_name]} managed-portfolio spectrum")
    ax.grid(False)
    ax.yaxis.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, results_folder / "eigenvalue_spectrum_raw.png")
    plt.close(fig)

    estimated_b = np.nan
    fit_start = np.nan
    fit_end = np.nan

    # Only Matérn gets a theoretically interpreted power-law fit.
    if kernel_name in matern_kernels:
        estimated_b, intercept, fit_mask = interior_matern_fit(ranks, values)
        if np.isfinite(estimated_b):
            fit_start = int(ranks[fit_mask][0])
            fit_end = int(ranks[fit_mask][-1])

            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            ax.loglog(
                ranks, values,
                color=KERNEL_COLORS[kernel_name],
                linewidth=1.6,
                label="Eigenvalues",
            )
            fit_ranks = ranks[fit_mask]
            fitted = np.exp(intercept) * fit_ranks ** (-estimated_b)
            ax.loglog(
                fit_ranks, fitted,
                color="#333333",
                linestyle="--",
                linewidth=1.4,
                label=rf"Interior fit: $\hat{{b}}={estimated_b:.3f}$",
            )
            ax.axvline(fit_start, color="#777777", linewidth=0.8, alpha=0.6)
            ax.axvline(fit_end, color="#777777", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("Eigenvalue rank")
            ax.set_ylabel(r"Raw eigenvalue $\hat{\mu}_j$")
            ax.set_title(
                f"{kernel_labels[kernel_name]} spectrum · fit ranks "
                f"{fit_start}–{fit_end}"
            )
            ax.legend(loc="upper right")
            ax.grid(False)
            ax.yaxis.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            save_figure(fig, results_folder / "eigenvalue_spectrum_fit.png")
            plt.close(fig)

            robust_matern_windows(ranks, values).to_csv(
                results_folder / "matern_b_window_sensitivity.csv", index=False
            )

    estimate = {
        "kernel": kernel_name,
        "test_year": last_test_year,
        "lengthscale": lengthscale,
        "estimated_b": estimated_b,
        "fit_start_rank": fit_start,
        "fit_end_rank": fit_end,
        "number_of_eigenvalues": len(values),
    }
    all_estimates.append(estimate)
    pd.DataFrame([estimate]).to_parquet(
        results_folder / "estimated_b.parquet", index=False
    )

    print(kernel_labels[kernel_name], estimate)

pd.DataFrame(all_estimates).to_parquet(
    project_folder / "results" / f"estimated_b_all_kernels_{characteristic_name}.parquet",
    index=False,
)
