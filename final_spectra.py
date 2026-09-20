"""Final managed-portfolio spectrum figures for the paper.

The main figure contains Linear, NTK, and Matérn 3/2. Eigenvalues are raw:
there is no normalization by the leading eigenvalue. The axes are log-log so
that the economically relevant tail remains visible.

For Matérn kernels the reported b is estimated on a fixed interior range,
ranks 5 through 450 whenever available. The fitted line is not drawn in the
main figure. Gaussian is treated as the fast-decay / b=infinity benchmark and
is not assigned a finite power-law slope.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import KERNEL_COLORS, save_figure, use_plot_style


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHARACTERISTIC_NAME = "all"

KERNEL_LABELS = {
    "linear": "Linear",
    "gaussian": "Gaussian",
    "ntk": "NTK",
    "matern12": "Matérn 1/2",
    "matern32": "Matérn 3/2",
    "matern52": "Matérn 5/2",
}

MAIN_KERNELS = ("linear", "ntk", "matern32")
APPENDIX_KERNELS = ("gaussian", "matern12", "matern52")
MATERN_KERNELS = {"matern12", "matern32", "matern52"}
FIT_START = 5
FIT_END = 450


def load_final_spectrum(kernel: str) -> pd.DataFrame:
    path = RESULTS / kernel / CHARACTERISTIC_NAME / "kernel_eigenvalues.parquet"
    data = pd.read_parquet(path)
    year = int(data["test_year"].max())
    data = data.loc[data["test_year"].eq(year)].copy()
    data = data.sort_values("eigenvalue_number")
    largest = float(data["eigenvalue"].max())
    data = data.loc[
        np.isfinite(data["eigenvalue"])
        & (data["eigenvalue"] > largest * 1e-10)
    ].copy()
    data["rank"] = np.arange(1, len(data) + 1)
    return data


def estimate_matern_b(spectrum: pd.DataFrame) -> dict:
    end = min(FIT_END, int(spectrum["rank"].max()))
    fit = spectrum.loc[
        spectrum["rank"].between(FIT_START, end)
    ].copy()
    if len(fit) < 10:
        raise ValueError("Too few eigenvalues for the pre-specified Matérn fit.")

    x = np.log(fit["rank"].to_numpy(dtype=float))
    y = np.log(fit["eigenvalue"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    denominator = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum(residual ** 2) / denominator

    return {
        "fit_start": FIT_START,
        "fit_end": end,
        "estimated_b": float(-slope),
        "r_squared": float(r_squared),
        "number_fit_eigenvalues": int(len(fit)),
    }


def save_individual(kernel: str, spectrum: pd.DataFrame, estimate: dict | None):
    folder = RESULTS / "final_spectra"
    folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    ax.loglog(
        spectrum["rank"],
        spectrum["eigenvalue"],
        color=KERNEL_COLORS[kernel],
        linewidth=1.7,
    )
    ax.set_xlabel("Managed-portfolio direction rank")
    ax.set_ylabel(r"Raw eigenvalue $\widehat{\mu}_j$")
    ax.set_title(f"{KERNEL_LABELS[kernel]} managed-portfolio spectrum")
    ax.grid(False)
    ax.yaxis.grid(True, which="both", alpha=0.25)

    if estimate is not None:
        ax.text(
            0.97,
            0.95,
            rf"$\widehat b={estimate['estimated_b']:.2f}$"
            + "\n"
            + rf"fit ranks {estimate['fit_start']}--{estimate['fit_end']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    elif kernel == "gaussian":
        ax.text(
            0.97,
            0.95,
            r"fast-decay benchmark ($b=\infty$)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, folder / f"{kernel}_raw_spectrum.png")
    plt.close(fig)


def main():
    use_plot_style()
    folder = RESULTS / "final_spectra"
    folder.mkdir(parents=True, exist_ok=True)

    spectra = {}
    estimates = {}

    for kernel in KERNEL_LABELS:
        spectrum = load_final_spectrum(kernel)
        spectra[kernel] = spectrum
        estimate = estimate_matern_b(spectrum) if kernel in MATERN_KERNELS else None
        estimates[kernel] = estimate
        save_individual(kernel, spectrum, estimate)

    rows = []
    for kernel, estimate in estimates.items():
        row = {
            "kernel": kernel,
            "number_of_eigenvalues": int(len(spectra[kernel])),
            "estimated_b": np.nan,
            "fit_start": np.nan,
            "fit_end": np.nan,
            "r_squared": np.nan,
        }
        if estimate is not None:
            row.update(estimate)
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder / "spectrum_summary.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    for axis, kernel in zip(axes, MAIN_KERNELS):
        spectrum = spectra[kernel]
        axis.loglog(
            spectrum["rank"],
            spectrum["eigenvalue"],
            color=KERNEL_COLORS[kernel],
            linewidth=1.6,
        )
        axis.set_title(KERNEL_LABELS[kernel])
        axis.set_xlabel("Direction rank")
        axis.grid(False)
        axis.yaxis.grid(True, which="both", alpha=0.25)

        if kernel == "matern32":
            estimate = estimates[kernel]
            axis.text(
                0.96,
                0.94,
                rf"$\widehat b={estimate['estimated_b']:.2f}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

    axes[0].set_ylabel(r"Raw eigenvalue $\widehat{\mu}_j$")
    figure.suptitle("Managed-portfolio spectra", fontsize=13)
    figure.tight_layout()
    save_figure(figure, folder / "main_spectra.png")
    plt.close(figure)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved final spectrum figures in {folder}")


if __name__ == "__main__":
    main()
