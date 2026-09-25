"""Paper figures for the spectrum-adaptive regularization experiment.

This script reads the annual output produced by spectral_lambda_experiment.py
and creates a compact set of paper figures. If the full annual eigenvalue file
is supplied, it also plots the evolution of the managed-portfolio spectrum
through time.

Typical 10k usage
-----------------
python plot_spectral_lambda.py \
    --annual-path analysis/10000/spectral_lambda/annual_lambda_paths.csv \
    --kernel gaussian \
    --output-dir analysis/10000/spectral_lambda/figures \
    --eigen-parquet restored/10000/results/gaussian/all/kernel_eigenvalues.parquet \
    --max-rank 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "gaussian": "Gaussian",
    "matern12": r"Matérn 1/2",
    "matern32": r"Matérn 3/2",
    "matern52": r"Matérn 5/2",
    "ntk": "NTK",
    "linear": "Linear",
}


def _label(kernel: str) -> str:
    return LABELS.get(kernel, kernel)


def _minmax(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if np.isclose(lo, hi):
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_learnability_path(
    annual: pd.DataFrame,
    output_path: Path,
    kernel: str,
) -> None:
    """One graph: shrinkage, effective complexity, and OOS Sharpe."""
    g = annual.sort_values("test_year").copy()

    less_shrinkage = 1.0 - _minmax(np.log(g["lambda_spectral_grid"]))
    complexity = _minmax(g["complexity_spectral"])
    sharpe_spectral = _minmax(g["oos_sharpe_spectral_year"])
    sharpe_cv = _minmax(g["oos_sharpe_cv_year"])

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    ax.plot(
        g["test_year"],
        less_shrinkage,
        marker="o",
        linewidth=1.6,
        markersize=4,
        label=r"Less shrinkage (spectral $\lambda_t$)",
    )
    ax.plot(
        g["test_year"],
        complexity,
        marker="s",
        linewidth=1.6,
        markersize=4,
        label="Effective complexity (spectral rule)",
    )
    ax.plot(
        g["test_year"],
        sharpe_spectral,
        marker="^",
        linewidth=1.6,
        markersize=4,
        label="Out-of-sample Sharpe (spectral rule)",
    )
    ax.plot(
        g["test_year"],
        sharpe_cv,
        linestyle="--",
        linewidth=1.25,
        label="Out-of-sample Sharpe (annual CV)",
    )

    ax.set_xlabel("Test year")
    ax.set_ylabel("Normalized index")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="best")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_complexity(
    annual: pd.DataFrame,
    output_path: Path,
) -> None:
    """Regularization versus the effective number of active directions."""
    g = annual.sort_values("test_year").copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.plot(
        g["complexity_cv"],
        g["lambda_cv"],
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Annual CV",
    )
    ax.plot(
        g["complexity_spectral"],
        g["lambda_spectral_grid"],
        marker="s",
        linewidth=1.4,
        markersize=4,
        label="Spectral rule",
    )

    ax.set_xlabel("Effective complexity")
    ax.set_ylabel(r"$\lambda$")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="best")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_over_time(
    annual: pd.DataFrame,
    output_path: Path,
) -> None:
    """Annual CV lambda and the spectrum-adaptive theoretical path."""
    g = annual.sort_values("test_year").copy()

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    ax.plot(
        g["test_year"],
        g["lambda_cv"],
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Annual CV",
    )
    ax.plot(
        g["test_year"],
        g["lambda_spectral_grid"],
        marker="s",
        linewidth=1.4,
        markersize=4,
        label="Spectral rule (nearest grid value)",
    )
    ax.plot(
        g["test_year"],
        g["lambda_spectral_target"],
        linestyle="--",
        linewidth=1.15,
        label=r"$cT^{-\hat b_t/(\hat b_t+1)}$",
    )

    ax.set_xlabel("Test year")
    ax.set_ylabel(r"$\lambda$")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="best")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_over_time(
    annual: pd.DataFrame,
    output_path: Path,
) -> None:
    """Effective complexity implied by CV and the spectral rule."""
    g = annual.sort_values("test_year").copy()

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    ax.plot(
        g["test_year"],
        g["complexity_cv"],
        marker="o",
        linewidth=1.4,
        markersize=4,
        label="Annual CV",
    )
    ax.plot(
        g["test_year"],
        g["complexity_spectral"],
        marker="s",
        linewidth=1.4,
        markersize=4,
        label="Spectral rule",
    )

    ax.set_xlabel("Test year")
    ax.set_ylabel("Effective complexity")
    ax.legend(frameon=False, loc="best")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_slope_over_time(
    annual: pd.DataFrame,
    output_path: Path,
) -> None:
    """Estimated polynomial spectral-decay exponent through time."""
    g = annual.sort_values("test_year").copy()

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    ax.plot(
        g["test_year"],
        g["b_hat"],
        marker="o",
        linewidth=1.5,
        markersize=4,
    )

    ax.set_xlabel("Test year")
    ax.set_ylabel(r"Estimated spectral slope $\hat b_t$")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_heatmap(
    eigen_path: Path,
    output_path: Path,
    max_rank: int,
) -> None:
    """Heatmap of the managed-payoff spectrum over time."""
    e = pd.read_parquet(eigen_path).copy()
    e = e[
        e["eigenvalue_number"].between(1, max_rank)
        & e["eigenvalue"].gt(0)
    ].copy()

    pivot = (
        e.pivot_table(
            index="eigenvalue_number",
            columns="test_year",
            values="eigenvalue",
            aggfunc="first",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    years = pivot.columns.to_numpy()
    ranks = pivot.index.to_numpy()
    values = np.log10(pivot.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    xidx = np.linspace(0, len(years) - 1, min(8, len(years))).astype(int)
    yidx = np.linspace(0, len(ranks) - 1, min(10, len(ranks))).astype(int)

    ax.set_xticks(xidx)
    ax.set_xticklabels(years[xidx])
    ax.set_yticks(yidx)
    ax.set_yticklabels(ranks[yidx])

    ax.set_xlabel("Test year")
    ax.set_ylabel("Eigenvalue rank")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}(\mu_{j,t})$")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-path", type=Path, required=True)
    parser.add_argument("--kernel", default="gaussian")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eigen-parquet", type=Path, default=None)
    parser.add_argument("--max-rank", type=int, default=100)
    args = parser.parse_args()

    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(args.annual_path)
    annual = annual.loc[annual["kernel"].eq(args.kernel)].copy()

    if annual.empty:
        raise ValueError(f"No annual rows found for kernel={args.kernel}")

    prefix = args.kernel

    plot_learnability_path(
        annual,
        args.output_dir / f"{prefix}_learnability_path.png",
        args.kernel,
    )
    plot_lambda_complexity(
        annual,
        args.output_dir / f"{prefix}_lambda_vs_complexity.png",
    )
    plot_lambda_over_time(
        annual,
        args.output_dir / f"{prefix}_lambda_over_time.png",
    )
    plot_complexity_over_time(
        annual,
        args.output_dir / f"{prefix}_complexity_over_time.png",
    )
    plot_spectral_slope_over_time(
        annual,
        args.output_dir / f"{prefix}_spectral_slope_over_time.png",
    )

    if args.eigen_parquet is not None:
        if not args.eigen_parquet.exists():
            raise FileNotFoundError(args.eigen_parquet)
        plot_spectrum_heatmap(
            args.eigen_parquet,
            args.output_dir / f"{prefix}_spectrum_over_time.png",
            args.max_rank,
        )

    print(f"Saved {_label(args.kernel)} figures in {args.output_dir}")


if __name__ == "__main__":
    main()
