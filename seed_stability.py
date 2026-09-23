"""Compare two finite-feature seeds after the clean uncapped retrain."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import save_figure, use_plot_style
from Utils.utils import compute_sharpe_ratio


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "seed_stability"
KERNELS = ("gaussian", "ntk", "matern12", "matern32", "matern52")
LABELS = {
    "gaussian": "Gaussian",
    "ntk": "NTK",
    "matern12": "Matérn 1/2",
    "matern32": "Matérn 3/2",
    "matern52": "Matérn 5/2",
}
SEEDS = (0, 1)


def folder(kernel, seed):
    base = ROOT / "results" / kernel / "all"
    return base if seed == 0 else base / f"seed_{seed}"


def load_returns(kernel, seed):
    data = pd.read_parquet(folder(kernel, seed) / "portfolio_returns.parquet")
    data = data.sort_values("return_date").copy()
    if not np.allclose(data["scale_factor"], 1.0):
        raise AssertionError(f"{kernel}/seed {seed}: expected uncapped weights")
    if not np.allclose(data["portfolio_return"], data["raw_portfolio_return"]):
        raise AssertionError(f"{kernel}/seed {seed}: capped and raw returns differ")
    return data


def stats(kernel, seed, data):
    r = data["raw_portfolio_return"].to_numpy(dtype=float)
    lengthscale = float(data["lengthscale"].iloc[0])
    return {
        "kernel": kernel,
        "label": LABELS[kernel],
        "seed": seed,
        "months": len(data),
        "sharpe": float(compute_sharpe_ratio(r)),
        "quadratic_criterion": float(np.mean((1.0 - r) ** 2)),
        "mean_ann": float(12.0 * np.mean(r)),
        "vol_ann": float(np.sqrt(12.0) * np.std(r, ddof=1)),
        "lengthscale": lengthscale,
        "median_gross": float(data["gross_exposure"].median()),
        "p95_gross": float(data["gross_exposure"].quantile(0.95)),
        "p99_gross": float(data["gross_exposure"].quantile(0.99)),
        "max_gross": float(data["gross_exposure"].max()),
        "max_abs_weight": float(data["max_abs_weight"].max())
        if "max_abs_weight" in data else np.nan,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    use_plot_style()

    rows = []
    monthly = {}
    for kernel in KERNELS:
        for seed in SEEDS:
            data = load_returns(kernel, seed)
            monthly[(kernel, seed)] = data
            rows.append(stats(kernel, seed, data))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "seed_performance.csv", index=False)

    correlations = []
    for kernel in KERNELS:
        left = monthly[(kernel, 0)][["return_date", "raw_portfolio_return"]]
        right = monthly[(kernel, 1)][["return_date", "raw_portfolio_return"]]
        merged = left.merge(right, on="return_date", suffixes=("_0", "_1"), validate="one_to_one")
        correlations.append({
            "kernel": kernel,
            "label": LABELS[kernel],
            "return_correlation": float(
                merged["raw_portfolio_return_0"].corr(merged["raw_portfolio_return_1"])
            ),
            "mean_absolute_return_difference": float(
                np.mean(np.abs(
                    merged["raw_portfolio_return_0"] - merged["raw_portfolio_return_1"]
                ))
            ),
        })
    corr = pd.DataFrame(correlations)
    corr.to_csv(OUT / "seed_return_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.arange(len(KERNELS))
    for seed, marker in [(0, "o"), (1, "s")]:
        part = summary.loc[summary["seed"].eq(seed)].set_index("kernel").reindex(KERNELS)
        axes[0].plot(x, part["sharpe"], marker=marker, label=f"Seed {seed}")
        axes[1].plot(x, part["quadratic_criterion"], marker=marker, label=f"Seed {seed}")
    axes[0].set_ylabel("OOS Sharpe")
    axes[1].set_ylabel("Response-one OOS loss")
    for ax in axes:
        ax.set_xticks(x, [LABELS[k] for k in KERNELS], rotation=25, ha="right")
        ax.grid(False)
        ax.yaxis.grid(True, alpha=0.35)
        ax.legend()
    axes[0].set_title("Seed stability: Sharpe")
    axes[1].set_title("Seed stability: portfolio loss")
    fig.tight_layout()
    save_figure(fig, OUT / "seed_performance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    width = 0.36
    for offset, seed in [(-width / 2, 0), (width / 2, 1)]:
        part = summary.loc[summary["seed"].eq(seed)].set_index("kernel").reindex(KERNELS)
        ax.bar(x + offset, part["p99_gross"], width=width, label=f"Seed {seed}")
    ax.set_xticks(x, [LABELS[k] for k in KERNELS], rotation=25, ha="right")
    ax.set_ylabel("99th percentile gross exposure")
    ax.set_title("Uncapped leverage stability across finite-feature seeds")
    ax.grid(False)
    ax.yaxis.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, OUT / "seed_gross_exposure.png")
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nSeed return correlations")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
