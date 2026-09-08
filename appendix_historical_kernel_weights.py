"""Internet Appendix: historical dual-weight and recency diagnostics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utils.empirical_analysis import load_result_bundle


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "matern32_rff_all_characteristics"
FIGURE_DIR = PROJECT_ROOT / "results" / "paper_figures"


def main(results_dir=RESULTS_DIR, figure_dir=FIGURE_DIR):
    models = load_result_bundle(results_dir)["models"]
    dual_paths = np.asarray(models["refit_dual_coefficients"], dtype=float)
    selected_indices = np.asarray(models["selected_indices"], dtype=int)
    selected_dual = np.column_stack(
        [dual_paths[window, :, index] for window, index in enumerate(selected_indices)]
    ).T
    absolute = np.abs(selected_dual[:, ::-1])
    shares = absolute / absolute.sum(axis=1, keepdims=True)
    effective_observations = 1.0 / np.sum(shares**2, axis=1)
    recent_12_month_share = shares[:, :12].sum(axis=1)
    lags = np.arange(1, shares.shape[1] + 1)

    lag_profile = pd.DataFrame(
        {
            "lag_months": lags,
            "median_absolute_weight_share": np.median(shares, axis=0),
            "q25_absolute_weight_share": np.quantile(shares, 0.25, axis=0),
            "q75_absolute_weight_share": np.quantile(shares, 0.75, axis=0),
        }
    )
    window_summary = pd.DataFrame(
        {
            "test_year": np.asarray(models["test_years"], dtype=int),
            "effective_historical_observations": effective_observations,
            "recent_12_month_share": recent_12_month_share,
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(lags, lag_profile["median_absolute_weight_share"])
    axes[0].fill_between(
        lags,
        lag_profile["q25_absolute_weight_share"],
        lag_profile["q75_absolute_weight_share"],
        alpha=0.2,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Historical lag (months)")
    axes[0].set_ylabel("Absolute dual-weight share")
    axes[0].set_title("A. Historical kernel weights")
    axes[0].grid(alpha=0.25)

    axes[1].hist(effective_observations, bins=12, color="#1f77b4", alpha=0.85)
    axes[1].axvline(np.median(effective_observations), color="black", linestyle="--")
    axes[1].set_xlabel("Effective historical observations")
    axes[1].set_ylabel("Rolling windows")
    axes[1].set_title("B. Effective history")
    axes[1].grid(alpha=0.25)

    axes[2].hist(recent_12_month_share, bins=12, color="#d62728", alpha=0.85)
    axes[2].axvline(np.median(recent_12_month_share), color="black", linestyle="--")
    axes[2].set_xlabel("Share on most recent 12 months")
    axes[2].set_ylabel("Rolling windows")
    axes[2].set_title("C. Recency diagnostic")
    axes[2].grid(alpha=0.25)
    figure.suptitle("Matérn-3/2 historical-weight diagnostics")
    figure.tight_layout()

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "appendix_historical_kernel_weights.png"
    lag_path = figure_dir / "appendix_historical_weight_profile.parquet"
    summary_path = figure_dir / "appendix_historical_weight_summary.parquet"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    lag_profile.to_parquet(lag_path, index=False)
    window_summary.to_parquet(summary_path, index=False)
    print(f"Saved: {figure_path}")
    print(f"Saved: {lag_path}")
    print(f"Saved: {summary_path}")
    return figure_path


if __name__ == "__main__":
    main()
