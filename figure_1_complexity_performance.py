"""Figure 1: OOS performance against effective portfolio complexity."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Utils.empirical_analysis import (
    block_bootstrap_performance,
    fixed_rho_curve,
    load_result_bundle,
    selected_point,
)


PROJECT_ROOT = Path(__file__).resolve().parent
from Utils.paper_figures import finish, style
RESULTS_DIR = PROJECT_ROOT / "results" / "matern32_rff_all_characteristics"
FIGURE_DIR = PROJECT_ROOT / "results" / "paper_figures"


def main(results_dir=RESULTS_DIR, figure_dir=FIGURE_DIR):
    style()
    bundle = load_result_bundle(results_dir, load_models=False)
    curve = fixed_rho_curve(bundle)
    bands = block_bootstrap_performance(bundle["monthly"])
    curve = curve.merge(bands, on="rho", validate="one_to_one").sort_values(
        "effective_dimension"
    )
    validation = selected_point(bundle)
    loss_optimum = curve.loc[curve["oos_loss_native"].idxmin()]
    sharpe_optimum = curve.loc[curve["oos_sharpe_capped"].idxmax()]

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)
    axes[0].plot(
        curve["effective_dimension"],
        curve["oos_loss_native"],
        color="#1f77b4",
        marker="o",
        label="Fixed relative-penalty paths",
    )
    axes[0].fill_between(
        curve["effective_dimension"],
        curve["loss_lower"],
        curve["loss_upper"],
        color="#1f77b4",
        alpha=0.18,
        label="Pointwise 95% block-bootstrap interval",
    )
    axes[0].scatter(
        validation["effective_dimension"],
        validation["oos_loss_native"],
        marker="D",
        s=70,
        color="#ff7f0e",
        label="Chronological validation choice",
        zorder=5,
    )
    axes[0].scatter(
        loss_optimum["effective_dimension"],
        loss_optimum["oos_loss_native"],
        marker="o",
        s=85,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Ex-post OOS optimum",
        zorder=5,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("OOS response-one loss")
    axes[0].set_title("A. Decision loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        curve["effective_dimension"],
        curve["oos_sharpe_capped"],
        color="#d62728",
        marker="o",
        label="Fixed relative-penalty paths",
    )
    axes[1].fill_between(
        curve["effective_dimension"],
        curve["sharpe_lower"],
        curve["sharpe_upper"],
        color="#d62728",
        alpha=0.18,
        label="Pointwise 95% block-bootstrap interval",
    )
    axes[1].scatter(
        validation["effective_dimension"],
        validation["oos_sharpe_capped"],
        marker="D",
        s=70,
        color="#ff7f0e",
        label="Chronological validation choice",
        zorder=5,
    )
    axes[1].scatter(
        sharpe_optimum["effective_dimension"],
        sharpe_optimum["oos_sharpe_capped"],
        marker="o",
        s=85,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Ex-post OOS optimum",
        zorder=5,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(r"Mean refit effective dimension $\widehat C(\lambda)$")
    axes[1].set_ylabel("OOS annualized Sharpe")
    axes[1].set_title("B. Economic performance (10% target volatility, gross cap 10)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.suptitle("Matérn-3/2: out-of-sample performance and learnable complexity")
    import pandas as pd
    return finish(figure, "figure_1_complexity_performance",
                  {"": curve, "_selected": pd.DataFrame([validation])}, figure_dir=figure_dir,
                  note="Intervals condition on fitted OOS paths; tuning is not repeated. Circles are hindsight diagnostics.\n"
                       "The diamond uses annual validation selection; its x-coordinate is an average.")


if __name__ == "__main__":
    main()
