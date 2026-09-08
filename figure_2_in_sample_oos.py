"""Figure 2: in-sample fit and out-of-sample value along complexity."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Utils.empirical_analysis import fixed_rho_curve, load_result_bundle
from Utils.paper_figures import finish, style


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "matern32_rff_all_characteristics"
FIGURE_DIR = PROJECT_ROOT / "results" / "paper_figures"


def main(results_dir=RESULTS_DIR, figure_dir=FIGURE_DIR):
    style()
    curve = fixed_rho_curve(load_result_bundle(results_dir, load_models=False)).sort_values(
        "effective_dimension"
    )
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)
    axes[0].plot(
        curve["train_effective_dimension"],
        curve["in_sample_loss"],
        marker="o",
        label="In-sample (120-month fits)",
    )
    axes[0].plot(
        curve["effective_dimension"],
        curve["oos_loss_native"],
        marker="o",
        label="OOS (180-month refits)",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Response-one loss")
    axes[0].set_title("A. Native response-one loss at each model's effective dimension")
    axes[0].plot(curve["train_effective_dimension"], curve["validation_loss"],
                 linestyle="--", label="Validation (120-month fits)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        curve["train_effective_dimension"],
        curve["in_sample_sharpe"],
        marker="o",
        label="In-sample: mean window Sharpe",
    )
    axes[1].plot(
        curve["effective_dimension"],
        curve["oos_sharpe_native"],
        marker="o",
        label="OOS: stitched monthly Sharpe",
    )
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("Mean effective dimension of the evaluated estimator")
    axes[1].set_ylabel("Annualized Sharpe (symlog)")
    axes[1].set_title("B. In-sample versus OOS Sharpe")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle("Matérn-3/2: statistical fit versus decision value")
    return finish(figure, "figure_2_in_sample_oos", {"": curve}, figure_dir=figure_dir,
                  note="Train and validation use train C; OOS uses refit C. Native returns throughout.\n"
                       "IS Sharpe averages overlapping windows; OOS Sharpe uses a single non-overlapping return series.")


if __name__ == "__main__":
    main()
