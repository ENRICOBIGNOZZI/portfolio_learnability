"""History length: empirical complexity, loss, and conditional uncertainty."""
import json
import numpy as np
import pandas as pd
from Utils.paper_figures import ROOT, PRIMARY, FIGURES, BLUE, ORANGE, style, finish
import matplotlib.pyplot as plt
from Utils.empirical_analysis import run_history_length_experiment, history_uncertainty

HISTORY_DIR = ROOT / "results/history_length"


def main(force=False, figure_dir=FIGURES):
    style()
    metadata = json.loads((PRIMARY / "metadata.json").read_text())
    summary = run_history_length_experiment(
        ROOT / "data/JKP_USA", HISTORY_DIR,
        ell=float(metadata["kernel_parameters"]["ell"]), force=force)
    extra = pd.DataFrame([history_uncertainty(p) for p in summary.result_dir])
    summary = pd.concat([summary.reset_index(drop=True), extra], axis=1)
    x = summary.history_months.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(x, summary.validation_selected_complexity, "D-", color=BLUE,
                 label="Validation choice: mean C")
    axes[0].fill_between(x, summary.selected_complexity_q25, summary.selected_complexity_q75,
                         color=BLUE, alpha=.15, label="Across-year IQR (descriptive)")
    axes[0].plot(x, summary.ex_post_optimal_complexity, "o-", color=ORANGE,
                 label="Hindsight loss minimum")
    axes[0].fill_between(x, summary.ex_post_complexity_lower, summary.ex_post_complexity_upper,
                         color=ORANGE, alpha=.15, label="Bootstrap minimum: 95% range")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean refit effective dimension (log)")
    axes[0].set_title("A. Selected and hindsight complexity")
    for col, prefix, label, color in [
        ("validation_selected_oos_loss", "selected_loss", "Validation-selected", BLUE),
        ("ex_post_minimum_oos_loss", "ex_post_loss", "Hindsight minimum", ORANGE)]:
        axes[1].plot(x, summary[col], "o-", color=color, label=label)
        axes[1].fill_between(x, summary[prefix+"_lower"], summary[prefix+"_upper"],
                             color=color, alpha=.15)
    axes[1].axhline(1, color="grey", ls="--", lw=1, label="Zero exposure: loss = 1")
    axes[1].set_ylabel("Native OOS response-one loss")
    axes[1].set_title("B. Does additional history improve OOS loss?")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(x, [str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel("Training months T (refit uses T + 60)")
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle("History length and portfolio learning: common evaluation period")
    return finish(fig, "figure_3_history_opens_directions", {"": summary}, figure_dir=figure_dir,
                  note="60 OOS formation months, 2020-2024. Same kernel map and bandwidth; rolling refit adds 60 validation months.\n"
                       "500 circular 12-month block draws; fits held fixed, hindsight rho re-minimized. No population rate is estimated.")


if __name__ == "__main__":
    main()
