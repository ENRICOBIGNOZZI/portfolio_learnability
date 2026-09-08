"""Annual validation surface and the stability of selected complexity."""
import numpy as np
import pandas as pd
from Utils.paper_figures import PRIMARY, FIGURES, BLUE, ORANGE, style, finish
import matplotlib.pyplot as plt
from Utils.empirical_analysis import load_result_bundle


def main(results_dir=PRIMARY, figure_dir=FIGURES):
    style()
    bundle = load_result_bundle(results_dir, load_models=False)
    d = bundle["diagnostics"].copy()
    d["validation_excess_loss"] = d.validation_loss - d.groupby("test_year").validation_loss.transform("min")
    surface = d.pivot(index="test_year", columns="rho", values="validation_excess_loss").sort_index()
    selected = d.loc[d.selected_by_validation].sort_values("test_year").copy()
    selected["refit_dimension_fraction"] = selected.effective_dimension / selected.n_refit_months
    selected["grid_boundary"] = selected.rho.isin([d.rho.min(), d.rho.max()])
    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[:, 0])
    im = ax.imshow(np.log10(1+surface.to_numpy()), aspect="auto", origin="lower",
                   extent=[np.log10(surface.columns.min())-.25,
                           np.log10(surface.columns.max())+.25,
                           surface.index.min()-.5, surface.index.max()+.5], cmap="cividis")
    ax.scatter(np.log10(selected.rho), selected.test_year, s=14, color="white",
               edgecolors="black", linewidths=.5, label="Validation argmin")
    ax.set(xlabel="log10 relative penalty rho", ylabel="OOS formation year",
           title="A. Validation surface")
    ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, label="log10(1 + excess validation loss)", shrink=.75)
    top = fig.add_subplot(grid[0, 1])
    top.step(selected.test_year, selected.rho, where="mid", color=BLUE)
    top.set(yscale="log", ylabel="Selected rho (log)", xlabel="OOS formation year",
            title="B. Selected relative penalty")
    top.grid(alpha=.2)
    bottom = fig.add_subplot(grid[1, 1])
    bottom.plot(selected.test_year, selected.train_effective_dimension, color=ORANGE, label="Train C (120 months)")
    bottom.plot(selected.test_year, selected.effective_dimension, color=BLUE, label="Refit C (180 months)")
    bottom.set(xlabel="OOS formation year", ylabel="Effective dimension",
               title="C. Active directions over time")
    bottom.legend(); bottom.grid(alpha=.2)
    fig.suptitle("How stable is chronological regularization selection?")
    return finish(fig, "figure_7_tuning_stability", {"": d, "_selected": selected},
                  figure_dir=figure_dir,
                  note="All choices minimize native validation loss. Heatmap colors show excess loss relative to each year's minimum.\n"
                       f"Grid-boundary selections: {selected.grid_boundary.sum()} / {len(selected)}. No OOS outcome is used for selection.")


if __name__ == "__main__":
    main()
