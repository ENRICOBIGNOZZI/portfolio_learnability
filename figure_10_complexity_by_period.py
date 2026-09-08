"""Does the apparent complexity optimum persist across calendar periods?"""
import pandas as pd
from Utils.paper_figures import PRIMARY, FIGURES, style, finish
import matplotlib.pyplot as plt
from Utils.empirical_analysis import load_result_bundle, fixed_rho_curve, selected_point

PERIODS = [(1978, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]


def main(results_dir=PRIMARY, figure_dir=FIGURES):
    style()
    bundle = load_result_bundle(results_dir, load_models=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    tables, points = [], []
    for (start, end), color in zip(PERIODS, ["#236B8E", "#C47728", "#39826D", "#845DA0", "#B34848"]):
        subset = {key: bundle[key].loc[bundle[key].test_year.between(start, end)]
                  for key in ("diagnostics", "monthly", "selected")}
        curve = fixed_rho_curve(subset).sort_values("effective_dimension")
        point = selected_point(subset)
        label = f"{start}-{end}"
        curve["formation_period"] = label
        tables.append(curve); points.append({"formation_period": label, **point})
        for ax, key in zip(axes, ["oos_loss_native", "oos_sharpe_capped"]):
            ax.plot(curve.effective_dimension, curve[key], color=color, label=label)
            ax.scatter(point["effective_dimension"], point[key], color=color, marker="D", s=35)
    axes[0].set(yscale="log", ylabel="Native response-one loss (log)",
                title="A. Statistical performance")
    axes[0].axhline(1, color="grey", ls="--", lw=.8)
    axes[1].set(ylabel="Implemented annualized Sharpe", title="B. Economic performance")
    axes[1].axhline(0, color="grey", lw=.8)
    for ax in axes:
        ax.set_xlabel("Mean refit effective dimension")
        ax.grid(alpha=.2)
    axes[1].legend(title="Formation period", loc="center left", bbox_to_anchor=(1.02, .5))
    fig.suptitle("The complexity-performance relation across fixed calendar periods")
    return finish(fig, "figure_10_complexity_by_period",
                  {"": pd.concat(tables), "_selected": pd.DataFrame(points)}, figure_dir=figure_dir,
                  note="Lines join fixed-rho strategies; complexity varies across years. Diamonds show annual validation selection.\n"
                       "Calendar splits are fixed in advance of this diagnostic. The curves do not establish a unique population optimum.")


if __name__ == "__main__":
    main()
