"""Paired uncertainty and prespecified calendar subperiod comparisons."""
import numpy as np
import pandas as pd
from Utils.paper_figures import PRIMARY, LINEAR, FIGURES, BLUE, RED, style, finish
import matplotlib.pyplot as plt
from Utils.empirical_analysis import load_result_bundle, paired_block_summary, performance_metrics

# Fixed calendar splits, never chosen using observed performance.
PERIODS = [(1978, 2024), (1978, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]


def main(figure_dir=FIGURES):
    style()
    a = load_result_bundle(PRIMARY, load_models=False)["selected"]
    b = load_result_bundle(LINEAR, load_models=False)["selected"]
    paired = a[["date", "portfolio_return"]].merge(
        b[["date", "portfolio_return"]], on="date", suffixes=("_matern", "_linear"),
        validate="one_to_one").sort_values("date")
    intervals, metrics = [], []
    for start, end in PERIODS:
        sample = paired.loc[paired.date.dt.year.between(start, end)]
        name = f"{start}-{end}"
        for block in (6, 12, 24):
            result = paired_block_summary(sample.portfolio_return_matern,
                                          sample.portfolio_return_linear, block_length=block)
            result["period"] = name
            intervals.append(result)
        for model in ("matern", "linear"):
            metrics.append({"period": name, "method": model,
                            **performance_metrics(sample[f"portfolio_return_{model}"])})
    intervals, metrics = pd.concat(intervals), pd.DataFrame(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    labels = [f"{a}-{b}" for a, b in PERIODS]
    y = np.arange(len(labels))
    for method, color, offset in [("matern", RED, -.12), ("linear", BLUE, .12)]:
        values = metrics.loc[metrics.method.eq(method)].set_index("period").reindex(labels)
        axes[0].scatter(values.annualized_sharpe, y+offset, color=color,
                        label="Matérn-3/2" if method == "matern" else "Linear ridge", s=45)
    axes[0].set_yticks(y, labels)
    axes[0].set(xlabel="OOS annualized Sharpe", title="A. Performance by formation period")
    for block, color, offset in [(6, "#B8B8B8", -.16), (12, BLUE, 0), (24, ORANGE_COLOR, .16)]:
        z = intervals.loc[intervals.metric.eq("annualized_sharpe_difference")
                          & intervals.block_length.eq(block)].set_index("period").reindex(labels)
        # Percentile intervals need not contain the point estimate.
        axes[1].hlines(y+offset, z.lower, z.upper, color=color, lw=2, label=f"{block}-month blocks")
        axes[1].scatter(z.estimate, y+offset, color=color, s=20)
    axes[1].set_yticks(y, labels)
    axes[1].set(xlabel="Sharpe difference: Matérn minus linear",
                title="B. Paired 95% bootstrap intervals")
    for ax in axes:
        ax.axvline(0, color="grey", lw=.8); ax.invert_yaxis()
        ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Does the nonlinear portfolio improve on direct linear ridge?")
    return finish(fig, "figure_9_benchmark_uncertainty",
                  {"_intervals": intervals, "_metrics": metrics, "_paired_returns": paired},
                  figure_dir=figure_dir,
                  note="1,000 paired circular block draws per period and block length. Intervals condition on existing fits.\n"
                       "Pointwise descriptive inference; no multiplicity correction. Subperiods use formation years; the last has 60 months.")


ORANGE_COLOR = "#C47728"
if __name__ == "__main__":
    main()
