"""Separate native response-one loss into direction and scale components."""
import pandas as pd
from Utils.paper_figures import PRIMARY, FIGURES, BLUE, RED, GREEN, style, finish
import matplotlib.pyplot as plt
from Utils.empirical_analysis import load_result_bundle, response_one_decomposition


def main(results_dir=PRIMARY, figure_dir=FIGURES):
    style()
    bundle = load_result_bundle(results_dir, load_models=False)
    complexities = bundle["diagnostics"].groupby("rho").effective_dimension.mean()
    rows = []
    for rho, sample in bundle["monthly"].groupby("rho"):
        rows.append({"rho": rho, "effective_dimension": complexities.loc[rho],
                     **response_one_decomposition(sample.portfolio_return_native)})
    data = pd.DataFrame(rows).sort_values("effective_dimension")
    selected = response_one_decomposition(bundle["selected"].portfolio_return_native)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(data.effective_dimension, data.native_loss, color=BLUE, label="Native response-one loss")
    axes[0].plot(data.effective_dimension, data.ex_post_scale_minimum_loss, color=GREEN,
                 label="Minimum under ex-post constant scaling")
    axes[0].plot(data.effective_dimension, data.scale_gap, color=RED, label="Nonnegative scale gap")
    axes[0].set_yscale("symlog", linthresh=.05)
    axes[0].set(ylabel="Empirical loss (symlog)",
                title="A. Exact loss decomposition")
    axes[0].legend()
    axes[1].plot(data.effective_dimension, data.ex_post_scale, color=BLUE)
    axes[1].axhline(1, color="grey", ls="--", label="Native scale a = 1")
    axes[1].axhline(0, color="grey", lw=.7)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set(ylabel="Ex-post multiplier a* (symlog)",
                title="B. Scale diagnostic: mean(R) / mean(R²)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Mean refit effective dimension"); ax.grid(alpha=.2)
    fig.suptitle("What drives response-one loss: direction or scale?")
    return finish(fig, "appendix_loss_calibration",
                  {"": data, "_selected": pd.DataFrame([selected])}, figure_dir=figure_dir,
                  note="Identity: Q(R) = min_a Q(aR) + mean(R²)(1-a*)². The minimum is an ex-post descriptive quantity.\n"
                       "It is not used to tune or trade, ignores the gross cap, and allows negative scaling; squared Sharpe loses return sign.")


if __name__ == "__main__":
    main()
