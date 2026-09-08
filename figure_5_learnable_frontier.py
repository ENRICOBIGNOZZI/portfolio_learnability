"""Empirical mean-risk rays, showing the observed gross-cap limit."""
import numpy as np
import pandas as pd
from Utils.paper_figures import PRIMARY, FIGURES, BLUE, GREEN, RED, style, finish, percent
import matplotlib.pyplot as plt
from Utils.empirical_analysis import load_result_bundle, fixed_rho_curve


def main(results_dir=PRIMARY, figure_dir=FIGURES):
    style()
    bundle = load_result_bundle(results_dir, load_models=False)
    curve = fixed_rho_curve(bundle)
    # Endpoints are chosen by complexity alone, without inspecting OOS returns.
    low = curve.loc[curve.effective_dimension.idxmin()]
    high = curve.loc[curve.effective_dimension.idxmax()]
    monthly = bundle["monthly"]
    strategies = [
        ("Strongest regularization", monthly.loc[monthly.rho.eq(low.rho)], BLUE),
        ("Validation-selected", bundle["selected"], GREEN),
        ("Weakest regularization", monthly.loc[monthly.rho.eq(high.rho)], RED),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    output, summary = [], []
    for label, frame, color in strategies:
        for ax, col, panel in [(axes[0], "portfolio_return_native", "native"),
                               (axes[1], "portfolio_return", "implemented")]:
            r = frame[col].to_numpy()
            mu, sigma = 12*r.mean(), np.sqrt(12)*r.std(ddof=1)
            sharpe = mu/sigma
            cap = float(bundle["metadata"]["max_gross_exposure"])
            feasible = cap / frame.sum_absolute_weights.max()
            x = np.linspace(0, .20 if panel == "native" else sigma*feasible, 100)
            ax.plot(x, x*sharpe, color=color, label=f"{label}: SR={sharpe:.2f}")
            if panel == "implemented":
                ax.scatter([sigma], [mu], color=color, marker="o", s=45, zorder=4)
                ax.scatter([x[-1]], [x[-1]*sharpe], color=color, marker="|", s=120)
            output.extend({"strategy": label, "panel": panel, "volatility": v,
                           "annualized_mean": v*sharpe} for v in x)
            summary.append({"strategy": label, "panel": panel, "annualized_mean": mu,
                            "annualized_volatility": sigma, "sharpe": sharpe,
                            "observed_cap_scaling_limit": feasible if panel == "implemented" else np.nan})
    for ax in axes:
        ax.axhline(0, color="grey", lw=.7)
        ax.set_xlabel("Annualized volatility")
        ax.set_ylabel("Annualized sample mean excess return")
        percent(ax); percent(ax, "x")
        ax.grid(alpha=.2); ax.legend(loc="upper left")
    axes[0].set_title("A. Native returns: unrestricted rescaling")
    axes[1].set_title("B. Implemented returns: observed cap limits")
    axes[1].legend(loc="upper right")
    fig.suptitle("Empirical risk-return geometry at three regularization rules")
    return finish(fig, "figure_5_learnable_frontier",
                  {"": pd.DataFrame(output), "_strategies": pd.DataFrame(summary)},
                  figure_dir=figure_dir,
                  note="These are sample mean-risk rays, not an estimated population efficient frontier.\n"
                       "Dots: realized strategies. End ticks: constant scaling compatible with cap 10 on observed holdings only.")


if __name__ == "__main__":
    main()
