"""Free portfolio exposures and the realized effect of implementation rules."""
import numpy as np
import pandas as pd
from Utils.paper_figures import PRIMARY, LINEAR, FIGURES, BLUE, RED, style, finish, percent, realization_dates
import matplotlib.pyplot as plt
from Utils.empirical_analysis import load_result_bundle


def main(figure_dir=FIGURES):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    tables, summaries = [], []
    for name, directory, color in [("Matérn-3/2", PRIMARY, RED), ("Linear ridge", LINEAR, BLUE)]:
        b = load_result_bundle(directory, load_models=False)
        s = b["selected"].sort_values("date").copy()
        s["return_month"] = realization_dates(s)
        s["method"] = name
        s["long_exposure"] = (s.sum_absolute_weights+s.sum_weights)/2
        s["short_absolute_exposure"] = (s.sum_absolute_weights-s.sum_weights)/2
        s["realized_volatility_36m"] = s.portfolio_return.rolling(36).std(ddof=1)*np.sqrt(12)
        s["cap_binding"] = s.leverage_scale < 1-1e-10
        axes[0, 0].plot(s.return_month, s.sum_absolute_weights, label=name, color=color)
        axes[0, 1].plot(s.return_month, s.sum_weights, label=name, color=color)
        axes[1, 0].plot(s.return_month, s.realized_volatility_36m, label=name, color=color)
        axes[1, 1].plot(s.return_month, s.turnover.rolling(12).mean(), label=name, color=color)
        summaries.append({"method": name, "max_gross": s.sum_absolute_weights.max(),
                          "mean_net": s.sum_weights.mean(), "mean_long": s.long_exposure.mean(),
                          "mean_short_absolute": s.short_absolute_exposure.mean(),
                          "cap_binding_months": int(s.cap_binding.sum()), "months": len(s),
                          "realized_annualized_volatility": s.portfolio_return.std(ddof=1)*np.sqrt(12),
                          "target_volatility": b["metadata"]["target_annualized_volatility"]})
        tables.append(s)
    axes[0, 0].axhline(10, color="grey", ls="--", label="Gross cap = 10")
    axes[0, 0].set(title="A. Gross exposure", ylabel="Sum of absolute weights")
    axes[0, 1].axhline(0, color="grey", lw=.8)
    axes[0, 1].set(title="B. Net exposure (unconstrained)", ylabel="Sum of signed weights")
    axes[1, 0].axhline(.10, color="grey", ls="--", label="10% ex-ante target")
    axes[1, 0].set(title="C. Realized risk versus target", ylabel="36-month annualized volatility")
    percent(axes[1, 0])
    axes[1, 1].set(title="D. Changes in target weights", ylabel="12-month mean L1 weight change")
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(loc="upper left"); ax.set_xlabel("Return realization month")
    fig.suptitle("Exposure and risk behind the OOS performance")
    return finish(fig, "figure_8_exposure_and_risk",
                  {"_monthly": pd.concat(tables), "_summary": pd.DataFrame(summaries)},
                  figure_dir=figure_dir,
                  note="Volatility scaling uses validation data and is fixed before each test window. Risk can exceed the target OOS.\n"
                       "L1 changes compare consecutive target weights; they are not drift-adjusted trading turnover. No costs.")


if __name__ == "__main__":
    main()
