"""Figure 6: economic outcomes and the direct linear-ridge benchmark."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from Utils.empirical_analysis import (
    load_result_bundle,
    performance_metrics,
    rolling_sharpe,
)
from Utils.paper_figures import finish, style, percent, realization_dates
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
MATERN_DIR = PROJECT_ROOT / "results" / "matern32_rff_all_characteristics"
LINEAR_DIR = PROJECT_ROOT / "results" / "linear_all_characteristics"
FIGURE_DIR = PROJECT_ROOT / "results" / "paper_figures"


def _selected_complexity(bundle):
    selected = bundle["diagnostics"].loc[
        bundle["diagnostics"]["selected_by_validation"]
    ]
    return float(selected["effective_dimension"].mean())


def main(
    matern_dir=MATERN_DIR,
    linear_dir=LINEAR_DIR,
    figure_dir=FIGURE_DIR,
):
    style()
    bundles = {
        "Matérn-3/2": load_result_bundle(matern_dir, load_models=False),
        "Direct linear ridge": load_result_bundle(linear_dir, load_models=False),
    }
    common_dates = None
    for bundle in bundles.values():
        dates = set(pd.to_datetime(bundle["selected"]["date"]))
        common_dates = dates if common_dates is None else common_dates & dates
    if not common_dates:
        raise ValueError("Matérn and linear results have no common OOS dates.")

    table_records = []
    prepared = {}
    for method, bundle in bundles.items():
        selected = bundle["selected"].copy()
        selected["date"] = pd.to_datetime(selected["date"])
        selected = selected.loc[selected["date"].isin(common_dates)].sort_values("date")
        prepared[method] = selected
        selected_complexity = _selected_complexity(bundle)
        metrics = performance_metrics(
            selected["portfolio_return"],
            turnover=selected["turnover"],
            gross_exposure=selected["sum_absolute_weights"],
        )
        table_records.append(
            {
                "method": method,
                "selected_effective_dimension": selected_complexity,
                **metrics,
            }
        )

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plotted = []
    colors = {"Matérn-3/2": "#d62728", "Direct linear ridge": "#1f77b4"}
    for method, selected in prepared.items():
        returns = selected["portfolio_return"]
        wealth = (1.0 + returns).cumprod()
        dates = realization_dates(selected)
        drawdown = wealth / wealth.cummax().clip(lower=1) - 1
        rolling = rolling_sharpe(returns, window=60)
        plotted.append(pd.DataFrame({"method": method, "formation_date": selected["date"],
                                     "return_month": dates, "excess_return_index": wealth,
                                     "drawdown": drawdown, "rolling_sharpe": rolling}))
        axes[0].plot(
            dates, wealth, label=method, color=colors[method], linewidth=2
        )
        axes[1].plot(
            dates,
            rolling,
            label=method,
            color=colors[method],
        )
        axes[2].plot(dates, drawdown, color=colors[method], label=method)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Compounded excess-return index (log)")
    axes[0].set_title("A. Excess-return index (10% volatility target, gross cap 10)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Annualized Sharpe")
    axes[1].set_title("B. Rolling 60-month Sharpe")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].set(xlabel="Return realization month", ylabel="Index drawdown",
                title="C. Drawdown from running peak (initial index = 1)")
    percent(axes[2]); axes[2].grid(alpha=.25); axes[2].legend()
    figure.suptitle("Nonlinear direction selection versus direct linear ridge")
    return finish(figure, "figure_6_economic_outcomes",
                  {"_table": pd.DataFrame(table_records), "_monthly": pd.concat(plotted)}, figure_dir=figure_dir,
                  note="Index compounds 1 + portfolio excess return; it is not total account wealth (cash returns are not included).\n"
                       "Predictable volatility target need not equal realized OOS volatility. No transaction costs.")


if __name__ == "__main__":
    main()
