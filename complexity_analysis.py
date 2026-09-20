"""Analyze saved runs and add C/T figures. This script never trains a model."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import complexity as charts
from plot_style import KERNEL_COLORS, save_figure
from Utils.utils import compute_sharpe_ratio


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/complexity_analysis"
BOOTSTRAPS = 1000


def bootstrap_curves(monthly_returns, lambdas):
    """Resample whole 12-month test windows; preserve paired lambda returns."""
    data = monthly_returns.copy()
    data["window"] = pd.to_datetime(data["formation_date"]).dt.year
    counts = data.groupby(["window", "lambda"]).size().unstack().reindex(columns=lambdas)
    if not (counts == 12).all().all():
        raise ValueError("Bootstrap requires complete 12-month test windows")
    sums = data.groupby(["window", "lambda"])["raw_portfolio_return"].sum().unstack()
    data["squared_return"] = data["raw_portfolio_return"] ** 2
    squares = data.groupby(["window", "lambda"])["squared_return"].sum().unstack()
    sums = sums.reindex(columns=lambdas).to_numpy()
    squares = squares.reindex(columns=lambdas).to_numpy()
    windows = len(sums)
    rng = np.random.default_rng(2024)
    weights = rng.multinomial(windows, np.full(windows, 1 / windows), size=BOOTSTRAPS)
    total = windows * 12
    mean = weights @ sums / total
    variance = (weights @ squares - total * mean**2) / (total - 1)
    sharpe = np.sqrt(12) * mean / np.sqrt(variance)
    return sharpe, weights


def analyze_kernel(kernel, characteristic_name="all", input_dimension=132):
    folder = ROOT / "results" / kernel / characteristic_name
    diagnostics = pd.read_parquet(folder / "lambda_diagnostics.parquet")
    monthly = pd.read_parquet(folder / "lambda_portfolio_returns.parquet")
    portfolio = pd.read_parquet(folder / "portfolio_returns.parquet").sort_values("return_date")
    data, summary = charts.plot_relative_graphs(diagnostics, monthly, kernel, folder)
    data.to_parquet(folder / "relative_complexity_diagnostics.parquet", index=False)
    summary = summary.sort_values("lambda").reset_index(drop=True)
    absolute = charts.make_summary(diagnostics, monthly).sort_values("lambda").reset_index(drop=True)
    assert np.allclose(summary.test_sharpe, absolute.test_sharpe)
    assert (data.relative_complexity >= 0).all() and (data.relative_complexity <= 1 + 1e-10).all()
    spectrum = pd.read_parquet(folder / "kernel_eigenvalues.parquet")
    dimensions = spectrum.groupby("test_year").size()
    Ts = data.groupby("test_year").estimation_months.first()
    feature_count = input_dimension + 1 if kernel == "linear" else 2000
    expected = np.minimum(Ts, feature_count)
    assert np.array_equal(dimensions.to_numpy(), expected.to_numpy())
    assert portfolio.gross_exposure.max() <= 2 + 1e-10

    draws, weights = bootstrap_curves(monthly, summary["lambda"].to_numpy())
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    summary["sharpe_ci_low"] = low
    summary["sharpe_ci_high"] = high
    summary["mean_window_oos_sharpe"] = data.groupby("lambda").out_of_sample_sharpe_raw.mean().to_numpy()
    peak_index = summary.test_sharpe.idxmax()
    peak = summary.loc[peak_index]
    boot_indices = draws.argmax(axis=1)
    window_q = data.pivot(index="test_year", columns="lambda", values="relative_complexity")
    boot_q = weights @ window_q.to_numpy() / len(window_q)
    q_at_peak = boot_q[np.arange(BOOTSTRAPS), boot_indices]
    summary.to_parquet(folder / "relative_complexity_summary.parquet", index=False)
    boot = pd.DataFrame({"peak_lambda": summary["lambda"].to_numpy()[boot_indices],
                         "peak_relative_complexity": q_at_peak,
                         "peak_sharpe_ex_post": draws[np.arange(BOOTSTRAPS), boot_indices]})
    boot.to_parquet(folder / "relative_complexity_bootstrap.parquet", index=False)

    ordered = summary.sort_values("complexity")
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    axis.fill_between(ordered.complexity, ordered.sharpe_ci_low, ordered.sharpe_ci_high,
                      color=KERNEL_COLORS[kernel], alpha=0.15, label="95% pointwise block-bootstrap interval")
    axis.plot(ordered.complexity, ordered.test_sharpe, color=KERNEL_COLORS[kernel],
               marker="o", markersize=3, label="Pooled OOS Sharpe")
    charts.mark_optimum(axis, peak.complexity, peak.test_sharpe, charts.optimum_label(peak.complexity, True))
    axis.set_xlabel(r"Average relative complexity $\overline{C/T}$")
    axis.set_ylabel("OOS Sharpe ratio")
    axis.set_title(charts.kernel_labels[kernel] + " · OOS Sharpe and uncertainty")
    axis.legend(loc="lower center", fontsize=8)
    axis.grid(False)
    axis.yaxis.grid(True, alpha=0.4)
    figure.tight_layout()
    save_figure(figure, folder / "relative_complexity_uncertainty.png")
    plt.close(figure)

    annual = data.loc[data.groupby("test_year").out_of_sample_sharpe_raw.idxmax()].sort_values("test_year")
    annual.to_parquet(folder / "relative_complexity_window_optima.parquet", index=False)
    test = annual.loc[annual.test_year == 2024].iloc[0]
    full = pd.read_parquet(folder / "estimated_b.parquet").iloc[0]
    capped = portfolio.portfolio_return.to_numpy()
    wealth = np.r_[1.0, np.cumprod(1 + capped)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    first = ordered.iloc[0]
    last = ordered.iloc[-1]
    same_mean_peak = summary.loc[summary.mean_window_oos_sharpe.idxmax()]
    row = {
        "kernel": kernel, "label": charts.kernel_labels[kernel], "T_min": Ts.min(), "T_max": Ts.max(),
        "windows": len(Ts), "months": len(portfolio), "lambdas": len(summary),
        "peak_lambda": peak["lambda"], "peak_q": peak.complexity, "peak_sharpe": peak.test_sharpe,
        "peak_sharpe_ci_low": peak.sharpe_ci_low, "peak_sharpe_ci_high": peak.sharpe_ci_high,
        "peak_q_boot_p025": np.quantile(q_at_peak, 0.025), "peak_q_boot_p975": np.quantile(q_at_peak, 0.975),
        "peak_q_boot_median": np.median(q_at_peak),
        "bootstrap_peak_at_endpoint_share": np.mean((boot_indices == 0) | (boot_indices == len(summary)-1)),
        "smallest_q_sharpe": first.test_sharpe, "largest_q": last.complexity,
        "largest_q_sharpe": last.test_sharpe, "drop_from_peak": peak.test_sharpe-last.test_sharpe,
        "drop_ci_low": np.quantile(draws[:,peak_index]-draws[:,0],0.025),
        "drop_ci_high": np.quantile(draws[:,peak_index]-draws[:,0],0.975),
        "train_sharpe_at_peak": peak.train_sharpe, "train_sharpe_at_largest_q": last.train_sharpe,
        "test2024_q": test.relative_complexity, "test2024_C": test.effective_dimension,
        "test2024_lambda": test["lambda"], "test2024_sharpe": test.out_of_sample_sharpe_raw,
        "annual_peak_at_endpoint_share": np.mean(annual["lambda"].isin([summary["lambda"].min(),summary["lambda"].max()])),
        "mean_window_peak_q": same_mean_peak.complexity, "mean_window_peak_sharpe": same_mean_peak.mean_window_oos_sharpe,
        "capped_selected_sharpe": compute_sharpe_ratio(capped), "capped_cagr": wealth[-1]**(12/len(capped))-1,
        "capped_max_drawdown": drawdown.min(), "max_gross_exposure": portfolio.gross_exposure.max(),
        "lengthscale": portfolio.lengthscale.iloc[0], "estimated_b": full.estimated_b,
        "first_return_date": str(portfolio.return_date.min().date()),
        "last_return_date": str(portfolio.return_date.max().date()),
    }
    print(kernel, 'q*',round(peak.complexity,3),'SR',round(peak.test_sharpe,3),flush=True)
    return row, annual, ordered


def main():
    global OUTPUT
    import argparse
    from download_JKP.read_dataset import DEFAULT_DATA_DIR, available_jkp_characteristics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characteristics", nargs="+", default=["all"])
    args = parser.parse_args()
    characteristic_name = "_".join(args.characteristics)
    input_dimension = (len(available_jkp_characteristics(DEFAULT_DATA_DIR))
                       if characteristic_name == "all" else len(args.characteristics))
    OUTPUT = ROOT / "results" / ("complexity_analysis" if characteristic_name == "all"
                                 else f"complexity_analysis_{characteristic_name}")
    OUTPUT.mkdir(exist_ok=True, parents=True)
    rows, annuals, curves = [], [], []
    for kernel in charts.kernel_labels:
        row, annual, curve = analyze_kernel(kernel, characteristic_name, input_dimension)
        rows.append(row)
        annuals.append(annual)
        curves.append((kernel, curve))
    pd.DataFrame(rows).to_parquet(OUTPUT / "kernel_comparison.parquet", index=False)
    pd.concat(annuals).to_parquet(OUTPUT / "window_optima.parquet", index=False)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for axis, (kernel, curve) in zip(axes.flat, curves):
        peak = curve.loc[curve.test_sharpe.idxmax()]
        axis.fill_between(curve.complexity, curve.sharpe_ci_low, curve.sharpe_ci_high,
                          color=KERNEL_COLORS[kernel], alpha=0.15)
        axis.plot(curve.complexity, curve.test_sharpe, color=KERNEL_COLORS[kernel], marker="o", markersize=2)
        charts.mark_optimum(axis, peak.complexity, peak.test_sharpe, "OOS optimum")
        axis.set_title(charts.kernel_labels[kernel])
        axis.set_xlabel(r"Average $C/T$")
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.4)
    axes[0,0].set_ylabel("Pooled OOS Sharpe")
    axes[1,0].set_ylabel("Pooled OOS Sharpe")
    figure.suptitle(f"Relative complexity · Six kernels · {characteristic_name}", fontsize=15)
    figure.text(0.5, 0.015, "Shading: 95% pointwise intervals, 1,000 paired test-window bootstrap draws. Unconstrained returns.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0,0.04,1,0.95))
    save_figure(figure, OUTPUT / "relative_complexity_all_kernels.png")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=characteristic_name == "all")
    for axis, annual, kernel in zip(axes.flat, annuals, charts.kernel_labels):
        axis.plot(annual.test_year, annual.relative_complexity, color=KERNEL_COLORS[kernel],
                   marker="o", markersize=3, linewidth=0.8)
        axis.set_title(charts.kernel_labels[kernel])
        axis.set_xlabel("Test formation year")
        if characteristic_name == "all":
            axis.set_ylim(-0.02,1.02)
        else:
            axis.set_ylim(0, max(annual.relative_complexity.max() * 1.08, 1e-6))
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.4)
    axes[0,0].set_ylabel(r"Window-specific optimal $C/T$")
    axes[1,0].set_ylabel(r"Window-specific optimal $C/T$")
    figure.suptitle("OOS optima across individual test windows", fontsize=15)
    figure.text(0.5,0.015,"Each optimum uses only 12 test returns and is measured ex post. No smoothing or theoretical fit.",
                ha="center",fontsize=9)
    figure.tight_layout(rect=(0,0.04,1,0.95))
    save_figure(figure, OUTPUT / "relative_complexity_optima_over_time.png")
    plt.close(figure)
    print(pd.DataFrame(rows).to_string(index=False))
    from complexity_report import build_report
    build_report(output=OUTPUT, characteristic_name=characteristic_name,
                 characteristics=args.characteristics)


if __name__ == "__main__":
    main()
