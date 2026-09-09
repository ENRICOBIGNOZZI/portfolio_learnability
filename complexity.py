"""Plot saved train and test results. No training is run here."""

from pathlib import Path
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_style import KERNEL_COLORS, save_figure, use_plot_style
from Utils.utils import compute_sharpe_ratio


use_plot_style()

project_folder = Path(__file__).resolve().parent
kernel_labels = {
    "linear": "Linear",
    "gaussian": "Gaussian",
    "ntk": "NTK",
    "matern12": "Matérn 1/2",
    "matern32": "Matérn 3/2",
    "matern52": "Matérn 5/2",
}


@lru_cache(maxsize=1)
def formation_dates():
    """Count actual monthly panels using the cleaned, zero-imputed cache."""
    dates = []
    for file in sorted((project_folder / "data/JKP_USA_clean").glob("*.parquet")):
        data = pd.read_parquet(file, columns=["eom", "ret_exc_lead1m"])
        valid = np.isfinite(data["ret_exc_lead1m"].to_numpy(dtype=float))
        dates.extend(pd.to_datetime(data.loc[valid, "eom"]).unique())
    if not dates:
        raise ValueError("No cleaned monthly panels found for computing T")
    return pd.DatetimeIndex(dates).unique().sort_values()


def add_relative_complexity(diagnostics):
    """Divide each window's C by its actual number of refit months."""
    data = diagnostics.copy()
    dates = formation_dates()
    counts = {}
    for year, window in data.groupby("test_year"):
        first = window.iloc[0]
        counts[year] = int(((dates.year >= first["train_start"]) &
                            (dates.year <= first["validation_end"])).sum())
    data["estimation_months"] = data["test_year"].map(counts)
    if (data["estimation_months"] <= 0).any():
        raise ValueError("Every estimation window must contain months")
    data["relative_complexity"] = data["effective_dimension"] / data["estimation_months"]
    return data


def optimum_label(value, normalized=False):
    if normalized:
        return rf"Optimal complexity OOS: $(C/T)^* = {value:.3f}$"
    return rf"Optimal complexity OOS: $C^* = {value:.1f}$"


def make_summary(diagnostics, monthly_returns, normalized=False):
    """Average train results and pool the saved monthly test returns."""
    column = "relative_complexity" if normalized else "effective_dimension"
    summary = diagnostics.groupby("lambda", as_index=False).agg(
        complexity=(column, "mean"),
        train_sharpe=("in_sample_sharpe", "mean"),
    )
    test_sharpe = monthly_returns.groupby("lambda")[
        "raw_portfolio_return"
    ].agg(lambda returns: compute_sharpe_ratio(returns.to_numpy()))
    return summary.merge(test_sharpe.rename("test_sharpe"), on="lambda")


def mark_optimum(axis, x, y, label):
    """Highlight only the maximum observed test Sharpe."""
    axis.scatter(
        x, y, marker="*", s=170, color="#E6B44C",
        edgecolor="#333333", linewidth=0.9, zorder=5, label=label,
    )


def plot_sharpe(data, kernel_name, title, x_label, note, output_file, normalized=False):
    """Use aligned panels so large train Sharpe cannot hide test Sharpe."""
    data = data.sort_values("complexity")
    optimum = data.loc[data["test_sharpe"].idxmax()]
    figure, axes = plt.subplots(
        2, 1, sharex=True, figsize=(7.2, 5.6),
        gridspec_kw={"height_ratios": [1, 1.35]},
    )

    for axis, column, color, panel_title in [
        (axes[0], "train_sharpe", "#64748B", "Train (IS)"),
        (axes[1], "test_sharpe", KERNEL_COLORS[kernel_name], "Test (OOS)"),
    ]:
        axis.plot(
            data["complexity"], data[column],
            color=color, linestyle="-", linewidth=1.5,
            marker="o", markersize=3, markeredgewidth=0,
        )
        axis.set_title(panel_title, loc="left", fontsize=10, pad=8)
        axis.set_ylabel("Sharpe ratio")
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.4)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.margins(x=0.035, y=0.22)

    mark_optimum(
        axes[1], optimum["complexity"], optimum["test_sharpe"],
        optimum_label(optimum["complexity"], normalized),
    )
    axes[1].legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0, fontsize=9,
    )
    axes[1].set_title("", loc="left")
    axes[1].set_ylabel("Test (OOS) Sharpe ratio")
    axes[1].set_xlabel(x_label)
    figure.suptitle(title, fontsize=13, y=0.98)
    figure.text(
        0.12, 0.025, note + "\nUnconstrained returns; OOS optimum measured ex post.",
        fontsize=7.5, color="#666666",
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.15, hspace=0.35)
    save_figure(figure, output_file)
    plt.close(figure)


def plot_lambda(summary, kernel_name, output_file, normalized=False):
    """Show complexity versus lambda with only the OOS optimum marked."""
    summary = summary.sort_values("lambda")
    optimum = summary.loc[summary["test_sharpe"].idxmax()]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.semilogx(
        summary["lambda"], summary["complexity"],
        color=KERNEL_COLORS[kernel_name], linestyle="-",
        marker="o", markersize=3, markeredgewidth=0,
    )
    mark_optimum(
        axis, optimum["lambda"], optimum["complexity"],
        optimum_label(optimum["complexity"], normalized),
    )
    axis.set_xlabel(r"Regularization $\lambda$")
    axis.set_ylabel(r"Average relative complexity $\overline{C/T}$" if normalized
                    else r"Average complexity $C(\lambda)$")
    axis.set_title(kernel_labels[kernel_name] + " · Complexity and regularization")
    axis.grid(False)
    axis.yaxis.grid(True, alpha=0.4)
    axis.legend(loc="upper right")
    axis.margins(y=0.12)
    figure.tight_layout()
    save_figure(figure, output_file)
    plt.close(figure)


def plot_oos_3d(diagnostics, kernel_name, output_file, test_year=2024, normalized=False,
                monthly_returns=None):
    """Plot a single test window, or pooled OOS returns when test_year is None."""
    import plotly.graph_objects as go

    column = "relative_complexity" if normalized else "effective_dimension"
    if test_year is None:
        if monthly_returns is None:
            raise ValueError("Pooled OOS Sharpe requires the monthly returns")
        data = make_summary(diagnostics, monthly_returns, normalized).rename(columns={
            "complexity": column, "test_sharpe": "out_of_sample_sharpe_raw",
        }).sort_values("lambda")
        months = monthly_returns["return_date"].nunique()
        series_label = f"Pooled OOS · {months} months"
        complexity_label = r"Average complexity $\overline{C}$"
        if normalized:
            complexity_label = r"Average complexity $\overline{C/T}$"
        interactive_complexity_label = "Average C/T" if normalized else "Average complexity C"
        sharpe_label = "Pooled OOS Sharpe ratio"
        note = (f"Sharpe of {months} concatenated monthly OOS returns; not a mean of annual Sharpes.\n"
                "One point per λ. Unconstrained returns; OOS optimum measured ex post.")
    else:
        data = diagnostics.loc[
            diagnostics["test_year"] == test_year,
            ["lambda", column, "out_of_sample_sharpe_raw"],
        ].sort_values("lambda")
        series_label = f"Test {test_year} (OOS)"
        complexity_label = r"Relative complexity $C/T$" if normalized else r"Complexity $C(\lambda)$"
        interactive_complexity_label = "Relative complexity C/T" if normalized else "Complexity C(λ)"
        sharpe_label = "OOS Sharpe ratio"
        note = "Each point is one λ. Coral star: maximum test Sharpe (ex post). Unconstrained returns."
    if data.empty:
        raise ValueError(f"No saved test results for {test_year}")

    optimum = data.loc[data["out_of_sample_sharpe_raw"].idxmax()]
    log_lambdas = np.log10(data["lambda"].to_numpy())
    complexity = data[column].to_numpy()
    sharpe = data["out_of_sample_sharpe_raw"].to_numpy()
    peak_x = np.log10(optimum["lambda"])
    peak_y = optimum[column]
    peak_z = optimum["out_of_sample_sharpe_raw"]
    title = f"{kernel_labels[kernel_name]} · {series_label}"

    figure = plt.figure(figsize=(8.5, 7))
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axis.plot(log_lambdas, complexity, sharpe, color="#486B78", linewidth=1.8, zorder=2)
    axis.scatter(
        log_lambdas, complexity, sharpe, c=sharpe, cmap="viridis",
        s=24, edgecolor="white", linewidth=0.3, depthshade=False, zorder=3,
    )
    axis.scatter(
        peak_x, peak_y, peak_z, marker="*", s=290, color="#EF476F",
        edgecolor="#182434", linewidth=1, depthshade=False, zorder=10,
    )
    axis.set_xlabel(r"Regularization $\lambda$ (log scale)", labelpad=13)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(
        lambda value, position: rf"$10^{{{value:.0f}}}$"
    ))
    axis.set_ylabel(complexity_label, labelpad=13)
    axis.text2D(-0.13, 0.55, sharpe_label, transform=axis.transAxes,
                rotation=90, va="center", fontsize=10)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.zaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.tick_params(labelsize=10, pad=3)
    axis.view_init(elev=28, azim=225)
    axis.set_proj_type("ortho")
    axis.set_box_aspect((1.25, 1.1, 0.85))
    for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate.set_pane_color((1, 1, 1, 0))
        coordinate._axinfo["grid"]["color"] = (0.65, 0.68, 0.72, 0.22)

    figure.suptitle(title, fontsize=15, y=0.97)
    figure.text(
        0.5, 0.915,
        optimum_label(peak_y, normalized)
        + f"   |   Sharpe = {peak_z:.2f}",
        ha="center", fontsize=11, color="#555555",
    )
    figure.text(0.5, 0.035, note, ha="center", fontsize=8, color="#666666")
    figure.subplots_adjust(left=0.12, right=0.88, bottom=0.17, top=0.86)
    save_figure(figure, output_file)
    plt.close(figure)

    interactive = go.Figure()
    interactive.add_trace(go.Scatter3d(
        x=log_lambdas, y=complexity, z=sharpe,
        customdata=data["lambda"].to_numpy(), mode="lines+markers",
        line={"color": "#486B78", "width": 5},
        marker={"size": 4, "color": sharpe, "colorscale": "Viridis", "showscale": False},
        name=series_label, showlegend=False,
        hovertemplate="λ: %{customdata:.3e}<br>"
                      + ("C/T: %{y:.4f}" if normalized else "Complexity: %{y:.1f}")
                      + f"<br>{sharpe_label}: %{{z:.3f}}<extra>{series_label}</extra>",
    ))
    interactive.add_trace(go.Scatter3d(
        x=[peak_x], y=[peak_y], z=[peak_z], mode="markers",
        marker={"size": 9, "color": "#EF476F", "line": {"color": "#182434", "width": 2}},
        name="Optimal complexity OOS", showlegend=False,
        hovertemplate=f"Optimal OOS λ: {optimum['lambda']:.3e}"
                      + ("<br>C/T: %{y:.4f}" if normalized else "<br>Complexity: %{y:.1f}")
                      + "<br>Sharpe: %{z:.3f}<extra></extra>",
    ))
    ticks = np.arange(np.ceil(log_lambdas.min()), np.floor(log_lambdas.max()) + 1, 2)
    interactive.update_layout(
        title={"text": title + "<br><sup>Drag to rotate · Scroll to zoom · "
                       "Coral point: OOS optimum</sup>", "x": 0.5},
        scene={
            "xaxis": {"title": "λ (log scale)", "tickvals": ticks.tolist(),
                      "ticktext": [f"10<sup>{int(tick)}</sup>" for tick in ticks]},
            "yaxis": {"title": interactive_complexity_label},
            "zaxis": {"title": sharpe_label},
            "camera": {"eye": {"x": -1.6, "y": -1.6, "z": 1.3}},
            "aspectmode": "manual", "aspectratio": {"x": 1.25, "y": 1.1, "z": 0.85},
        },
        template="plotly_white", font={"family": "Georgia, serif", "size": 13},
        height=800, margin={"l": 30, "r": 30, "t": 100, "b": 100},
    )
    interactive.add_annotation(
        text=note.replace("Coral star", "Coral point").replace("\n", "<br>"), x=0.5, y=-0.12,
        xref="paper", yref="paper", showarrow=False,
        font={"size": 12, "color": "#666666"},
    )
    interactive.write_html(
        output_file.with_suffix(".html"), include_plotlyjs=True, full_html=True,
        config={"scrollZoom": True, "displaylogo": False,
                "toImageButtonOptions": {"format": "png", "scale": 3}},
    )


def plot_relative_graphs(diagnostics, monthly_returns, kernel_name, folder):
    """Add normalized versions without replacing the original figures."""
    data = add_relative_complexity(diagnostics)
    summary = make_summary(data, monthly_returns, normalized=True)
    label = kernel_labels[kernel_name]
    months = monthly_returns["return_date"].nunique()
    plot_sharpe(
        summary, kernel_name, label + " · Sharpe ratio and relative complexity",
        r"Average relative complexity $\overline{C/T}$",
        f"C/T computed per window, then averaged. Test: pooled over {months} months.",
        folder / "sharpe_vs_relative_complexity.png", normalized=True,
    )
    year = 2024
    one = data.loc[data["test_year"] == year, [
        "lambda", "relative_complexity", "in_sample_sharpe", "out_of_sample_sharpe_raw",
    ]].rename(columns={"relative_complexity": "complexity", "in_sample_sharpe": "train_sharpe",
                       "out_of_sample_sharpe_raw": "test_sharpe"})
    T = int(data.loc[data["test_year"] == year, "estimation_months"].iloc[0])
    plot_sharpe(
        one, kernel_name, f"{label} · Test window {year}", r"Relative complexity $C/T$",
        f"Single window. T = {T} refit months (train + validation).",
        folder / f"sharpe_vs_relative_complexity_{year}.png", normalized=True,
    )
    plot_lambda(summary, kernel_name, folder / "relative_complexity_vs_lambda.png", normalized=True)
    plot_oos_3d(data, kernel_name, folder / "relative_complexity_3d.png", normalized=True)
    plot_oos_3d(data, kernel_name, folder / "pooled_oos_relative_complexity_3d.png",
                test_year=None, normalized=True, monthly_returns=monthly_returns)
    return data, summary


def plot_complexity(kernel_name, characteristic_name="all"):
    folder = project_folder / "results" / kernel_name / characteristic_name
    diagnostics = pd.read_parquet(folder / "lambda_diagnostics.parquet")
    monthly_returns = pd.read_parquet(folder / "lambda_portfolio_returns.parquet")
    summary = make_summary(diagnostics, monthly_returns)
    months = monthly_returns["return_date"].nunique()
    label = kernel_labels[kernel_name]

    plot_sharpe(
        summary, kernel_name, label + " · Sharpe ratio and complexity",
        r"Average complexity $C(\lambda)$",
        f"Train: mean across windows. Test: pooled over {months} months.",
        folder / "sharpe_vs_complexity.png",
    )

    last_year = diagnostics["test_year"].max()
    one_window = diagnostics.loc[
        diagnostics["test_year"] == last_year,
        ["lambda", "effective_dimension", "in_sample_sharpe", "out_of_sample_sharpe_raw"],
    ].rename(columns={
        "effective_dimension": "complexity",
        "in_sample_sharpe": "train_sharpe",
        "out_of_sample_sharpe_raw": "test_sharpe",
    })
    plot_sharpe(
        one_window, kernel_name, f"{label} · Test window {last_year}",
        r"Complexity $C(\lambda)$", "Single expanding window.",
        folder / f"sharpe_vs_complexity_{last_year}.png",
    )
    plot_lambda(summary, kernel_name, folder / "complexity_vs_lambda.png")
    plot_oos_3d(diagnostics, kernel_name, folder / "complexity_3d.png")
    plot_relative_graphs(diagnostics, monthly_returns, kernel_name, folder)
    print(f"{label}: 9 PNG plots saved in {folder}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characteristics", nargs="+", default=["all"])
    characteristic_name = "_".join(parser.parse_args().characteristics)
    for kernel_name in kernel_labels:
        plot_complexity(kernel_name, characteristic_name)
