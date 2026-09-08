"""Plot Sharpe ratio, lambda and complexity for every kernel."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import KERNEL_COLORS, save_figure, use_plot_style
from Utils.utils import compute_sharpe_ratio


use_plot_style()

characteristics = "all"
kernel_names = [
    "linear",
    "gaussian",
    "ntk",
    "matern12",
    "matern32",
    "matern52",
]
kernel_labels = {
    "linear": "Linear",
    "gaussian": "Gaussian",
    "ntk": "NTK",
    "matern12": "Matern 1/2",
    "matern32": "Matern 3/2",
    "matern52": "Matern 5/2",
}

project_folder = Path(__file__).resolve().parent
if characteristics == "all":
    characteristic_name = "all"
else:
    characteristic_name = "_".join(characteristics)


def make_summary(results_folder):
    """Create one row for every lambda."""
    diagnostics = pd.read_parquet(
        results_folder / "lambda_diagnostics.parquet"
    )
    monthly_returns = pd.read_parquet(
        results_folder / "lambda_portfolio_returns.parquet"
    )

    summary = diagnostics.groupby(
        "lambda",
        as_index=False,
    ).agg(
        effective_dimension=("effective_dimension", "mean"),
        in_sample_sharpe=("in_sample_sharpe", "mean"),
        validation_sharpe=("validation_sharpe_raw", "mean"),
        validation_loss=("validation_loss_raw", "mean"),
    )

    out_of_sample_sharpe = monthly_returns.groupby(
        "lambda"
    )["raw_portfolio_return"].agg(
        lambda returns: compute_sharpe_ratio(
            returns.to_numpy()
        )
    )
    out_of_sample_sharpe = (
        out_of_sample_sharpe
        .rename("out_of_sample_sharpe")
        .reset_index()
    )

    return summary.merge(
        out_of_sample_sharpe,
        on="lambda",
    )


def mark_special_points(
    axis,
    grid_boundary,
    validation_optimum,
    oos_peak,
):
    """Mark the selected complexity, grid boundary and OOS peak."""
    axis.axvline(
        validation_optimum["effective_dimension"],
        color="#009E73",
        linestyle="--",
        alpha=0.8,
    )
    axis.axvline(
        grid_boundary["effective_dimension"],
        color="#888888",
        linestyle=":",
        alpha=0.8,
    )
    axis.scatter(
        grid_boundary["effective_dimension"],
        grid_boundary["out_of_sample_sharpe"],
        marker="X",
        s=220,
        color="#888888",
        edgecolor="black",
        linewidth=1.5,
        zorder=6,
        label="Grid boundary",
    )
    axis.scatter(
        validation_optimum["effective_dimension"],
        validation_optimum["validation_sharpe"],
        marker="D",
        s=110,
        color="#009E73",
        edgecolor="black",
        zorder=7,
        label=r"Optimal complexity $C^*$ (validation)",
    )
    axis.scatter(
        oos_peak["effective_dimension"],
        oos_peak["out_of_sample_sharpe"],
        marker="*",
        s=140,
        color="#F0E442",
        edgecolor="black",
        zorder=8,
        label="OOS Sharpe peak (ex post)",
    )
    axis.text(
        grid_boundary["effective_dimension"],
        0.98,
        f"Grid boundary = {grid_boundary['effective_dimension']:.1f}",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        color="#666666",
    )
    axis.annotate(
        "Validation optimum\n"
        f"C* = {validation_optimum['effective_dimension']:.1f}\n"
        f"lambda = {validation_optimum['lambda']:.1e}",
        (
            validation_optimum["effective_dimension"],
            validation_optimum["validation_sharpe"],
        ),
        xytext=(42, 38),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    axis.annotate(
        "OOS peak\n"
        f"C = {oos_peak['effective_dimension']:.1f}\n"
        f"lambda = {oos_peak['lambda']:.1e}",
        (
            oos_peak["effective_dimension"],
            oos_peak["out_of_sample_sharpe"],
        ),
        xytext=(-125, -68),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )


def plot_complexity(kernel_name):
    """Create the three complexity graphs for one kernel."""
    kernel_label = kernel_labels[kernel_name]
    results_folder = (
        project_folder
        / "results"
        / kernel_name
        / characteristic_name
    )
    summary = make_summary(results_folder)
    summary = summary.sort_values("effective_dimension")

    grid_boundary = summary.loc[
        summary["effective_dimension"].idxmax()
    ]
    validation_optimum = summary.loc[
        summary["validation_sharpe"].idxmax()
    ]
    loss_optimum = summary.loc[
        summary["validation_loss"].idxmin()
    ]
    oos_peak = summary.loc[
        summary["out_of_sample_sharpe"].idxmax()
    ]

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(
        summary["effective_dimension"],
        summary["validation_sharpe"],
        label="Validation (mean window Sharpe)",
        color=KERNEL_COLORS[kernel_name],
        marker="o",
        markersize=3,
    )
    axis.plot(
        summary["effective_dimension"],
        summary["out_of_sample_sharpe"],
        label="OOS (pooled over 564 months)",
        color="#333333",
        marker="o",
        markersize=3,
    )
    mark_special_points(
        axis,
        grid_boundary,
        validation_optimum,
        oos_peak,
    )
    in_sample_axis = axis.twinx()
    in_sample_axis.plot(
        summary["effective_dimension"],
        summary["in_sample_sharpe"],
        label="IS (right axis)",
        color="#888888",
        linestyle="--",
    )
    axis.set_xlabel(r"Average complexity  $C(\lambda)$")
    axis.set_ylabel("Validation and OOS Sharpe")
    in_sample_axis.set_ylabel(
        "IS Sharpe",
        color="#666666",
    )
    in_sample_axis.tick_params(
        axis="y",
        colors="#666666",
    )
    in_sample_axis.spines["right"].set_visible(True)
    axis.set_title(
        f"{kernel_label} kernel: Sharpe ratio and complexity"
    )
    handles, marker_labels = axis.get_legend_handles_labels()
    axis.legend(
        handles + in_sample_axis.get_lines(),
        marker_labels + ["IS (right axis)"],
        loc="lower center",
        ncol=2,
        fontsize=9,
    )
    figure.tight_layout()

    sharpe_file = results_folder / "sharpe_vs_complexity.png"
    save_figure(figure, sharpe_file)
    plt.close(figure)

    summary["is_validation_sharpe_peak"] = (
        summary["lambda"] == validation_optimum["lambda"]
    )
    summary["is_validation_loss_minimum"] = (
        summary["lambda"] == loss_optimum["lambda"]
    )
    summary["is_oos_sharpe_peak"] = (
        summary["lambda"] == oos_peak["lambda"]
    )
    summary["is_grid_boundary"] = (
        summary["lambda"] == grid_boundary["lambda"]
    )
    summary_file = results_folder / "complexity_summary.parquet"
    summary.to_parquet(summary_file, index=False)

    diagnostics = pd.read_parquet(
        results_folder / "lambda_diagnostics.parquet"
    )
    last_test_year = diagnostics["test_year"].max()
    one_window = diagnostics[
        diagnostics["test_year"] == last_test_year
    ].copy()
    one_window = one_window.rename(
        columns={
            "validation_sharpe_raw": "validation_sharpe",
            "validation_loss_raw": "validation_loss",
            "out_of_sample_sharpe_raw": "out_of_sample_sharpe",
        }
    )
    one_window = one_window.sort_values(
        "effective_dimension"
    )

    one_grid_boundary = one_window.loc[
        one_window["effective_dimension"].idxmax()
    ]
    one_validation_optimum = one_window.loc[
        one_window["validation_sharpe"].idxmax()
    ]
    one_oos_peak = one_window.loc[
        one_window["out_of_sample_sharpe"].idxmax()
    ]

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot(
        one_window["effective_dimension"],
        one_window["validation_sharpe"],
        label="Validation Sharpe ratio",
        color=KERNEL_COLORS[kernel_name],
        marker="o",
        markersize=2.5,
    )
    axis.plot(
        one_window["effective_dimension"],
        one_window["out_of_sample_sharpe"],
        label="OOS Sharpe ratio",
        color="#222222",
        marker="o",
        markersize=2.5,
    )
    mark_special_points(
        axis,
        one_grid_boundary,
        one_validation_optimum,
        one_oos_peak,
    )
    in_sample_axis = axis.twinx()
    in_sample_axis.plot(
        one_window["effective_dimension"],
        one_window["in_sample_sharpe"],
        label="IS Sharpe ratio (right axis)",
        color="#888888",
        linestyle="--",
    )
    axis.set_xlabel(r"Complexity  $C(\lambda)$")
    axis.set_ylabel("Validation and OOS Sharpe ratio")
    in_sample_axis.set_ylabel("IS Sharpe ratio", color="#666666")
    in_sample_axis.tick_params(axis="y", colors="#666666")
    in_sample_axis.spines["right"].set_visible(True)
    axis.set_title(
        f"{kernel_label} kernel: Single test window ({last_test_year})"
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles + in_sample_axis.get_lines(),
        labels + ["IS Sharpe ratio (right axis)"],
        loc="lower center",
        ncol=2,
        fontsize=9,
    )
    figure.tight_layout()

    one_window_file = (
        results_folder
        / f"sharpe_vs_complexity_{last_test_year}.png"
    )
    save_figure(figure, one_window_file)
    plt.close(figure)

    summary = summary.sort_values("lambda")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.semilogx(
        summary["lambda"],
        summary["effective_dimension"],
        color=KERNEL_COLORS[kernel_name],
        marker="o",
        markersize=3,
    )
    axis.scatter(
        grid_boundary["lambda"],
        grid_boundary["effective_dimension"],
        marker="X",
        s=220,
        color="#888888",
        edgecolor="black",
        linewidth=1.5,
        zorder=6,
        label="Grid boundary",
    )
    axis.scatter(
        validation_optimum["lambda"],
        validation_optimum["effective_dimension"],
        marker="D",
        s=110,
        color="#009E73",
        edgecolor="black",
        zorder=7,
        label=r"Optimal complexity $C^*$ (validation)",
    )
    axis.scatter(
        oos_peak["lambda"],
        oos_peak["effective_dimension"],
        marker="*",
        s=140,
        color="#F0E442",
        edgecolor="black",
        zorder=8,
        label="OOS Sharpe peak (ex post)",
    )
    axis.set_xlabel("Lambda")
    axis.set_ylabel(r"Average complexity  $C(\lambda)$")
    axis.set_title(
        f"{kernel_label} kernel: Complexity and regularization"
    )
    axis.legend(loc="best")
    figure.tight_layout()

    dimension_file = (
        results_folder / "effective_dimension_vs_lambda.png"
    )
    save_figure(figure, dimension_file)
    plt.close(figure)

    figure = plt.figure(figsize=(7.2, 5.4))
    axis = figure.add_subplot(111, projection="3d")
    log_lambda = np.log10(summary["lambda"])

    axis.plot(
        log_lambda,
        summary["effective_dimension"],
        summary["in_sample_sharpe"],
        label="IS",
        color="#888888",
        linestyle="--",
    )
    axis.plot(
        log_lambda,
        summary["effective_dimension"],
        summary["validation_sharpe"],
        label="Validation",
        color=KERNEL_COLORS[kernel_name],
    )
    axis.plot(
        log_lambda,
        summary["effective_dimension"],
        summary["out_of_sample_sharpe"],
        label="OOS",
        color="#333333",
    )
    axis.scatter(
        np.log10(grid_boundary["lambda"]),
        grid_boundary["effective_dimension"],
        grid_boundary["out_of_sample_sharpe"],
        marker="X",
        s=180,
        color="#888888",
        edgecolor="black",
        label="Grid boundary",
    )
    axis.scatter(
        np.log10(validation_optimum["lambda"]),
        validation_optimum["effective_dimension"],
        validation_optimum["validation_sharpe"],
        marker="D",
        s=90,
        color="#009E73",
        edgecolor="black",
        label=r"Optimal $C^*$ (validation)",
    )
    axis.scatter(
        np.log10(oos_peak["lambda"]),
        oos_peak["effective_dimension"],
        oos_peak["out_of_sample_sharpe"],
        marker="*",
        s=110,
        color="#F0E442",
        edgecolor="black",
        label="OOS Sharpe peak",
    )
    axis.set_xlabel("log10(lambda)")
    axis.set_ylabel(r"Average complexity  $C(\lambda)$")
    axis.set_zlabel("Annualized Sharpe ratio")
    axis.set_title(f"{kernel_label} kernel: Three-dimensional view")
    axis.legend()
    figure.tight_layout()

    three_dimensional_file = results_folder / "complexity_3d.png"
    save_figure(figure, three_dimensional_file)
    plt.close(figure)

    print()
    print("Kernel:", kernel_label)
    print(
        "Largest complexity in grid:",
        grid_boundary["effective_dimension"],
    )
    print(
        "Optimal validation complexity:",
        validation_optimum["effective_dimension"],
    )
    print("Optimal validation lambda:", validation_optimum["lambda"])
    print("Minimum-loss complexity:", loss_optimum["effective_dimension"])
    print("Minimum-loss lambda:", loss_optimum["lambda"])
    print("OOS peak complexity:", oos_peak["effective_dimension"])
    print("OOS peak lambda:", oos_peak["lambda"])
    print("Average Sharpe graph:", sharpe_file)
    print("Single-window Sharpe graph:", one_window_file)
    print("Dimension graph:", dimension_file)
    print("3D graph:", three_dimensional_file)
    print("Summary:", summary_file)


for kernel_name in kernel_names:
    plot_complexity(kernel_name)
