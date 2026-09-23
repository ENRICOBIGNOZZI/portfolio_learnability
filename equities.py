"""Plot all kernel equity lines in one graph."""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    KERNEL_COLORS,
    KERNEL_LINESTYLES,
    save_figure,
    use_plot_style,
)
from Utils.utils import compute_equity_line


use_plot_style()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--characteristics", nargs="+", default=["all"])
characteristics = parser.parse_args().characteristics
if characteristics == ["all"]:
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

figure, axis = plt.subplots(figsize=(7.2, 4.6))

series = {}
use_additive = False
for kernel_name in kernel_names:
    returns_file = (
        project_folder
        / "results"
        / kernel_name
        / characteristic_name
        / "portfolio_returns.parquet"
    )
    results = pd.read_parquet(returns_file).sort_values("return_date")
    returns = results["portfolio_return"].to_numpy(dtype=float)
    equity = compute_equity_line(returns)
    if np.any(equity <= 0):
        use_additive = True
    series[kernel_name] = (results["return_date"], returns, equity)

for kernel_name in kernel_names:
    dates, returns, equity = series[kernel_name]
    y = np.cumsum(returns) if use_additive else equity
    axis.plot(
        dates,
        y,
        label=kernel_labels[kernel_name],
        color=KERNEL_COLORS[kernel_name],
        linestyle=KERNEL_LINESTYLES[kernel_name],
    )

axis.set_title(f"Out-of-sample uncapped paths by kernel · {characteristic_name}")
axis.set_xlabel("Date")
if use_additive:
    axis.set_ylabel("Cumulative excess return (additive)")
else:
    axis.set_ylabel("Cumulative wealth - log scale")
    axis.set_yscale("log")
axis.legend(ncol=3, loc="upper left")
axis.text(
    0.99,
    0.02,
    "Uncapped monthly portfolio weights",
    transform=axis.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    color="#555555",
)
figure.tight_layout()

plot_file = (
    project_folder
    / "results"
    / f"equities_{characteristic_name}.png"
)
save_figure(figure, plot_file)
plt.close(figure)

print("Equities saved in:", plot_file)
