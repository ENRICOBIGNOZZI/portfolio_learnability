"""Plot the out-of-sample equity line."""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from Utils.utils import compute_equity_line


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--characteristics", nargs="+", default=["be_me"])
args = parser.parse_args()
kernel_name = args.kernel
characteristics = args.characteristics

project_folder = Path(__file__).resolve().parent
characteristic_name = "_".join(characteristics)
results_folder = (
    project_folder
    / "results"
    / kernel_name
    / characteristic_name
)

returns_file = (
    results_folder
    / "portfolio_returns.parquet"
)

results = pd.read_parquet(returns_file)
results = results.sort_values("return_date")

results["equity"] = compute_equity_line(
    results["portfolio_return"],
    initial_value=1.0,
)

plt.figure(figsize=(10, 5))
plt.plot(
    results["return_date"],
    results["equity"],
)
plt.title(
    f"Equity line - {kernel_name} - {characteristic_name}"
)
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(alpha=0.3)
plt.tight_layout()

plot_file = results_folder / "equity.png"
plt.savefig(
    plot_file,
    dpi=200,
)
plt.close()

print("Equity plot saved in:", plot_file)
