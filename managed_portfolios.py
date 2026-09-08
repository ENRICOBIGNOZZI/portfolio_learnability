"""Plot one-characteristic managed payoffs for the available kernels.

This file is intentionally illustrative: ``be_me`` is used only to visualize
the monthly managed-payoff direction

    X_t(z) = sum_i k(z, z_it) r^e_{i,t+1}.

The empirical Gaussian training is in ``train_gaussian_model.py`` and uses all
cleaned characteristics.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Kernels.kernel_function import PortfolioKernel
from Utils.utils import _estimate_median_length_scale, build_monthly_panels
from download_JKP.read_dataset import available_jkp_years, load_jkp_value


PROJECT_ROOT = Path(__file__).resolve().parent
CHARACTERISTIC = "be_me"


def plot_managed_portfolios(
    data_dir=PROJECT_ROOT / "data" / "JKP_USA",
    output_dir=PROJECT_ROOT / "results" / "managed_portfolios",
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    latest_year = available_jkp_years(data_dir)[-1]
    frame = load_jkp_value(
        data_dir,
        CHARACTERISTIC=CHARACTERISTIC,
        years=[latest_year],
    )
    panels = build_monthly_panels(frame, characteristics=CHARACTERISTIC)
    if not panels:
        raise ValueError("No monthly panel is available for the managed-payoff plot.")
    panel = panels[-1]
    x = panel["x"]
    returns = panel["r"]
    grid = np.linspace(-0.5, 0.5, 301)[:, None]

    ell = _estimate_median_length_scale([panel], random_state=0)
    kernel_values = {
        "Linear": PortfolioKernel.linear_gram(grid, x),
        f"Gaussian (ell={ell:.3f})": PortfolioKernel.gaussian_gram(
            grid, x, ell=ell
        ),
        "Two-layer ReLU NTK": PortfolioKernel.ntk_gram(grid, x),
    }

    output = pd.DataFrame({CHARACTERISTIC: grid[:, 0]})
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    for axis, color, (name, gram) in zip(
        axes, colors, kernel_values.items(), strict=True
    ):
        managed_payoff = gram @ returns
        output[name] = managed_payoff
        axis.plot(grid[:, 0], managed_payoff, color=color, linewidth=2)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_ylabel(r"$X_t(z)$")
        axis.set_title(name)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Cross-sectional rank of be_me")
    figure.suptitle(
        f"Managed payoff directions - {panel['date']:%Y-%m-%d}",
        y=0.995,
    )
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "be_me_managed_payoffs_by_kernel.png"
    data_path = output_dir / "be_me_managed_payoffs_by_kernel.parquet"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    output.to_parquet(data_path, index=False)
    print(f"Saved: {figure_path}")
    print(f"Saved: {data_path}")
    return {"figure": figure_path, "data": data_path}


if __name__ == "__main__":
    plot_managed_portfolios()
