"""Regenerate expanding-window figures after exact response-target rescaling."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import expanding_window_gaussian as gaussian
import expanding_window_matern as matern
from plot_style import save_figure, use_plot_style

ROOT = Path(__file__).resolve().parent
TARGET = 0.1


def gaussian_validation_plot(annual: pd.DataFrame, folder: Path):
    use_plot_style()
    figure, axes = plt.subplots(2, 1, figsize=(7.3, 6.3), sharex=True)

    for rule in ("annual_validation", "fixed", "inverse_T"):
        data = annual.loc[annual["rule"].eq(rule)].sort_values("T")
        axes[0].plot(
            data["T"],
            data["C_over_T_ma5"],
            label=gaussian.RULE_LABELS[rule],
            linewidth=1.8,
            linestyle=":" if rule == "annual_validation" else "-",
        )
        axes[1].plot(
            data["T"],
            data["annual_oos_loss_ma5"],
            label=gaussian.RULE_LABELS[rule],
            linewidth=1.8,
            linestyle=":" if rule == "annual_validation" else "-",
        )

    axes[0].set_ylabel(r"$\mathcal{C}_T/T$")
    axes[0].set_title("Relative complexity")
    axes[1].set_ylabel(r"Response-target loss $Q_{0.1}$")
    axes[1].set_xlabel(r"Expanding estimation history $T$ (months)")
    axes[1].set_title("Out-of-sample quadratic criterion")

    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)

    axes[0].legend(loc="best")
    figure.text(
        0.5,
        0.012,
        "Response target c=0.1. Five-year centered moving averages shown for "
        "readability; underlying annual values are saved separately.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(
        figure,
        folder / "gaussian_expanding_validation_benchmark.png",
    )
    plt.close(figure)


def main():
    gfolder = ROOT / "results" / "expanding_gaussian" / "all"
    gannual = pd.read_parquet(gfolder / "annual_diagnostics.parquet")
    gaussian.plot_main(gannual, gfolder)
    gaussian_validation_plot(gannual, gfolder)
    gaussian.plot_raw_appendix(gannual, gfolder)

    mfolder = ROOT / "results" / "expanding_matern" / "all"
    mannual = pd.read_parquet(mfolder / "annual_diagnostics.parquet")
    for kernel in matern.MATERN_NU:
        matern.plot_kernel_main(kernel, mannual, mfolder)
        matern.plot_raw_appendix(kernel, mannual, mfolder)
    matern.plot_all_matern(mannual, mfolder)

    print("Regenerated expanding-window figures for response target c=0.1")


if __name__ == "__main__":
    main()
