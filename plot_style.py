"""Shared style for all result plots."""

import matplotlib.pyplot as plt


KERNEL_COLORS = {
    "linear": "#1F4E79",
    "gaussian": "#B45F06",
    "ntk": "#38761D",
    "matern12": "#A61C00",
    "matern32": "#741B47",
    "matern52": "#674EA7",
}

KERNEL_LINESTYLES = {
    "linear": "-",
    "gaussian": "--",
    "ntk": "-.",
    "matern12": ":",
    "matern32": (0, (5, 2)),
    "matern52": (0, (3, 1, 1, 1)),
}


def use_plot_style():
    """Use a clean journal-style format."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.titleweight": "normal",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure, png_file):
    """Save a high-resolution PNG and a vector PDF."""
    figure.savefig(png_file, dpi=300)
    figure.savefig(png_file.with_suffix(".pdf"))
