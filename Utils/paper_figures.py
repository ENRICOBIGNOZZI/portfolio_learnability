"""Small, shared plotting helpers. All analysis runs on saved monthly tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results/matern32_rff_all_characteristics"
LINEAR = ROOT / "results/linear_all_characteristics"
FIGURES = ROOT / "results/paper_figures"
BLUE, ORANGE, GREEN, RED = "#236B8E", "#C47728", "#39826D", "#B34848"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8, "axes.spines.top": False,
        "svg.hashsalt": "portfolio-paper-v1",
        "axes.spines.right": False, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def finish(figure, stem, tables=None, *, figure_dir=FIGURES, note=None):
    """Write a PNG preview, editable vector SVG, and auditable data tables."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if note:
        figure.text(0.025, 0.015, note, fontsize=8, va="bottom")
    figure.tight_layout(rect=(0, 0.08 if note else 0, 1, 0.97))
    for suffix in ("png", "svg"):
        figure.savefig(figure_dir / f"{stem}.{suffix}", dpi=180,
                       metadata={"Date": None} if suffix == "svg" else None)
    plt.close(figure)
    for name, table in (tables or {}).items():
        table.to_parquet(figure_dir / f"{stem}{name}.parquet", index=False)
        table.to_csv(figure_dir / f"{stem}{name}.csv", index=False)
    print(f"Saved: {figure_dir / stem} (PNG, SVG, tables)")
    return figure_dir / f"{stem}.png"


def percent(axis, which="y"):
    getattr(axis, f"{which}axis").set_major_formatter(PercentFormatter(1.0))


def realization_dates(frame):
    """JKP eom is formation month; ret_exc_lead1m is next month's return."""
    return pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(1)
