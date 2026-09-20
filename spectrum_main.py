"""Create the compact main-text spectrum figure.

Reads the raw-spectrum outputs created by eigenvalues.py. Main text keeps
Linear, NTK, and Matérn 3/2. Gaussian and the other Matérn kernels remain
available as robustness figures.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_style import KERNEL_COLORS, save_figure, use_plot_style

use_plot_style()
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "spectrum_main.png"

kernels = [
    ("linear", "Linear"),
    ("ntk", "NTK"),
    ("matern32", "Matérn 3/2"),
]

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))

for ax, (name, label) in zip(axes, kernels):
    folder = ROOT / "results" / name / "all"
    eig = pd.read_parquet(folder / "kernel_eigenvalues.parquet")
    year = int(eig["test_year"].max())
    s = eig.loc[eig["test_year"].eq(year)].sort_values("eigenvalue_number").copy()
    largest = float(s["eigenvalue"].max())
    s = s.loc[s["eigenvalue"] > largest * 1e-10]

    ax.plot(
        s["eigenvalue_number"],
        s["eigenvalue"],
        color=KERNEL_COLORS[name],
        linewidth=1.5,
    )
    ax.set_title(label)
    ax.set_xlabel("Direction rank")
    ax.grid(False)
    ax.yaxis.grid(True, alpha=0.3)

axes[0].set_ylabel(r"Raw eigenvalue $\hat{\mu}_j$")
fig.suptitle("Managed-portfolio spectra", y=1.01, fontsize=13)
fig.tight_layout()
save_figure(fig, OUT)
plt.close(fig)

print(OUT)
