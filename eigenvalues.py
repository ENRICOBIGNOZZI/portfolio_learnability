"""Plot the spectrum and estimate b for every kernel."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import KERNEL_COLORS, save_figure, use_plot_style


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
all_estimates = []


for kernel_name in kernel_names:
    kernel_label = kernel_labels[kernel_name]
    results_folder = (
        project_folder
        / "results"
        / kernel_name
        / characteristic_name
    )
    eigenvalues_file = (
        results_folder
        / "kernel_eigenvalues.parquet"
    )
    eigenvalues = pd.read_parquet(
        eigenvalues_file
    )

    last_test_year = eigenvalues["test_year"].max()
    spectrum = eigenvalues[
        eigenvalues["test_year"] == last_test_year
    ].copy()
    spectrum = spectrum.sort_values(
        "eigenvalue_number"
    )
    lengthscale = spectrum["lengthscale"].iloc[0]

    largest_eigenvalue = spectrum["eigenvalue"].max()
    spectrum = spectrum[
        spectrum["eigenvalue"]
        > largest_eigenvalue * 1e-10
    ]

    ranks = spectrum["eigenvalue_number"].to_numpy()
    values = spectrum["eigenvalue"].to_numpy()
    normalized_values = values / values.max()

    estimated_b = np.nan
    fitted_values = None

    if len(values) >= 2:
        slope, intercept = np.polyfit(
            np.log(ranks),
            np.log(values),
            1,
        )
        estimated_b = -slope
        fitted_values = (
            np.exp(intercept)
            * ranks ** (-estimated_b)
        )
        fitted_values = fitted_values / values.max()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.5),
    )

    histogram_weights = np.full(
        len(normalized_values),
        100.0 / len(normalized_values),
    )
    axes[0].hist(
        normalized_values,
        bins=np.linspace(0.0, 1.0, 61),
        weights=histogram_weights,
        color=KERNEL_COLORS[kernel_name],
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel(r"Normalized eigenvalue  $\mu_j / \mu_1$")
    axes[0].set_ylabel("Share (%) - log scale")
    axes[0].set_yscale("log")
    axes[0].set_title("Distribution")

    axes[1].loglog(
        ranks,
        normalized_values,
        label="Eigenvalues",
        color=KERNEL_COLORS[kernel_name],
    )

    if fitted_values is not None:
        axes[1].loglog(
            ranks,
            fitted_values,
            linestyle="--",
            color="#333333",
            label=rf"Power-law fit: $\hat{{b}}={estimated_b:.3f}$",
        )

    axes[1].set_xlabel("Eigenvalue rank")
    axes[1].set_ylabel(r"Normalized eigenvalue  $\mu_j / \mu_1$")
    axes[1].set_title("Ranked spectrum and fit")
    axes[1].legend(loc="upper right")

    title = (
        f"{kernel_label} kernel: Normalized eigenvalue distribution"
    )
    if kernel_name in {
        "gaussian",
        "matern12",
        "matern32",
        "matern52",
    }:
        title = (
            f"{title}  (lengthscale = {lengthscale:.3g})"
        )
    figure.suptitle(title)
    figure.tight_layout()

    plot_file = (
        results_folder
        / "eigenvalue_spectrum.png"
    )
    save_figure(figure, plot_file)
    plt.close(figure)

    estimate = {
        "kernel": kernel_name,
        "test_year": last_test_year,
        "lengthscale": lengthscale,
        "estimated_b": estimated_b,
        "number_of_eigenvalues": len(values),
    }
    all_estimates.append(estimate)

    estimate_file = (
        results_folder
        / "estimated_b.parquet"
    )
    pd.DataFrame(
        [estimate]
    ).to_parquet(
        estimate_file,
        index=False,
    )

    print()
    print("Kernel:", kernel_label)
    print("Estimated b:", estimated_b)
    print("Spectrum:", plot_file)


all_estimates_file = (
    project_folder
    / "results"
    / f"estimated_b_all_kernels_{characteristic_name}.parquet"
)
pd.DataFrame(all_estimates).to_parquet(
    all_estimates_file,
    index=False,
)

print()
print("All estimates saved in:", all_estimates_file)
