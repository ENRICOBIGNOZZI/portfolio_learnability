"""Figure 4: managed-payoff spectrum and response-one target alignment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Utils.empirical_analysis import load_result_bundle
from Utils.paper_figures import finish, style


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "matern32_rff_all_characteristics"
FIGURE_DIR = PROJECT_ROOT / "results" / "paper_figures"


def main(results_dir=RESULTS_DIR, figure_dir=FIGURE_DIR):
    style()
    bundle = load_result_bundle(results_dir)
    models = bundle["models"]
    eigenvalues = np.asarray(models["refit_eigenvalues"], dtype=float)[:, ::-1]
    alignment = np.asarray(models["refit_target_alignment"], dtype=float)[:, ::-1]
    valid = np.isfinite(eigenvalues)
    normalized = np.divide(
        eigenvalues,
        eigenvalues[:, [0]],
        out=np.full_like(eigenvalues, np.nan),
        where=eigenvalues[:, [0]] > 0,
    )
    ranks = np.arange(1, eigenvalues.shape[1] + 1)
    spectrum_median = np.nanmedian(np.where(valid, normalized, np.nan), axis=0)
    spectrum_lower = np.nanquantile(np.where(valid, normalized, np.nan), 0.25, axis=0)
    spectrum_upper = np.nanquantile(np.where(valid, normalized, np.nan), 0.75, axis=0)
    alignment = np.where(valid, alignment, np.nan)
    cumulative_alignment = np.nancumsum(alignment, axis=1) / np.nansum(
        alignment, axis=1
    )[:, None]
    alignment_median = np.nanmedian(cumulative_alignment, axis=0)
    alignment_lower = np.nanquantile(cumulative_alignment, 0.25, axis=0)
    alignment_upper = np.nanquantile(cumulative_alignment, 0.75, axis=0)

    selected_lambda = (models["rho_grid"][models["selected_indices"]]
                       * models["refit_spectral_scales"])
    filters = eigenvalues / (eigenvalues + selected_lambda[:, None])
    models.close()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    axes[0].plot(ranks, spectrum_median, color="#1f77b4")
    axes[0].fill_between(
        ranks, spectrum_lower, spectrum_upper, color="#1f77b4", alpha=0.2
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Eigendirection rank")
    axes[0].set_ylabel(r"Normalized eigenvalue $\widehat\mu_j/\widehat\mu_1$")
    axes[0].set_title("A. Economic spectrum")
    axes[0].grid(alpha=0.25)

    axes[1].plot(ranks, alignment_median, color="#d62728")
    axes[1].fill_between(
        ranks, alignment_lower, alignment_upper, color="#d62728", alpha=0.2
    )
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("Eigendirection rank")
    axes[1].set_ylabel("Cumulative response-one target alignment")
    axes[1].set_title("B. Empirical target projection")
    axes[1].grid(alpha=0.25)
    axes[2].plot(ranks, np.nanmedian(filters, axis=0), color="#39826D")
    axes[2].fill_between(ranks, np.nanquantile(filters, .25, axis=0),
                         np.nanquantile(filters, .75, axis=0), color="#39826D", alpha=.2)
    axes[2].axhline(.5, color="grey", ls="--", lw=.8)
    axes[2].set(xlabel="Within-window eigendirection rank", ylabel="Selected filter: mu / (mu + lambda)",
                title="C. Directions retained by validation", ylim=(0, 1.02))
    axes[2].grid(alpha=.25)
    figure.suptitle("Matérn-3/2 managed-payoff spectrum and target alignment")
    figure.tight_layout()

    data = pd.DataFrame(
        {
            "rank": ranks,
            "spectrum_median": spectrum_median,
            "spectrum_q25": spectrum_lower,
            "spectrum_q75": spectrum_upper,
            "cumulative_alignment_median": alignment_median,
            "cumulative_alignment_q25": alignment_lower,
            "cumulative_alignment_q75": alignment_upper,
            "selected_filter_median": np.nanmedian(filters, axis=0),
            "selected_filter_q25": np.nanquantile(filters, .25, axis=0),
            "selected_filter_q75": np.nanquantile(filters, .75, axis=0),
        }
    )
    return finish(figure, "figure_4_economic_spectrum", {"": data}, figure_dir=figure_dir,
                  note="Lines: medians; shading: across-window IQR, not confidence intervals. Ranks are local to each window.\n"
                       "Target projections are sample diagnostics; they do not identify the population source parameter r or spectral exponent b.")


if __name__ == "__main__":
    main()
