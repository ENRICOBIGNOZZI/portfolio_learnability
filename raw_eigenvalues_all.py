"""Generate raw managed-portfolio eigenvalue plots for all six paper kernels.

This script reproduces the current empirical preprocessing, selects stationary-kernel
lengthscales using the first chronological validation block exactly as train_model.py,
and then computes the managed-payoff spectrum on the final pre-2024 estimation
sample (1963-2023). Eigenvalues are NOT normalized.
"""

from __future__ import annotations

import ast
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2

from download_JKP.read_dataset import build_clean_jkp_cache
from Kernels.kernel_function import (
    PortfolioKernel,
    fit_lambda_grid,
    kernel_eigenvalues,
    make_complexity_lambda_grid,
    make_return_matrix,
    make_return_row,
)
from median_lengthscale import median_lengthscale
from Utils.utils import build_monthly_panels

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "JKP_USA"
OUT_DIR = ROOT / "results" / "raw_eigenvalues"
START_YEAR = 1963
END_YEAR = 2024
FINAL_ESTIMATION_END_YEAR = 2023
N_RANDOM_FEATURES = 1000

KERNELS = [
    "linear",
    "gaussian",
    "ntk",
    "matern12",
    "matern32",
    "matern52",
]

LABELS = {
    "linear": "Linear",
    "gaussian": "Gaussian",
    "ntk": "NTK",
    "matern12": "Matérn 1/2",
    "matern32": "Matérn 3/2",
    "matern52": "Matérn 5/2",
}


def repo_wrds_credentials():
    """Read the credentials already used by the repository without duplicating them."""
    path = ROOT / "download_JKP" / "dataset.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"WRDS_USERNAME", "WRDS_PASSWORD"}:
                values[target.id] = ast.literal_eval(node.value)
    return values["WRDS_USERNAME"], values["WRDS_PASSWORD"]


def factor_characteristics():
    url = (
        "https://raw.githubusercontent.com/"
        "bkelly-lab/jkp-data/main/"
        "src/jkp/data/resources/factor_details.xlsx"
    )
    details = pd.read_excel(url)
    return (
        details.loc[details["abr_jkp"].notna(), "abr_jkp"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def ensure_raw_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [RAW_DIR / f"JKP_USA_{year}.parquet" for year in range(START_YEAR, END_YEAR + 1)]
    if all(path.exists() for path in wanted):
        return

    user, password = repo_wrds_credentials()
    chars = factor_characteristics()
    base = [
        "id", "eom", "excntry", "gvkey", "permno", "size_grp",
        "me", "ret_exc_lead1m",
    ]
    columns = ", ".join(base + chars)
    filters = """
        common = 1
        AND exch_main = 1
        AND primary_sec = 1
        AND obs_main = 1
        AND excntry = 'USA'
    """

    connection = psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        dbname="wrds",
        user=user,
        password=password,
        sslmode="require",
    )
    try:
        for year, path in zip(range(START_YEAR, END_YEAR + 1), wanted):
            if path.exists():
                continue
            print(f"Downloading {year}...", flush=True)
            sql = f"""
                SELECT {columns}
                FROM contrib.global_factor
                WHERE {filters}
                  AND eom >= '{year}-01-01'
                  AND eom < '{year + 1}-01-01'
                ORDER BY eom, id
            """
            frame = pd.read_sql_query(sql, connection, parse_dates=["eom"])
            frame.to_parquet(path, index=False, compression="zstd")
            print(f"  {len(frame):,} rows", flush=True)
    finally:
        connection.close()


def clean_manifest():
    clean_dir, manifest = build_clean_jkp_cache(RAW_DIR)
    return clean_dir, manifest


def year_file_map(manifest):
    ans = {}
    for name in manifest["output_files"]:
        year = int(Path(name).stem.rsplit("_", 1)[1])
        ans[year] = name
    return ans


def load_year_panels(clean_dir, files_by_year, year, characteristics):
    frame = pd.read_parquet(
        clean_dir / files_by_year[year],
        columns=["id", "eom", "permno", "me", "ret_exc_lead1m", *characteristics],
    )
    frame.attrs["characteristics_rank_standardized"] = True
    return build_monthly_panels(
        frame,
        characteristics=characteristics,
        characteristics_are_ranked=True,
    )


def first_validation_panels(clean_dir, files_by_year, characteristics):
    panels = []
    for year in range(1963, 1978):
        panels.extend(load_year_panels(clean_dir, files_by_year, year, characteristics))
    train = [p for p in panels if p["formation_date"].year <= 1972]
    validation = [p for p in panels if 1973 <= p["formation_date"].year <= 1977]
    return train, validation


def choose_lengthscale(kernel_name, train, validation):
    if kernel_name not in {"gaussian", "matern12", "matern32", "matern52"}:
        return 1.0, np.nan

    median = median_lengthscale(train)
    best_ell = None
    best_loss = np.inf

    for multiplier in (0.5, 1.0, 2.0):
        ell = median * multiplier
        kernel = PortfolioKernel(
            kernel=kernel_name,
            ell=ell,
            n_random_features=N_RANDOM_FEATURES,
            random_state=0,
        )
        train_matrix = make_return_matrix(train, kernel)
        validation_matrix = make_return_matrix(validation, kernel)
        eig = kernel_eigenvalues(train_matrix)
        grid = make_complexity_lambda_grid(eig, number_of_lambdas=60)
        betas, _ = fit_lambda_grid(train_matrix, grid)

        this_best = np.inf
        for lam in grid:
            r = validation_matrix @ betas[lam]
            loss = float(np.mean((1.0 - r) ** 2))
            if loss < this_best:
                this_best = loss

        print(
            f"{LABELS[kernel_name]} ell={ell:.8g}: validation loss={this_best:.8g}",
            flush=True,
        )
        if this_best < best_loss:
            best_loss = this_best
            best_ell = ell

        del kernel, train_matrix, validation_matrix, betas
        gc.collect()

    return float(best_ell), float(best_loss)


def compute_final_rows(clean_dir, files_by_year, characteristics, kernels):
    rows = {name: [] for name in kernels}
    for year in range(START_YEAR, FINAL_ESTIMATION_END_YEAR + 1):
        print(f"Managed-payoff rows: {year}", flush=True)
        panels = load_year_panels(clean_dir, files_by_year, year, characteristics)
        for month in panels:
            for name, kernel in kernels.items():
                rows[name].append(make_return_row(month, kernel))
        del panels
        gc.collect()
    return rows


def save_spectrum(kernel_name, eigenvalues, ell):
    positive = np.asarray(eigenvalues, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    ranks = np.arange(1, len(positive) + 1)

    frame = pd.DataFrame({
        "kernel": kernel_name,
        "rank": ranks,
        "eigenvalue": positive,
        "lengthscale": ell,
        "sample_start": f"{START_YEAR}-01",
        "sample_end": f"{FINAL_ESTIMATION_END_YEAR}-12",
    })
    frame.to_csv(OUT_DIR / f"{kernel_name}_raw_eigenvalues.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ranks, positive, linewidth=1.4)
    ax.set_xlabel("Ordered managed-portfolio direction")
    ax.set_ylabel(r"Raw eigenvalue $\hat{\mu}_j$")
    title = f"{LABELS[kernel_name]} managed-portfolio spectrum"
    if kernel_name in {"gaussian", "matern12", "matern32", "matern52"}:
        title += f"  (lengthscale={ell:.3g})"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{kernel_name}_raw_eigenvalues.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_raw_data()
    clean_dir, manifest = clean_manifest()
    characteristics = list(manifest["kept_characteristics"])
    print(f"Kept characteristics: {len(characteristics)}", flush=True)
    files_by_year = year_file_map(manifest)

    train, validation = first_validation_panels(
        clean_dir, files_by_year, characteristics
    )

    selected = {}
    diagnostics = []
    for kernel_name in KERNELS:
        ell, validation_loss = choose_lengthscale(kernel_name, train, validation)
        selected[kernel_name] = ell
        diagnostics.append({
            "kernel": kernel_name,
            "lengthscale": ell,
            "initial_validation_loss": validation_loss,
        })

    pd.DataFrame(diagnostics).to_csv(
        OUT_DIR / "lengthscale_selection.csv", index=False
    )

    del train, validation
    gc.collect()

    kernels = {
        name: PortfolioKernel(
            kernel=name,
            ell=selected[name],
            n_random_features=N_RANDOM_FEATURES if name != "linear" else None,
            random_state=0,
        )
        for name in KERNELS
    }

    rows = compute_final_rows(
        clean_dir, files_by_year, characteristics, kernels
    )

    all_frames = []
    summary = []
    for name in KERNELS:
        matrix = np.vstack(rows[name])
        eig = kernel_eigenvalues(matrix)
        frame = save_spectrum(name, eig, selected[name])
        all_frames.append(frame)
        summary.append({
            "kernel": name,
            "months": len(matrix),
            "number_positive_eigenvalues": len(frame),
            "largest_eigenvalue": float(frame["eigenvalue"].iloc[0]),
            "trace": float(frame["eigenvalue"].sum()),
            "lengthscale": selected[name],
        })
        del matrix
        gc.collect()

    pd.concat(all_frames, ignore_index=True).to_csv(
        OUT_DIR / "raw_eigenvalues_all.csv", index=False
    )
    pd.DataFrame(summary).to_csv(
        OUT_DIR / "spectrum_summary.csv", index=False
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
