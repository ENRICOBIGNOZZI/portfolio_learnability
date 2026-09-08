# Portfolio learnability

Reproducible code for direct, characteristic-based portfolio learning using
kernel ridge regression. The headline empirical specification uses an
all-characteristic Matérn-3/2 random-feature policy; direct linear ridge is the
benchmark.

The estimator learns a portfolio policy from managed payoffs, selects relative
regularization only by chronological response-one validation loss, then refits
on training and validation months. Economic outputs use a predictable 10%
annual volatility target and a monthly gross-exposure cap of 10. Transaction
costs are not included.

## Quick start

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python -m pytest -q
```

`data/` is intentionally excluded from Git. Place the local JKP USA parquet
files under `data/JKP_USA/`. The training scripts build their cleaned cache
locally and write outputs under `results/`, also excluded from Git.

```bash
python train_matern_model.py
python train_linear_model.py
python run_paper_figures.py
```

For the full run order, output descriptions, memory design, and reproducibility
manifest, read [EMPIRICAL_WORKFLOW.md](EMPIRICAL_WORKFLOW.md). The substantive
interpretation and limitations of the empirical diagnostics are in
[PAPER_FIGURE_REVIEW.md](PAPER_FIGURE_REVIEW.md).

## Repository contents

- `Kernels/` — linear, Gaussian, Matérn-3/2 and NTK feature maps.
- `Utils/` — shared estimator, analysis, plotting and reproducibility helpers.
- `download_JKP/` — local JKP cleaning/loading utilities.
- `train_*.py` — model entry points.
- `figure_*.py` and `appendix_*.py` — separate reproducible research figures.
- `tests/` — estimator and analysis checks.

No JKP data, local results, credentials, or downloaded PDF are committed.
