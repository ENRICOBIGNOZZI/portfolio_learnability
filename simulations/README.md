# Canonical balanced-factor simulations

This directory is the sole simulation system for *The Law of Portfolio Learnability*.
The complete run passed final acceptance: 3,500 replications, 123 repository tests, 66 PDF figures, and a 91-page integrated manuscript. Deliverables: [full manuscript](outputs/paper/paper/The_Law_of_Portfolio_Learnability.pdf), [final report](outputs/paper/audit/final_report.txt), and [machine-readable acceptance report](outputs/paper/audit/final_report.json).

The economic model has exactly three factors. Nyström rank is a numerical RKHS
approximation dimension, never a number of economic factors.

The exact DGP is in `dgp/balanced.py`; its derivation and E1–E6 arguments are in
`docs/dgp_methodology.md`. Legacy cleanup is documented in
`../simulation_cleanup_manifest.md`. Empirical JKP code and outputs remain separate.

Baseline audit:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m simulations.diagnostics.baseline
```

Reproduce the smoke and complete Monte Carlo profiles:

```sh
PYTHONPYCACHEPREFIX=/tmp/codex_dgp_pycache VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.run --profile smoke
PYTHONPYCACHEPREFIX=/tmp/codex_dgp_pycache VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.run --profile paper --workers 4
```

The paper profile fixes 500 headline and 200 replications in each of 15 robustness
configurations, ten training sample sizes, 96 positive penalties, 720 independent
OOS periods, and a 25% chronological validation block followed by a full-history
refit. N is varied on {99,300,501,999}: nearby multiples of three replace the
suggested nonmultiples, preserving the exact triplet construction. Rank robustness
uses 128, 256 and 512 nested inducing points. No seeds or grids are chosen from
realized curves. `config/design.py` is the frozen design.

Outputs are separated by profile. Every replication is saved atomically under
`outputs/{profile}/data/{environment}/replications/`; completed replications are
reused only when the scientific source/configuration hash matches. Each checkpoint
contains the complete T-by-lambda population and OOS loss/Sharpe/complexity surfaces,
validation losses, exact decomposition terms, benchmark results and seeds. CSV
summaries retain mean, median, MC SE and pointwise 95% intervals. Bootstrap rate
intervals resample whole independent replications and reapply selection and the
ensemble oracle inside each resample. `--render-only` rebuilds summaries and figures
from completed checkpoints; `--only baseline` is explicitly a partial run.

`docs/execution_status.md` records completed checks and disclosed limitations; `docs/requirements_matrix.md` maps every requested section to final evidence.


Verification and paper-preview commands:

```sh
python3 -m simulations.diagnostics.verify --profile smoke
python3 -m simulations.diagnostics.verify --profile paper
python3 -m simulations.paper --profile smoke
python3 -m simulations.paper --profile paper
```

The LaTeX preview is clearly labeled and is not the full theoretical manuscript.
The source in `Portfolio 8` was identified by its unique exact E1–E6 match to the supplied brief; this choice is recorded as an inference, with confirmation requested. Imported theoretical files remain byte-for-byte unchanged. Economic `DGPParameters.nu=1.5`
locks the headline convention; spectral robustness changes the independent
kernel representation setting `Environment.nu` to 0.5 or 2.5, keeping D=6 and
all three economic factors unchanged.

The original mathematical PDF and brief are preserved in `docs/`. Population
spectral OLS errors are descriptive regression errors across eigenvalue ranks;
window sensitivity is reported separately and is not a confidence interval for
the asymptotic exponent.

`--workers` changes execution concurrency only. The scientific configuration hash, random streams and cached replication results are independent of that override. The initial two-worker baseline was allowed to finish its queued replications before resuming with four workers; no seed or sample-size cell was discarded.

The simulation runtime is pinned separately in `simulations/requirements.txt` (the executed environment uses Python 3.13.3). This does not change the empirical dependency file. Install these packages into a dedicated environment for version-matched reproduction; BLAS/thread metadata and versions are also saved in each run manifest.

The full manuscript is assembled in `../paper/main.tex`, using unchanged theory from the matching E1–E6 source and the unchanged empirical sections already in this repository. After numerical acceptance, compile it with `python3 -m simulations.manuscript`. `--smoke-layout` produces an explicitly labeled layout preview and cannot certify production. Compilation checks hashes of all imported theoretical content and all 112 protected empirical files. The current theory gives the Sharpe-level benchmark exponent `-b/(b+1)` under E6; rate annotations follow that equation.

The final closeout command `python3 -m simulations.diagnostics.closeout` checks numerical acceptance, theory/empirical preservation, the compiled PDF hash and a matching all-page visual review before writing `outputs/paper/audit/final_report.json`. It refuses to certify missing or partial deliverables.
