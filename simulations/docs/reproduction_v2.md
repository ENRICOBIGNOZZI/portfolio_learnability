# V2 reproduction and evidence boundaries

Run from the repository root using the pinned `requirements.txt`. Local evidence
was produced with Python 3.13.3, NumPy 2.3.5 and SciPy 1.16.3. CI additionally
checks Python 3.12. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and, on macOS,
`VECLIB_MAXIMUM_THREADS=1`. On an iCloud-backed desktop, a local
`PYTHONPYCACHEPREFIX=/tmp/codex_dgp_pycache` avoids delayed cache hydration.
Prefer a fully local checkout for execution: iCloud can also delay source and
Git metadata reads. The final V2 stages were moved to a local checkout after
verifying the frozen scientific hashes and all 552 saved NPZ files. The transfer
and interrupted render event are recorded in `audit/local_workspace_transition.json`
and `run_manifest.json`; production paths and the numerical protocol were unchanged.

Fast verification from a clone, without checkpoints:

```sh
python -m simulations.diagnostics.reconstruct
python -m simulations.diagnostics.reconstruct_confirmation --aggregates-only
```

The second command checks V2 method summaries against per-path CSVs and checks
the plugin oracle against the published penalty-grid curves. It does not
reconstruct coefficients, test payoffs or bootstrap intervals from raw arrays.
Neither clone-only command certifies an experiment newly recomputed from seeds.

Full historical raw reconstruction:

```sh
python -m simulations.diagnostics.raw_archive fetch --destination .
python -m simulations.diagnostics.reconstruct --full
```

The download is public and needs no login. The stable release is
https://github.com/ENRICOBIGNOZZI/portfolio_learnability/releases/tag/synthetic-checkpoints-cdfbd11-v1
and the archive SHA256 is
`c639bcf88fe8b5b6e8e8deef0f09486608841109a705f1d61c9aaac412807de0`.
Per-file checksums and the source commit are in
`outputs/confirmation_v2/audit/raw_archive.json`. The public download was actually
verified against all 3,521 NPZ members; its evidence is in
`raw_archive_download_verification.json`. No licensed empirical inputs are present.

The full verifier independently reconstructs all six CSVs per historical
environment, including bootstrap rate intervals, plus combined aggregate tables.
It does not call the production summarizer. V1 raw surfaces omit stock returns
and coefficient arrays; consequently raw-to-aggregate verification is distinct
from a stock-level fitting replay. The V2 stores selected coefficients and
actual independent/future payoff arrays and tests the evaluator boundary.

The separately versioned confirmation archive is indexed by
`simulations/outputs/confirmation_v2/audit/confirmation_raw_archive.json`.
The public release is
https://github.com/ENRICOBIGNOZZI/portfolio_learnability/releases/tag/synthetic-confirmation-v2-719034dd1fd3
with archive SHA256
`5c96fc50c1f2a77c6f5a7c9338a6fcc39b19c31e8efbbb8d8e7b9d4d4a707b8d`.
An actual unauthenticated download verified all 561 members on 8 October 2026.
Run the following commands sequentially to download it and reconstruct the V2 evidence:

```sh
python -m simulations.diagnostics.raw_archive fetch --destination . --index simulations/outputs/confirmation_v2/audit/confirmation_raw_archive.json --report simulations/outputs/confirmation_v2/audit/confirmation_raw_archive_download_verification.json
python -m simulations.diagnostics.reconstruct_confirmation
python -m simulations.diagnostics.reconstruct_resolution
python -m simulations.diagnostics.publication_v2
```

The index records the stable asset URL, archive checksum and every member's
checksum. Archive preparation alone is not proof of public availability; the
download-verification report records an actual successful public fetch. The
fetch command refuses to overwrite differing local files. Full reconstruction
uses archived arrays without rerunning the expensive production simulation.

Fresh smoke and adversarial tests:

```sh
python -m simulations.diagnostics.test_evidence
python -m simulations.run --profile smoke --output /tmp/simulation-smoke --workers 1
```

Versioned confirmation checkpoints, followed by aggregation:

```sh
python -m simulations.run --profile confirmation_v2 --compute-only --workers 1
python -m simulations.run --profile confirmation_v2 --render-only
```

`--compute-only` explicitly leaves numerical and publication acceptance pending.
Changing workers does not alter the scientific identity. A renderer change can
regenerate presentation artifacts without labeling old arrays as new simulation.
Changed scientific source or configuration requires a fresh output directory.
The old `outputs/paper`, `outputs/smoke` and `outputs/audit` are preserved.

Numerical checks use the same economic generator and estimator:

```sh
python -m simulations.diagnostics.exact_kernel
python -m simulations.diagnostics.resolution_v2 rank --environment baseline_original --workers 1
python -m simulations.diagnostics.resolution_v2 rank --environment rich6d --workers 1
python -m simulations.diagnostics.resolution_v2 quadrature --environment baseline_original
python -m simulations.diagnostics.resolution_v2 quadrature --environment rich6d
python -m simulations.diagnostics.spectrum_v2
python -m simulations.diagnostics.reconstruct_resolution
```

Run these numerical commands sequentially. Following the user's memory request,
production uses one worker and the rank/quadrature evaluator accumulates 256
independent groups per batch. The points, seed families and population formulas
are unchanged; only floating-point summation order changes. Agreement with the
canonical evaluator is tested for both loading maps, including partial batches.
Rank checkpoints from the interrupted dense evaluation are preserved under
`audit/rank_paths`; new diagnostics use `audit/rank_paths_low_memory` and include
both diagnostic source hashes. There is no silent migration of those caches.

The spectral cache additionally binds all scientific dependencies, numerical
runtime, group count and actual quadrature seed. Six early cases with an
incomplete dependency fingerprint remain under
`audit/spectrum_before_dependency_binding`; the accepted spectrum is recomputed
under `audit/spectrum`. These early values are not relabeled as newly computed
or accepted on a rewritten manifest. The independent verifier checks the bound
runtime of the archived computation, allowing read-only reconstruction on a
different supported runtime without pretending to have regenerated the arrays.

Kernel matrices above 128 MiB use a temporary local file and 256-row reads for
the full matrix operator. This changes storage, not the kernel entries, nodes or
seeds. Controlled partial eigensolves avoid dense copies at the two larger
resolutions. Products and managed-operator eigenvalues are checked against dense
references; each computed eigenpair also retains its actual residual. Temporary
kernel files are closed and removed after each spectral case.

The independent numerical verifier reconstructs all rank and quadrature table
cells, checks the 50 paired paths and four independent quadrature seeds, and
recomputes quadrature risk and Sharpe from coefficients and saved moments.
It reconstructs descriptive spectral fits from stored eigenpairs; this does not
independently rerun the large eigensolver or certify the infinite kernel.

`oracle_optimism_bootstrap.csv` reports percentile intervals with whole-path
resampling and recalibration of plugin and cross-fitted oracles on every draw.
It retains the practical and extended cohorts separately, using the same frozen
bootstrap seeds and draws as the rate analysis. The original pointwise paired
MCSE summaries remain in `oracle_optimism.csv` as conditional descriptions;
they do not include the uncertainty from fitting the oracle penalties.

After full computation, regenerate and independently check publication material:

```sh
python -m simulations.run --profile confirmation_v2 --render-only --workers 1
python -m simulations.diagnostics.reconstruct_confirmation
python -m simulations.plotting.confirmation
python -m simulations.paper --profile confirmation_v2
python -m simulations.diagnostics.publication_v2
```

Final acceptance also requires a clean-checkout manuscript build, actual review
of its rendered pages, both public archive download reports and the full test
transcript bound to the final source files. The closeout command checks these
dependencies and refuses missing or stale evidence:

```sh
python -m simulations.manuscript
python -m simulations.diagnostics.execution_timing
python -m simulations.diagnostics.closeout_v2
```

Compilation alone does not create a successful visual-review record. The final
report keeps all thirteen requested conditions separate. A failed numerical
tolerance is reported with its exact region and cannot be relabeled as a pass.
Elapsed times distinguish the initial run, the memory-related interruption,
resume and rendering from summed completed-replication times.

The rank comparison covers the practical T grid. The extra horizons remain
uncertified by that rank check even if their selected-policy quadrature passes.
All source/test/compile evidence is versioned; a historical `passed` flag cannot
certify modified code. CI separates empirical verification, synthetic aggregate
verification, fresh synthetic smoke and the manually dispatched raw archive job.

## Completed acceptance

The 8 October 2026 final report records all 13 requested conditions as met.
The final local suite passed 189 tests, and the complete manuscript contains
94 visually reviewed pages. The original theory, 112 empirical files and
3,891 historical artifacts retain their recorded bytes. Actual GitHub CI
evidence, including the public download and full raw reconstruction job, is
in `outputs/confirmation_v2/audit/ci_verification.json`.

Five practical-grid policy cells remain uncertified by the rank criterion:
holdout at T=60 and T=360 for both maps, and rolling validation at T=60 for
rich6d. These are retained in the paper and in
`audit/claimed_resolution_regions.csv`. All six headline policy cells at T=1440
pass both rank and quadrature checks. Extra training horizons retain their
uncertified rank status. Acceptance records completed and honestly scoped
checks; it does not turn these failed tolerances into numerical passes.
