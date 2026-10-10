EXTENDED ROUGH RICH6D STUDY — PARTIAL PRODUCTION, PRODUCTION RECOVERY ACTIVE

Reference: rich6d_rough_sr3_annual_monthly_v1, commit 2d59c3324569.
The reference economy keeps annualized optimal Sharpe 3, the existing factor
means/covariance, Fourier truncation 128, and the original baseline ridge constant.
N robustness preserves those economic parameters, so its analytical optimum is
not artificially reset to 3. Persistence robustness preserves the population
operator exactly.

All 12 paired rank pilots and their audits are complete. The fixed-rank production
protocol is frozen. The latest independent checkpoint audit verifies 57 baseline
replications and no completed production robustness batches. The study is NOT
complete: all 300 paths in each economy, final summaries and all ten final figures
remain required.

Recovery resumed on 2026-10-10 after disk availability returned to approximately
8.3 GiB. The primary supervisor retains one worker and persistent stdout/stderr
file logging. Its previous exit status 120 remains of undetermined cause; the
resource-blocked recovery record is preserved in commit 8e2a8b2. Existing completed
checkpoints are reused under the unchanged frozen identity. Recovery process IDs,
log path, environment and the latest audit count are recorded in
production_recovery_current.json.

A bounded supplementary batch completed the missing indices 290 and 289, and
both passed the independent checkpoint audit. Its state, execution source, log
path and resource observations are retained in production_recovery_supplement*
JSON records. The next bounded batch covers the twenty existing baseline indices
287 down through 268. Its launch source and status are in
production_recovery_batch_287_268_launch.json and
production_recovery_batch_287_268.json. This allows at most two production workers.
The supplement checks a 3 GiB disk reserve before each replication and stops if
the primary queue approaches its indices. All seeds and numerical calculations
remain unchanged. Shared-host timings are operational observations, not controlled
performance benchmarks; process peak RSS in a multi-replication batch is cumulative
from process start, not a separate per-replication memory measurement.

As the shared host became busier and disk availability fell to about 4 GiB, an
operational watcher was started to stop the supplement immediately after index
286 has saved both its outcome and resource record. The primary supervisor
continues with one worker. production_after_checkpoint_stop.json records whether
this stop is still pending or has occurred; its launch record preserves the exact
watcher source. All remaining indices stay in the primary deterministic queue.
This changes execution concurrency only, not the scientific protocol.

The supplementary stop completed after saving index 286; the primary supervisor
continues alone. A manual GitHub Actions compatibility workflow now checks the
frozen source hashes, one full baseline economic history and a sixteen-period
paired prefix in all five economies on standard Linux and macOS runners. Its
canonical reference derives from the audited baseline index 0 and local paired
histories. The local probe passes. This is a host-compatibility diagnostic only:
a complete paired fitted-policy reproduction is still required before any remote
production. The probe code is simulations/extended/remote_compatibility.py; the
manual workflow is .github/workflows/rich6d-compatibility.yml.
The first remote compatibility run completed: macos-26 matches every checked
history hash, whereas both Linux runners fail this exact compatibility gate.
This does not claim that Linux arithmetic is economically incorrect; it excludes
those hosts from this frozen bitwise history design. Full results and the workflow
URL are in remote_compatibility_results.json. The next macOS fitted-policy check
and its tolerances were declared before execution in remote_reproduction_plan.json.
No remote production has been started.
The fitted-policy gate and three local boundary tests are implemented. Numerical
caches are listed by exact size and SHA-256 in remote_input_manifest.json; the
separate input prerelease will carry those auxiliary files without adding large
cache blobs to Git. The remote download checks every cache and all supporting
calibration/population files before fitting. The manual reproduction workflow
runs only on macos-26 and uses a fresh isolated output directory.

At the user's request, pip's regenerable download cache was purged, removing 14
files (23.2 MB reported by pip). This frees disk space, not process RAM. Required
simulation caches, code, data, completed results and unrelated jobs were preserved.

Frozen scientific source hashes were verified before restart and in each audit.
Process-local PYTHONDONTWRITEBYTECODE=1 and a separate PYTHONPYCACHEPREFIX under
/tmp bypass the cloud-backed bytecode cache that had stalled imports, without
changing source. production_status.json counts checkpoints acknowledged by the
primary supervisor, including reused checkpoints; checkpoint_integrity.json
records the most recent independent audit of all completed files.
The read-only checkpoint audit can be rerun during production with
  python3 -m simulations.extended.verify_checkpoints
It writes checkpoint_integrity.json with the unique completed count by economy,
per-file hashes, protocol/seed/shape checks, numerical identities and available
cross-stage history pairing. It does not replace final numerical verification,
the approximation audits, statistical summaries or visual review. Its rejection
of a wrong seed and a missing penalty was checked on isolated temporary copies;
the evidence is in checkpoint_verifier_validation.json.
The frozen run identity is
e9f1f005be1a58ee736a09cb711d5be87457b9875d1e26dab56c03ccdca61d8d.

Production uses rank 4096 in every environment, the 13 required baseline horizons
through T=4860, and the common 12-horizon robustness grid through T=3240. Optional
T=7290 was investigated in the pilots and excluded on numerical grounds. The
predeclared additional main diagnostic path at T=2160 passed the two pilot 5%
conditions and the quadrature check; it remains subject to the qualifications below.

All 62 pilot quadrature cells pass the unchanged 5% criterion; the largest paired
discrepancy bound is approximately 3.26% of regret. The approximation floor still
fails at baseline T=3240,4860; N300 T=1080,1440,2160,3240; and both alternative-rho
environments at T=3240. No investigated common rank resolves every required cell.
Rank 4096 also lacks an independent higher-rank certificate. These limitations
must remain explicit in the final figures and scientific conclusions.

For a fresh independent reproduction, set RICH6D_EXTENDED_OUTPUT to a new empty
output directory before running these commands. Keep the checked-in reference
experiment unchanged. The same seed design will be used in that new directory.

Workflow (from the repository root; limit BLAS threads on this 8 GB host):
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m simulations.extended.preflight
  python3 -m simulations.extended.freeze --rank <rank selected from audits>
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m simulations.extended.production --workers 1
  python3 -m simulations.extended.summarize
  python3 -m simulations.extended.verify --data-only
  python3 -m simulations.extended.report
  python3 -m simulations.extended.figures
  python3 -m simulations.extended.archive
  python3 -m simulations.extended.verify

Final delivery also requires inspection of all ten rendered figures and the
critical scientific report; the verifier does not substitute for visual review.

The preflight supervisor resumes completed population cases and replication
checkpoints. It stops before production, so rank selection and resource evidence
can be reviewed. A failed or missing process is diagnosed before restarting;
observation timeouts do not imply termination.

An optional advance supervisor can observe an already-running preflight:
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m simulations.extended.advance --preflight-pid <preflight PID>
It checks all 12 completed paired pilots, applies the declared rank/quadrature
gates, freezes the current scientific code in a fresh process, and runs production,
summaries, figures and verification. If no investigated common rank passes the
rank tests at every required horizon, it retains the highest investigated rank
and explicitly records the unresolved cells, as required by the study protocol.
It never relaxes the 5% tolerance. Its terminal ready_for_visual_review status
still requires human-readable scientific review, inspection of all ten figures,
and committing/pushing the intended deliverables before reporting completion.
Live progress is recorded in preflight_status.json, advance_status.json and
production_status.json as each corresponding stage becomes available.
The initial production supervisor uses one worker. After disk space recovered,
a bounded second-worker probe computed the already-declared baseline index 299
with the unchanged production.execute function and frozen run identity. Its
timing and memory evidence are in production_concurrency_probe.json. Subsequent
bounded batches use distinct high indices, with a 2 GiB disk-reserve check before
each replication and a conservative separation from the primary supervisor.
They add no seeds or replications to the fixed 300-path scientific design.
Batch records identify their indices and completion status. During this phase,
production_status.json counts checkpoints acknowledged by the primary supervisor;
the number of unique rep_*.npz files includes supplemental completed indices.
The primary supervisor will validate and reuse those checkpoints when it reaches
them. Final summaries still require all 300 unique indices in every environment.
A further single-replication probe at index 288 tested three concurrent workers.
It completed successfully but took about 425 seconds, versus about 157-184
seconds in the recent two-worker observations. That phase retained two total workers;
see production_three_worker_probe.json for timings and their operational limits.
The current primary supervisor uses one worker; the bounded supplements
above temporarily restore two concurrent workers after resource recovery.
During this probe, a fresh Python import waited on a cloud-backed bytecode file
that macOS had made dataless. Targeted local-download requests restored the cache
and source without changing their contents. runtime_read_recovery.json records
the diagnosis, recovery and confirmation that the source matches committed Git.
After freezing and checking the recorded audit hashes, the advance supervisor
removes regenerable pilot-only caches to reserve disk space for production.
It retains the chosen-rank basis, all production operators and the rank-512
baseline operator needed for comparison with the previous experiment. Every
removal is recorded in preproduction_cache_cleanup.json; spectra, pilot outcomes,
audits, calibration and the frozen protocol are preserved.

Primary uncertainty bands are the central 95% replication percentiles. Mean MCSE
is distinct. Slopes use complete within-replication cross-T covariance, with a
leave-one-replication-out jackknife cross-check. All predeclared windows remain.

Storage: pilot/basis_*, pilot/integrals_* and pilot/operator_* are regenerable
numerical caches, excluded from Git. Raw stock/managed histories for all 1,500
economies are reproducible from seeds and are not duplicated on disk. Every
relevant fitted-policy penalty outcome is retained. Population spectra, audit
results, seed identities and calibration are retained. No previous experiment
or unrelated user artifact is deleted or overwritten.

The pilot integrity check verifies all 48 checkpoints, paired histories across
ranks, independent replication seeds, separation from the production seeds and
complete predetermined grids. It does not substitute for numerical acceptance.
Projection floors, paired rank differences and their uncertainty are separate
diagnostics. Finite Fourier truncation is smooth and does not prove the r=1 source
condition; observed rate agreement cannot prove a minimax or asymptotic theorem.
