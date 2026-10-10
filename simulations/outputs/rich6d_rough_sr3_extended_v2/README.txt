EXTENDED ROUGH RICH6D STUDY — PARTIAL PRODUCTION, RESOURCE RECOVERY PENDING

Reference: rich6d_rough_sr3_annual_monthly_v1, commit 2d59c3324569.
The reference economy keeps annualized optimal Sharpe 3, the existing factor
means/covariance, Fourier truncation 128, and the original baseline ridge constant.
N robustness preserves those economic parameters, so its analytical optimum is
not artificially reset to 3. Persistence robustness preserves the population
operator exactly.

All 12 paired rank pilots and their audits are complete. The fixed-rank production
protocol is frozen. There are 38 verified baseline checkpoints; the study is NOT
complete. The previous production supervisor exited with status 120, and neither
that supervisor nor the supplementary batch remains alive. The supplementary
batch record now identifies its unfinished indices explicitly.

A bounded single-replication recovery was prepared with persistent file logging,
but was not launched: free disk and already allocated free swap did not provide
a stable margin above 1.25 times the measured maximum baseline resident memory
plus a 512 MiB reserve. production_recovery_current.json records all three resource
readings and the prior failure. This is an operational resource guard, separate
from the unchanged scientific 5% approximation gates. Small regenerable preview
files were inspected but not deleted because their size would not resolve the
resource shortage. No unrelated processes or artifacts were altered.

Until a new worker is confirmed live, the old production_status.json is a
historical checkpoint counter, not evidence that production is running. Resume
from the missing deterministic indices once resources permit; retain stdout and
stderr in an execution log independent of the interactive tool console.
The cloud-backed bytecode cache caused another import wait during recovery.
Frozen scientific imports and hashes were successfully checked with
PYTHONDONTWRITEBYTECODE=1 and a separate PYTHONPYCACHEPREFIX under /tmp. These
process-local environment settings avoid the old bytecode cache without changing
scientific sources. The tested resume environment is recorded in the recovery
JSON; it has not yet been used to start another simulation.
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
seconds in the recent two-worker observations. Two total workers are retained;
see production_three_worker_probe.json for timings and their operational limits.
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
