EXTENDED ROUGH RICH6D STUDY — PREPRODUCTION CHECKPOINT

Reference: rich6d_rough_sr3_annual_monthly_v1, commit 2d59c3324569.
The reference economy keeps annualized optimal Sharpe 3, the existing factor
means/covariance, Fourier truncation 128, and the original baseline ridge constant.
N robustness preserves those economic parameters, so its analytical optimum is
not artificially reset to 3. Persistence robustness preserves the population
operator exactly.

This directory currently contains preproduction evidence and executable workflow
code. It is NOT the final 300-replication study. No production protocol is valid
until protocol.json has been frozen after all declared pilots and quadrature gates.

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

The first 4096-rank pilot completed all required horizons and the optional 7290
resource horizon. Preliminary projection floors do not establish numerical
resolution: compare them with actual fitted-policy regret, paired rank differences
and their uncertainty. A highest investigated rank has no independent larger-rank
certificate. Finite Fourier truncation is smooth and does not prove the r=1 source
condition; observed rate agreement cannot prove a minimax or asymptotic theorem.
