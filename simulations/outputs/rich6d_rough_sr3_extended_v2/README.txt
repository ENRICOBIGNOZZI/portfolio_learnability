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

Workflow (from the repository root; limit BLAS threads on this 8 GB host):
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m simulations.extended.preflight
  python3 -m simulations.extended.freeze --rank <rank selected from audits>
  VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m simulations.extended.production --workers 1
  python3 -m simulations.extended.summarize
  python3 -m simulations.extended.figures

The preflight supervisor resumes completed population cases and replication
checkpoints. It stops before production, so rank selection and resource evidence
can be reviewed. A failed or missing process is diagnosed before restarting;
observation timeouts do not imply termination.

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
