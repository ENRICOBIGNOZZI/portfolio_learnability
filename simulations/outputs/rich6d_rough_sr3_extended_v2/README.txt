EXTENDED ROUGH RICH6D STUDY — BASELINE COMPLETE, ROBUSTNESS IN PRODUCTION

Status recorded on 2026-10-10 after the baseline completion audit. All 300
baseline replications are independently verified and committed. The four paired
robustness economies are assigned to remote run 38075206545. The study remains
incomplete until all five economies have 300 verified replications, all summaries
pass the independent audit, and the ten final figures receive visual review.
For current counts, consult checkpoint_integrity.json and the corresponding
remote_collection_production_robustness_38075206545.json, rather than this snapshot.

Scientific design

Reference: rich6d_rough_sr3_annual_monthly_v1, commit 2d59c3324569.
The baseline retains annualized optimal Sharpe 3, the factor mean and covariance,
Fourier truncation 128, Matérn-3/2 lengthscale 1 and the original independently
calibrated ridge constant. Annualization is sqrt(12) times monthly marginal
Sharpe, not the Sharpe of compounded annual returns. N robustness preserves the
economic parameters and uses each economy's own analytical optimum; it does not
force every N optimum to 3. The rho variants share the stationary population
operator, analytical optimum and penalty constant exactly.

The frozen identity is
e9f1f005be1a58ee736a09cb711d5be87457b9875d1e26dab56c03ccdca61d8d.
protocol.json and seed_manifest.json fix 300 independent replication indices,
paired innovations, nested histories, one common rank 4096, the 13 baseline
horizons through T=4860 and the common 12-horizon robustness grid through T=3240.
Each checkpoint retains all 96 predetermined diagnostic penalties plus the exact
a*T^(-0.6) choice. No Monte Carlo outcomes select the reported strategy.

Numerical interpretation

All 12 paired rank pilots and their audits are complete. The 62 required
selected-rank population-quadrature cells pass the unchanged 5% criterion.
The largest recorded pilot quadrature discrepancy bound, including optional
T=7290, is approximately 3.26% of regret. The pilot projection-floor criterion
fails at baseline T=3240,4860; N300 T=1080,1440,2160,3240; and both alternative-rho
environments at T=3240. No investigated common rank resolves every required cell.
Rank 4096 also lacks an independent higher-rank certificate. Final reporting
recomputes the floor fractions using the complete 300-path mean regrets and
retains the pilot paired-discrepancy evidence separately.

Optional T=7290 was investigated and excluded on numerical grounds. The frozen
additional main diagnostic path T=2160 passed both checked pilot 5% conditions
and quadrature; this is not an independent higher-rank certificate. The economic
optimum is never replaced by the finite-rank optimum. More Monte Carlo paths do
not remove the approximation floor. Finite Fourier truncation is smooth and does
not prove the r=1 source condition, minimax optimality or an asymptotic rate.

Primary bands are the central 95% replication percentiles, not confidence
intervals for the mean. Tables retain the mean, median, sample standard deviation,
both percentiles and mean MCSE for every horizon and penalty. Slope uncertainty
uses the full within-path cross-T covariance, checked by replication jackknife.
All four predeclared windows are reported. Full baseline slopes through T=4860
and matched robustness slopes through T=3240 are separate tables; comparisons
across economies use identical horizons. Local population elasticity, window
regression slopes and empirical complexity remain distinct quantities.

Execution and provenance

The completed baseline run is
https://github.com/ENRICOBIGNOZZI/portfolio_learnability/actions/runs/38064621892
It preserved 68 locally completed paths and computed the remaining 232 paths in
15 disjoint batches. All 300 checkpoints passed the independent scientific audit.
remote_baseline_completion.json freezes the evidence used to admit robustness.
remote_artifact_receipts_production_baseline_38064621892.json retains all 15
artifact identities, transfer-manifest hashes and assigned replication indices.

The robustness run is
https://github.com/ENRICOBIGNOZZI/portfolio_learnability/actions/runs/38075206545
Its source commit is 2eb27a92a2153e5c8bd249b3adda46404ba51b48. There are 15 batches
of 20 indices with at most five standard macOS jobs in parallel. Each index
contains N300, N1200, rho000 and rho075 over the common frozen grid. The collector
retrieves completed artifacts and independently validates identities, checksums,
seeds, grids, economic bounds, decomposition, ridge residuals and history pairing
before publishing checkpoints. Launch and collection JSON files record its PID,
source hash, exact command and log location. An observation timeout does not
justify restarting a live job. Do not restart the former local production queue
alongside this remote allocation.

Remote production was admitted only after full fitted-policy reproduction on
macos-26 with the exact dependency versions and predeclared numeric tolerances.
remote_reproduction_results.json records the successful qualification; the
earlier failed comparator attempt remains in its separate attempt record.
Linux runners were excluded by the exact history-hash gate, not by a claim that
their arithmetic is economically incorrect. remote_input_manifest.json and
remote_input_publication.json identify the five numerical caches, public input
prerelease and verified server SHA-256 digests. No tolerance was relaxed.

Earlier local recovery, bounded-worker probes, cache cleanup and the completed
local-to-remote handoff remain documented in their original JSON records. They
are historical execution evidence, not instructions to restart those processes.
Shared-host timings are operational observations, not controlled benchmarks;
peak RSS in a batch is cumulative from process start.

Remaining postprocessing

Once the collector has verified 300 paths for every economy, run from the
repository root with these process-local settings on the shared Mac:
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPYCACHEPREFIX=/tmp/rich6d_no_bytecode_cache
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  python3 -m simulations.extended.summarize
  python3 -m simulations.extended.verify --data-only
  python3 -m simulations.extended.report
  python3 -m simulations.extended.figures
  python3 -m simulations.extended.verify

The summary audit independently recomputes every penalty distribution, paired
difference and slope window, including all matched-grid robustness slopes.
Synthetic integrity tests also reject corrupted percentiles, incorrect paired
summaries, diagonal-only covariance uncertainty and unequal-grid substitutions.
Passing those tests is not a claim that unfinished production tables are valid.

Inspect all ten actual PNG/PDF figures, record visual review and refresh the
manifest after any deliverable changes. Commit and push all intended artifacts.
Final presentation must show the four main figures, then the three N figures,
then the three rho figures directly in chat with the seven requested scientific
answers and all prespecified slope windows. Neither the numerical verifier nor
pilot previews substitute for this final review and presentation.

Storage and independent reproduction

The portable population-operator archive in population/operators is already
complete, committed and losslessly checked (population_archive_validation.json).
It need not be exported again for final postprocessing. The archive includes
baseline rank 512 for the paired comparison with the previous 100-path study,
as well as the three unique rank-4096 population operators.

Large pilot basis/integral/operator caches are regenerable and excluded from
Git; frozen seeds, calibration, spectra, audits and every fitted-policy outcome
are retained. Raw return histories are reproduced from seeds rather than saved
1,500 times. Previous experiments, manuscript files and empirical results remain
outside this task's changes.

For a new independent full reproduction, set RICH6D_EXTENDED_OUTPUT to an empty
directory before invoking preflight, reviewing its numerical/resource evidence,
freezing the rank and running production. Keep the checked-in reference run
unchanged. The preflight and advance supervisors are for that separate workflow;
do not invoke them to duplicate the already allocated production run.
