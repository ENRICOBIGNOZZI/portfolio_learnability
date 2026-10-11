This closes out the archived initial-CV Rich6D simulation without editing its scientific protocol or re-running the 1500-history production. All additional diagnostics are post hoc. The historical outputs and frozen scientific source hashes remain immutable; the original summarizer now also includes regret in its slope quantities.

From the repository root, first ensure the original ignored NPZ checkpoints and population archives are available and match their tracked manifests. Their original workflow IDs, source commits and transfer hashes are in `rich6d_initial_cv60_v1/collection_{pilot,production}.json` and the extended experiment's reproduction/archive records. The verification fails on changed inputs; do not recreate a frozen manifest to bypass that failure.

```sh
python3 -m simulations.closeout.results
python3 -m simulations.closeout.numerical freeze
python3 -m simulations.closeout.cache_spectrum
```

`results` reads all 300 histories per economy, preserves every horizon, reproduces the historical comparison, distinguishes oracle Q from oracle SR, exports the actual selected penalties and per-history normalized quantities, and computes the existing four slope windows using full covariance and whole-history jackknife uncertainty. Its output directory is `simulations/outputs/rich6d_initial_cv60_closeout_v1`.

The numerical audit is explicitly bounded. The protocol declares twelve representative **production** indices and separate stress indices determined by the smallest archived initial constant, with index tie-breaking. These are not the old rank-pilot streams. A keeps the archived constant fixed at both ranks; B reruns CV once on the same initial60 at each rank; C provides the historical comparison. An explicit worker request is required for each history, so the reproduction commands never launch a full production campaign or a cloud job.

```sh
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 \
  python3 -m simulations.closeout.numerical fit --index 0
# Initial60 breadth audit for representative indices 1..11:
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 \
  python3 -m simulations.closeout.numerical fit --index 1 --short
# Repeat the preceding explicitly requested worker for each delivered index.
python3 -m simulations.closeout.numerical evaluate
python3 -m simulations.closeout.numerical compact
python3 -m simulations.closeout.audit_results
python3 -m simulations.closeout.components
python3 -m simulations.closeout.fourier
python3 -m simulations.closeout.fourier_paired
python3 -m simulations.closeout.publication
python3 -m simulations.closeout.verify
```

The delivered `numerical_status.json` and worker receipts enumerate the exact completed indices. Do not treat these example commands as a claim that all planned full histories or stress tests have run. The basis and population eigensystem caches are temporary, derived from the fixed kernel and checksummed original operators. Fit sidecars retain raw policy coefficients, actual penalties, fold scores, training seeds, return hashes, recovered monthly moments and low-rank population-ridge coefficients. They permit repeat evaluation without replaying the histories. No jitter is introduced. Independent quadrature evaluates the **fixed** learned policies rather than using a quadrature-fitted optimum as a certificate of the economic optimum.

The paired Fourier diagnostic is deliberately limited to index0 and T=60 in all five economies: M128/512/2048 share innovations and characteristics, policies are refit at actual archived constants and at each new initial choice, and payoff moments are evaluated in the corresponding Fourier economy. This does not certify final-horizon policy or CV-choice stability. The deterministic operator bounds are separately evaluated at **every** archived CV lambda.

```sh
python3 -m pytest -q tests/test_simulation_closeout.py tests/test_initial_cv.py \
  tests/test_rank_followup.py tests/test_extended_simulation.py \
  tests/test_extended_summary_integrity.py tests/test_low_memory_quadrature.py
python3 -m simulations.closeout.build_paper
python3 -m simulations.closeout.delivery
```

`build_paper` compiles an isolated staging copy of the current paper, changing only its simulation-output link. It keeps the empirical and theorem source files byte-identical and also produces a standalone simulation section/appendix. `delivery` verifies original and delivered artifacts and writes the final report, reproduction/limitation statuses and hash manifest. The four principal figures are spectrum/filter/energy, Sharpe recovery/frontier, empirical/population complexity, and Sharpe against complexity. Alternative economies and normalized distributions appear only in the appendix. Bands show replication outcomes; mean intervals are tabulated separately. Oracle bands and slopes are descriptive, not selection adjusted.

The mathematical note is preserved in `sources/` and reviewed in `theory_review.md`. A fixed positive projection floor blocks convergence to the full economic optimum at a fixed rank. Neither an attractive slope, a finite-grid frozen-CV rate proposition for the full kernel estimator, nor a P4096/P8192 comparison is a certificate of the maximal numerical rank. `resolved`, `failed`, `unresolved` and `not_run` refer to separate claims, never to whether a plotted exponent equals -0.6.

`components` also recovers the uniquely identified positive-ray loss, scale loss and normalized directional error from the signed monthly Sharpe and regret of all archived policies, without reconstructing ambiguous u/v roots. It computes numerical population-ridge bias at every actual lambda using the archived operator spectrum and mean projections, and labels the remaining sum displacement plus signed cross. It independently checks these quantities against the regenerated coefficient/moment sidecars.

`numerical compact` losslessly moves regenerable initial60 managed arrays to the local cache after checking all retained policy fields bit for bit. Public sidecars preserve policy coefficients, actual penalties, fold scores, seeds, history hashes and recovered moments, so financial diagnostics and quadrature can be repeated without replay. Initial60 training caches remain local; their checksums are recorded. This reduces the public sidecar payload while keeping the complete financial audit inputs. The twelve fixed representative histories supply the paired MCSE. Additional minimum-a stress cases are labeled separately and never pooled into a representative IID uncertainty calculation.
