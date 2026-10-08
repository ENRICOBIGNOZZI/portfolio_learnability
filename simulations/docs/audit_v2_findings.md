# V2 audit findings

Starting reference: `cdfbd11a5390c15c54cace9b95e7d5e40c2ec35e`, branch `main`.
The initial tree, runtime, 3,891 output hashes, 112 empirical hashes and protected
theory hashes are recorded in `outputs/confirmation_v2/audit/`.
Historical evidence stays in place and is read-only to the revised runner.

| Finding | Location and evidence | Severity/type | Minimal intervention and verification | Status |
|---|---|---|---|---|
| Orchestration omitted from cache identity | `run.context_for`; old SCIENCE_SOURCES omitted run.py and audit dependencies | P0 verification vulnerability | Conservative science fingerprint, explicit effective kernel, execution-only workers; mutation tests in `test_simulation_integrity_v2.py` | Implemented, tests passed |
| Audit cache trusts stale `passed` | `run.environment_audit`, `ensure_baseline`; no complete source/settings binding | P0 verification vulnerability | Schema/source/settings/kernel identities; baseline audit moved inside versioned output | Implemented; fresh end-to-end smoke and all 189 tests passed |
| Manifest overwrites execution history | `run.run`; resume time replaces original runtime | P0 provenance gap | Preserve original event, append resume/render events, validate before replacing; atomic writes and exclusive output lock | Implemented |
| Production assertions disappear under optimization | `diagnostics.verify`, `diagnostics.closeout` | P0 verification vulnerability | Explicit runtime gates; subprocess `python -O` mutation test | Corrected; strict V2 closeout completed with all 13 conditions met |
| Incomplete raw reconstruction | `diagnostics.verify`; OOS, MCSE, benchmark and rate cells largely unchecked | P0 verification vulnerability | Independent scalar/spectral reconstruction and centered-sum/OLS/bootstrap algorithms in `diagnostics.reconstruct`; no call to summarizer | All 3,500 historical and 400 confirmation evaluations reconstructed; 462 publication cells and 280 paired rows verified |
| Synthetic NPZ files absent from clone | `.gitignore` excludes NPZ | P0 reproducibility gap | Versioned public release with archive and individual checksums; separate fast/full verification commands | V1 and V2 published and actually downloaded; 3,521 historical and 561 confirmation archive members verified |
| Constant rate series produces undefined SciPy stderr | `experiments.reporting.rate_bootstrap`; actual small fixture gave NaN for constant log series | Confirmed software defect | Exact constant-series OLS residual SE is zero; preserve slopes, reject nonfinite aggregates | Corrected, mutation fixture passed |
| Leakage test mutates unused array | `tests/test_simulation_pipeline.py` | P1 test gap | Actual stock test-return, population moment, W-star and OOS-length mutations preserve training/selection fingerprints | End-to-end mutation tests passed |
| Spectral slope band used as production truth gate | `diagnostics.baseline.audit_spectrum` | P1 scientific interpretation | Preserve descriptive fit/band; acceptance uses operator identity rather than closeness to exponent | Corrected; nine cases, 21,600 spectral rows and 81 fixed-window fits independently reconstructed; all fixed windows resolved, with no slope-closeness gate |
| Rank check reselects oracle and omits entire curve | `diagnostics.approximation`; 256→512 test limited to selected/oracle regret | P1 numerical limit | Preserve V1 test; add fixed-lambda exact Gram, nested ranks, payoff/complexity/floor and uncertainty | Exact reference: 288 comparisons. Independent rank reconstruction: 50 paths per map, 990 curve cells and 60 policy cells per map; 115/120 policy cells meet the criterion, five remain uncertified |
| Quadrature check uses nested seed and 20 paths | `diagnostics.quadrature` | P1 numerical limit | Four independent seeds, two group resolutions, at least 50 paired paths | Executed and independently reconstructed: 80/80 policy cells pass per map, with 32,000 fixed-policy evaluations per map |
| Oracle fit is not a theory-path experiment | `experiments.reporting` | P1 missing experiment | Frozen population calibration and exact analytic lambda(T), separate oracle and selectors | All 400 frozen evaluations computed and independently reconstructed, including whole-path bootstrap intervals |
| Mean lambda obscures skewness and boundaries | `reporting`, `plotting.figures` | P1 reporting limitation | Separate lower/upper frequency, mean/median/IQR/geometric mean and contributions | Implemented; all six environments aggregated and independently reconstructed |
| Baseline depends on one loading coordinate | `dgp.balanced.beta` | P2 proposed extension, not baseline defect | Canonical rich6d phase, eta=0 preservation, analytic gradients and independent dense solve | Algebra tests passed; full-cube gradient moments audited on 131,072 points |
| PDF provenance omits figures/nested inputs | `manuscript.compile_manuscript` | P0 provenance gap | Hash compiler `.fls` inputs, including figures, tables, bibliography and class/style files | Implemented; clean tracked checkout aad53a7 compiled, 199 compiler inputs hashed, all 94 final PDF pages visually reviewed |
| Large quadrature intermediates and concurrent numerical jobs consume memory | `resolution_v2`, execution settings; user requested lower memory | Resource constraint, not a scientific defect | One worker, sequential heavy jobs, 256-group accumulation and file-backed large spectral kernels; new diagnostic cache namespace preserves interrupted files | Five equivalence tests and seven summary mutation tests passed; rank/quadrature/spectral execution and independent reconstruction complete; relocation to a verified local checkout avoids repeated iCloud source-read stalls |
| New spectral cache initially omitted imported scientific dependencies | `spectrum_v2.run`; diagnostic-only file hash did not bind kernel/DGP/runtime or quadrature case | P0 verification vulnerability | Bind science sources, Python/NumPy/SciPy, groups and actual seed; preserve six old cases separately and recompute, without cache migration. Rank/quadrature caches also bind numerical runtime. | Corrected; cache-loader mutation tests and full rerun tracked in V2 audit |
| Log penalty axes conceal negative Gaussian interval bounds | `plotting.confirmation.render_environment`; headline T=2160 validation penalty lower bounds cross zero | P1 presentation defect | Preserve every interval bound; use labeled symmetric-log axes with fixed threshold 1e-5 where necessary | Corrected; actual theory-path renders inspected |
| Figure annotations overlap | `plotting.confirmation`; theory-path legend covers data and numerical-resolution titles collide | P2 layout defect | Move shared legends outside panels, wrap titles and typeset mathematical axis labels | Corrected in inspected renders and all 94 final PDF pages |

Final acceptance completed on 8 October 2026. `outputs/confirmation_v2/audit/final_report.json`
records all thirteen conditions; `paper/visual_review.json` binds the actual
94-page review to the compiled PDF. The final test transcript records 189 passes.
A final acceptance bug incorrectly treated Python's `-m pytest` launcher as a
pytest marker filter. The check now inspects pytest arguments separately; eleven
regression cases cover full-suite launchers and rejected filters/subsets. This
change does not alter simulation, aggregation or manuscript inputs.

The independent reconstruction agrees with all recorded historical numerical
surfaces and all six per-environment aggregate CSVs, including whole-path
bootstrap intervals. This does not prove the old fitting from stock returns:
V1 checkpoints did not store coefficients or return paths. No cache vulnerability
is treated as evidence that the historical estimates are wrong.

There are 3,500 historical environment/rank evaluations but 3,100 distinct
economic replication seeds: 400 rank evaluations reuse baseline paths.
The baseline at T=1440 has mean selected lambda about 8.6553e-4, median about
1.2743e-6 and combined boundary frequency 8.4%. These are distinct summaries.

The V2 rank/quadrature reconstruction is recorded separately in
`outputs/confirmation_v2/audit/rank_quadrature_reconstruction.json`. The subsequent
full `numerical_reconstruction.json` also verifies all 21,600 spectral rows and
81 fixed-window regressions. The unresolved policy
cells are `holdout25` at T=60 and T=360 for both maps and `rolling3` at T=60 for
rich6d. Every audited policy at T=1440 passes the rank and quadrature criteria.
These statements are relative to the tested rank-1024 reference and quadrature,
not an equality with the infinite kernel. The full penalty curves and extra
training horizons are not certified by those selected-policy results.
