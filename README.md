# Portfolio learnability: final empirical rebuild

The active experiment contains only **Linear, Gaussian and Matérn-3/2**. Old
cleaned data, fitted models, performance statistics and figures are not valid
inputs. Licensed raw snapshots and stock-level outputs stay outside Git.

## Execution status

The raw WRDS snapshot contains **3,687,389 stock-month observations**, covering
January 1963 through January 2025, acquired on 1 October 2026. Its private
`data/raw/manifest.json` records every file checksum. Credentials are not saved.

By user-confirmed convention, stock-months with missing or nonfinite JKP
`ret_exc_lead1m` are excluded **after the formation metadata and characteristic
missingness filters, before monthly ranking and calculation of N_t**. This is
our complete-case sample choice, not a documented Didisheim rule. The resulting
sample is conditional on future payoff availability. No payoff is imputed or
recovered from another field. Private `data/clean/excluded_returns.csv` records
every exclusion; `universe_counts.csv` reports monthly before/after counts.
The previous stop-on-missing-payoff policy is superseded.

Preparation is complete: **2,496,913 retained stock-months**, **744 months**,
**7,489 exclusions**, and **3,356.07 stocks per month** on average. The initial
coverage sample contains 223,762 observations; restricting feature selection to
1963–1972 changes 12 selected characteristics compared with full-period coverage.
**The real-data run is complete:** all three representations have 47 refits and
564 OOS monthly payoffs. The five PDF/PNG figures, performance table, empirical
LaTeX section and reproduction manifest are in `paper/`. All five PDFs were
rendered and visually inspected. The implementation passes 57 tests.

| Policy | OOS Sharpe | Maximum drawdown |
|---|---:|---:|
| Linear | 3.0911 | -15.7741% |
| Gaussian | 3.2999 | -15.6100% |
| Matérn-3/2 | 3.4610 | -13.6346% |

The common lengthscale is 4.3921077846 and kappa is 0.0374000023. The descriptive
2024 test maxima occur at effective complexities 76.7566 (Gaussian) and 89.5107
(Matérn); these ex-post peaks do not select the fitted policies.

`paper/temporal_audit.json` records the temporal, accounting, source and memory
checks. The combined resident-memory peak sampled across the two model workers
was 1,277.4 MiB; per-process OS peaks and the one-second sampling convention are
recorded separately. Annual caching and 256-stock feature blocks bound memory.

The exact 130 author variables were not found in the official materials searched.
The 130 best-covered characteristics are selected from the fixed 153-variable
JKP research dictionary using **only the initial training period 1963–1972**.
Alphabetical names break ties. This deliberately differs from the paper's full
1963–2023 coverage selection to avoid using later observations. Validation,
test and the 2024 extension cannot affect feature selection. The selected names,
all missing shares and provenance are saved in `characteristic_selection.json`.
The fixed published dictionary itself is a retrospective research specification,
not a claim that all 153 signals were discovered by 1972 or that the current
JKP snapshot is a historical data vintage.

Linear uses the user-confirmed affine specification `[1, Z]`: 130 input
characteristics and 131 coefficients, including the intercept.

## Reproduce locally

Python 3.12 is used in CI. Unit and integration fixtures are synthetic tests,
not empirical results. The commands below describe reproduction in fresh output
directories; existing prepared data are never silently overwritten. The current convention drops missing payoffs explicitly during `prepare`.

```sh
python3 -m pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests

# Set WRDS_USERNAME and WRDS_PASSWORD securely in the process environment.
# Do not put their values in source, command-line arguments or a tracked file.
python3 download.py --names characteristics.json --out data/raw --max-connections 1
python3 run_empirics.py prepare --raw data/raw --clean data/clean
python3 run_empirics.py risk-free --raw data/raw --out data/risk_free.csv
export VECLIB_MAXIMUM_THREADS=2
python3 run_empirics.py fit --clean data/clean --out results/final --kernel linear
python3 run_empirics.py fit --clean data/clean --out results/final --kernel gaussian
python3 run_empirics.py fit --clean data/clean --out results/final --kernel matern32
python3 run_empirics.py report --results results/final --risk-free data/risk_free.csv --out paper
```

Memory use is bounded by a one-year panel cache and nonlinear feature blocks of
256 stocks. On macOS the double-precision vector cosine uses Apple Accelerate;
other platforms use NumPy. Tests compare both paths to double-precision NumPy,
and compare batched scores and managed returns to dense feature calculations.
The feature bank, 10,000-feature count and statistical specification are unchanged.

The downloader makes at most one connection by default and downloads annual
parquet files, including January 2025. A resumed raw file is accepted only with
its recorded checksum. Failed database exception text is not printed or saved.
GitHub Actions runs tests only; acquisition belongs on an authorized reachable
local machine. No stock-level artifacts or secrets are uploaded by CI.

Preparation verifies schema, checksums and duplicate security-months before
filtering. Formation uses U.S. CRSP common shares 10/11/12 on exchanges 1/2/3,
JKP common/primary/main flags and non-nano status. Rows with more than 39 of 130
missing characteristics are excluded, followed by missing/nonfinite forward
payoffs. Monthly ranks and N_t use the remaining sample. Observed values use average ranks mapped by
`(rank-1)/(n_observed-1)-0.5`; singleton and residual missing values become
neutral zero.

JKP excess returns already incorporate its documented source-level delisting
construction. The legacy SAS code documents a -30% convention for specified missing
performance-related CRSP delistings; this pipeline consumes JKP values and does
not independently invent such returns. Missing forward returns are dropped,
with no alternative-source recovery. All retained returns must be finite;
incomplete calendars and corrupt prepared data still block fitting.

The calendar audit compared 3,642,027 finite raw lead/current-return pairs from
adjacent months with zero discrepancies. For the 346 retained observations
without a next-month characteristic row, CRSP CIZ confirms the next-calendar
payoff: 345 match the monthly CIZ return less the frozen cash rate, and one
matches the delisting compounding documented in the
[current JKP Python code](https://github.com/bkelly-lab/jkp-data/blob/666e8960ed81a664f1e9f189f92925ab2eeffcd5/src/jkp/data/aux_functions.py).
The legacy CRSP tables alone did not reconcile these cases; WRDS documents the
[format transition](https://wrds-www.wharton.upenn.edu/pages/data-announcements/changes-to-crsp-data/).
These read-only cross-checks did not modify returns or sample membership.
The frozen protocol retains its legacy SAS reference; the supplementary audit
records the CIZ evidence separately, rather than claiming an unobserved WRDS
build version. This is not certification of historical database vintages.

## Frozen design

- Formation sample: January 1963–December 2024. Train 1963–1972, validation
  1973–1977 initially; then expanding training, five validation years, annual
  refits. Exactly 47 OOS windows / 564 monthly observations.
- Refits occur at the January formation close. By user-confirmed convention,
  **κ is frozen on 31 January 1978, before the first February OOS payoff**.
  This permits use of the December 1977 formation's January 1978 realized return.
- Direct maximum-Sharpe portfolio learning uses the quadratic ridge criterion.
  Training estimates each candidate; validation minimizes the quadratic portfolio
  loss; refitting then uses training plus validation. Test returns never select
  penalties. Each representation freezes its own 120 numerical lambda candidates
  from its initial training spectrum.
- Gaussian and Matérn-3/2 each use exactly 10,000 fixed RFF, seed 0. Both share
  the median pairwise Euclidean distance of 1,000 initial-training vectors.
  Matérn frequencies use independent normal vectors divided by `sqrt(chi2(3)/3)`.
- Weights equal `f(Z)/N`. The pipeline checks stock payoffs against managed-feature
  payoffs every test month. The same positive κ, calibrated to Linear validation
  median gross 1.8, scales all models and months. No monthly normalization or cap.
- Cash returns are recovered as JKP current total minus current excess return
  from the same snapshot, verifying consistency across CRSP stocks each month.
  Wealth compounds `1 + Rf + κ * portfolio_excess`. Drawdown uses that same wealth
  with a unit starting point, and is independently checked. Maximum drawdown
  is signed. Excess-only wealth is saved as a diagnostic.

## Final artifacts

Only a successful new run generates:

1. `paper/figures/fig01_wealth_drawdown.{pdf,png}`
2. `paper/figures/fig02_exposure.{pdf,png}`
3. `paper/figures/fig03_complexity_gaussian.{pdf,png}`
4. `paper/figures/fig04_complexity_matern32.{pdf,png}`
5. `paper/figures/fig05_managed_spectrum.{pdf,png}`

The compact table is `paper/tables/performance.{csv,tex}`. The report also writes
`paper/empirics.tex`, aggregate figure inputs, selected-characteristic provenance,
drawdown episodes and crisis diagnostics, and `paper/reproduction_manifest.json`.
The LaTeX section requires `graphicx` and `booktabs` in the host manuscript.
The manifest binds raw, code, sample, lambda, scale, figure-input and figure-output
checksums. The final complexity panels contain **only** the twelve subsequent
payoffs for 2024 formations (February 2024–January 2025), not pooled returns.

Publication stops if any total monthly return is at or below -100%, data or
calendars fail validation, or Sharpe is at least 2.5 while the worst pre-2020
drawdown is shallower than 5%. This last threshold is a review trigger, not a
proof that other results are plausible. The committed run passed the real-data accounting diagnostics and final PDF/PNG
visual inspection. High Sharpe alone is not evidence that source-vintage biases
have been eliminated.

## Sources and security

Primary construction sources and pinned checksums are in `protocol.json`:
[AIPT, September 2024, §2.5](https://www.nber.org/system/files/working_papers/w33012/revisions/w33012.rev0.pdf),
[JKP documentation](https://jkpfactors-data.s3.amazonaws.com/documents/Documentation.pdf),
and [JKP return construction](https://github.com/bkelly-lab/ReplicationCrisis/blob/67174c7f170bf8b952b3c86876f187fd9970b5ba/GlobalFactors/project_macros.sas).
The active tree contains no credentials. Deleting an active file does not erase
secrets from historical commits or rotate credentials; history rewriting and
credential rotation are separate operations.

## Initial-C0 Matérn complexity experiment

`complexity_schedule.py` produces four additional, separate empirical figures in
`paper/complexity_schedule/`. This experiment implements the manuscript's
r = 1 rate with an annually estimated spectral exponent:
`lambda_T = C0 * T ** (-b_T / (b_T + 1))`, where T counts historical monthly
managed payoffs. It differs from the original three-model results above, which
select lambda again on each preceding five-year validation window.

C0 is selected **once** using 1963–1972 training and 1973–1977 formation
validation, available at the January 1978 close. Its grid retains all 120
original initial-training lambda candidates, adds 80 intermediate effective
complexity points, and converts the resulting 200 penalties to C0 using the
initial training T and b. Subsequent OOS performance cannot change C0.
Each January close, the coefficients and b use the expanding history through
the previous December formation (whose payoff is then observed).

The spectral estimate is an OLS slope of log second-moment eigenvalues on log
rank, using the fixed 10%–60% band of positive ranks, with a relative 1e-12
numerical cutoff. Both the band and cutoff are fixed for every year; the fit
is not tuned against OOS performance. Estimates at or below b=1 cause a stop,
not a silent bound adjustment. This is an explicit finite-sample plug-in
convention: the supplied manuscript gives the asymptotic rate, but does not
specify a spectral-slope estimator or the finite-sample proportionality constant.
The appendix's general r-dependent rate is specialized to r=1 as in the main
text. Re-estimating b can create local increases in lambda; none are suppressed.

The four figures show all 47 annual OOS loss curves, all annualized OOS Sharpe
curves, the loss heatmap, and the selected complexity/regularization path.
Each annual curve evaluates the 12 payoffs from February through the following
January. Loss is the unscaled held-out mean of `(1 - portfolio_excess)**2`,
without an OOS ridge penalty. These curves do not use the common wealth-plot
scale kappa. Heatmap interpolation is restricted to each year's observed
complexity support; there is no extrapolation, smoothing or imposed U-shape.
The source limitations of the main experiment continue to apply.

```sh
VECLIB_MAXIMUM_THREADS=2 python3 complexity_schedule.py \
  --clean data/clean --cache results/schedule_cache \
  --out paper/complexity_schedule
```

Use a fresh output directory when reproducing. The private managed-payoff cache
is hash-checked against its frozen source manifest and feature bank. It is
excluded from Git. Actual first/final-window future-payoff perturbations are
checked before the figures are published; additional synthetic tests compare
predictions with an independent primal ridge calculation. All figure inputs,
the 200 initial validation scores, annual b diagnostics and selected monthly
payoffs accompany the four PDFs/PNGs, with source/code/output hashes.

The four companion `C/T` figures use each annual fit's **historical monthly**
sample size, not the number of stocks, stock-month rows, or the 12 test months.
T grows from 180 to 732. Generate them from the completed experiment with:

```sh
python3 complexity_schedule.py --out paper/complexity_schedule --normalized-only
```

This writes `paper/complexity_schedule/normalized/`, retaining all OOS losses,
Sharpes, years and fixed-C0 selections. Its CSVs include the original complexity,
T and their ratio, and its manifest links to the original input hashes.


### Grid resolution audit and final diagnostic curves

The initial policy still selects C0 from the 200 frozen candidates. A separate
validation-only audit evaluated 3,001 points over a range 1,000 times wider at
both ends and refined the best interval numerically. Its minimum improves the
initial validation loss by only 0.0004464%; the selected strategy is preserved.

The 200-point curves were too coarse in some regions: the largest discrepancy
between a plotted Sharpe segment and a recalculated intermediate value was
1.123. Final figures therefore use **1,201 diagnostic candidates per year**,
including every original candidate and a wider log-spaced grid. An additional
56,400 midpoint evaluations check their drawing accuracy. The audit checks
both the unregularized payoff and the strong-regularization Sharpe limit for
every annual fit. Boundary optima are recorded as boundary/limit behavior,
not presented as identified interior optima. These OOS diagnostics never choose
the investment policy or revise the initial C0.

After the original fit, reproduce the audited figures in this order:

```sh
VECLIB_MAXIMUM_THREADS=2 python3 audit_schedule_grid.py
python3 complexity_schedule.py --out paper/complexity_schedule --normalized-only
```

The grid audit updates the four original figure exports from saved managed
payoffs. Run the normalized export into a fresh `normalized` subdirectory.
`grid_audit.json`, `grid_endpoint_checks.csv`, and `diagnostic_paths.csv` contain
all resolution checks, endpoint comparisons and extended figure inputs.
