# Portfolio learnability: final empirical rebuild

The active experiment contains only **Linear, Gaussian and Matérn-3/2**. Old
cleaned data, fitted models, performance statistics and figures are not valid
inputs. Licensed raw snapshots and stock-level outputs stay outside Git.

## Execution status

**The real-data run is incomplete. No new empirical results are claimed.**
The local historical raw parquet files omit the universe flags, CRSP share and
exchange codes, and current returns needed for the mandatory audit. They cannot
substitute for a complete raw snapshot. `WRDS_USERNAME` and `WRDS_PASSWORD` were
absent from the local process on 1 October 2026. No fresh authentication or
repeated GitHub-hosted acquisition has been attempted.

The exact 130 author variables were not found in the official materials searched.
`protocol.json` records the sources and the fallback: best coverage among the
153 candidates in `characteristics.json`, measured over **1963–2023**, the sample
in Section 2.5 of the September 2024 AIPT paper. Alphabetical names break ties.
This reference selection uses historical research information; it is not a
claim of real-time feature discovery. The 2024 extension cannot affect selection.
The final selected names and all 153 missing shares are generated only from the
new complete raw snapshot, in `data/clean/characteristic_provenance.json`.

Linear currently retains the existing affine specification `[1, Z]` (130 input
characteristics, 131 coefficients). The theoretical manuscript is absent from
this checkout, so equivalence to its exact linear specification remains to be
verified before accepting the empirical run.

## Reproduce locally

Python 3.12 is used in CI. Unit and integration fixtures are synthetic tests,
not empirical results.

```sh
python3 -m pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests

# Set WRDS_USERNAME and WRDS_PASSWORD securely in the process environment.
# Do not put their values in source, command-line arguments or a tracked file.
python3 download.py --names characteristics.json --out data/raw --max-connections 1
python3 run_empirics.py prepare --raw data/raw --clean data/clean
python3 run_empirics.py risk-free --raw data/raw --out data/risk_free.csv
python3 run_empirics.py fit --clean data/clean --out results/final --kernel linear
python3 run_empirics.py fit --clean data/clean --out results/final --kernel gaussian
python3 run_empirics.py fit --clean data/clean --out results/final --kernel matern32
python3 run_empirics.py report --results results/final --risk-free data/risk_free.csv --out paper
```

The downloader makes at most one connection by default and downloads annual
parquet files, including January 2025. A resumed raw file is accepted only with
its recorded checksum. Failed database exception text is not printed or saved.
GitHub Actions runs tests only; acquisition belongs on an authorized reachable
local machine. No stock-level artifacts or secrets are uploaded by CI.

Preparation verifies schema, checksums and duplicate security-months before
filtering. Formation uses U.S. CRSP common shares 10/11/12 on exchanges 1/2/3,
JKP common/primary/main flags and non-nano status. Payoff availability never
changes formation or monthly ranks. Rows with more than 39 of 130 missing
characteristics are excluded. Observed values use average ranks mapped by
`(rank-1)/(n_observed-1)-0.5`; singleton and residual missing values become
neutral zero.

JKP excess returns already incorporate its documented source-level delisting
construction. The source code includes a -30% convention for specified missing
performance-related CRSP delistings; this pipeline consumes JKP values and does
not independently invent such returns. A missing lead can be recovered only from
the same snapshot's actually observed next-calendar-month excess return.
Unresolved payoffs produce **`data/clean/unresolved_returns.csv` and stop the run**.
No fit accepts that panel. A corrected complete source snapshot must be prepared
into a fresh output directory before proceeding.

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
proof that other results are plausible. Real-data diagnostics and final PDF/PNG
visual inspection remain required before claiming the experiment is complete.

## Sources and security

Primary construction sources and pinned checksums are in `protocol.json`:
[AIPT, September 2024, §2.5](https://www.nber.org/system/files/working_papers/w33012/revisions/w33012.rev0.pdf),
[JKP documentation](https://jkpfactors-data.s3.amazonaws.com/documents/Documentation.pdf),
and [JKP return construction](https://github.com/bkelly-lab/ReplicationCrisis/blob/67174c7f170bf8b952b3c86876f187fd9970b5ba/GlobalFactors/project_macros.sas).
The active tree contains no credentials. Deleting an active file does not erase
secrets from historical commits or rotate credentials; history rewriting and
credential rotation are separate operations.
