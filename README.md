# Portfolio learnability — reproducible empirical analysis

Only **Linear, Gaussian, and Matérn-3/2** belong to this experiment. There is one cleaner, one estimator, and one figure/report generator. Old results must never be passed to the new report generator.

## Current execution status

The first raw-data attempt on 1 October 2026 failed at the WRDS connection: **one connection attempt, zero SELECT calls, OperationalError**. Run: https://github.com/ENRICOBIGNOZZI/portfolio_learnability/actions/runs/36896896876 . It has not been retried automatically. The new empirical backtest, its numerical results and its real-data RFF figures have **not** been produced. Passing unit tests is not an empirical rerun.

## What the cleaning follows

Primary reference: Didisheim, Ke, Kelly and Malamud, *APT or AIPT? The Surprising Dominance of Large Factor Models*, NBER 33012, September 2024 version, Section 2.5, printed p. 15:
https://www.nber.org/system/files/working_papers/w33012/revisions/w33012.rev0.pdf

PDF SHA256: `5ba2ce32de3ac9a21e5dd6396dc82d85378e48814910baf536a1af013372f424`.

That passage specifies 153 JKP candidates; NYSE/AMEX/NASDAQ and CRSP share codes 10/11/12; exclusion of nano stocks; the 130 characteristics with the fewest missing values; exclusion of rows with more than 30% missing characteristics; monthly ranks in [-0.5,0.5]. It does **not** specify our former <=1/3 characteristic cutoff yielding 132 variables.

**Declared adaptation, not literal replication:** the 130 names are chosen using **1963–1972 only**, after the formation-universe filters, with deterministic alphabetical tie-breaking. Their names then remain fixed. A separate coverage table records the calculation. We do not assert that this is the authors' exact 130-name list or that adopting a later-published information set eliminates all historical research-selection issues. Formation dates extend to 2024, beyond the reference sample. The standard JKP primary/common/main-observation flags are also retained. These choices are not the distinct JKMP 115-variable, NYSE-large-stock, entry/deletion protocol.

The 153 candidates in `characteristics.json` are frozen from JKP's published Factor Details dictionary (source XLSX SHA256 `4c579e4dcb93eed0897941d8f5f84e9cc784a2d3fc696be68a29a24e68206141`). We do not classify arbitrary metadata columns as predictors.

Observed monthly values use `(average_rank-1)/(observed_count-1)-0.5`. Residual missing values are mapped to **neutral zero**. With ties, zero need not be the empirical median. This implementation convention is explicit; it is not attributed as an exact algorithm stated in the source paragraph.

Formation universes and N_t never depend on future-return availability. Missing lead returns may be recovered from an actually observed next-calendar-month return in the same raw snapshot. Unresolved payoffs are **not** silently set to zero and securities are **not** dropped to make the backtest run. The fit is blocked until documented return corrections reconcile them. Publication lags and source-level delisting completeness remain properties of the JKP snapshot, not guarantees established by these unit tests.

## Frozen experiment

- First train: 1963–1972; validation: 1973–1977. Expanding training, preceding five formation years for validation, annual refits.
- OOS formation: January 1978–December 2024. Realized returns: February 1978–January 2025. A refit takes place after the first formation-month close; labels available at that close may be used.
- Response-one ridge loss; 120 positive penalties built from the initial managed-payoff spectrum and held fixed. Validation, not test Sharpe, chooses lambda.
- Main nonlinear specification: 10,000 RFF, seed zero. Median Euclidean distance among 1,000 initial-training characteristic vectors fixes the common bandwidth.
- P sensitivity: 250, 500, 1,000, 2,000, 4,000 and 10,000. Separate random streams for Gaussian frequencies, Matérn radial scales and phases ensure **nested prefixes**. The sqrt(2/P) normalization changes with P. Main P is not selected from OOS performance.
- Numerical approximation error uses only initial-training vectors; its range across seeds 0/1/2 is not a confidence interval. Economic P sensitivity uses seed-zero policies, not a seed ensemble.
- One common positive portfolio scale is calibrated from the first Linear validation portfolio, targeting median gross 1.8 there, and is fixed before the first OOS return. Reported gross is its **time mean**, not median. No time-varying gross cap or ex-post volatility rescaling.
- Turnover is the L1 difference between target weights over the union of consecutive universes. The optional 25-bp net Sharpe is a **turnover-cost proxy**, not drift-adjusted execution accounting. It never chooses complexity.

Effective complexity is a trace of the ridge filter, not the number of stocks or profitable factors. A test-window maximum is descriptive and ex post. The report checks whether the maximum is interior; it does not insert that conclusion irrespective of the data. Finite RFF spaces for different kernels are not asserted to be nested.

## Reproduce

Python 3.12 is the pinned CI interpreter.

```sh
python -m pip install -r requirements.txt
python -m pytest -q tests

# Only after an explicit authorization for a fresh authentication:
# export WRDS_USERNAME=... and WRDS_PASSWORD=... securely, never in source files.
python download.py --names characteristics.json --out data/raw

# No subsequent command connects to WRDS.
python run.py prepare --raw data/raw --clean data/clean
python run.py fit --clean data/clean --out results/new --kernel linear
python run.py fit --clean data/clean --out results/new --kernel gaussian
python run.py fit --clean data/clean --out results/new --kernel matern32
python run.py risk-free --out data/risk_free.csv
python run.py report --results results/new --risk-free data/risk_free.csv --out paper

cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The downloader makes a single authenticated connection and a single streaming SELECT. A failed attempt leaves a sentinel; it cannot silently retry in the same directory. A complete cache is reused only after checksum validation. The report refuses mixed source manifests, incomplete calendars, corrupt inputs or unresolved payoffs. It generates one empirical section without subsections, all new figures, numerical CSVs, BibTeX, and a standalone empirical-section PDF. This PDF is not the full theoretical manuscript.

Figures: log cumulative wealth; colored gross/net exposure for all three policies; Gaussian and Matérn historical/test complexity panels for the 2024 window; their managed spectra; kernel approximation error versus P; OOS portfolio Sharpe versus P. No correlation figure, stacking discussion, NTK, other Matérn specifications, rolling-window variants or simulated empirical results.

## Data and security

Raw and cleaned stock-level data and saved security weights are licensed/private inputs. They are gitignored and only uploaded encrypted in CI. Public artifacts contain aggregate portfolio returns, statistics, figures and source manifests. Do not print or commit WRDS credentials. Historical commits previously contained credentials: removing active files does **not** rotate credentials or erase Git history. Credential rotation and any coordinated history purge are separate operations.

The new source tree deliberately contains no prior empirical plots or old pipeline copies. Git history is the recovery mechanism, not an `archive` or `legacy` directory in the active code.
