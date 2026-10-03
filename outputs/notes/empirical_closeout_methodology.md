# Empirical close-out: retrospective sample sensitivities

## Status of the evidence

The proposed three figure roles are E1 (managed-payoff spectrum), E2 (row-normalized
year-by-C/T learnability), and E3 (common-period window tradeoff). **All three current
figures are explicitly labeled complete-payoff sample sensitivities. None is presented
as pristine real-time investment evidence.** Rebuilding the formation universe is
possible; identifying every realized payoff from the current snapshot is not.
The missing-payoff limitation is isolated rather than silently filled or reclassified.

The private formation-only panel now contains 2,504,402 stock-months
in 744 months, before any future-payoff filter. Ranks are constructed
from formation metadata and characteristics only, retaining unknown returns as NaN.
Changing all future payoffs to missing leaves identities and ranks exactly unchanged
(maximum perturbation error 0.0).
The maximum characteristic-rank change versus the earlier complete-payoff construction
is 0.00587510 on the [-0.5,0.5] scale.
This measures the cross-sectional rank effect, not the resulting portfolio-performance bias.

There are 7,489 missing forward payoffs. The audit matches each
security to its **exact next calendar month** in the frozen raw source and tries the
source's current excess return. It recovers 0;
7,489 remain unidentified across 638 formation
months. No nearest-date substitution, zero return, sample reweighting, or undocumented
delisting imputation is applied. Calculating complete managed returns for this
formation-only panel would require additional source evidence or an explicit missing-return
assumption. The original fitted cache is therefore retained only for the labeled
sensitivity diagnostics, avoiding a misleading claim that reranking alone solves missing
payoffs. These numbers do not prove the missing payoffs are zero or economically irrelevant.

The previous CRSP checks concerned **finite** JKP forward returns lacking a next-month
characteristic row; they do not resolve these 7,489 missing outcomes.
No new WRDS fetch was used in this close-out. Private rebuilt files are in
`data/formation_only`; `formation_timing_by_month.csv` publishes aggregate counts only.

## Characteristic selection and timing

The 130-variable list was recomputed from the 153 frozen candidates using only
1963-1972 formation metadata and characteristic coverage, with alphabetical ties.
It exactly matches the stored list: 223,762 reference
observations, no forward-return availability or OOS outcome in selection.
This meets the pre-OOS coverage-freeze convention. It does not reconstruct historical
vintages: candidate discovery, revised accounting values and current JKP snapshot
availability cannot be certified as known in real time. The list is a retrospective
research specification, clearly separate from outcome-dependent feature selection.

Annual fitting continues to use the already audited January decision chronology:
T payoffs through December y-1; first T-V train, last V=min(60,floor(T/3)) validate;
refit all T, then evaluate January-December y. The 1,201-point lambda paths include
zero and use inner-training information only. Representation settings remain frozen
from the initial 1963-1972 sample. Existing future-payoff perturbation checks show
that later portfolio payoffs cannot change earlier fitted grids or selections.
That estimator-level check does not cure stock-source selection or vintage limitations.

## E1: managed economic spectrum

The three curves use the same December 31, 2023 payoff cutoff for T=60,120,240.
They diagonalize the uncentered second-moment operator G'G/T via GG'/T.
Formation rows run through November 2023. Neither centered covariance nor a spectrum
of raw characteristics is substituted for the managed-payoff operator.

## E2: row-normalized heatmaps and selection quality

For each year and T, the ex-post oracle minimizes OOS response-one loss across the
entire saved lambda path; ties choose largest lambda. This diagnostic never selects
the portfolio. Default color is DeltaQ=Q-min(Q); the appendix includes natural
log(Q/min(Q)) and the original raw loss, each separately labeled. The natural-log
version reveals smaller within-year differences more clearly, while DeltaQ retains
an immediately interpretable loss scale. All paths here have positive minima.

The 60 equal-width C/T bins select the observed path point nearest the bin center
within the bin. Empty cells remain blank. Minima are computed on the **full** paths
before binning, not from displayed cells. Saved cell tables identify candidate indices.
Validation overlays are white circles with black borders, without connecting lines.
Near-one means C/T >=0.99; near-zero means <=0.01, fixed before inspecting frequencies.
The full table includes absolute C/T gaps and fractions within 1%, 5%, and 10% of oracle.

| T_months | years | mean_validation_regret | mean_relative_validation_regret | spearman_C_over_T | within_10_percent | fraction_near_one |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | 47 | 0.1838 | 0.3317 | -0.1637 | 0.4255 | 0.4468 |
| 84 | 47 | 0.1880 | 0.3509 | 0.2481 | 0.4681 | 0.2766 |
| 120 | 47 | 0.1914 | 0.4125 | 0.0466 | 0.4255 | 0.3191 |
| 180 | 46 | 0.1757 | 0.4339 | 0.0366 | 0.4130 | 0.1957 |
| 240 | 41 | 0.2409 | 0.5781 | -0.1448 | 0.4146 | 0.2439 |
| 360 | 31 | 0.2383 | 0.4692 | -0.1815 | 0.3871 | 0.0968 |

Validation is not consistently interior and does not track the ex-post optimum reliably.
Its mean relative regret ranges from approximately 33% to 58%; Spearman correlations
range from approximately -0.18 to +0.25. Ex-post selection among 1,201 candidates using
only 12 OOS returns also makes the oracle optimistic as a regime diagnostic.
These are diagnostics of a noisy tuning problem, not a claim of implementable oracle gains.

## E3: exact common-period comparison

Every row uses exactly the same 372 monthly payoffs, January 1994-December 2024.
Response-one loss is the mean of (1-r)^2; volatility uses the sample standard deviation
(ddof=1), and annualized Sharpe is sqrt(12)*mean(r)/std(r). Raw portfolio payoffs are
unscaled excess returns from the response-one estimator, without a cash overlay,
transaction costs or an additional leverage normalization. Their magnitudes should
not be interpreted as returns on a standard unlevered stock index. These values are
not averages of 31 annual Sharpes. Mean C and C/T use the 31 annual selections.

| T_months | months | response_one_loss | monthly_mean | monthly_volatility | annualized_sharpe | mean_C_over_T |
| --- | --- | --- | --- | --- | --- | --- |
| 60 | 372 | 0.8140 | 0.4735 | 0.7336 | 2.2355 | 0.6818 |
| 84 | 372 | 0.8022 | 0.4426 | 0.7020 | 2.1840 | 0.5496 |
| 120 | 372 | 0.7817 | 0.4721 | 0.7102 | 2.3028 | 0.5568 |
| 180 | 372 | 0.7074 | 0.5004 | 0.6775 | 2.5584 | 0.5204 |
| 240 | 372 | 0.7353 | 0.5358 | 0.7219 | 2.5710 | 0.6025 |
| 360 | 372 | 0.7042 | 0.5359 | 0.7001 | 2.6518 | 0.5317 |

Uncertainty uses 5,000 circular moving-block bootstrap replicates with 12-month blocks
and seed 20261003. The **same monthly indices** resample all six windows in each
replicate. Pairwise percentile 95% intervals for all 15 loss differences are saved;
each includes zero in this sample, so the point-estimate ranking is not a statistically
clear winner under this procedure. Intervals describe uncertainty of the realized
performance series, conditional on already fitted policies; the bootstrap does not
rerun training and is not a proof of stationarity across 1994-2024.
The question is more observations versus older, potentially less relevant regimes,
not the stationary asymptotic T-law. Turnover is unavailable because local annual
stock weights were not retained by this pipeline; it is not set to zero.

## Reproduction and proposed placement

Run `VECLIB_MAXIMUM_THREADS=1 python3 audit_formation_timing.py --raw data/raw --destination data/formation_only`
then `VECLIB_MAXIMUM_THREADS=1 python3 empirical_closeout.py`.
No fitting is repeated for these diagnostics; their input paths and cache hashes
are pinned in `manifest_empirical_closeout.json`. The source-timing audit took
794.5 seconds on the recorded local run.

Use E1-E3 as the proposed three empirical figure roles **with their sensitivity status
visible**. The existing absolute-C and selected-path figures and the new raw/log-loss
heatmaps belong in an appendix. Gaussian cache replication remains an available
pipeline option, not an additional completed result of this close-out.
