# Local learnability heatmaps: matern32

**Evidence status:** complete-payoff sample sensitivity. The subsequent formation-only
rebuild, unresolved missing-payoff audit, and common-period diagnostics are documented
in `empirical_closeout_methodology.md`. These cached results are not pristine real-time OOS evidence.

T is the length of trailing monthly history treated as locally informative for the current regime.
It is not the total history of a globally stationary economy, nor a count of stock-month rows.

## Chronology and validation

Decision year y starts after the December 31 close of y-1. The trailing history contains T **payoffs** through that December, corresponding to formations through November y-1. The OOS block is January-December y (formations December y-1 through November y). For example, y=2000 and T=60 use January 1995-December 1999 payoffs; default inner training ends April 1998, validation is May 1998-December 1999, and OOS is January-December 2000.
At the information cutoff we assume month-end inputs can be used for portfolio formation;
execution delay and trading costs are not modeled. Every fit uses exactly T consecutive months.
Within that history the earliest T-V months train the candidate policy and the latest V months
validate it. By default V=min(60,floor(T/3)); `--validation-months` supplies a fixed override.
The candidate minimizing mean `(1 - raw portfolio excess payoff)^2` on validation is selected
(exact ties use the largest lambda). It is then refitted on **all T months**, and the overlay
uses the resulting full-window complexity. No OOS loss or Sharpe participates in selection.
This rolling-validation design replaces the fixed-C0 schedule for this experiment only.

## Estimation and grid

The existing fixed feature bank and managed payoffs are reused: 130 characteristics,
10,000 fixed RFFs and seed zero; representation inputs were fixed using 1963-1972.
Write G for the T-by-feature managed-payoff matrix and M=GG'. The repository solves
`alpha=(M+T*lambda*I)^(-1)1`. Complexity is `sum(mu/(mu+lambda))`, with mu the
eigenvalues of M/T, exactly equivalent to `tr[M(M+T*lambda*I)^(-1)]`.
The grid has 1200 positive log-spaced penalties from `mu_min*1e-6` to `mu_max*1e6`,
using only the **inner-training** spectrum, plus exact lambda=0. All 1201 candidates
are eligible for validation selection. Zero uses a minimum-norm pseudoinverse; numerical
rank uses a relative 1e-12 cutoff. Thus C/T is at most one. OOS loss uses the 12 held-out
monthly raw excess payoffs with response one; neither a ridge penalty nor kappa is added.
Annual OOS Sharpe is a secondary, noisy statistic computed from those same 12 payoffs.

## Heatmap construction

The main panels use T=60,120,240. Their common relative grid has 60 equal-width bins
on [0,1]. Each occupied year-bin uses the **observed** path point nearest its center,
among points inside that bin. Empty bins and unavailable years stay transparent; no
interpolation or extrapolation is performed. A saved cells table identifies every point
used. The absolute-complexity companion rescales these same bin edges by each panel's T;
its horizontal limits consequently differ. Both figures share the same full-path OOS-loss
color range and logarithmic normalization. Cividis has monotonic luminance for grayscale
printing; white circles with black borders identify validation selections.

## Coverage

| T (months) | V (months) | Decision years | Annual fits |
|---:|---:|---|---:|
| 60 | 20 | 1978-2024 | 47 |
| 84 | 28 | 1978-2024 | 47 |
| 120 | 40 | 1978-2024 | 47 |
| 180 | 60 | 1979-2024 | 46 |
| 240 | 60 | 1984-2024 | 41 |
| 360 | 60 | 1994-2024 | 31 |

23 requested year-window combinations are unavailable; reasons and available
history are recorded in the skipped-windows CSV. Plots retain the requested year axis.
The summary gives arithmetic means across each T's **available years**; different coverage
means those rows are not a matched-period causal comparison. Mean annual Sharpe is not
the Sharpe of the concatenated monthly history.

## Provenance and limits

The manifest records source/cache/code hashes, numerical settings and perturbation checks.
Changing OOS and later payoffs must leave earlier grids, validation selections and fitted
coefficients unchanged; Gram and primal implementations are independently compared in tests.
The inherited sample excludes missing future stock payoffs before ranking. Its dependence
on future payoff availability, the retrospective characteristic dictionary and the current
JKP snapshot remain source-level limitations. The overlay introduces no additional future
information. Local windows do not by themselves establish a stationary local regime.

Reproduce with `python3 local_learnability.py --kernel matern32 --timing calendar`.
Use `--kernel gaussian` for the Gaussian representation (its cache is built when absent).
Use a fresh `--output` directory for a new run; source data and private caches are excluded
from Git. The published path parquet contains portfolio-level aggregates only.
