# One-Constant Theory-Guided Spectral Shrinkage

Recommend excluding this method from the paper: it does not add convincing financial value over annual CV.

## Frozen design and chronology

Starting commit: `0b565955a1f9ffd7babf8d53f6549b210c3f863b` on `main`. This isolated run reuses the real JKP cache, 130 characteristics, the fixed 10,000-feature Gaussian bank (seed 0, bandwidth 4.39210778461642), and portfolio.py's annual_splits, fit_windows and dual_path. The original annual CV minimizes validation response-one squared loss, not validation Sharpe. No stock preprocessing, feature selection, bandwidth tuning or manuscript change is included. The inherited complete-payoff stock sample can itself depend on payoff availability; this experiment preserves that limitation rather than claiming point-in-time universe purity (see the hashed formation-timing audit).

The 120 positive penalties are fixed from January 1963-December 1972 formation observations. Bounds remain [5.168051289907321e-11, 0.00477969090655684]. Training for year y ends in y-6, validation uses formation years y-5 through y-1, and refitting uses all past formation observations from January 1963 through December y-1. The last refit label is realized at January close y, when the policy is formed; the first OOS payoff is February y. Each annual test ends in January y+1. Thus decision years 1978-2024 contain 564 unique OOS months (February 1978-January 2025). The primary 1979-2024 comparison contains 552 months (February 1979-January 2025), excluding the anchored decision year. Annual dates are in annual_selection.csv and every formation/return pair is in monthly_oos.csv. Performance annualizes concatenated excess returns using sample standard deviations; annual Sharpes are descriptive only.

## Algebra and calibration

Use unscaled, uncentered managed payoffs, mu=eig(gg'/T), C(lambda)=sum mu/(mu+lambda), and minimize F(lambda)=lambda+(rho/T)C(lambda). The spectrum and penalty share raw units; the frozen display scale kappa=0.03740000227285338 multiplies both policies only after fitting. Zero eigenvalues contribute zero. Meaningfully negative Gram eigenvalues are rejected; only roundoff negatives within 1e-10 times the largest absolute Gram eigenvalue are clipped. Active rank uses a 1e-12 relative threshold for diagnostics only, without truncating the proxy spectrum.

For a nonzero spectrum, F''(lambda)=2(rho/T)sum mu/(mu+lambda)^3 is strictly positive. Derivative signs identify the constrained boundary optimum; otherwise Brent solves the unique root in log(lambda), with fixed bounds. This is an algebraic result about the proxy, not a theorem about financial optimality.

The only calibration uses the 1978 CV decision: train 1963-1972, validate 1973-1977, refit T0=180 through December 1977 formation (last known return January 31, 1978). Its interior raw penalty is lambda0=2.515462907417361e-07, C0=19.494340129, S0=41434574.715117186; rho0=T0/S0=4.3441980818576606e-06. The constant is frozen for all future years. The root matches lambda0 to relative error 2.22e-16. Calibration has no 1978 test-payoff argument and a future-perturbation test verifies isolation. A boundary anchor would fail with an explicit identification limitation rather than silently pick rho.

## Primary common-month evidence

| Scenario | CV SR | Spectral SR | Difference | 12-month paired 95% interval |
|---|---:|---:|---:|---:|
| gross | 3.2772 | 3.3301 | +0.0529 | [-0.2847, +0.3772] |
| target_trade25 | 2.6355 | 2.7247 | +0.0892 | [-0.1958, +0.3611] |
| borrow30 | 3.2403 | 3.2921 | +0.0518 | [-0.2817, +0.3706] |
| target_trade25_borrow30 | 2.5966 | 2.6858 | +0.0892 | [-0.1926, +0.3569] |
| trade25_borrow30 | 2.5871 | 2.6759 | +0.0887 | [-0.1939, +0.3565] |

Gross annual mean/volatility: CV 27.160%/8.288%; spectral 21.483%/6.451%. Gross maximum drawdown: CV -15.610%, spectral -9.988%. All means, volatilities, drawdowns, exposures and fees by scenario and subperiod are in summary.csv. Wealth uses total returns including the same cash rate and is normalized at the primary-window start.

Paired circular-block uncertainty uses 5000 shared month-index draws per period, seed 20261010; differences are spectral minus CV. Percentile intervals are conditional on the fitted paths, not whole-pipeline sampling inference; there is no retraining or recalibration in the bootstrap. Arbitrary nonstationarity limits a blanket bootstrap interpretation, and subperiod comparisons are descriptive without multiple-comparison adjustment.

Gross block sensitivity:

- 12 months: [-0.2847, +0.3772], bootstrap SE 0.1666.
- 6 months: [-0.3085, +0.3715], bootstrap SE 0.1731.
- 24 months: [-0.3004, +0.3940], bootstrap SE 0.1776.

Gross subperiod Sharpe:

- 1979-1989: CV 5.541, spectral 5.621.
- 1990-1999: CV 5.813, spectral 5.402.
- 2000-2009: CV 2.812, spectral 2.617.
- 2010-2019: CV 2.978, spectral 2.658.
- 2020-2024: CV 0.759, spectral 1.511.

The discrete-grid proxy uses exactly the same 120 frozen candidates and rho. Its primary gross Sharpe is 3.3321, versus 3.3301 for the continuous rule. Its annual choices and paired intervals are retained as a small grid robustness check, without any OOS grid-hunting.

## Complexity, numerical checks and cost accounting

Primary mean C: CV 126.808; spectral 30.644. Spectral complexity ranges from 19.494 to 40.733; lambda ranges from 1.30667696e-07 to 2.51546291e-07. Boundary counts: {'interior': 47}. The annual table contains C, lambda, active rank, PSD clipping, derivative/KKT residuals, ridge residuals, chronology, and each method's annual realized metrics. Increasing rho under a fixed spectrum increases the selected penalty and cannot increase active complexity; this is covered by deterministic tests.

All 47 annual CV selections and all 564 payoffs reproduce the original fixed-bandwidth Gaussian snapshot and the three-experiment snapshot, at rtol=1e-7, atol=1e-9. The clean-manifest and feature-bank identities match exactly. The cache's managed-array hash differs from the original array hash: the cache uses 64-stock reductions while original managed_matrix uses 256, so float64 accumulation orders differ. The first real managed month is independently recomputed with the original reduction routine, and the full selected payoff paths reconcile. Both hashes and the first-month check are recorded in baseline_compatibility.json before comparison; this byte mismatch is disclosed rather than treating the arrays as bitwise identical. Maximum original/legacy raw payoff errors are 8.79e-11/2.22e-16; maximum root KKT residual 5.41e-14; maximum complexity discrepancy 4.73e-12; maximum ridge residual 2.09e-11. Independent direct ridge solves at 1978, 2000 and 2024 and feature-space payoff aggregation agree. The entire calculation uses deterministic float64 CPU routines and explicitly limited BLAS threads, with no GPU dependency.

Both policies were reconstructed from the licensed stock panels, using the identical frozen feature bank. Monthly weights, identifiers, features and model arrays remain private under results/. CV weights and the original drift-adjusted net account were independently reconciled. Stock-level check maxima are in audit/cost_verification.json and tables/stock_reconstruction_checks.csv. Target turnover is the two-sided sum of absolute changes in signed target weights, including entry, exits, and annual refits. The requested illustration subtracts 0.0025 times target turnover and 0.003/12 times short notional from gross excess returns. The repository self-financing sensitivity separately solves c=0.0025*||(1-c)w-d||_1 and deducts the short fee on post-cost NAV using core.accounting_step. Each policy has its own weights and fee path; costs are never inferred from baseline turnover. Entry from cash is charged in February 1978; the primary 1979-2024 comparison carries that account state. Terminal positions are marked without forced liquidation. These are illustrative fees, not observed transaction or lending costs; financing uses the frozen cash rate without impact or an extra funding spread.

## Limits of the theoretical interpretation

- The starting A*lambda+B*C(lambda)/T is a theoretical upper envelope / fixed-lambda bound under the paper's assumptions, not an identity for exact Sharpe regret. Calibration does not estimate A or B independently; rho is one historically calibrated tuning ratio and need not be structural under nonstationarity.
- A data-dependent empirical-complexity argmin does not automatically inherit a theorem proved only for fixed deterministic lambda. A uniform-in-lambda concentration result would be required.
- Gaussian managed spectra need not be polynomial. This rule uses the full empirical spectrum and never estimates b or extrapolates the sample tail.
- A constrained numerical minimum need not be an unconstrained oracle optimum. Finite sample rank can alter behavior near zero regularization. Bounds must not be expanded after observing returns; frequent boundary solutions would be a limitation.
- Changing complexity does not prove changing population spectral exponents or establish optimality for nonstationary returns.
- The 1978 agreement is guaranteed by construction and supplies no independent evidence of predictive superiority. It is excluded from the primary comparison.

## Reproduction and output scope

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m empirical_final.theory_guided_lambda --output outputs/theory_guided_lambda_20261010_reproduction
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_theory_guided_lambda.py tests/test_final_empirical.py tests/test_audit.py tests/test_bandwidth_tuning.py tests/test_three_empirical_experiments.py
```

The default bootstrap count is 5,000. A new output directory is required for a fresh full run. Selection writes annual checkpoints, then stores private coefficients under results/<output-name>/. `--phase costs` reruns stock reconstruction from verified coefficients; `--phase report` rebuilds summaries and figures after hash verification. Public cost checkpoints are refreshed after every stock year. Input hashes, original source hashes, software/thread details, the anchor and numerical checks are under audit/. Hashes publish no licensed observations or holdings. Public output is restricted to portfolio-level aggregates, figures, this report, tests, and a separate candidate subsection. The main manuscript remains unchanged. The candidate is intentionally at most one page and is not inserted automatically.

The exact package versions used for this run are in `audit/requirements_runtime.txt` (including the additional `threadpoolctl` runtime dependency). With the licensed inputs in place, the optional environment and one-page rendering commands are:

```sh
python3 -m pip install -r outputs/theory_guided_lambda_20261010/audit/requirements_runtime.txt
cd outputs/theory_guided_lambda_20261010/publication
pdflatex -interaction=nonstopmode -halt-on-error -jobname=one_page_algorithm_appendix one_page_preview.tex
```

The standalone preview uses the manuscript's letter-paper geometry, 12-point Times font and 1.5 line spacing. It is a candidate only. Final test counts and visual/page verification are in `audit/tests.json` and `audit/visual_review.json`.
