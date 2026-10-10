# Lengthscale, managed spectral shape and portfolio shrinkage

Recommendation: **NO-GO**. The amplitude identification gate fails on the initial pretest history. Amplitude transfer is a finite-range falsification exercise; it is not an identified population-amplitude estimator. All explicit scientific answers are in DECISION_MEMO.md.

## Chronological protocol and controls

The requested mission and diagnostic protocol are frozen in audit/research_mission.md and audit/protocol.json. Starting commit is `d5744e045e1803702f051cac4f88aa1f46e80bde`. Both existing 5 × 744 × 10,000 float64 managed-payoff caches are reused, with 130 characteristics, seed 0 and lengthscale 4.392107784616423 times [0.25,0.5,1,2,4]. The same Gaussian/Matérn random draws and phases are scaled across lengths; no new independent banks, stock preprocessing, architectures or simulated research returns are introduced. The inherited JKP complete-payoff stock sample is a retrospective sensitivity, not a certified vintage point-in-time universe.

Every original annual CV and joint CV selection/payoff is reproduced. Original fixed-CV maximum raw errors are 9.38e-11 (Matérn) and 8.79e-11 (Gaussian); joint errors are 1.7e-10/1.33e-10. The prior Gaussian spectral path agrees to 2.22e-16. Cache file hashes, bank metadata, source checksums, original grids and anchor cutoff dates are recorded. Raw eigenvalues are those of g g'/T, without centering or display scaling; the common frozen kappa is 0.03740000227285338.

The 1978 base-lengthscale chronological CV decisions calibrate each kernel's one constant once. Gaussian lambda/rho are 2.515462907417361e-07/4.3441980818576606e-06; Matérn are 1.5594482461490982e-07/1.5896429646518706e-06. Sharing rho across lengthscales is a hypothesis. The original 120-point positive bounds remain frozen separately per representation. The fixed-lambda primary control always uses its baseline raw anchor. Cross-length fixed penalties use the same raw units and are constrained to each representation's original interval; clipping is explicit.

For transfer, the fixed empirical slopes are 1.421083230 (Matérn) and 1.623402358 (Gaussian), each from its own 1978 baseline ranks 11-60. The robust amplitude statistic is median(log(mu_j)+b log(j)) on those same ranks. The Gaussian transfer was explicitly requested after the original Matérn-primary results and is a documented exploratory amendment; no polynomial population law is imposed on Gaussian. Nominal b=133/130 is a Matérn-only unverified sensitivity; b=1.5,2,3, alternate ranges and an OLS mean-intercept variant are finite-range sensitivities for both. No invalid b is clipped. First-history identification gates and their failures are retained; the term c_proxy is used throughout. The adjusted transfer is a separate additional-information experiment with five 1978 CV FOC calibrations; boundary calibrations identify interval endpoints only.

Candidate policies are genuinely OOS from their 1978 fit onward. Primary adaptive selection starts in 1984 using the last 60 realized candidate returns from decisions 1979-1983, minimum mean (1-raw_return)^2 and the original median-length tie rule. Validation 1973-1977 is not recycled, and the anchored decision 1978 is excluded from candidate selection. Every selection cutoff and first/last candidate return is exported. Thirty-six-month windows are secondary, on the same 1984+ evaluation; none select a final rule from full-sample outcomes.

## Primary common-month economic evidence

| Kernel / method | Gross SR | Exact net SR | Mean C | Annual net trades |
|---|---:|---:|---:|---:|
| matern32 / Fixed lambda | 3.256 | 2.626 | 38.68 | 15.92 |
| matern32 / Annual CV | 3.285 | 2.589 | 196.73 | 23.25 |
| matern32 / One-anchor spectrum | 3.318 | 2.654 | 50.88 | 17.25 |
| matern32 / Joint CV | 3.392 | 2.597 | 298.94 | 25.74 |
| matern32 / Amplitude transfer (heuristic) | 3.289 | 2.651 | 63.59 | 16.76 |
| matern32 / Adaptive full spectrum | 3.244 | 2.624 | 54.76 | 16.26 |
| gaussian / Fixed lambda | 3.129 | 2.540 | 24.35 | 14.53 |
| gaussian / Annual CV | 3.114 | 2.461 | 135.21 | 21.97 |
| gaussian / One-anchor spectrum | 3.191 | 2.563 | 31.80 | 15.99 |
| gaussian / Joint CV | 3.480 | 2.729 | 251.64 | 23.49 |
| gaussian / Amplitude transfer (heuristic) | 3.244 | 2.618 | 43.13 | 15.23 |
| gaussian / Adaptive full spectrum | 3.219 | 2.601 | 37.14 | 15.14 |

This sample has 492 months, February 1984-January 2025, corresponding to decisions 1984-2024. Sharpe uses concatenated monthly excess returns and ddof=1, not averaged annual Sharpes. The full fixed-policy context (564 months) and post-anchor fixed-policy context (552 months) are reported separately. All primary common-window net accounts start from cash at the 1984 first formation close and charge entry; terminal holdings are marked without forced liquidation.

Every leading method's signed stock weights are reconstructed with shared original frequencies/phases and verified against its managed payoff. The old CV weights and full-history net accounts reconcile independently. Exact net uses core.accounting_step: c=0.0025*||(1-c)w-d||_1, borrowing charge 0.003/12 times short notional on post-cost NAV, and the identical cash financing rate. The target-turnover scenario instead subtracts 0.0025*||w_t-w_(t-1)||_1 plus 0.003/12*short from gross excess returns. They are separate illustrative scenarios, without observed impact, lending spreads or executable liquidity claims. No new cost is inferred from another portfolio's turnover. Licensed rows, IDs and holdings remain private.

Paired circular block-bootstrap comparisons use 5,000 shared draws, seed 20261010, 12-month blocks and 6/24-month sensitivities. References are joint CV, annual CV and fixed lambda. Intervals condition on fitted paths; they are not full-pipeline inference and have no multiple-comparison adjustment. Full block lengths, subperiods and failed/inconclusive contrasts are in paired_comparisons.csv.

## Shape, amplitude, constants and penalty diagnostics

Rank intervals [1,10], [11,40], [41,100], [11,60], [61,120], [121,180] were declared before calculations. Diagnostics at 1978, 2000 and 2024 use common observations, expanding history and available 120/180/360/720-month windows, relative rank thresholds 1e-10/1e-12/1e-14, and existing feature prefixes 1,000/5,000/10,000. Prefix managed vectors multiply sqrt(10000/P); failing to do that would change the penalty units. OLS slope standard errors are descriptive fits, not confidence intervals for a population spectral exponent. Figure-2 whiskers show sensitivity across rank intervals.

The ratio curves compare leading, middle and finite tail ranks. Flatness statistics and sign-invariant temporal subspace overlap are exported; sorted-rank ratios alone do not track the same economic eigenportfolio. Separate finite-range slopes, a common slope pooled across lengthscales, frozen historical slopes and robust intercepts are compared for both kernels. Gaussian fits are exploratory finite-range diagnostics, not population power-law estimates. The theoretical Matérn amplitude law ell^(-3) requires the additional ordinary-operator/Weyl and economic-covariance assumptions in the mathematical note; those assumptions are not established by the cache.

The penalty diagnostic retains predicted transfer, full-spectrum and CV penalties, original-grid ex-post loss minima and Sharpe maxima, log prediction errors, OOS loss and annual risk. A hindsight grid peak over 12 returns is not a population oracle and never feeds calibration or selection. FOC-implied rho values and regularized historical policy norms/score-covariance traces describe why eigenvalues alone do not identify bias/noise constants; they are not estimators of the population RKHS norm or structural A/B.

The representation ablation holds raw penalties fixed while changing joint-selected lengthscale, and holds baseline lengthscale fixed while applying joint-selected penalties. Together with fixed-length transfer and spectral/fixed-lambda controls, this separates observed representation and shrinkage effects as far as the protocol permits. Selection interactions prevent a causal decomposition. Secondary transfer choices, 36-month paths, all candidate returns and the complete 120-grid OOS payoff paths are exported, without selecting a favorable sensitivity.

Quantitative cross-length penalty-effect errors are in lengthscale_effect_summary.csv, exact descriptive 2x2 representation/penalty contrasts in representation_penalty_contrasts.csv, and shorter-selection/stale-history limitations in selection_history_sensitivity.csv and DECISION_MEMO.md. Finite history and normalized feature-prefix slope changes are summarized in finite_sample_sensitivity.csv. Crossed raw penalties are not retuned and may fall outside the receiving grid; these secondary controls never expand an actual penalty search.

Alternate b/rank ranges, the five-anchor adjusted transfer, 36-month selection and crossed-penalty policies are gross-only secondary diagnostics. Actual costs are reconstructed for every method in the mandatory primary comparison and the fixed-length transfer control. No secondary net result is inferred from another portfolio.

Primary-method boundary counts: {('gaussian', 'adaptive_spectral'): 0, ('gaussian', 'annual_cv'): 0, ('gaussian', 'fixed_lambda'): 0, ('gaussian', 'joint_cv'): 8, ('gaussian', 'one_anchor'): 0, ('gaussian', 'spectral_transfer'): 0, ('matern32', 'adaptive_spectral'): 0, ('matern32', 'annual_cv'): 1, ('matern32', 'fixed_lambda'): 0, ('matern32', 'joint_cv'): 14, ('matern32', 'one_anchor'): 0, ('matern32', 'spectral_transfer'): 0}. Gaussian has finite-range proxy b/c statistics but no identified population polynomial b or amplitude. Missing numeric cells in annual diagnostics mean a quantity is inapplicable (e.g. a KKT residual for CV), not a skipped payoff; every actual monthly payoff/account is finite and verified.

## Mathematical limits and deliverables

The mathematical note supplies one central conditional proposition and proof, the corrected lambda and complexity formulas, an exact same-spectrum/different-risk-oracle counterexample, the difference between two-sided bounds and an asymptotic constant, conservative norm/source/tail comparisons, and a regularized-score estimation target with its assumptions. It does not conflate ordinary and managed operators. It does not infer D=6, polynomial Gaussian decay, an exact Sharpe-regret identity, structural rho or data-dependent oracle optimality from the paper's fixed-lambda upper bound.

Four main vector-PDF/high-resolution-PNG figures, the Gaussian shape negative control and the requested Gaussian transfer/financial appendix figure are generated from exported tables. Mathematical and empirical PDF reports, a separate one-page candidate, public audit and this decision memo remain isolated from the manuscript. The final recommendation uses demonstrated economics and identification, not smoothness of complexity paths.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m empirical_final.lengthscale_spectral --output outputs/lengthscale_spectral_constant_20261010_reproduction
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_lengthscale_spectral.py tests/test_theory_guided_lambda.py tests/test_bandwidth_tuning.py tests/test_final_empirical.py
```

With licensed inputs present, --phase estimate/costs/summary/publication provides explicit stages and year-by-year checkpoints. New output paths inherit the committed frozen diagnostic protocol; source/input checksums are revalidated. Private coefficients are stored under results/<output-name>/. Runtime package versions and final tests/visual/completion audits are in audit/. Only the new code/tests, public aggregates, figures, reports and candidate are committed. The main manuscript and unrelated work are preserved.
