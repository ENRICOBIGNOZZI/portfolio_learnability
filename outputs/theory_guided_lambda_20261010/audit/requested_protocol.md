# CODEX EXECUTION BRIEF — One-Constant Theory-Guided Spectral Shrinkage

**Repository:** `ENRICOBIGNOZZI/portfolio_learnability` (the current working branch).
**Research task:** Test a compact, direct consequence of the paper's effective-complexity bound: select the annual Gaussian portfolio ridge penalty from its *past-only empirical spectrum*, with **one** constant calibrated only once, instead of reselecting the penalty each year by five-year cross-validation.

## Deliverable / completion condition

Implement and actually run the experiment on the **real JKP managed-payoff cache** already used by the empirical paper. Produce verified annual/monthly comparisons, cost-aware performance where the licensed stock panel is available, figures, tests and a 1-page candidate appendix subsection. Do **not** change the main paper or claim a theorem that is not proved. If execution is blocked by missing private data, clearly report the missing input and provide a smoke-test implementation; do not manufacture empirical results.

## 0. Read and freeze the baseline

Inspect the current branch, then inspect, at minimum:

- `portfolio.py`: `annual_splits`, `complexity_grid`, `dual_path`, `fit_windows`, `sharpe`;
- `empirical_final/run.py`: `clean_manifest`, `load_managed('gaussian', ...)`, source hashes, raw-versus-scaled payoffs;
- `empirical_final/three_experiments.py`: existing 47-year expanding-window evaluation;
- `bandwidth_tuning.py`: fixed-bandwidth Gaussian baseline and canonical penalty grid;
- `outputs/empirical_three_experiments_20261009/tables/e1_selected_monthly.csv` and `e1_annual_paths.csv`;
- existing tests and data-timing audit.

Record the starting commit, input hashes, scale convention, all formation and return dates and the exact canonical baseline. Reuse `portfolio.py` matrix functions rather than duplicating the estimator. Do not overwrite the existing runs, empirical figures, input files, or manuscript. Use fixed Gaussian bandwidth and **the same 10,000 fixed random Fourier features/seed** as the baseline. No neural nets, factor selection, stock preprocessing changes, simulations, estimation of b, or kernel/bandwidth optimization in this focused experiment.

The current repository uses a **120-candidate positive ridge grid** fixed from the 1963–1972 training block; chronological validation uses 1973–1977 for the first decision. Yearly choices use training through `year-6`, last five formation years as validation, then refit on all available past data. Current decision-year 1978 means returns February 1978 through January 1979. Confirm the actual timing from code. The first selected raw penalty in the previous Gaussian snapshot was approximately `2.515462907417361e-07`, with selected C approximately `19.49434`; use these numbers **only as reconciliation checks**, not as hardcoded targets if the canonical run differs.

## 1. Mathematically specified algorithm

For year `y`, let `H_y` be the indices of managed payoff observations available at the January-close decision, after the original refit protocol (training plus validation). Let `T_y = len(H_y)` and `g_y = managed[H_y]` (rows are monthly managed vectors). Let

```
Gram_y = g_y @ g_y.T
mu_y   = eigenvalues(Gram_y / T_y), nonnegative (with PSD numerical tolerance)
C_y(lam) = sum_j mu_y[j] / (mu_y[j] + lam)
S_y(lam) = sum_j mu_y[j] / (mu_y[j] + lam)**2
```

Use the **raw managed-payoff units**, exactly the same normalization for `mu_y`, `lambda`, and the ridge solve. Do not apply portfolio display scale `kappa` when computing these quantities. Do not infer b from Gaussian eigenvalues.

The bound has the schematic form `A*lambda + B*C(lambda)/T`. Define `rho = B/A > 0`, an unknown ratio. The proposed **plug-in selector**, not a proven exact oracle, is

```
lam_spec_y = argmin_{lam in [lam_min,lam_max]} [lam + (rho / T_y) * C_y(lam)]
```

`lam_min`, `lam_max` must be the *same positive penalty search limits* available to the original yearly CV benchmark, fixed before OOS analysis. In the primary analysis use the min and max of the original frozen 120-point baseline grid, without grid-hunting on future returns. **Compute the continuous minimizer** with a monotone scalar root solver and clip only when the minimum is at a search boundary. The first-order equation for an interior optimum is

```
1 = (rho / T_y) * S_y(lam)
```

`F_y(lam)` is strictly convex for positive eigenvalues. Implement boundary cases via derivative signs: if `F'(lam_min)>=0`, return `lam_min`; if `F'(lam_max)<=0`, return `lam_max`; else solve `F'(lam)=0` with robust bisection or `scipy.optimize.brentq` in the fixed interval. Return boundary flags and derivative residuals. Compare a discrete-grid evaluation in a small appendix robustness check to ensure that a continuous-versus-grid distinction does not drive the outcome.

**Interpretation:** the spectrum changes annually. `rho` is a *single historically calibrated tuning ratio*; it is not a consistently estimated structural constant of the economy, and it does not make the Gaussian spectrum polynomial. The objective is the plug-in upper-bound proxy, not the true Sharpe oracle.

## 2. One-time calibration: anchor to the first valid CV decision

For **1978 only**, use the *original* chronological CV to select the anchor penalty `lam0` (1963–1972 train, 1973–1977 validation). Refit on the complete first pretest history as in the baseline; compute its refit eigenvalues `mu0`, `T0=180`, and set the constant by the exact first-order equation:

```
S0   = sum(mu0 / (mu0 + lam0)**2)
rho0 = T0 / S0
```

Freeze `rho0` for **all future decision years 1979–2024**. This algebraically makes the spectral root equal to the 1978 anchor (provided the anchor is interior and both policies use the same refit spectrum and bounds). Verify agreement numerically; this first-year equality is *by construction*, not evidence of predictive superiority.

Do **not** recalibrate `rho` annually, use future 1978+ realized returns to select it, pick it from 2020–2024 results, or fit it to the OOS Sharpe curve. If `lam0` is on a grid boundary, state explicitly that the anchor does not point-identify `rho` and use a documented one-sided interval or report an identification limitation; do not silently choose a preferred interior value. The existing first-year snapshot is interior, but verify on current data.

This one-time anchor is the **primary experiment**. Optionally describe, but do not prioritize, fully nested rolling recalibration (it would be a different algorithm and higher-dimensional study).

## 3. Out-of-sample experiment, no leakage

For each decision year 1978–2024:

1. Construct exactly the same `H_y` and Gram matrix as baseline after the annual refit.
2. Compute past-only eigenvalues and `lam_spec_y` with frozen `rho0`.
3. Solve the ridge policy using `dual_path(Gram_y, [lam_spec_y])` / existing trained coefficients; apply it to the same 12 future monthly managed payoffs as the baseline, with the same kernel and display scale.
4. Independently reproduce the **annual rolling-CV baseline** using `fit_windows`, whose `lambda_cv_y` uses the prior five-year holdout for each year. Do not replace the baseline with a later, different snapshot.
5. Record `year, T, lambda_spectral, lambda_cv, C_spectral, C_cv, rho0, active_rank, boundary_flags, numerical_residuals, first/last observation dates` and every monthly OOS return for **both** methods.
6. Estimate annualization only from the concatenated common realized monthly returns; do not average annual Sharpes. Compute out-of-sample Sharpe, means, volatility, maximum drawdown, and baseline-versus-spectral paired differences. Display full 1978–2024 as context and use **1979–2024 as the primary comparison**, excluding the trivially anchored decision year.
7. If the licensed stock-level data and existing feature caches are available, reconstruct both policies' signed stock weights, compute gross/net exposure, turnover, and *identical* 25-bp target-turnover and 30-bp/year short-notional illustrative scenarios, reusing the repository's exact cost definitions. Never approximate new transaction costs from the old baseline's turnover. A more realistic trading-cost adjustment is optional only after the original convention is reproduced. If unavailable, give gross-only results, label the missing input and do not claim net performance.
8. Compute paired circular block-bootstrap uncertainty for Sharpe differences on identical OOS months (12-month blocks, with 6 and 24 months as sensitivities), explicitly conditional on fitted paths, not whole-pipeline sampling inference. Report the sign and uncertainty, even if spectral performs worse.

A falsifiable question: **Does the single-anchor spectral rule match or improve the risk-adjusted, net-of-cost OOS results of annual CV while producing a coherent adaptively changing complexity path?** Do not assume success, or treat a correlation between lambda and C as evidence of economic value.

## 4. Tests and guardrails

Implement `tests/test_theory_guided_lambda.py` covering:

- Algebraic check of `C_y(lam)` against existing `dual_path(...).complexity`, and Gram versus feature-primal eigenspectra on a small deterministic example.
- Nonnegative eigenvalues after controlled numerical PSD clipping; no silent clipping of meaningfully negative Gram eigenvalues.
- `C_y(lam)` decreasing, `F'_y(lam)` increasing, positive-root uniqueness or correct boundary selection; `rho` increasing should not increase chosen active complexity under fixed `mu,T`.
- A deterministic known-spectrum case with an analytically verified root.
- 1978 anchor: `lam_spec_1978` matches `lam_cv_1978` within numerical tolerance, and no 1978 test return enters calibration.
- Cross-run baseline: selected CV monthly payoffs reproduce the existing fixed-bandwidth Gaussian snapshot to stored tolerance, or record and explain an input-version mismatch before comparing.
- `lambda_spectral` determined only by past-year managed payoffs and frozen `rho0` (future test-payoff perturbation test). Exact formation/return-date timing and 47 decision years / 564 OOS months.
- Portfolio aggregation and scaling used identically for baseline and spectral; gross/net cost accounting tested against actual weights where available.
- No NaNs or dependence on hidden default GPU/non-deterministic randomness; frozen manifest/hash audit.

Numerical stability: `mu_y` are eigenvalues of `Gram_y/T_y`; use stable symmetric eigensolvers, documented handling of zero eigenvalues, and scalar root finding (no repeated dense inversions along a lambda grid). If lambda hits bounds for many years, report this as a limitation, not a victory. Do not expand the grid based on realized test outcomes.

## 5. Outputs and delivery

Create isolated, versioned outputs, e.g. `outputs/theory_guided_lambda_20261010/`:

- `audit/config.json`, `audit/input_hashes.json`, `audit/verification.json`, `audit/anchor.json` (including `lam0`, `T0`, `rho0`, `S0` and exact cutoff dates).
- `tables/annual_selection.csv`, `tables/monthly_oos.csv`, `tables/summary.csv`, `tables/paired_bootstrap.csv`, optional `tables/costs.csv`.
- `figures/complexity_paths.pdf`+`.png`: selected C by year for CV versus one-time spectral rule; optional companion lambda chart with correctly scaled log axis.
- `figures/wealth_comparison.pdf`+`.png`: comparable gross (and available net) cumulative equity paths.
- `REPORT.md`: statement of what is proven algebraically, what is only a statistical proposal, chronological protocol, performance estimates with uncertainty, annual differences, cost effects, clipping frequency, failure modes and reproducibility.
- `publication/one_page_algorithm_appendix.tex`: **maximum one manuscript page**, including the two equations, one-time calibration, and main empirical verdict; do not insert it into the manuscript automatically. If the method adds no compelling value, recommend leaving it out of the paper.

Create one entry point `python3 -m empirical_final.theory_guided_lambda` (with `--help` and configurable `--output`), reuse existing data loading/code, and document an exact reproduction command and test command. Provide status messages and checkpoints for a long run, and conduct a lightweight deterministic test before expensive full execution.

Commit and push **only** the new module, relevant tests, public aggregates, figures, report and candidate appendix file; do not publish licensed stock data, holdings, data manifests containing secrets or large private model arrays, and do not modify unrelated files. Print the verified remote commit hash and exact links to the outputs.

## 6. Mathematical caveats (must appear in the report)

- The starting `A*lambda + B*C(lambda)/T` is a **theoretical upper envelope / fixed-lambda bound** under the paper's assumptions, **not an identity for exact Sharpe regret**. Calibrating `rho=B/A` from the first selected CV lambda does not estimate A or B independently, and rho need not remain structural under nonstationarity.
- Using `C_hat_y(lambda)` in a data-dependent argmin does **not automatically inherit** a nonasymptotic theorem proved only for fixed deterministic lambda. A uniform-in-lambda concentration result would be needed for that claim.
- Gaussian managed spectra need not be polynomial. The main experiment uses the *full empirical spectrum* rather than estimating b. If exploring Matérn later, keep it an explicitly separate robustness exercise; do not infer b by unverified extrapolation of the finite sample tail.
- A numerical minimum on a constrained penalty interval need not be an unconstrained oracle optimum. A finite-rank sample spectrum can make the rule act differently from the population operator near zero regularization.
- Do not claim that a changing selected complexity proves changing population spectral exponents or that the algorithm is optimal for nonstationary returns.
- A good-looking 1978 match is guaranteed by construction and is not to be counted as independent evidence.

**Stop after this focused experiment.** No new neural-network experiment, no nine-panel figure, no extensive manuscript rewrite. Preserve the compactness of the current Journal of Finance submission.
