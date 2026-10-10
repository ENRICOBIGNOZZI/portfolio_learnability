# CODEX RESEARCH MISSION — Can Kernel Lengthscale Determine Optimal Portfolio Shrinkage?

## Overall objective

Work in the connected repository:

`ENRICOBIGNOZZI/portfolio_learnability`

Conduct a rigorous theoretical and real-data investigation into the relationship between:

- Kernel lengthscale \(\ell\).
- Eigenvalue decay of the managed-payoff operator.
- Spectral amplitude \(c(\ell)\).
- Effective portfolio complexity.
- Optimal ridge penalty \(\lambda\).
- Out-of-sample maximum-Sharpe portfolio performance.

**The central scientific question is whether changing the lengthscale can be translated into a predictable change in optimal shrinkage, using spectral information and at most one historical calibration.**

A successful result would allow the portfolio manager to select \(\lambda\) without repeating a large penalty-grid optimization each year.

Investigate this question thoroughly. Do not assume the proposed relationship is true. Distinguish exact mathematical results, asymptotic statements, finite-sample approximations, and empirical observations.

This is a **standalone research experiment**, not a request to rewrite the paper.

Do not modify the main manuscript. Do not introduce unnecessary new machine-learning architectures, feature-selection procedures, simulations, or unrelated research questions.

Reuse the current real JKP data and the existing fitted caches. Keep confidential stock-level observations private.

---

## 1. Inspect the existing code and reproduce the benchmarks

Before making changes, inspect:

`kernels.py`

`portfolio.py`

`bandwidth_tuning.py`

`empirical_final/theory_guided_lambda.py`

`empirical_final/three_experiments.py`

`empirical_final/compact_protocol.json`

`outputs/bandwidth_tuning/`

`outputs/theory_guided_lambda_20261010/`

The latest one-anchor experiment was published in commit:

`473e8ff951cd507cea3d9470a48aa33304b736ab`

The Gaussian experiment uses:

- 130 characteristics.
- 10,000 random Fourier features.
- Feature seed 0.
- Initial lengthscale \(\ell_0=4.392107784616423\).
- Annual expanding refits from 1978 through 2024.
- Chronological validation with the prior 60 formation months.
- OOS returns February 1978 through January 2025.

Its one-anchor rule is:

\[
\widehat\lambda_t
=
\arg\min_{\lambda>0}
\left\{
\lambda+\frac{\rho_0}{T_t}
\widehat{\mathcal C}_t(\lambda)
\right\},
\]

where \(\rho_0=4.344198081857661\times10^{-6}\), calibrated from the initial 1978 decision.

This method obtains approximately:

- Gross Sharpe 3.330 versus 3.277 for annual CV on the common 1979–2024 evaluation.
- Mean effective complexity 30.6 versus 126.8 for CV.
- Net Sharpe 2.676 versus 2.587 under the existing trading/borrowing-cost sensitivity.

The Sharpe differences are statistically inconclusive.

Importantly, the spectral rule produces a relatively stable complexity path of approximately 19–41, whereas annual CV varies substantially more.

The purpose of this new investigation is to understand whether **lengthscale adaptation and its spectral amplitude can provide genuine economic adaptation**, rather than merely another stable penalization rule.

Reproduce the relevant benchmarks and verify input hashes, dates, payoffs, feature normalization, and penalty conventions before proceeding.

Do not tune any rule against the full historical OOS outcome.

---

## 2. First solve the theoretical question

Consider a Matérn kernel with fixed smoothness \(\nu\), variable lengthscale \(\ell\), and its actual managed-payoff operator:

\[
\Sigma_\ell
=
\mathbb E[X_{t,\ell}\otimes X_{t,\ell}],
\]

where

\[
X_{t,\ell}
=
\frac1{N_t}
\sum_{i=1}^{N_t}
R^e_{i,t+1}K_\ell(\cdot,Z_{i,t}).
\]

We want to understand under what conditions:

\[
\boxed{
\mu_j(\ell)\sim c(\ell)j^{-b},
\qquad j\to\infty,
}
\]

with an exponent \(b\) independent of \(\ell\).

### Questions requiring mathematical answers

**A.** For a Matérn kernel with fixed smoothness, does changing \(\ell\) preserve the polynomial eigenvalue-decay exponent?

**B.** Is this property valid for the ordinary kernel integral operator, the actual managed-payoff operator, or both?

**C.** Which assumptions on the distribution of characteristics, stock-return covariance, factor structure, and operator comparison are required?

**D.** Can we derive an explicit theoretical dependence of \(c(\ell)\) on \(\ell\), perhaps using the high-frequency Matérn spectral density?

**E.** Does a two-sided spectral bound

\[
c_1(\ell)j^{-b}
\leq\mu_j(\ell)
\leq c_2(\ell)j^{-b}
\]

actually imply the existence of a unique asymptotic constant \(c(\ell)\)? If not, state the stronger assumptions required.

**F.** Does changing \(\ell\) also modify the source norm of the population-optimal portfolio policy and the stochastic constants appearing in the learning bound?

**G.** Are the associated Matérn RKHS function spaces identical as sets on a compact domain, with lengthscale-dependent norms? Explain the consequences for representation error and regularization.

**H.** Is there a simple counterexample showing that two economies with identical managed-payoff eigenvalues can nevertheless have different optimal ridge penalties because their expected-payoff signal or noise differs?

Do not confuse the kernel integral operator with the managed-payoff second-moment operator.

For standard Matérn asymptotics, a candidate exponent is

\[
b=1+\frac{2\nu}{D}.
\]

Here the empirical characteristic dimension is \(D=130\), not the six-dimensional illustrative factor economy. For \(\nu=3/2\), this nominal expression gives \(b=1+3/130\).

Determine whether the assumptions needed for this expression are actually satisfied. Do not replace \(D=130\) with a fitted effective dimension without a separate mathematical justification.

The mathematical result may be that \(b\) is preserved but the asymptotic constant is not identifiable from our finite history. That is an acceptable conclusion.

### Gaussian comparison

Repeat the conceptual analysis for Gaussian, but **do not impose a polynomial decay law on Gaussian**.

Determine whether changing Gaussian lengthscale modifies the spectral-decay speed, rather than only a multiplicative constant.

Use Gaussian as a useful negative control for the fixed-\(b\) hypothesis.

Deliver a concise theorem/proposition with proof, or an explicit counterexample if the proposed statement fails. Do not manufacture a proof by relying on numerical similarity.

---

## 3. Derive the precise lengthscale-to-lambda relationship

Our theoretical risk envelope is

\[
\mathcal R_T(\lambda,\ell)
\lesssim
A(\ell)\lambda
+
\frac{B(\ell)}{T}
\mathcal C_\ell(\lambda).
\]

Assume temporarily

\[
\mu_j(\ell)\sim c(\ell)j^{-b},\qquad b>1.
\]

Then verify

\[
\mathcal C_\ell(\lambda)
\sim
\kappa_b
\left(\frac{c(\ell)}{\lambda}\right)^{1/b},
\]

where

\[
\kappa_b=\frac{\pi}{b\sin(\pi/b)}.
\]

Derive the minimizer of the approximate envelope:

\[
\boxed{
\lambda_T^{\mathrm{bound}}(\ell)
=
\left[
\frac{B(\ell)\kappa_b}
{A(\ell)b}
\frac{c(\ell)^{1/b}}{T}
\right]^{b/(b+1)}.
}
\]

Verify the algebra and all necessary uniformity conditions.

**The main question is whether the dependence on \(\ell\) can be simplified.**

If

\[
\frac{B(\ell)}{A(\ell)}
\]

is constant across lengthscales, the theory predicts

\[
\boxed{
\frac{\lambda_T^{\mathrm{bound}}(\ell_2)}
{\lambda_T^{\mathrm{bound}}(\ell_1)}
=
\left[
\frac{c(\ell_2)}{c(\ell_1)}
\right]^{1/(b+1)}.
}
\]

If the ratio \(B(\ell)/A(\ell)\) changes, derive the corrected formula and quantify what additional information is necessary.

Also derive the implied scaling of optimal effective complexity.

**Very important:** distinguish an optimal tuning parameter for an upper bound from the actual risk-minimizing oracle \(\lambda\). The first relationship does not automatically establish the second.

Determine whether estimating the empirical spectrum alone could identify the oracle constant, and explain rigorously why or why not.

Try to derive meaningful bounds on the variation of \(A(\ell)\) and \(B(\ell)\) using the existing assumptions. If they cannot be determined sharply, document that limitation rather than inventing a formula.

Keep the theory concise: one central proposition and a short proof are preferable to pages of loosely connected calculations.

---

## 4. Test whether lengthscale changes only spectral amplitude

Use the existing managed-payoff caches for both:

- Matérn-3/2.
- Gaussian, as a comparison.

Lengthscale multipliers:

\[
\{0.25,\ 0.5,\ 1,\ 2,\ 4\}
\]

relative to the initial median-distance bandwidth.

Do not generate an entirely new feature bank independently for each lengthscale. Preserve the existing shared random frequencies/phases and documented scaling whenever possible.

For selected historical decision dates, including early, middle, and recent periods, compute the positive empirical eigenvalues of

\[
G_{t,\ell}G_{t,\ell}^{\top}/T_t.
\]

Use identical training observations when comparing lengthscales.

### Diagnostic 1 — Is the exponent stable?

Estimate local log–log slopes of the empirical spectrum over several **predeclared rank ranges**.

Do not choose the fitting window after seeing which one supports the desired hypothesis.

Compare estimates of \(b\) across lengthscales and years.

Report numerical rank, eigenvalue cutoffs, goodness of fit, and sensitivity to the fitting interval.

Do not interpret finite-sample slopes as proof of the population asymptotic exponent.

### Diagnostic 2 — Is the ratio of spectra approximately constant?

For each lengthscale compute

\[
r_{j,t}(\ell)
=
\frac{\widehat\mu_{j,t}(\ell)}
{\widehat\mu_{j,t}(\ell_0)}.
\]

If changing \(\ell\) only changes amplitude, this ratio should be approximately flat across the relevant ranks.

Create a graph showing the ratios across rank.

Compare leading directions, intermediate ranks, and the empirical tail.

Quantify deviations from constant-ratio behavior.

This may be the single most informative diagnostic.

### Diagnostic 3 — Estimate \(c(\ell)\)

If a stable polynomial range exists, estimate the amplitude from

\[
\log\widehat\mu_j(\ell)
=
\log c(\ell)-b\log j+\varepsilon_j.
\]

Compare:

1. Separate slope estimates for each lengthscale.
2. A common slope across lengthscales.
3. Robust amplitude estimates based on a fixed rank interval.
4. Alternative rank intervals to assess numerical stability.

Test whether the estimated amplitudes follow a simple scaling law in \(\ell\), including any law suggested by the theoretical Matérn spectral density.

If the power-law approximation is unstable, say so explicitly and do not use estimated \(c(\ell)\) as if it were identified.

### Diagnostic 4 — Finite versus infinite spectral behavior

Quantify how the result changes with:

- Training length \(T\).
- Positive-rank threshold.
- Rank interval.
- Number of random Fourier features, using already available smaller feature prefixes if practical.

Remember that the empirical operator has rank at most \(T\), even with 10,000 features.

Explain whether an estimated spectral tail reflects underlying kernel regularity or finite-sample effects.

---

## 5. Construct three theory-guided lambda rules

The goal is to test whether estimated spectral amplitude really predicts useful changes in the penalty.

### Rule A — Existing one-anchor full-spectrum rule

Reuse the already implemented algorithm:

\[
\widehat\lambda_{t,\ell}
=
\arg\min_{\lambda>0}
\left\{
\lambda+
\frac{\rho_0}{T_t}
\widehat{\mathcal C}_{t,\ell}(\lambda)
\right\}.
\]

For the fixed-kernel control, reproduce the existing result exactly.

For cross-lengthscale comparisons, treat the assumption that \(\rho_0\) is shared across lengthscales as an explicit hypothesis, not a mathematical fact.

Use the same units for managed payoffs, penalties, and spectral eigenvalues.

### Rule B — Spectral-amplitude transfer

This is the primary new hypothesis.

Select one baseline kernel lengthscale \(\ell_0\) and use only its original 1978 chronological-validation decision to obtain \(\lambda_0\).

For a fixed common exponent \(b\), construct

\[
\boxed{
\widehat\lambda_{t,\ell}^{\,\mathrm{transfer}}
=
\lambda_0
\left(\frac{T_0}{T_t}\right)^{b/(b+1)}
\left(
\frac{\widehat c_t(\ell)}
{\widehat c_0(\ell_0)}
\right)^{1/(b+1)}.
}
\]

This is the proposed transfer formula under the hypothesis that the bias-to-variance constant ratio remains unchanged.

Do not silently incorporate a new CV-selected constant for each \(\ell\).

Test alternative predeclared choices of \(b\), including the theoretically justified value if its assumptions apply, and a training-only empirical estimate where reliable.

When \(b\) cannot be estimated defensibly, do not force this method to produce an apparently precise answer.

If the estimated constant ratio varies by lengthscale, derive and test the adjusted prediction separately. It must be clearly labelled as a different method requiring additional information.

No OOS return may influence the fitted exponent, amplitude, or anchor.

### Rule C — Fixed-lambda control

Reuse the initial selected \(\lambda_0\) in every subsequent annual refit for the baseline fixed lengthscale.

This is indispensable.

The previous one-anchor spectral experiment produced almost constant penalization. We need to know whether updating the spectrum improves anything beyond leaving \(\lambda\) unchanged.

Do not omit this control.

For any cross-lengthscale fixed-penalty comparison, clearly specify the convention used to make penalties comparable.

---

## 6. How should we choose lengthscale?

**Do not choose lengthscale by minimizing spectral complexity alone.**

That objective could favor an overly simple representation even if it loses valuable investment opportunities.

We want to compare two economically meaningful approaches.

### Approach 1 — Lengthscale selected by prior portfolio performance

For each candidate lengthscale, compute an entire past-only sequence of portfolio policies using its theory-guided penalty.

Track their genuinely out-of-sample realized payoffs by decision year.

Select the next year's lengthscale from each candidate's trailing historical OOS performance, using the same portfolio objective as the existing benchmark.

The anchor calibration uses the 1973–1977 validation period. Do not treat those same returns as fresh independent evidence for choosing lengthscale.

Use a predetermined trailing 60-month candidate-performance window after the anchor and start the principal adaptive-lengthscale comparison only when that history is genuinely available, approximately decision year 1984.

Do not compute hypothetical validation results for an earlier date using a calibration that only became available later.

The final model for year \(t\) is fitted only with observations known by its decision cutoff.

No return from year \(t\)'s OOS evaluation can select its lengthscale.

A shorter trailing window may be used as an explicitly secondary robustness test.

### Approach 2 — Standard joint chronological cross-validation

Reuse the existing benchmark from `bandwidth_tuning.py`:

- Five lengthscales.
- Ridge penalty selected through chronological validation.
- Same stock universe, dates, return units, kernel features, and costs.

This is the strongest existing practical baseline.

The comparison must be fair: selecting five candidates for the theory-guided method is still hyperparameter selection, even if its ridge parameter is obtained from a formula.

Do not claim a completely validation-free algorithm if lengthscale is chosen from past portfolio returns.

---

## 7. Mandatory benchmark comparison

Compare the following on identical OOS dates:

| Method | Lengthscale | Lambda |
|---|---|---|
| Fixed lambda | Initial fixed value | Fixed from 1978 |
| Annual CV | Fixed | Annual CV |
| One-anchor spectral | Fixed | Full empirical spectrum |
| Joint CV | Chronological CV | Chronological CV |
| Spectral transfer | Chronological past-only selection | Estimated \(c(\ell)\) and \(b\) |
| Full-spectrum adaptive | Chronological past-only selection | Full spectrum with frozen \(\rho_0\) |

Do this for Matérn-3/2 as the primary mathematical investigation.

Use Gaussian as a secondary negative-control comparison. Do not fit Gaussian spectra to a polynomial merely for symmetry with Matérn.

For each method report:

- Monthly OOS excess returns.
- Annualized maximum-Sharpe performance.
- Annual mean and volatility.
- Average and annual effective complexity.
- Annual \(\lambda\) and selected lengthscale.
- Gross exposure and turnover.
- Net Sharpe after the existing illustrative trading and borrowing costs.
- Maximum drawdown.
- Paired uncertainty relative to the main CV benchmark and fixed-lambda control.

Use identical common-month samples, especially for the adaptive-lengthscale experiments whose evaluation can only begin after sufficient historical candidate performance is available.

Reconstruct actual stock weights for the leading methods; do not infer transaction costs from a different portfolio's turnover.

Compute Sharpe from concatenated monthly returns, not averages of annual Sharpe ratios.

Include the existing block-bootstrap sensitivity procedures. Treat the intervals as conditional retrospective evidence, not full-pipeline inference under arbitrary nonstationarity.

Report negative and statistically inconclusive results without adjustment to make them attractive.

---

## 8. Does spectral amplitude actually explain the best lambda?

This is the most important scientific diagnostic.

For each year and candidate lengthscale, compare:

1. The ridge penalty predicted by spectral-amplitude transfer.
2. The full-spectrum theoretical penalty.
3. The penalty selected by chronological validation.
4. The ex-post OOS best penalty, **strictly as a hindsight diagnostic**.

The ex-post best penalty must never enter an actual portfolio decision or calibration.

Report the prediction errors on a logarithmic penalty scale and their consequences for OOS Sharpe and risk.

Do not equate an ex-post maximizer over only 12 monthly returns with the population oracle.

Determine whether:

- The effect of changing lengthscale on the penalty is predicted by \(c(\ell)^{1/(b+1)}\).
- The full empirical spectrum predicts it better than the asymptotic power-law approximation.
- Neither approach does meaningfully better than a constant penalty.
- The main obstacle is unstable estimation of \(b\), unstable \(c(\ell)\), variation in bias/noise constants, or the use of stale observations.

These alternatives must be assessed rather than assuming one conclusion.

Specifically investigate whether the improvement from lengthscale tuning is due mainly to a better representation, better shrinkage, or both.

Use controlled cross-comparisons to separate those effects as far as the observed protocol permits. Do not claim a causal decomposition unless it is identified.

---

## 9. Can we estimate the remaining constants instead of calibrating them?

Study whether the theoretical coefficients

\[
A(\ell),\qquad B(\ell)
\]

or at least their ratio

\[
\rho(\ell)=B(\ell)/A(\ell)
\]

can be estimated from historical managed payoffs.

The relevant objects include:

- The RKHS norm of the population policy.
- The covariance of the optimality score.
- The regularized inverse of the managed-payoff operator.
- Interactions between empirical operator estimation and score estimation.

Derive a candidate estimator only if its statistical meaning and assumptions are clear.

Do not treat the norm of a noisy unregularized estimate as the true population RKHS norm.

Do not estimate unknown constants using the future test sample.

Distinguish estimating a conservative bound constant from estimating the exact oracle-optimal penalty.

If the ratio cannot be identified robustly, supply a clear impossibility argument or counterexample.

The objective is to determine whether the method can become **fully data-driven without repeated validation**, not to force such a formula to exist.

A rigorous negative result is acceptable.

---

## 10. Required figures

Create a small set of publication-quality figures, saved as vector PDF and high-resolution PNG.

**Figure 1 — Spectral shape across lengthscales**

A comparison of Matérn empirical eigenvalue curves for all five lengthscales, on common axes and a common historical sample.

Include either ratio-to-baseline curves or an adjacent panel making it visually obvious whether \(\ell\) changes spectral amplitude only or also changes shape.

**Figure 2 — Spectral amplitude and exponent**

Estimated \(b(\ell)\), estimated \(c(\ell)\), theoretical scaling where applicable, and uncertainty/sensitivity across rank-fitting intervals.

**Figure 3 — Predicted versus selected lambda**

Compare the annual penalty paths of CV, fixed lambda, full-spectrum spectral shrinkage, and amplitude-transfer shrinkage.

Display effective complexity in a separate panel or aligned figure.

**Figure 4 — Financial results**

Gross and net OOS performance of the competing strategies, including the strongest chronological benchmark.

Include a compact uncertainty summary.

Keep detailed robustness figures in the appendix.

Do not produce dozens of redundant charts.

---

## 11. Mathematical and numerical verification

Implement rigorous tests covering:

- Recovery of the original 1978 anchors for Gaussian and Matérn.
- Eigenvalue computation and PSD tolerance.
- Normalization of the uncentered managed-payoff operator.
- Exact effective-complexity calculation.
- The relationship between primal and dual ridge predictions.
- Monotonicity and KKT conditions for the full-spectrum proxy.
- Algebraic scaling of the asymptotic penalty with \(c(\ell)\) and \(T\).
- Correct treatment of fixed versus variable \(b\).
- Invariance of results to eigenvector signs.
- Lack of future-return leakage.
- Fully chronological lengthscale selection.
- Reproduction of existing baseline monthly payoffs.
- Actual stock-level cost accounting.
- Output tables reconstructing plotted values.
- Numerical stability when the estimated spectral exponent approaches one.

Do not artificially clip an estimated exponent to make the desired theorem applicable without reporting the failure.

Keep the original baseline and the new experimental outputs separate.

Use numerical sanity checks on mathematically controlled matrices where useful, but do not substitute simulated portfolio returns for the requested empirical investigation.

---

## 12. Outputs and delivery

Create a new isolated versioned directory, for example:

`outputs/lengthscale_spectral_constant_20261010/`

Deliver:

**A. Mathematical note**

A concise standalone PDF, preferably 5–8 pages, explaining:

1. What can actually be proved about lengthscale and eigenvalue decay.
2. When \(c(\ell)\) exists and can be estimated.
3. The exact relationship between spectral amplitude and the minimizer of the theoretical risk envelope.
4. Why this relationship may fail for the actual oracle penalty.
5. How the empirical algorithm is constructed and calibrated.

Use the notation of *The Law of Portfolio Learnability*.

**B. Empirical report**

A compact PDF with the figures, tables, main economic findings, statistical uncertainty, and failures.

**C. Full reproducible code**

Use existing modules wherever possible, rather than duplicating kernels, managed-payoff construction, cross-validation or cost-accounting utilities.

Run the complete real-data experiments. Reuse and validate existing caches to avoid unnecessary expensive recomputation.

**D. Audit and data tables**

Export the estimated slopes, amplitudes, annual lambda choices, complexity, selected lengthscales, OOS returns and paired comparisons.

Public outputs must not include confidential licensed stock-level information.

**E. Final decision memo**

Give explicit answers to these questions:

- Is \(b\) empirically stable across Matérn lengthscales?
- Does the managed-payoff spectrum change mainly through \(c(\ell)\), or does its shape materially change?
- Is \(c(\ell)\) reliably estimable from the historical sample?
- Does the predicted scaling of \(\lambda\) match the observed changes in useful regularization?
- Does lengthscale-adaptive theory-guided shrinkage beat a fixed lambda?
- Does it beat annual cross-validation?
- Does it beat jointly tuned lengthscale and lambda?
- Does it improve net Sharpe rather than only reduce volatility or exposure?
- Can \(B(\ell)/A(\ell)\) reasonably be treated as constant?
- Is there a mathematically defensible one-page contribution for the current paper?

Give a clear **GO / NO-GO** recommendation for inclusion in the Journal of Finance manuscript, based on demonstrated value rather than cosmetic novelty.

Do not change the manuscript unless the evidence justifies inclusion and the result fits within one page of main-text exposition. Even then, prepare a candidate LaTeX insert separately and await approval.

Verify the final reports, test results, figures, and remote Git commit. Commit and push only permitted changes, preserving unrelated work.

---

## Final scientific standard

The investigation is successful if it establishes which of the following statements is true:

**Case 1 — Strong positive result:** Matérn lengthscale primarily changes spectral amplitude in the relevant regime, the theoretically predicted penalty scaling is empirically reliable, and the resulting rule is financially competitive.

**Case 2 — Qualified positive result:** The amplitude approximation works only over selected ranges, but the full-spectrum rule performs well and has a clear practical interpretation.

**Case 3 — Negative result:** Lengthscale changes the relevant spectral geometry or the signal/noise constants too substantially for amplitude transfer to recover useful penalties.

All three outcomes are scientifically informative.

**Do not force Case 1. The task is to discover whether the proposed spectral relationship is true, when it is true, and whether it is useful for real portfolio decisions.**