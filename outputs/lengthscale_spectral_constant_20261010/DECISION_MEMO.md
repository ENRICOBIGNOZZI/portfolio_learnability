# Decision memo: NO-GO

The evidence does not support a structural amplitude-only lengthscale-to-shrinkage rule. Recommend excluding this extension from the current Journal of Finance manuscript.

**Is b empirically stable across Matérn lengthscales?**

No over the predeclared finite ranges. On ranks 11-60 in the first refit alone, slopes span 0.667 to 1.716. These are finite-range slopes, not population exponents. Fixed-smoothness norm equivalence preserves an assumed infinite-operator polynomial order; it does not justify a common finite-sample slope or identify b=133/130.

**Does the managed spectrum mainly change amplitude or shape?**

It changes finite-sample shape as well. Across nonbaseline lengths at the three diagnostic dates, the middle-rank 90th/10th percentile ratio of eigenvalue ratios ranges from 1.13 to 2.48, whereas amplitude-only scaling requires one. Leading, middle and tail ratios, temporal subspace overlaps and history/feature-prefix sensitivity are exported.

**Is c(ell) reliably estimable from this history?**

No identified population amplitude is established. The training-only identification gate failed before OOS analysis. Separate slopes, pooled common slopes, fixed historical slope, median/mean intercepts and six rank intervals disagree. c_proxy is a finite-range statistic; its numeric transfer paths are retained to falsify the hypothesis, not presented as precise structural estimates.

**Does predicted lambda scaling match useful regularization?**

The full prediction-error and hindsight-loss diagnostics are in prediction_summary.csv. Primary transfer/CV log-penalty RMSE is 3.892; the full-spectrum value is 4.029. This compares heterogeneous historical CV choices, not a known oracle. The nominal ell^(-3) tail law is a conditional infinite-kernel prediction and is not a supported description of these finite spectra.

**Does adaptive theory-guided shrinkage beat fixed lambda?**

Matérn amplitude-transfer net Sharpe difference versus fixed lambda is +0.025 [-0.060, +0.107]; adaptive full-spectrum difference is -0.002 [-0.083, +0.067]. These paired intervals, with 6/24-month sensitivities, determine the strength of evidence. A small point difference is not a demonstrated financial gain.

**Does it beat annual CV?**

Matérn transfer net difference is +0.062 [-0.184, +0.300]; adaptive full spectrum is +0.035 [-0.221, +0.281]. All methods share February 1984-January 2025 and all primary net accounts enter from cash together.

**Does it beat joint CV?**

Matérn transfer net difference is +0.055 [-0.290, +0.352]; adaptive full spectrum is +0.027 [-0.338, +0.342]. Joint CV is the strongest pre-existing practical benchmark; selecting five theory-guided candidates from trailing returns remains hyperparameter selection.

**Does it improve net Sharpe rather than merely reduce risk or exposure?**

Primary Matérn transfer gross/net Sharpe is 3.289/2.651. Annual means, volatility, drawdowns, actual signed exposure, turnover and fees are all compared in performance_summary.csv. The recommendation requires net paired evidence, not a smoother complexity path or lower exposure.

**Can B(ell)/A(ell) be treated as constant?**

It is a tested hypothesis, not established by RKHS equivalence. Historical per-lengthscale CV-implied FOC ratios vary, and source/score geometry changes with the RKHS norm. The additional five-anchor adjusted rule is explicitly separate; it consumes extra validation information and cannot identify a unique structural bound constant.

**Is there a defensible one-page contribution for the paper?**

NO-GO. The conditional spectral-order/transfer proposition and the signal/noise counterexample are defensible mathematics, but they do not establish a useful one-constant oracle formula for this experiment. A separate one-page candidate records the negative diagnostic; do not insert it or expand the manuscript without approval.

The mathematical note distinguishes exact norm/order comparisons, conditional Weyl/asymptotic statements, finite-rank fits and retrospective OOS evidence. No oracle optimality is claimed.

**Does amplitude explain the cross-length change in useful lambda?**

On the common 1984+ diagnostic panel, the amplitude-predicted cross-length shift has log-effect RMSE 2.309 against CV shifts, versus 0.267 against full-spectrum shifts. Full absolute penalty errors and OOS-loss consequences remain in the prediction tables. The contrast with CV is diagnostic, not proof that CV or a 12-month hindsight grid peak equals a population oracle. Conditional mean/noise constants are not identified merely by this residual.

**Representation, shrinkage and stale history**

For Matérn, the descriptive 2x2 gross-Sharpe difference of joint versus fixed-length CV is +0.107; symmetric representation/penalty contrasts are +0.115/-0.008, with interaction +0.173. These are exact arithmetic contrasts of selected paths, not causal contributions. Crossed raw penalties are held fixed and can lie outside the receiving representation's search grid; they are controlled diagnostics, not expanded optimization.

- matern32 adaptive_spectral: 36-versus-60-month selection changes gross Sharpe by +0.003.
- matern32 spectral_transfer: 36-versus-60-month selection changes gross Sharpe by -0.006.
- gaussian adaptive_spectral: 36-versus-60-month selection changes gross Sharpe by -0.082.
- gaussian spectral_transfer: 36-versus-60-month selection changes gross Sharpe by -0.047.

Across the predeclared history/feature-prefix cases, baseline-length middle-rank slope changes range from -0.118 to +0.106. The finite approximation and rank-range sensitivity are observable obstacles. The expanding-refit effect of stale observations and a time-varying source/score constant cannot be separately identified by these controls; the shorter performance-window sensitivity is assessed without claiming to solve that identification problem.

The additional 1978 CV anchor at multiplier 0.25 lies on a boundary for both kernels. The adjusted transfer uses the corresponding historical FOC interval endpoint, not a point-identified rho; explicit lower/upper one-sided intervals are in initial_constant_diagnostics.csv. The primary baseline anchors are interior. Different CV-implied FOC values do not by themselves prove that a sharp population B/A varies.

**Requested Gaussian extension**

The same finite-range transfer pipeline is now executed for Gaussian using its own single 1978 anchor and frozen first-history slope 1.623402. This is explicitly an exploratory amendment requested after the original Matérn-primary run; it does not assume a polynomial Gaussian population spectrum. Gaussian transfer gross/net Sharpe is 3.244/2.618. Its paired net difference is +0.078 [-0.023, +0.175] versus fixed lambda, +0.157 [-0.131, +0.447] versus annual CV, and -0.111 [-0.407, +0.171] versus joint CV. The Gaussian identification gate fails; no nominal Matérn exponent is assigned to Gaussian. Source density changes decay speed, so the common finite-rank exponent is a falsifiable surrogate, not a Gaussian theorem.
