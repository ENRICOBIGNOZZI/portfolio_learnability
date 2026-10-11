# Rich6D initial-CV closeout — 12 October 2026

The complete 300-history experiment in each of five economies is preserved. The economic model, M=128, Matérn lengthscale, one-time initial60 CV, 71 constants, forward folds, exponent -0.6, pairing, horizons, boundary choices and analytical optima are unchanged. All new diagnostics are post hoc. No full campaign or cloud workflow was launched. Existing unrelated empirical work is excluded from this task.

## Quantities covered by the rate

On the 13 baseline means, regret slope = -0.4378196; Sharpe-gap slope = -0.3986480. Loss versus Sharpe alone does not explain the finite slow decay. All four established windows and full/common grids are exported with whole-history covariance MCSE and jackknife checks. Slopes regress log Monte Carlo means, not individual-history slopes.

Normalized regret T^0.6 regret, monthly/annual Sharpe gaps times T^0.6, and empirical/population C/T^0.4 are retained for every history and summarized by mean, median, SD, 2.5/97.5 percentiles and mean MCSE. A finite plot does not certify or disprove an asymptotic O_P upper bound. No cutoff is chosen by proximity to -0.6.

## What initial CV improves

- baseline, T=4860, regret: initial CV 0.011296; historical 0.022205; paired change -0.010909, 95% mean interval [-0.011399, -0.010419]. Oracle Q 0.008327; oracle SR 0.008665.
- baseline, T=4860, annual_SR: initial CV 2.932497; historical 2.934774; paired change -0.002277, 95% mean interval [-0.005079, +0.000525]. Oracle Q 2.951808; oracle SR 2.953426.
- N300, T=3240, regret: initial CV 0.025292; historical 0.033922; paired change -0.008630, 95% mean interval [-0.013779, -0.003480]. Oracle Q 0.015551; oracle SR 0.016467.
- N300, T=3240, annual_SR: initial CV 2.815167; historical 2.838549; paired change -0.023382, 95% mean interval [-0.041627, -0.005136]. Oracle Q 2.861294; oracle SR 2.863383.
- N1200, T=3240, regret: initial CV 0.008765; historical 0.026467; paired change -0.017701, 95% mean interval [-0.018494, -0.016909]. Oracle Q 0.005453; oracle SR 0.005783.
- N1200, T=3240, annual_SR: initial CV 2.974204; historical 2.965526; paired change +0.008678, 95% mean interval [+0.004675, +0.012682]. Oracle Q 2.995382; oracle SR 2.997172.
- rho000, T=3240, regret: initial CV 0.015380; historical 0.029037; paired change -0.013657, 95% mean interval [-0.017025, -0.010289]. Oracle Q 0.009187; oracle SR 0.009739.
- rho000, T=3240, annual_SR: initial CV 2.915776; historical 2.921605; paired change -0.005829, 95% mean interval [-0.017817, +0.006159]. Oracle Q 2.947726; oracle SR 2.949870.
- rho075, T=3240, regret: initial CV 0.014634; historical 0.029092; paired change -0.014458, 95% mean interval [-0.017217, -0.011699]. Oracle Q 0.009226; oracle SR 0.009782.
- rho075, T=3240, annual_SR: initial CV 2.918416; historical 2.921493; paired change -0.003078, 95% mean interval [-0.012181, +0.006026]. Oracle Q 2.947497; oracle SR 2.949641.

Baseline T=60: annualized Sharpe is 2.598814 under initial CV versus 2.067570 historically; paired difference +0.531244, 95% mean interval [+0.494082,+0.568406]. All later paired horizons are retained in the delivered CSV, including the loss of a clear final Sharpe benefit.


Initial constant distribution, including boundaries (all 300 histories each):

- baseline: median a=5.0118723e-06, central95% [7.9432823e-07,1.2589254e-05], lower-bound selections 0, upper-bound selections 0; no removal.
- N300: median a=6.3095734e-06, central95% [1.1229896e-06,2.5118864e-05], lower-bound selections 2, upper-bound selections 1; no removal.
- N1200: median a=3.1622777e-06, central95% [7.0855852e-07,1e-05], lower-bound selections 0, upper-bound selections 0; no removal.
- rho000: median a=5.0118723e-06, central95% [1.2589254e-06,1.5848932e-05], lower-bound selections 2, upper-bound selections 0; no removal.
- rho075: median a=5.0118723e-06, central95% [7.0855852e-07,1.2589254e-05], lower-bound selections 2, upper-bound selections 0; no removal.

In the baseline the final loss improvement is clear, while the final Sharpe interval includes zero. The original short-history benefit remains tabulated. Oracle Q minimizes mean loss and oracle SR maximizes mean Sharpe on the same original 96-point grid, one common penalty per economy/T. Even/odd cross-fitting selects on one half of complete histories and assesses on the other. Neither oracle enters tuning; oracle bands and slopes do not account for selection.

## Measured mechanisms and scope

Baseline T=4860, A (initial CV), **1 recovered audit histories**, monthly quantities: regret=0.0098476626, u=0.42937588, v=0.44002799, c=1.001877, d=0.0098461526, radial_gap=1.5099887e-06, normalized_directional_error=0.040054613, ray_gap=0.0095897984, scale_gap=0.00025786423, projection_floor=0.0017771061, regularization_bias_within_P=0.0036774466, estimation_displacement=0.0043699786, signed_cross_term=2.3131278e-05.
Baseline T=4860, C (historical), **1 recovered audit histories**, monthly quantities: regret=0.020265504, u=0.362138, v=0.31597007, c=0.84498866, d=0.0099675686, radial_gap=0.010297935, normalized_directional_error=0.057003603, ray_gap=0.013519683, scale_gap=0.0067458211, projection_floor=0.0017771061, regularization_bias_within_P=0.019338313, estimation_displacement=0.0011582166, signed_cross_term=-0.0020081318.

These are measured audit-history decompositions, not a reconstruction of all 300 final policies. The scale/ray and radial/angular identities hold policy by policy. Direction is normalized because raw d changes with positive rescaling. Population-ridge bias is relative to the optimum in the same subspace, and the signed cross term is retained. The displacement norm is not labeled variance. A numerical difference in this subset is descriptive, not a causal diagnosis of the complete Monte Carlo mean.


All 300 endpoint baseline policies have positive signed Sharpe. Without choosing ANY u/v root, the identified per-policy ray gap is q-SR_monthly^2/(1+SR_monthly^2), scale gap is regret-ray_gap, and normalized directional error is SR_star_monthly^2/SR_monthly^2-1. These are uniquely identified even when two distinct moment pairs produce the same Q/SR.

initial_cv, **all 300 baseline endpoint histories**: ray_gap=0.011118805, scale_gap=0.00017688868, normalized_directional_error=0.046769925, projection_floor=0.0017771061, regularization_bias_within_P=0.0031794424, estimation_displacement_plus_cross=0.006339145.
old_fixed, **all 300 baseline endpoint histories**: ray_gap=0.010732949, scale_gap=0.011471944, normalized_directional_error=0.044978646, projection_floor=0.0017771061, regularization_bias_within_P=0.019338313, estimation_displacement_plus_cross=0.0010894733.

Initial CV sharply reduces scaling error while the mean scale-invariant directional loss is slightly larger. This explains the response-one/Sharpe discrepancy for the actual complete Monte Carlo means. Population-ridge bias also falls, but the combined displacement plus signed cross grows. The latter is not called variance; only recovered coefficients separate its two terms. Raw monthly u/v/c/d remain unavailable for the unregenerated histories.

Finite full-grid component slopes (post hoc, no horizon removal): ray_gap=-0.407614, scale_gap=-0.923496, regularization_bias_within_P=-0.336122, representation_adjusted_regret=-0.470399, representation_adjusted_ray_gap=-0.440013. The projection floor explains only part of the finite slow slope: subtracting it changes descriptive regret decay from about -0.438 to about -0.470, still distinct from -0.6. The complete-grid empirical benchmark always remains the analytic economic optimum.

## Rank and quadrature

The new CV audit re-estimates full audit horizons for 1 original production histories and only T=60 for 15 additional histories. There are 0 failed numerical metric comparisons and 450 comparisons with insufficient independent histories for MCSE. Rank-specific initial CV changes 8 of 80 checked selections. Representative-history MCSE is computed separately from minimum-a stress diagnostics; stress cases are never pooled into an IID MCSE. The twelve-history full-horizon protocol and separate stress coverage are incomplete; uncertainty at untested cells remains unresolved. Rank 8192 has no forward-rank certificate. Population-ridge decomposition at rank 8192 has not run. Quadrature differences are systematic evaluation diagnostics; replication MCSE does not quantify uncertainty over quadrature seeds.

Full audit histories: [0]; additional initial60 audit histories: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 33, 50, 70, 232]. Missing full representative indices: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Missing full stress indices: [1, 33, 50, 70, 232].

Both A (same archived constant across ranks) and B (initial CV repeated at each rank on the same initial60) are kept separate. C checks the historical rule. Primary and independent quadratures use fixed learned policies and different seeds, with analytical factor/noise integration. An estimated subspace optimum is never treated as the exact economic optimum. Normal residuals and reproduction of old loss/Sharpe are checked. Paired bounds with fewer than two histories are marked not estimable rather than given an invented MCSE.

- baseline, T=3240: projection floor 0.0017771061, mean regret 0.013111, ratio 13.554%; 5% condition fails.
- baseline, T=4860: projection floor 0.0017771061, mean regret 0.011295693, ratio 15.733%; 5% condition fails.
- N300, T=3240: projection floor 0.0034043193, mean regret 0.025292483, ratio 13.460%; 5% condition fails.
- N1200, T=3240: projection floor 0.00090822786, mean regret 0.0087652606, ratio 10.362%; 5% condition fails.
- rho000, T=3240: projection floor 0.0017771061, mean regret 0.01538041, ratio 11.554%; 5% condition fails.
- rho075, T=3240: projection floor 0.0017771061, mean regret 0.014634299, ratio 12.143%; 5% condition fails.

All failed original checks remain in `historical_numerical_warnings.csv`. The new four-specification `numerical_reliability_mask.csv` preserves the complete grid and asserts no global rank certificate. No ranks are mixed along the archived curves.

## Fourier and theory

The unchanged supplied note and compiled PDF are in sources/. Its ideal infinite-economy r=1 proof, operator-range comparison, and conditional finite-positive-grid frozen-CV argument were checked against the DGP and current manuscript; see theory_review.md. The manuscript mixing/stability assumptions and sufficient asymptotic threshold remain conditions. The finite M128 economy is not shown to have maximal source order one. A fixed numerical subspace with positive representation error cannot inherit convergence of regret to zero against the full economic optimum. The Nyström computational-error issue is also explicit in [Rudi, Camoriano and Rosasco](https://papers.nips.cc/paper/5936-less-is-more-nystrom-computational-regularization.pdf); their assumptions are not automatically transferred to this dependent unbounded-feature setting.

M128 uniform Fourier tail bound = 6.699081605e-09; the conservative second-operator perturbation/lambda ratio has maximum 0.682215 across actual CV penalties and exceeds 5% in 0.4317% of cells. Small pointwise error does not certify a ridge inverse or argmin.
Paired M128/512/2048 refits and economy-specific moment evaluation: {'indices': [0], 'times': [60], 'terms': [128, 512, 2048], 'rank': 4096, 'groups': 32768, 'quadrature_seed': 1237722080, 'tolerance': 0.05, 'scope': 'Post hoc initial60 diagnostic; actual CV and frozen-constant fits; no endpoint sensitivity certification.'}. All checked initial choices unchanged: True. Final-horizon sensitivity and second independent quadrature for the Fourier variants remain not run.
Within this single initial60 pilot, maximum relative RKHS policy change = 1.4964404e-10; maximum relative moment/gap change = 6.687229e-11. These small changes do not certify the untested endpoints.

## Status by claim

- **resolved**: verified full 1500-history input provenance; all-horizon regret and Sharpe-rate diagnostics and normalized distributions; distinct loss/Sharpe common-grid oracles; whole-history uncertainty; preserved initial constant; exact helper identities and recovered low-rank population-ridge reconstruction; reviewed ideal/theorem distinction; generated four-figure section and appendix.
- **failed**: endpoint projection-floor criteria wherever the exported ratio exceeds 5%. The new paired numerical audit has 0 failed comparisons in its checked scope; late single-history comparisons lack MCSE. Thresholds are unchanged.
- **unresolved**: transfer of the full-kernel asymptotic upper rate to the fixed Nyström implementation; full-horizon twelve-history representative/stress audit; rank8192 forward accuracy; complete-300-history raw u/v/c/d and separate population-ridge displacement/cross decomposition.
- **not_run**: uncompleted audit workers and rank8192 population-ridge decomposition; final-horizon paired Fourier fitted-policy/CV sensitivity; new 1500-history campaign (deliberately not launched).

## Minimum further work

Complete the declared missing paired representative/stress histories at the same fixed audit horizons, using saved sidecars for independent evaluation. If the 5% criterion still fails, assess higher-rank cost and benefit before any new production. To recover raw moments and separate the population-ridge displacement/cross for all 300 histories, recover their original coefficients/moments or regenerate the missing histories with the original seeds; do not choose an arbitrary loss/Sharpe root. The complete-300 ray/scale split already requires no such root. Extend paired Fourier sensitivity only where the operator bound and initial60 diagnostic cannot close the requested penalty-specific check. A full rerun is a later scientific decision, not an automatic action.

## Reproduction and editorial package

Run the commands in simulations/closeout/README.md. Worker receipts preserve indices, original training seeds, history hashes, selected constants, fold losses, coefficients and moments. closeout_verification.json checks CSV arithmetic independently. artifact_manifest.json hashes the intended public artifacts; regenerable large basis/operator caches are excluded. Four principal figures have vector PDF and PNG copies; extra economy and normalized-outcome figures appear in the appendix. All plotted bands are replication outcomes, with mean intervals separate.

The paper is compiled in an isolated staging copy before the one-line simoutput link changes. Empirical and theorem inputs are hash-checked and unchanged. Both a standalone simulation document and a full staged manuscript are delivered. Visual QA and test receipts record the final verified versions.
