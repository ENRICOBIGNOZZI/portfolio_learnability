# Persistent characteristic-factor economy

## Economic construction and exact identities

N=5,000 stocks have stationary latent Gaussian AR(1) characteristics, rho_Z=0.90.
At formation t, U=(rank-0.5)/N, using characteristics only. The actual stock audit
constructs three months for each economy/J configuration and five random policies per
month. Phi_ij=sqrt(2)cos(pi*j*U_i), j=1,...,J<N, satisfies Phi'Phi/N=I.
R_(t+1)=Phi_t F_(t+1)+epsilon_(t+1). Independent stock shocks with sigma_e=0.15
are projected as epsilon=sigma_e*(v-Phi Phi'v/N); the identity normalization makes this
exactly the requested orthogonal-complement projector. Stock residual noise is nonzero.
For weights W=Phi theta/N, W'R=theta'F up to the audited numerical precision.
The admissible policies are static functions of the current characteristic; theta*
is their unconditional population optimum, not an optimum over arbitrary history-dependent
strategies. Positions are unconstrained long-short exposures, not normalized unlevered
index weights. For each finite J the coefficient norm defines the finite-dimensional
RKHS with kernel sum_j phi_j(u)phi_j(v). This is a controlled finite-rank approximation,
not a proof of a uniformly bounded infinite-J point-evaluation kernel.
After checking this identity, every Monte Carlo path uses these same managed factors
without rebuilding the observationally irrelevant stock matrices. It is the same DGP.

mu_j=0.0004*j^(-b), b in {1.25,1.5,2}. The NL target is c_theta/j (a=1).
The L sanity target has only coordinate one. In each configuration c_theta normalizes
theta*'S theta*=0.16<1; mean factors are fbar=S theta*. Stationary covariance is
V_F=S-fbar fbar', NOT S. A diagonal-minus-rank-one square root samples this covariance
without a dense Cholesky. Factor AR persistence is 0.30, with stationary initialization
and innovation covariance (1-rho_F^2)V_F. Thus E[FF']=S exactly, theta* is optimal,
and Q(theta*)=0.84. One 200,000-month stationary path audits each configuration:
all coordinate means/second-moment diagonals, a whitened 16x16 second-moment block,
and aggregate centered AR coefficient. Recorded empirical/target means and moments
are included in the JSON, together with every parameter and residual volatility.
Predeclared long-path tolerances: maximum standardized mean error <6.5, maximum
relative diagonal error <0.035, 16x16 block error <0.025, AR error <0.005.

| Economy | b | J | c_theta | min eig(V_F) | max orthogonality error | max payoff error | estimated AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| NL | 1.5 | 2000 | 18.8417 | 4.47214e-09 | 1.07e-13 | 3.76e-14 | 0.29999 |
| NL | 1.25 | 2000 | 18.5763 | 2.9907e-08 | 1.07e-13 | 2.73e-14 | 0.30005 |
| NL | 2.0 | 2000 | 19.2243 | 1e-10 | 1.08e-13 | 4.06e-14 | 0.30004 |
| L | 1.5 | 2000 | 20 | 4.47214e-09 | 1.07e-13 | 4.15e-14 | 0.30001 |
| NL | 1.5 | 1000 | 18.8417 | 1.26491e-08 | 5.69e-14 | 7.30e-15 | 0.30012 |
| NL | 1.5 | 4000 | 18.8417 | 1.58114e-09 | 1.86e-13 | 8.07e-14 | 0.30000 |

## Estimation, tuning and uncertainty

T={60,90,120,180,240,360,540,720,1080,1440}. Headline NL b=1.5 J=2000 has
500 independent paths; other b values, L, and J=1000/4000 have 200 each.
Within one replica the T paths are nested prefixes; replicas are independent.
Seeds use SeedSequence([20261003,100*b,J,economy_code,replica]); no seed is selected
for attractive results. The population risk is the exact quadratic
(theta_hat-theta*)'S(theta_hat-theta*), not a noisy holdout loss.
The estimator uses uncentered sample second moments and mean response-one payoffs.
Population ridge-target regret and empirical complexity diagnostics are saved too.

Each spectrum uses 360 positive logarithmic penalties from mu_J*1e-5 to mu_1*1e3.
The script automatically extends/reruns the grid if any MC oracle is on a boundary.
Simulation lambda never equals zero: it includes near-unregularized positive ridge.
Population C=sum mu/(mu+lambda); unlike sample rank, population C/T can exceed one.
No large-C portion is dropped. Both heatmaps use log x axes over the full sampled range.
Their geometric bin edges lie inside the sampled endpoints; no curve smoothing or
out-of-range extrapolation is used. Color is log10(mean exact regret / row minimum).

Validation trains on the first T-V, V=min(60,floor(T/3)), selects minimum response-one
loss on the last V, then refits all T. Exact ties choose largest lambda. Neither
population oracle nor regret enters validation. The validation table reports selected
medians, mean population regret, MCSE, ratio to oracle and boundary frequency.
Bootstrap resamples whole independent replicas, paired across T and lambda, 500 times.
Intervals reselect the oracle within each bootstrap. They describe Monte Carlo error,
not sampling uncertainty of a single investment history. Grid discreteness remains.
Slopes report OLS SE and bootstrap SE/95% intervals. The predeclared headline subset
is the upper half, T=360,...,1440; full-grid and largest-four estimates are also saved.
OLS regression errors alone do not account for paired Monte Carlo dependence.

In the headline economy, mean validation-selected regret is
2.749 times oracle regret at T=60 and
5.249 times oracle regret at T=1440.
At T=1440 it is 0.030339, versus oracle
0.005780. Median complexity markers near the oracle curve
therefore must not be interpreted as near-oracle mean performance: they conceal
loss dispersion and costly selection errors. The validation window is capped at
60 observations; it does not grow with T beyond 180. This experiment does not
establish rate-optimality of that finite-validation tuning rule.

## Honest comparison with the r=1 envelope

| b | Quantity | fitted slope | OLS SE | MC bootstrap SE | R squared | r=1 reference |
|---:|---|---:|---:|---:|---:|---:|
| 1.5 | lambda_oracle | -0.4777 | 0.0162 | 0.0109 | 0.9965 | -0.6000 |
| 1.5 | C_oracle | +0.3279 | 0.0108 | 0.0075 | 0.9968 | +0.4000 |
| 1.5 | regret_oracle | -0.6344 | 0.0125 | 0.0178 | 0.9988 | -0.6000 |
| 1.25 | lambda_oracle | -0.4473 | 0.0152 | 0.0265 | 0.9965 | -0.5556 |
| 1.25 | C_oracle | +0.3438 | 0.0114 | 0.0204 | 0.9967 | +0.4444 |
| 1.25 | regret_oracle | -0.5922 | 0.0114 | 0.0245 | 0.9989 | -0.5556 |
| 2 | lambda_oracle | -0.5144 | 0.0326 | 0.0344 | 0.9881 | -0.6667 |
| 2 | C_oracle | +0.2821 | 0.0182 | 0.0190 | 0.9877 | +0.3333 |
| 2 | regret_oracle | -0.6502 | 0.0117 | 0.0441 | 0.9990 | -0.6667 |

RKHS membership supplies a bias upper bound, not a matching bias lower bound for
every fixed target. The supplied manuscript explicitly makes this distinction in
proofs_learning.tex, in the proof of effective complexity (around lines 950-953).
This simulation therefore checks the controlled operator/estimator and compares
rates with the envelope; it must not be described as establishing equality with a
worst-case minimax rate for this particular target.

Here the population ridge bias is exactly sum_j mu_j theta_j^2 [lambda/(mu_j+lambda)]^2.
For a=1, infinite power-law spectra and b>1 give bias of order lambda^(1+1/b),
which is smaller than the general O(lambda) bound. Balancing this pointwise bias
with order lambda^(-1/b)/T suggests lambda of order T^(-b/(b+2)), C of order
T^(1/(b+2)), and regret of order T^(-(b+1)/(b+2)). These are explanatory asymptotic
calculations, not fitted constraints or replacements for the requested r=1 comparison.
Finite T, AR dependence, empirical covariance estimation and finite J can alter slopes.
The smooth one-direction L target is reported separately and is not used to claim
the nonlinear law. No r parameter was added to the baseline.

## Finite-rank check and reproducibility

J=1000,2000,4000 are compared using the same prespecified T subsets, with independent
seeds across J. Stability tolerances declared before results: maximum upper-half slope
difference from J=2000 <0.08; largest-T mean-regret difference <10%.
Observed check: {"max_slope_difference": 0.03623313160352282, "slope_tolerance": 0.08, "max_largest_T_relative_regret_difference": 0.045248074803436955, "regret_tolerance": 0.1, "passed": true}.
If this check fails, the finite-rank acceptance condition remains unmet; a failure
cannot be concealed by switching the slope interval. The JSON and CSV retain it.

Compute command: `VECLIB_MAXIMUM_THREADS=1 python3 -u simulation_characteristic_factor.py --workers 2`.
The initial run used 2 BLAS threads/one process, then resumed from deterministic
checkpoints using four, subsequently two, one-thread workers to limit memory pressure;
results do not depend on scheduling.
Render: `VECLIB_MAXIMUM_THREADS=1 python3 render_characteristic_factor.py`.
Recorded completed-checkpoint compute time: 3227.3 seconds.
The run manifest records code hashes, counts, seeds and timing. Licensed empirical
inputs are unrelated to the artificial economy and are never required for this run.
