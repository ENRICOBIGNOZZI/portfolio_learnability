# Monte Carlo design and interpretation

The frozen design is `config/design.py`. The full profile contains 500 baseline
replications and 200 in each of 15 robustness configurations. Training sizes are
60, 90, 120, 180, 240, 360, 540, 720, 1080, and 1440. The same 96 log-spaced
positive penalties from 1e-12 to 1e-2 are used everywhere. The grid includes
strongly regularized and nearly unregularized policies; boundary selections are
retained and reported. No plots or fitted slopes feed back into this design.

## Numerical representation and actual stock returns

The baseline economic factor dimension is always three. A 256-dimensional
Nyström subspace approximates the Matérn RKHS. Its inducing points are scrambled
Sobol points drawn independently of all training paths. If G_A is their kernel
Gram matrix, phi(z)=k(z,A) G_A^(-1/2). Ridge penalizes the Euclidean norm of
coefficients in this orthonormal RKHS basis. No diagonal rescaling or numerical
jitter changes the kernel. Summing stock returns against kernel sections before
whitening reduces computation without changing the managed payoff.

Every simulated date contains the actual N stock returns B F + epsilon with
ordinary independent idiosyncratic shocks. No artificial managed factors replace
stock-level simulation. The saved raw Monte Carlo surfaces retain every
replication, T and penalty, as well as the seeds needed to reconstruct the
underlying stock panel; retaining the full stock panels is unnecessary.

Within a replication, training histories are nested across T. Replications have
independent SeedSequence streams. The rank-128/256/512 experiments deliberately
share economic paths and nested inducing points to enable paired approximation
comparisons. This sharing does not create dependence across replication indices.

## Selection, evaluation and benchmarks

The last quarter of each training history is chronological validation. Policies
trained on the earlier block compete using validation response-one loss. The
chosen penalty is then refitted on the complete T observations. The selector's
API accepts neither population moments, W*, nor OOS returns.

An independent stationary 720-period stock-return path provides genuine OOS
loss and Sharpe. Separately, independent population group quadrature evaluates
policy means and second moments, integrating factor and idiosyncratic shocks
analytically. It preserves the exact within-triplet dependence and cross-group
factor contribution. This produces precise population regret for rate analyses
without using population information for validation. Sharpe is unannualized.

The ensemble oracle minimizes mean population regret across independent fitted
replications on the same penalty grid. Pathwise oracle penalties are also saved.
The high- and low-regularization endpoints diagnose underfitting and estimation
error. The validated affine characteristic benchmark asks whether nonlinear
policy flexibility matters; equal weighting gives a simple investable fixed
policy comparison. Neither benchmark changes the headline estimator.

## Decomposition and uncertainty

Let a_lambda be the population ridge coefficients and a_hat the estimated ones.
Population moments in the numerical basis are m and S. Then

    Q(a_hat)-Q* = [Q(a_lambda)-Q*]
                 + (a_hat-a_lambda)' S (a_hat-a_lambda)
                 + 2 (a_hat-a_lambda)' (S a_lambda-m).

The last term is signed; Q(a_hat)-Q(a_lambda) need not be nonnegative.
Population bias includes the measured numerical approximation floor. This
identity is checked pathwise for every saved T and lambda, not just in averages.

Curve files contain means, medians, MC standard errors and pointwise 95% normal
intervals for means. Rate regressions use exactly three predeclared windows:
the full T grid, its upper half and its largest four values. Bootstrap draws
resample whole replication indices, preserving each T-by-lambda surface. Both
the ensemble oracle and validation selections are recomputed in each resample.
OLS slope errors and bootstrap percentile intervals are reported separately.

Dashed theoretical slopes are rate benchmarks, not enforced equalities for this
fixed smooth target. A minimax rate is not a statement that every smooth DGP
attains that worst-case rate. Boundary choices, finite-sample transitions and
numerical approximation can affect observed slopes; none is suppressed.

## Robustness and acceptance scope

N varies through 99, 300, 501 and 999, preserving the required divisibility by
three. Signal/noise, persistence and kernel smoothness change separately. The
economic DGP remains three-factor. Kernel nu=0.5 and 2.5 imply b=7/6 and 11/6,
respectively; their spectra are computed rather than prescribed.

The empirical-rank variant has exact trigonometric balance without ties but
finite support, so its asymptotic E5 interpretation is deliberately violated.
Bounded heteroskedastic Gaussian volatility keeps Gaussian tail control but
generally destroys baseline conditional pricing. Its population reference is
the optimum within the same numerical subspace, not the baseline W*. Separate
numerical reports verify these intended differences rather than labeling all
variants theorem-compatible.

Increasing Nyström rank from 256 to 512 must change paired mean oracle and
validation regret by no more than 10% at every T. The first 200 baseline paths
are used for this paired comparison; rank 128 is a lower-resolution diagnostic.
The threshold was fixed before these rank-robustness outcomes were generated.
This controls the reported oracle and selected-policy comparisons, not every
penalty uniformly. At numerical rank r, empirical complexity is bounded by
min(T,r). Near-unregularized curves can therefore remain rank-sensitive even
when the selected-region check passes. The complete rank-robustness curves are
retained; fixed numerical rank is not an infinite-rank asymptotic experiment.

Scientific seeds, configuration and source hashes are invariant to the worker
count. Execution concurrency affects only runtime. Atomic checkpoints permit
resumption after interruptions without changing the sample or discarding paths.


For heteroskedastic shocks, sigma(z)^2 = sigma_eps^2 exp(0.7 z_2) lies
between sigma_eps^2 exp(-0.7) and sigma_eps^2 exp(0.7). The E5 sandwich
therefore holds with these lower and upper idiosyncratic multipliers divided
by N, plus c_beta in the upper bound. This spectral statement remains exact
even though the baseline conditional pricing identity fails. The numerical
subspace reference satisfies its unconditional normal equation, not a claim
of state-by-state pricing for the unrestricted population problem.

Curve horizontal coordinates are across-replication mean empirical complexity
at each fixed penalty. Mean and median OOS outcomes refer to those same penalty
indices. The rate tables distinguish population trace complexity at the selected
penalty from empirical trace complexity; both are saved. Different summaries
in the selected-complexity/penalty panel are labeled explicitly and do not
represent a synthetic median policy.
