# Interpreting rates without a predetermined result

The theorem gives a risk envelope under its assumptions, including the source
condition r=1. At fixed N and D=6, a Matérn kernel with smoothness nu has the
benchmark b=1+2nu/6. Choosing lambda proportional to T^(-b/(b+1)) is a separate
experiment from fitting a slope to an ex-post loss oracle or a validation rule.
The population calibration s_ref fixes numerical units once; it cannot be
estimated from confirmation loss or changed separately at each T.

Both loading maps are smooth fixed targets. Six-coordinate dependence neither
establishes a least-favourable minimax family nor requires risk to attain the
envelope exponent with equality. Faster convergence is possible. A selected
penalty may have a skewed distribution; slopes of its mean, median and geometric
mean are different summaries and must be labeled separately.

The empirical complexity is bounded by min(T,numerical rank), whereas population
complexity divided by T can exceed one. A projection floor can dominate at large
T. Fixed-rank slopes cannot establish an infinite-kernel asymptotic law when
numerical accuracy is unresolved. A fitted eigenvalue slope on a finite window
is descriptive: the operator sandwich and analytic Matérn result support E5,
not proximity to a historical tolerance band. The lower sandwich constant
depends on N, which stays fixed within each learning-rate experiment.

Regret decomposes into population regularization bias (including approximation
floor), a nonnegative estimation norm and a signed cross term. The signed
difference between sample and population ridge losses is not a variance.
Negative sample OOS excess loss is retained: it can arise from test-sample noise
even though population regret is nonnegative. Pointwise Monte Carlo intervals
do not include quadrature or anchor uncertainty unless explicitly augmented.

The old 3500 evaluations remain historical evidence. The V2 confirmation uses
new seed families and a smaller, prospectively recorded local-resource cohort.
It does not retroactively preregister the old results, and no selector is
retuned after viewing the confirmation outcomes.
