# Rough Rich6D with annualized optimal Sharpe three

This experiment combines the normalized rough loading map with the explicit
monthly convention from the separate SR3 experiment. The analytical target is
`SR_star_annualized = sqrt(12) * SR_star_monthly = 3`, so the monthly optimum is
`0.8660254037844386`. This is a marginal-Sharpe reporting convention, not the
Sharpe of compounded annual returns.

```sh
VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough_sr3 --workers 2
```

Outputs go to `simulations/outputs/rich6d_rough_sr3_annual_monthly_v1/`.
The original rough run and the separate smooth SR3 run remain unchanged.

The factor mean vector is calibrated analytically, preserving its direction:
`mu_F = (0.5490699705067791, 0.32944198230406746, 0.16472099115203373)`.
The multiplier relative to the original means is `5.490699705067791`.
The factor covariance becomes `I - mu_F mu_F'`, retaining `E[FF']=I` and a
positive definite Gaussian covariance. Stock returns and learned policies are
regenerated and refitted; old fitted outcomes are not multiplied to manufacture
a new simulation. Everything else in the rough economic model is retained.

The independent population pilot, production and sensitivity operators are
recomputed. Their second moments are invariant to this calibration because
`E[FF']=I` remains fixed; their means change. Thus the recomputed penalty scale
and population spectrum can legitimately coincide with the original rough run.
The new protocol explicitly records the economic parameters and monthly versus
annualized conventions before production.

The established 100 independent replications, ten horizons, rank-512 main
estimator, 50 nested rank-1024 audit paths, 96 diagnostic penalties, exact
`theory_1` penalties, four quadrature seeds, and bootstrap windows are retained.
Seed families also remain fixed, pairing innovations with the original rough
experiment, while remaining independent across replications and across pilot,
evaluation, and training families. Process workers only change execution
concurrency, not the random streams or scientific design.

The 128/512/2048 Fourier comparison is rerun before production. Because the
analytical policy scales linearly with the factor means, its absolute numerical
tolerance is the original `1e-7` times the analytically known multiplier. This
preserves the original relative policy precision and is set before simulation.
Other tolerances are unchanged. No Fourier coefficient or loading amplitude is
retuned. Finite Fourier truncations remain smooth and cannot prove the exact
source condition `r=1`.

Figures 1 and 3, Sharpe gaps, captions, and the narrative report use annualized
Sharpe units. The explicitly suffixed `_annualized` CSV columns supply the
plotted values and confidence bounds. Exact raw checkpoint metrics and the
original estimator summaries remain monthly, with their units declared in the
protocol. Loss, regret, penalty, eigenvalues, and complexity are not annualized.
Both raw and presentation tables are retained to make the conversion auditable.

Numerical failures remain visible. A larger signal can change the inherited
5% regret-based rank audit even if the second-moment spectrum is invariant.
The report and figures therefore use new audit outcomes, not certification flags
from the old run. The manuscript and empirical section are not outputs.

The additional audit command verifies all checkpoint hashes and annualized
columns and, when the old rough run is available, confirms that all 100 stock
return hashes changed despite paired innovations:

```sh
VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.diagnostics.rough_sr3_verification
```
