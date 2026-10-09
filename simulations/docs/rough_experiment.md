# Rough Rich6D numerical experiment

The `rich6d_rough` loading map adds the requested period-two Fourier target to the
existing balanced three-factor economy. `rich6d` and `baseline_original` retain
their previous loading functions. No paper, proof, or empirical file is an output
of this experiment.

```sh
VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough --workers 2
```

The default destination is `simulations/outputs/rich6d_rough_r1_v1/`. The runner
checks Fourier truncation before freezing its protocol and then executes all
100 independent economic replications on the original ten-horizon grid. It
reuses the repository's Matérn basis, ridge solver, analytical optimal policy,
and independent balanced-group population evaluation. It creates no holdout or
rolling selector. The 96 grid penalties are diagnostics; the actual strategy
is fitted separately at the exact value of `a*T**(-0.6)`.

`dgp/rough.py` uses a vectorized Clenshaw cosine-series evaluation. Parseval
normalization fixes the RMS to that of a sine, independently of performance.
The deterministic normalization sum extends through 65,536 with an explicit
negligible tail bound. `eta=0.35` is unchanged. Every finite truncation is smooth;
these outputs cannot mathematically verify the source condition `r=1`.

`protocol.json` contains the executable configuration, source hashes, separate
seed families, frozen amplitude, Fourier truncation, and theory penalty scale.
The population pilot retains the existing `theory_1` convention: `a` is the
largest eigenvalue of the rank-512 population second-moment approximation.
It uses fresh independent groups, not production evaluation draws. This is a
population-calibrated theoretical benchmark, not a historical tuning procedure.
Economic parameters and per-period Sharpe units are unchanged.

The first 50 predeclared replications additionally fit a nested rank-1024
estimator with the identical economic histories. Their policies are compared
on a common, independently recomputed 32,768-group evaluator. Four further
independent quadrature seeds compare 8,192 versus 32,768 groups on fixed theory
policies. The inherited 5% regret-based criteria are retained. Failed cells
remain in the output and are explicitly marked in diagnostic curves.

The four figures use only new results. A single fixed rank-1024 population
spectrum supplies the spectral panels and population complexity coordinates.
This compressed finite-rank spectrum is not the exact infinite-dimensional
spectrum. Empirical rank-512 complexity and population rank-512 complexity are
separately labeled. Pointwise Monte Carlo bands condition on fixed numerical
approximations; numerical sensitivity is reported separately.

Read `critical_report.txt`, `figures/captions.txt`, `observed_slopes.json`, and
`verification.json` in the output directory. Figure source CSVs, full path
summaries, raw checkpoints, calibration and sensitivity records, and a hashed
reproducibility manifest accompany the four vector PDFs and 320-dpi PNGs.
Checkpoints retain exact float64 rank-512 training histories, seeds and stock
return hashes, fitted theory coefficients, all 96 diagnostic metrics, and
paired rank audit results. High-rank histories can be regenerated from seeds.

`--prepare-only` stops after the preflight and independent population operators.
`--render-only` reconstructs the complete 100-path summaries and figures from
checkpoints. A protocol mismatch stops resumption rather than silently reusing
results under different scientific code. To run a different scientific design,
use a different output directory and freeze a new protocol.

The original simulations and saved results are never loaded or overwritten.
