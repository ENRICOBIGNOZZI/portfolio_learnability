# Boundary-target extension and four DGP performance figures

The new headline uses theta_j proportional to 1/[sqrt(j) log(j+1)], with natural
logarithm, normalized to theta*'S theta*=0.16. There are 500 paths at J=2000 and
200 each at J=1000 and J=4000, all with b=1.5. The original six configurations
and their 1,500 independent paths are retained; total unique paths are 2,400.
The original smooth nonlinear 1/j economy now has an explicit appendix figure.

The source exponent statement applies to the infinite sequence. Finite J and
logarithmic corrections prevent this experiment from certifying exact equality
with all pure-power minimax r=1 benchmarks. The sample sizes, 360-point lambda
grid, slope subsets, and normalization are not changed to improve agreement.

| Quantity | upper-half slope | bootstrap SE | bootstrap 95% interval | minimax r=1 benchmark |
|---|---:|---:|---|---:|
| lambda_oracle | -0.477722 | 0.021407 | [-0.480969, -0.431573] | -0.600 |
| C_oracle | +0.327895 | 0.014579 | [+0.296570, +0.331010] | +0.400 |
| regret_oracle | -0.639432 | 0.018284 | [-0.675479, -0.601774] | -0.600 |

Finite-J stability passes: maximum slope difference 0.037358;
largest-T relative regret difference 0.030803.
The previously declared tolerances remain 0.08 and 0.10.

The four requested reference-style figures use exact population loss versus
complexity, exact population Sharpe versus complexity, the raw loss heatmap,
and chronological-validation selections of C and lambda. T replaces year.
The performance extension re-estimates the SAME 500 headline paths to recover
moments not retained in the original checkpoint. These are not 500 extra
independent paths. Every fitted regret and empirical complexity is compared
with the original saved path; maximum loss identity error is
1.377e-14. No population outcome enters validation.
The original oracle/validation/slopes tables are bit-for-bit identical numerically;
only recomputed population ridge bias differs, by at most 8.33e-17.

## Commands

    VECLIB_MAXIMUM_THREADS=1 python3 -u simulation_characteristic_factor.py --workers 2
    VECLIB_MAXIMUM_THREADS=1 python3 -u simulation_performance.py --workers 2
    VECLIB_MAXIMUM_THREADS=1 python3 render_simulation_performance.py
    VECLIB_MAXIMUM_THREADS=1 python3 render_characteristic_factor.py
    VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
    VECLIB_MAXIMUM_THREADS=1 python3 verify_final_outputs.py

All 88 tests pass. Independent aggregate verification passes for 32,400 main
surface rows and 3,600 performance rows, all three finite-rank sizes, fixed
validation selections, original empirical accounting checks and source/output hashes.
No licensed empirical inputs or returns are recomputed by this extension.

Checkpoint computation times (seconds): {"R1_b150_J4000_R200": 635.1322549999968, "R1_b150_J1000_R200": 227.22472774999915, "R1_b150_J2000_R500": 1451.7003627919985}.
Additional moment calculation: 816.1 seconds.
Both jobs use two worker processes and one BLAS thread per process.

Original-results preservation detail:

{
  "full_surface.parquet": {
    "identical_except_floating_point_reductions": {
      "population_ridge_regret": 8.326672684688674e-17
    }
  },
  "oracle_by_T.csv": {
    "identical_except_floating_point_reductions": {}
  },
  "validation_by_T.csv": {
    "identical_except_floating_point_reductions": {}
  },
  "rate_slopes.csv": {
    "identical_except_floating_point_reductions": {}
  }
}
