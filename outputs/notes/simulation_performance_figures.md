# Four DGP performance figures

The four PDFs/PNGs reproduce the reference figure roles for the simulated R1 boundary economy. Training sample size T replaces calendar year; the DGP is stationary. There are 500 fitted paths per T, nested across T within each independent replication.

For each fitted coefficient vector beta, m = beta.T S theta*, s2 = beta.T S beta, Q = 1 - 2m + s2 = 0.84 + regret, and SR = sqrt(12) m / sqrt(s2 - m^2). The average of these policy-specific ratios is reported, not a ratio of averaged moments. The sqrt(12) convention does not equal the Sharpe of a twelve-month sum under AR dependence. Evaluation integrates a fresh stationary population draw; it is not conditional on the last training observation and is not an observed next-year return sequence.

The selected-lambda panel uses exactly the existing chronological validation choices. Population moments do not affect tuning. C and lambda medians and their interquartile ranges are computed separately from their actual replication-level values. They summarize selection dispersion, not the parameters of one synthetic median policy. The validation window is capped at 60 observations; the main simulation note documents its substantial regret relative to the oracle.

Every recomputed fit is checked against the saved original regret and empirical-complexity arrays. The maximum loss identity discrepancy is 1.377e-14. The population Sharpe bound is sqrt(12*0.16/0.84). No curve shape, downturn, or zero Sharpe endpoint is imposed. All 360 positive penalties remain visible; the smallest is near the unregularized finite-rank limit, not exactly zero.

Reproduce calculation: VECLIB_MAXIMUM_THREADS=1 python3 simulation_performance.py --workers 2.
Reproduce figures from public aggregates: VECLIB_MAXIMUM_THREADS=1 python3 render_simulation_performance.py.
