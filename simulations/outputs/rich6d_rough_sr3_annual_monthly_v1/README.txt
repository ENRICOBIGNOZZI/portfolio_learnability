Rough Rich6D. Sharpe ratios and gaps are annualized as sqrt(12) times marginal monthly values, not the Sharpe of compounded annual returns.
Read critical_report.txt and figures/captions.txt for numerical scope and limitations.
Reproduce: VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough_sr3 --workers 2
