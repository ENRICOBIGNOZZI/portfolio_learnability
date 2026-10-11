# Final theory-guided regularization experiment

The original 3.56133432162478 Sharpe baseline is reproduced before the controlled 2 x 2 design. See `DECISION_MEMO.md` and `publication/final_theory_report.pdf` for the findings. Original published outputs and the manuscript are preserved.

## Reproduction

Requires the existing licensed JKP clean panels and verified managed-payoff caches, Python/numpy/pandas/scipy/pyarrow/threadpoolctl/matplotlib, reportlab for the PDF, and pytest. Run from the repository root with single BLAS threads. The complete reproduction wrapper requires fresh destinations and preserves the delivered package.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m empirical_final.final_theory_reproduce \
  --output outputs/final_theory_regularization_reproduction \
  --private results/final_theory_regularization_reproduction
python3 -m pytest -q tests/test_final_theory_regularization.py tests/test_final_theory_dense.py
```

The wrapper calls replication, estimation, actual-holdings costs, paired inference, rolling diagnostics, figures, the separately specified denser search, and the report in order. Both destinations must be absent. Individual phase commands are also available in the underlying modules. The tests reconcile the default delivered package. Never delete or overwrite a completed output to rerun. PDF rendering and visual QA use `pdftoppm`; rendered pages are temporary QA artifacts.

## Tables and units

- `frozen_constants.csv`, `initial_validation_candidates.csv`: all five initial calibrations, including inadmissible candidates. Numeric C0 for b<=1 is diagnostic extrapolation and not used by primary strategies.
- `candidate_losses.csv`: inner/full slopes and dates, single scheduled inner penalties, availability and historical candidate losses.
- `annual_policies.csv`, `annual_spectra.csv`, `lambda_decomposition.csv`: annual choices, penalties, complexity, exact sample/b/representation log decomposition and explicit fixed-b penalty counterfactuals.
- `monthly_payoffs.csv`, `monthly_accounts.csv`: raw and fixed-scale gross payoffs; strategy-specific actual-holdings gross, target-change cost sensitivity and exact self-financing net accounts.
- `performance_summary.csv`, `annual_performance.csv`, `cumulative_performance.csv`: annual/cumulative mean, volatility, Sharpe, loss, turnover, leverage and drawdown; gross and net common windows. Month labels are return dates; period labels are annual decision years.
- `paired_bootstrap.csv`, `interactions.csv`: conditional paired 95% circular-block intervals at 6, 12 and 24 months, 5,000 replicates; point comparisons are fitted-strategy differences, not causal effects.
- `exclusion_sensitivity_*`: outside-theorem positive-b extrapolation, gross only.
- `rolling_*`: window-specific pretest translations and monthly spectrum-only diagnostics; no monthly portfolio backtest.
- `performance_figure_data.csv`: exact primary gross/net wealth curves used by the performance figure. Annual and rolling diagnostic figures plot their exported path tables directly.
- `dense_lengthscale/`: separate exploratory denser representation search requested during execution, retaining the same initial calibration, spectral rule and trading implementation.

Response-one loss uses raw portfolio payoff units. For net inference, net excess returns are divided by the same fixed pretest exposure scale. Reported investment mean/volatility/wealth use that scale and the frozen cash series. Net account states begin in cash February 1978 and carry into subperiods, without terminal liquidation. Stock identifiers and holdings are excluded from public exports. Private coefficient and holdings hashes provide reconciliation receipts.

## Interpretation

The baseline regression is a finite-rank diagnostic, not identification of a population spectral exponent. The theorem assumes b>1; no clipping or CV replacement is performed. The initial calibration block also selects the anchor representation. Primary inference excludes 1978; the 1984 comparison removes early overlapping calibration/validation history. The extra denser grid is explicitly exploratory because it was proposed after seeing coarse-grid gross results. No OOS ranking selects a deployment hyperparameter.
