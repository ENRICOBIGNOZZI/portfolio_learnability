# Final empirical consolidation: E1–E3

Canonical baseline: `536b0736ae548370592273ad8667c04b547f2edb`.
Final publication revision: **`publication/v2`**. The received request is archived
in `notes/request.txt`; it ends mid-sentence in E3. No later experiment is inferred.

## Deliverables and interpretation

- `publication/v2/manuscript_real_data.pdf`: the original theory and proofs with
  a new real-data section and accounting/spectral appendix. The wrapper excludes
  the postponed simulation sections. Original `paper/main.tex` is unchanged.
- `publication/v2/figures`: five vector PDF figures and matching PNG previews.
- `publication/v2/performance_table.tex`: one consolidated E1 performance table.
- `publication/v2/spectral_table.tex`: nested OOS losses and nonadditive Sharpe comparisons.
- `tables/`: aggregate numerical evidence, with no licensed stock identifiers or holdings.
- `audit/`: frozen protocol, accounting, spectral and missing-payoff checks, final verification.
- `notes/reference_audit.md`: historical PDF, archive and secondary CTF reconciliation.

All comparisons use February 1978–January 2025 (564 months). Linear retains
131 coefficients including the intercept; each nonlinear model retains 10,000
fixed seed-zero features. No bootstrap, new confidence band, inference claim,
ex-post model selection, or new synthetic research experiment is introduced.

| Policy | Gross Sharpe | 25 bps trading | Trading + 30 bps/year borrowing |
|---|---:|---:|---:|
| Linear | 3.091 | 2.511 | 2.476 |
| Gaussian | 3.300 | 2.644 | 2.605 |
| Matérn-3/2 | 3.461 | 2.766 | 2.725 |

Costs are **explicit hypotheses**, authorized by the user, not observed execution
or lending fees. Two-sided trades use actual stored weights and drift; execution
costs solve a self-financing equation. Cash is included once. Initial entry is
charged; there is no forced final liquidation. There is no inferred impact model,
capacity estimate, or extra financing spread.

E2 fixes cumulative training-rank fractions at 10%, 50%, 90%, and full before
computing the new OOS comparisons. The last group lowers Gaussian response-one
loss by 0.00312 but raises Matérn loss by 0.00114. Sharpe changes include covariance
effects and are not additive contributions. Nested net returns are not inferred:
only full-policy stored stock weights support E1 cost accounting.

E3 reports actual annual C and C/T. Gaussian C/T rises 22 times and falls 24 times;
Matérn rises 20 times and falls 26 times. No monotone path or fixed-b rate is imposed.
The descriptive b fit is not a population exponent estimate with inference.

**Identification limits remain:** 7,489 forward payoffs are unavailable and the
canonical sample removes them before ranks and N_t. They represent 0.299% of
otherwise eligible stock-months, but their strategy-return bias is unidentified.
The JKP snapshot is retrospective, not verified historical-vintage information.
The output is therefore an auditable conditional backtest and cost sensitivity,
not a submission-readiness claim or a verified implementable trading record.

## Data lineage

| Aggregate file | Unit and use |
|---|---|
| `monthly_accounting.csv` | Model × scenario × payoff month; E1 fees, drift-adjusted turnover, exposures and returns |
| `performance_scenarios.csv` | Model × scenario summaries; E1 table |
| `annual_complexity.csv` | Model × January decision; 141 fits, T, lambda, C/T, dates, concentration, slope, validation and OOS metrics |
| `managed_spectra.csv` | Model × five historical cutoffs × positive rank; E2 spectrum figure uses 1978, 2000 and 2024 |
| `nested_spectral_monthly.csv` | Model × four frozen stages × payoff month; selected filters, counts, gaps and payoff increments |
| `spectral_contributions.csv` | Model × stage; E2 loss identity with cross terms and nested-policy performance |
| `missing_payoff_materiality.csv` | Formation month; counts and identified market-cap share, without imputation |

The phase reports bind numerical source files and private input checksums. The
render manifest binds figure/table inputs and all generated sources. The build
report binds theory, bibliography, empirical sources and the final PDF. The final
verifier checks preservation of 161 pre-existing tracked files outside the
postponed directory, and independently reconstructs aggregate accounting/loss
identities. Private data access remains required for the original stock-level
reconstruction; published aggregate checks alone cannot validate missing outcomes.

## Reproduction, from repository root

Use the repository Python dependencies and a TeX installation with `latexmk`.
The original run used Python 3.13.3. The numerical wrapper forces one thread and
processes kernels and yearly panels sequentially. It refuses an existing output
folder; the renderer and final PDF builder also refuse completed destinations.

```sh
# Full private-data reproduction into a NEW destination:
python3 -m empirical_final.reproduce --out outputs/final_empirical_rerun
python3 -m empirical_final.build --publication outputs/final_empirical_rerun/publication/v1
python3 -m empirical_final.verify --out outputs/final_empirical_rerun --revision v1

# Verify the delivered package with the local authorized input files:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -m empirical_final.verify --revision v2

# Aggregate/source checks in a checkout without licensed data:
OPENBLAS_NUM_THREADS=1 python3 -m empirical_final.verify --revision v2 --public-only

# Re-render from the delivered aggregate CSVs into a NEW publication revision:
OPENBLAS_NUM_THREADS=1 python3 -m empirical_final.render --revision v3
python3 -m empirical_final.build --publication outputs/final_empirical_20261008/publication/v3

# Focused local tests; no postponed experiment tests or workflows:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_final_empirical.py tests/test_pipeline.py tests/test_audit.py
```

The original `empirical_final.run` default paths are deliberately immutable;
use `reproduce` for a fresh run. Figure layout v1 is retained for provenance;
v2 fixes a long diagnostic axis label. It does not alter numerical results.
Existing GitHub workflows execute postponed experiments, so this empirical
change must not trigger them; local empirical verification is recorded instead.
