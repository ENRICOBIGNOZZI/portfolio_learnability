This delivery contains only the revised main Section VI, **Empirical Portfolio Learnability**. The theory, other manuscript sections, new theory-guided experiments, and pre-existing uncommitted work are outside this revision.

The compiled PDF and complete LaTeX source ZIP are in `publication/`. The wrapper starts at Section VI and Figure 3 so that the nine historical panels are Figure 6. It includes no appendix. The editable section, numeric macros, three tables, and five vector PDF figures are also available individually there.

The opening comparison uses Linear and the existing chronologically joint-validated Gaussian and Matérn-3/2 paths, on the same 564 realized months (February 1978–January 2025). Tuned stock holdings reproduce those managed returns; stock-level drift, trades, cash and short fees are recomputed. Every later spectrum, regularization profile and economic decomposition uses the original fixed Gaussian lengthscale and selected policy.

The diagnostic grid contains 1,522 policies per annual refit: all 120 original penalties, 1,401 additional positive penalties, and exact zero. It does not change validation eligibility or the selected policy. `ridge_monthly_paths.csv` contains every actual OOS evaluation. The relative eigenvalue cutoff is 1e-12; zero uses the symmetric Moore–Penrose pseudoinverse. Independent feature-space minimum-norm checks cover the first and final histories; all 47 endpoints have small training residuals and agree with the smallest positive-penalty evaluations. A constant interpolated historical payoff has unbounded historical Sharpe: no finite rounding artefact is displayed for that limit. All OOS endpoint Sharpes are observed and retained.

`tables/` contains canonical public aggregates, including costs, paired uncertainty, incremental spectral loss terms, figure observations and endpoint checks. `audit/inputs.json` binds the source exports, managed caches and private holdings to SHA-256 hashes; `audit/source_map.json` maps figures, tables and numerical prose to results. Licensed observations and individual stock holdings remain under the ignored `data/` and `results/` directories.

The complete-payoff sample depends on future-payoff availability. The formation-only audit leaves 7,489 unidentified payoffs among 2,504,402 eligible stock-months. Selection chronology does not remove this sample limitation. Costs are illustrative self-financing sensitivities, with upstream corporate-action treatment, rather than observed execution costs.

Reproduce with the repository's existing private inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m empirical_final.section_vi_finalize
```

Regenerate figures and compile from the verified public aggregates:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m empirical_final.section_vi_finalize --public-only
```

Compile the self-contained extracted source package with `tectonic Section_VI_Empirical_Portfolio_Learnability.tex`, or run `pdflatex` twice. Rendering uses Poppler's `pdftoppm`. Numerical verification runs during both reproduction commands; the final visual review is recorded separately in `audit/visual_review.json`.
