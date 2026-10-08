# Canonical experiment and reference reconciliation

The canonical implementation remains `portfolio_learnability` at baseline commit
`536b0736ae548370592273ad8667c04b547f2edb`. The new empirical consolidation is
isolated; all pre-existing results and theoretical source files are preserved.

The supplied request ends in section E3, mid-sentence. The missing continuation
was requested; the executed scope is the received E1-E3 specification. The user
explicitly authorized sensitivity analyses under declared cost assumptions.

## Manuscript reference and historical archive

`Portfolio_Paper (24).pdf` reports gross Sharpes of approximately 3.20, 3.35 and
3.25 (Linear, Gaussian, Matérn) and excess-return wealth. Those values are from
a different historical experiment. They are not targets for this reconstruction.
The current canonical output instead has gross Sharpes 3.0911, 3.2999 and 3.4610
and wealth including the same frozen cash return exactly once.

Page 62 of the reference supplies the sensitivity assumptions: 25 basis points
per unit of turnover and 30 basis points per year on short notional. They are
not stock-specific observed execution or lending costs. Our main accounting
measures drift-adjusted traded notional, charges both purchases and sales, and
solves the self-financing execution-cost equation. Target-weight turnover is
also retained as a separate diagnostic. Historical numerical net-return figures
from the reference are not imported.

`protfolio (11).zip` contains 144 archive entries spread over multiple paper
versions, including Portfolio 2, Portfolio 4 and Final_version 9. Its numerical
CSVs and figures are not merged across experiments. Existing baseline figures,
local-window figures and their methodologies were inspected before new figures
were designed. Earlier local-window uncertainty bands are not reused.

## Secondary repository

The inspected `PORTFOLIO-PAPER` revision is
`42f9646a6781d12bb171d5aa9d110e1e0d3e3d1d`. Its CTF module correctly expresses
the normalized dual response-one solve using `F F'/T`, which is checked here
against the canonical fits at five historical cutoffs. This is an algebraic
cross-check using the canonical real-data managed matrices, not a rerun or a
validation of the full CTF research experiment.

Its defaults differ materially: paired cosine/sine features, a different
feature-count convention, percentile-rank centering, and a Linear map without
the canonical explicit intercept. Its raw preprocessing and ex-post diagnostic
best-index helper are therefore not adopted. No CTF scoring, new strategy,
bootstrap or leaderboard claim is introduced.

## Scope of the evidence

The 564 OOS observations are February 1978-January 2025 for every baseline policy
and every nested spectral diagnostic. Annual decisions occur at January
formation closes; the December formation payoff is known at that close. Earlier
local-window outputs covering January-December realized returns are not pooled
with these annual windows. All baseline comparison cells are reconstructed from
the actual stored weights, selected coefficients and managed-payoff histories.

The inherited sample excludes 7,489 unknown next-month returns before ranks and
N_t are constructed. The new analyses therefore remain complete-payoff sample
sensitivities. Neither successful accounting checks nor small missing-return
counts establish historical-vintage validity or identify the missing-payoff
performance bias. This package does not claim submission readiness.

Kozak, Nagel and Santosh (2020), *Shrinking the cross-section*, motivate studying
signal in managed-portfolio principal components and shrinking low-variance
directions. Our diagnostic uses the uncentered response-one second-moment
operator and is not their centered covariance/SDF estimator. Incremental losses
retain OOS cross terms; Sharpe differences are not additive contributions.
Sources: https://doi.org/10.1016/j.jfineco.2019.06.008 and
https://www.nber.org/papers/w24070.
