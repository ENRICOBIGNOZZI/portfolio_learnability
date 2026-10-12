# Annual Figure 6 manuscript revision - 12 October 2026

This version aligns Figure 6 and Table II with the same nine annual decision years in
`outputs/manuscript_final_clean_20261012/Portfolio`. The obsolete paragraph
explaining why annual CV could not be plotted on a pooled-block curve is
absorbed into the revised introduction. The build README points to the new
reproduction command. Table II now reports the annual maxima and zero endpoints displayed in Figure 6,
and its following descriptive summary uses these annual values. Theory, proofs,
bibliography, other tables, other figures, and the remaining empirical prose
are unchanged.
Baseline files and unrelated ongoing work are preserved.

## Deliverables

- `Portfolio_Paper_Annual_Figure6.pdf`: complete manuscript, 66 pages.
- `Empirics_Section_VI_Annual_Figure6.pdf`: standalone empirical section, 8 pages.
- `Portfolio_Paper_Annual_Figure6_LaTeX.zip`: self-contained compilable LaTeX project.

Figure 6 is on page 34 of the complete manuscript and page 5 of the standalone
section. It displays exactly nine annual paths: 1980, 1985, 1990, 1995, 2000,
2005, 2010, 2015, and 2021. All requested years are available; no substitution
was necessary. There are 1,522 evaluated penalties per annual path (13,698
vertices), including the original positive penalties, extended near-zero
positive penalties and exact minimum-norm zero endpoint. Every vertex is
retained, with no smoothing, extrapolation or plotting-path simplification.
The nine panels share linear axes covering all observed paths and endpoints.

Each annual Sharpe is independently reconstructed as sqrt(12) times the mean
of the year's 12 monthly returns divided by their sample standard deviation
(ddof=1). The OOS window is February of the decision year through January of
the next year. Complexity uses that decision year's saved effective-complexity
value, never a block mean. The fixed Gaussian lengthscale is 4.392107784616423,
with the original feature bank and seed 0.

Black diamonds are checked against the original prior-validation loss
minimizers, saved selected monthly returns and gross-accounting Sharpes.
Every diamond lies on its annual path. Orange stars use the maximum over all
1,522 annual candidates, including zero. Open circles show exact zero-penalty
endpoints at the annual effective rank. Full numerical checks, original input
hashes and per-year marker coordinates are in `audit/annual_profiles.json`,
`audit/annual_curves.csv`, and `audit/annual_summary.csv`. No backtest was run.

Table II reports the same selected years as Figure 6. Its columns are decision
year, annual complexity at the retrospective maximum, maximum annual OOS
Sharpe, annual zero-endpoint complexity, zero-endpoint OOS Sharpe, and the
peak-to-zero Sharpe decline. The caption, note and following descriptive summary
now use annual values. No multi-year complexity averages or pooled Sharpes
remain in this table. Only the HistoricalFindings macro in numbers.tex changes;
all other reported numbers and both other tables are byte-identical.

## Reproduction and checks

From the repository root:

    python3 -m empirical_final.figure6_annual

To compile the ZIP independently, extract it, enter `Portfolio/`, then run:

    tectonic main.tex
    tectonic Empirics_only.tex

Both entry points were compiled directly and from the actual ZIP extracted
into a separate directory. All dependencies and references resolve, and the
independent extracted builds reproduce the delivered PDF text. Source-byte
preservation, the permitted prose diff and compilation receipts are in
`audit/source_preservation.json`, `audit/manuscript_source.diff`,
`audit/dependencies.json`, and `audit/compilation.json`.

All eight standalone pages and full-manuscript pages 33-35 were rendered and
visually checked. The annual figure's panel labels, common scales, full tails,
three markers per panel, and caption are readable and unclipped. The complete
manuscript retains the baseline's 7.09189-point overfull box in an unchanged
proof; the empirical section has no overfull boxes or unresolved references.

The annual reconstruction, marker geometry and Table II coordinate tests, narrow source/archive
preservation tests and five existing manuscript/empirical test files pass:
26 tests total. See `audit/tests.json` and `audit/visual_review.json`.
