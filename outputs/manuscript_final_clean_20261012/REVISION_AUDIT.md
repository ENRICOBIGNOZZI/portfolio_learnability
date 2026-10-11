# Manuscript revision audit — 12 October 2026

The baseline is the newest integrated manuscript,
`Portfolio_Paper_con_Empirics_Aggiornata.zip`, downloaded on 12 October 2026.
Its exact original archive is retained in `sources/`; its SHA-256 and source
comparisons are recorded in `audit/source_preservation.json`. This revision
leaves the older repository `paper/` project and unrelated ongoing simulation
work untouched. All historically published Section VI outputs remain unchanged.
This version integrates the canonical empirical results from
`outputs/section_vi_final_20261011/` without running new experiments.

## Scientific correction

Reconstruction from the saved validation losses confirms all 47 original
past-only CV choices and their 564 monthly OOS payoffs. Each fixed-candidate
curve uses the same candidate penalty across a block's annual refits, whereas
the implemented CV policy changes penalties across refits. Every old diamond
matches that dynamic CV policy; none matches a fixed candidate's block-level
coordinates. This is an expected difference between policies, not a numerical
or indexing error. Figure 6 therefore omits the diamonds and their legend item.
Figure 5 retains its independently verified 2024 diamond: candidate 465,
complexity 11.3912682700898, OOS Sharpe 1.83216307909457.

Figure 6 still draws all 1,522 evaluated candidates in each of nine panels:
13,698 observations, including every exact minimum-norm zero endpoint. Maxima
are recalculated over the entire grid. Sharpe uses concatenated monthly
returns; complexity uses evaluated-month weights. There is no smoothing,
extrapolation, path simplification, or artificial downturn. See
`audit/figure6_scientific_audit.json` and `section_vi/audit/figure6_render.json`.

## Sources and removed material

- Changed the Figure 6 generator in `empirical_final/section_vi_finalize.py`
  and its caption and introduction in `empirical_final/templates/final_section_vi.tex`.
- Added `empirical_final/figure6_audit.py`, `empirical_final/manuscript_cleanup.py`,
  and scientific and manuscript acceptance tests in `tests/test_manuscript_cleanup.py`.
- Synchronized the versioned publication and integrated `Portfolio/empirics.tex`
  and `figure6_caption.tex`. Preserved the integrated manuscript's float placement.
- Deleted only the obsolete cost discussion and figure in `Portfolio/appendice_2.tex`,
  immediately before the factor-economy subsection. Removed its old gross Sharpes
  3.20/3.35/3.25 and superseded commission/borrowing estimates.
- Excluded six unused assets from the final project: `equity_with_costs.pdf`,
  `cumulative_wealth.pdf`, `complexity_gaussian.png`, `complexity_matern32.png`,
  `economic_spectrum.png`, and `gross_and_net_exposure.png`. Original research
  outputs and the original manuscript archive are preserved.
- Resolved the inherited literal `table ....` placeholder in the opening of
  `Portfolio/direct_port_learning.tex` to the existing portfolio-representation
  table. This is its sole change; all mathematical content is byte-identical.

All other theory sources, bibliography, `main.tex`, theoretical illustrations,
and the exact appendix byte spans before and after the deletion are unchanged.
The portfolio-representation and learning-rate tables, propositions, theorem
statements, proofs, and mathematical cross-references remain intact.
Source diffs and SHA-256 checks are in `audit/manuscript_source.diff`,
`audit/generating_source.diff`, and `audit/source_preservation.json`.

Every empirical CSV remains byte-identical, including the ridge monthly and
annual paths, nine-panel curves and summary, representation accounts and
performance, cost accounts, spectral decomposition, paired bootstrap results,
and missing-payoff audits. Figures 3, 4, 5, and 7 remain byte-identical in both
PDF and PNG. The jointly validated comparison still reports gross/net Sharpes
3.091/2.476, 3.646/2.857, and 3.587/2.747. The fixed Gaussian analysis remains
separate, and the limitation of 7,489 unidentified payoffs remains explicit.

## Compilation and verification

The complete paper has 66 pages and standalone Section VI has eight. Both
compile with bibliography and resolved inputs, figures, tables, labels, and
cross-references. The actual final ZIP was extracted into a separate directory;
both entry points compiled there and their extracted text matches the delivered
PDFs. The original full paper had 67 pages. The five-figure empirical sequence
and original 12-point, 1.5-spaced full-paper format are preserved.

All empirical pages and all eight standalone pages were rendered and visually
reviewed, including Figure 6 at 150 dpi and the appendix deletion boundary.
Markers, period labels, tails, captions, and tables are readable and unclipped.
The full paper retains the original 7.09189-point overfull-box warning in an
unchanged proof on page 49; inspection confirms visible text without clipping.
There are no undefined commands or references, or empirical overfull boxes.

The requested four existing test files plus `tests/test_manuscript_cleanup.py`
pass: **35 passed**. Evidence: `audit/pytest_results.txt`, `audit/compilation.json`,
`audit/manuscript_dependencies.json`, and `audit/visual_review.json`.

Reproduce in the repository: `python3 -m empirical_final.manuscript_cleanup`.
Compile the ZIP independently inside `Portfolio/`: `tectonic main.tex` and
`tectonic Empirics_only.tex`. See `Portfolio/README_BUILD.txt` for alternatives.

Git source-and-deliverables commit: recorded after committing, in the audit-only follow-up.
