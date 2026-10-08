# Execution status and completion checklist

Completed and independently verified on 7 October 2026. The full request is preserved in `implementation_request.txt`.

- [x] Remove 75 legacy simulation paths; retain the requested cleanup manifest.
- [x] Implement the canonical balanced three-factor DGP and derive its population optimum.
- [x] Validate E1–E6, exact balance, conditional pricing, Gaussian MGF and fourth moments.
- [x] Implement the paper estimator and chronological validation with no oracle leakage.
- [x] Run and independently verify the smoke profile.
- [x] Run 500 baseline and 200 replications for each of 15 robustness cases: 3,500 total.
- [x] Verify identities on all 7,560,000 simulated dates.
- [x] Produce population/OOS curves, signed regret decomposition, benchmarks and uncertainty.
- [x] Estimate rates on all three predeclared windows with 1,000 whole-path bootstrap draws.
- [x] Pass paired rank sensitivity: maximum change 4.64%, against the fixed 10% tolerance.
- [x] Pass population quadrature sensitivity: maximum change 1.58%, against 5%.
- [x] Produce 66 PDF figures and 11 CSV tables, including all five requested main tables.
- [x] Preserve 11 imported theoretical entries and 112 protected empirical files byte-for-byte.
- [x] Integrate simulations into the complete manuscript; compile and visually review all 91 pages.
- [x] Pass final closeout; document exact reproduction commands and remaining interpretive limits.

Evidence: `outputs/paper/audit/final_report.json`, `outputs/paper/audit/final_numerical_verification.json`, `outputs/paper/paper/visual_review.json`, and `outputs/audit/test_verification.json`. All 123 repository tests passed; the nine pipeline tests were also rerun after the audit-cache correction. The smoke profile passes with final source hashes. `git diff --check` passes.

The complete PDF is `outputs/paper/paper/The_Law_of_Portfolio_Learnability.pdf`; a readable final report is `outputs/paper/audit/final_report.txt`. Reproduction commands are in `../README.md`.

## Provenance and limits

The repository initially contained empirical fragments. The imported theory is `/Users/enrico/Downloads/Portfolio 8`, the only available source with the exact managed quadratic-MGF E2 in the supplied brief. This source selection is a disclosed inference; explicit user confirmation was not received. Its theoretical source bytes were preserved. The older source named in the historical empirical manifest uses an incompatible bounded-return proof and was not adopted. No theoretical formula was changed to resolve this version mismatch.

The results use a finite Nyström representation and population quadrature, each with a separately passed sensitivity gate. Near-unregularized curves remain rank-sensitive; the rank acceptance concerns oracle/selected regret rather than uniform agreement across the whole penalty grid. Spectral slopes are finite-resolution diagnostics. A fixed smooth target need not attain minimax rate bounds with equality: the baseline oracle-regret slope is -0.518 on the full grid and -0.692 on its upper half, versus the -0.600 theoretical benchmark. All predeclared windows, boundary choices, and configurations are retained. Empirical ranks and heteroskedasticity are explicitly labeled departures from specified baseline assumptions.

Execution used deterministic checkpoints and a local Python bytecode cache to avoid iCloud placeholder stalls. Two cached alternate-kernel preflights initially had 12 states; final verification required 24, so they were extended and passed without changing any simulation draw or acceptance threshold. Final publication edits were followed by regenerated outputs and independent verification. Only the spectral-table page changed in the last PDF revision; the other 90 pages were pixel-identical to the prior reviewed render.
