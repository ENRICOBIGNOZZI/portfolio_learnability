# Requirement-by-requirement completion matrix

Status records evidence rather than intentions. Paths below are relative to `simulations/` unless otherwise stated.

| Request section | Requirement | Current state | Evidence / verification target |
|---|---|---|---|
| 0 | Preserve current theory and notation | Verified; source selection remains a disclosed inference | manuscript_preservation.json; all 112 empirical hashes verified |
| 1 | Remove all legacy simulations | Complete and independently verified | simulation_cleanup_manifest.md; cleanup_inventory.json; independent smoke verifier |
| 2 | Canonical three-factor DGP | Complete and independently verified | dgp/balanced.py; baseline audit; unit tests |
| 3 | Derive population optimum | Complete and independently verified | docs/dgp_methodology.md; conditional identities; tests |
| 4 | E1–E6 audit | Complete and independently verified | outputs/audit/baseline; paper/data/*/environment_audit.json |
| 5 | Exact estimator and approximation diagnostics | Complete and independently verified | Full numerical verification; rank and quadrature sensitivity reports |
| 6 | Loss/Sharpe versus effective complexity | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 7 | Four-panel main figure | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 8 | Population and empirical complexity figures | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 9 | Bias/estimation/regret decomposition | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 10 | Wide-grid rates and bootstrap intervals | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 11 | Chronological validation and refit | Complete and independently verified | estimator/ridge.py; leakage tests; saved validation losses |
| 12 | N robustness | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 13 | Signal/noise robustness | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 14 | Persistence robustness | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 15 | Spectral-b robustness | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 16 | Misspecification robustness and assumption status | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 17 | Six meaningful benchmarks | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 18 | 500 headline and 200 per robustness, explicit seeds | Complete and independently verified | 3,500 saved and verified replications; 7,560,000 dates |
| 19 | MC SE, curve CI, whole-path bootstrap | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 20 | All main and appendix figures | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 21 | Five requested tables | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 22 | Acceptance gates and all-date identities | Complete and independently verified | Full numerical verification; rank and quadrature sensitivity reports |
| 23 | Single canonical subtree | Complete and independently verified | simulations/; no legacy engines remain |
| 24 | Reproducible one-command profiles and metadata | Complete and independently verified | outputs/paper/data, figures and tables; final_numerical_verification.json |
| 25 | Paper integration and complete compiled paper | Complete and independently verified | final_report.json; manuscript_compilation.json; all 91 pages visually reviewed |
| 26 | No ex-post design/seed/range changes | Complete and independently verified | Frozen scientific config/source hashes; all grid points retained |
| 27 | Final deliverables and report | Complete and independently verified | final_report.json; manuscript_compilation.json; all 91 pages visually reviewed |
| 28 | Required execution order | Complete and independently verified | Execution records and preflight reports |
| 29 | Requirement-by-requirement completion audit | Complete and independently verified | final_report.json; manuscript_compilation.json; all 91 pages visually reviewed |

Final acceptance evidence: `outputs/paper/audit/final_report.json` and the readable `final_report.txt`. See `execution_status.md` for provenance and finite-resolution limitations.
