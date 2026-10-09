# Discovery

Repository: git@github.com:ENRICOBIGNOZZI/portfolio_learnability.git, local `/Users/enrico/Desktop/PHD/portfolio/paper_codice`.
Initial HEAD: `6c72c9badee2f7824635b399100066bf7cad2367`. Initial working tree had an unrelated untracked `.gitignore 2`, left untouched. No applicable AGENTS.md was found from filesystem root through this repository or inside it. No branches/checkouts/remotes/CI changes.

User destination: `non stationarity/adaptive_memory_equities/`. The attachment asked for a root-level package; the user's explicit enclosing-folder instruction takes precedence. Imports/CLI use `PYTHONPATH='non stationarity'` from the project root.

Read the full prompt TXT and byte-identical MD, all extracted text of Portfolio_Paper_Final.pdf (51 pages), protfolio (15).pdf (46 pages), Adaptive_Memory_Deep_NN_FINAL.pdf (22 pages), README, data_pipeline.py, audit_formation_timing.py, run_empirics.py, empirical_closeout.py and relevant tests/manifests. Extracted texts are private local artifacts. The matching adaptive-paper code was not found under the document's name; no unrelated code archive was substituted. PDF versions remain distinct.

The final portfolio PDF uses fixed N for its proof, response-one and Sharpe at page 8, annual calendar-year evaluations at page 24, and unresolved-payoff limits on pages 29-30. The earlier draft explicitly uses N_t at page 6, has different historical tables and a weaker stated Sharpe bound. This implementation uses N_t operationally without importing fixed-N theorems. The adaptive regression PDF studies independent regression with drift in the conditional mean, bounded outputs and a different guard; its simulation results are not equity evidence and its clipping/projection are not inherited.

Hardware: `macOS-26.2-arm64-arm-64bit-Mach-O`, 8 logical CPUs, 8 GiB unified memory. Installed torch 2.9.1 reports MPS available; CPU float64 and one thread are deliberately used for reproducible numerical checks. Initial swap used about 5.07 GiB: resource monitoring is material. Development training peak 399.5 MiB, full audit peak 1534.8 MiB; exact logs in private_runs.

Original safe functions reused: `data_pipeline.formation_mask`, `rank_months`. Original `load_panels` rejects incomplete payoffs and describes the retrospective complete-payoff convention, so it is not used as a formation-only loader. New PanelStore reads existing verified annual clean or formation-only files and creates private row-contiguous annual memory maps with monthly views. Original loader and snapshot are not modified.

Data paths: `data/raw/manifest.json`, 63 annual/terminal parquet files, 3,687,389 raw stock-months, January 1963-January 2025; acquired 2026-10-01. Raw fingerprint `01dc5d2771a9426534526bc963a44ccb31f437f694df1205c0c4bd79abe9e676`. Existing `data/clean` has 62 files and 2,496,913 retained stock-months; `data/formation_only` has 62 files and 2,504,402 stock-months. Same-snapshot cash is in `data/risk_free.csv` with its own checksum/source manifest. No downloads or external data services were used.

Original schema includes id, permno, eom, excntry, size_grp, me, common, primary_sec, obs_main, exch_main, crsp_shrcd, crsp_exchcd, ret_exc_lead1m, current_excess_return and current_total_return. Prepared files contain id/eom/return_date/r plus ranked features. `ret_exc_lead1m` is already next-calendar-month excess, decimal units. No second shift or second risk-free subtraction.

Feature allowlist: 130 actual selected names from `characteristic_selection.json`, frozen 1963-1972 from the fixed 153-characteristic dictionary. Hash `d043f8a2e7fdbf55b0ee46beb133ac8c6af7e197629d124f5403b6ae102d6f08`. The complete ordered list is in every run source.json. No IDs, future-availability variables, target returns or normalized calendar time enter the NN. Rank transform: observed-value average ranks mapped to [-0.5,0.5]; singletons and residual missing features neutral zero. Remove >30% missing before ranks. USA/common/primary/main flags, CRSP codes and non-nano rules are inherited exactly.

Calendar discrepancy is explicit: original equities protocol uses January formation-close annual refits and February-January payoffs, while the final PDF and local_learnability default use December information and January-December payoffs. The new experiment inherits the original January-close convention. Common dates are selected by actual realization date, never by a year label.

The new audit recomputed source counts and exact next-calendar matches. Historical database vintages, publication lags of revisions and discovery dates of the retrospective feature dictionary remain uncertified. Licensed raw data and stock-level artifacts never leave local ignored storage.
