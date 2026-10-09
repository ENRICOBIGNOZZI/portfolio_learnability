# Execution and evidence

All changes are confined to `non stationarity/adaptive_memory_equities/`. The full prompt (TXT and identical MD) and the three PDFs were read. No original tracked artifact or the preexisting untracked `.gitignore 2` changed. No branch, commit, push, order, publication, cloud purchase or external data download was made.

## Completed work and limits

| Component | Status | Actual evidence |
|---|---|---|
| Direct deep NN and temporal memories | IMPLEMENTED / TESTED | All hidden layers train; monthly response-one; no network penalty; training-only scalar calibration |
| Independent raw audit | PAYOFF_COMPLETENESS_FAILED | 63 verified source files; 744 months; 2,504,402 formation stock-months; 130 characteristics |
| Formation-only run attempt | EXTERNAL_BLOCKER | Explicit failure on unidentified historical portfolio payoffs; no imputations or after-decision renormalization |
| Final real smoke | REAL_RUN_COMPLETED / RETROSPECTIVE_ONLY | 12 seed-specific fits, 2 January origins, 3 experts, 24 payoffs: February 1978–January 1980 |
| Reduced real comparison | REAL_RUN_COMPLETED / RETROSPECTIVE_ONLY | 414 fits, 3 origins, 35 payoffs: February 1993–December 1995 |
| Reduced common-calendar slice | RETROSPECTIVE_ONLY | Exactly 24 payoffs: January 1994–December 1995; 11 earlier startup observations |
| Stationary/break controls | SYNTHETIC_RUN_COMPLETED | 360 actual fits, 4 scenario streams × 180 payoff dates; two independent innovation seeds, paired across scenarios |
| Legacy gross references | PREEXISTING_VERIFIED_OUTPUT | Linear, Gaussian, Matérn-3/2; existing seed-0 outputs, no new fits |
| Full 1994–2024 comparison | NOT_RUN | No 372-month result; full grid plans 31,104 fits before exclusions/deduplication |
| Terminal 2020–2024 | NOT_RUN | No new terminal evaluation; historical use prevents a pristine-holdout claim |
| Legacy net-cost comparison | NOT_RUN | Only verified raw-scale, zero-cost legacy payoffs imported |

The raw audit **recounted 7,489 unresolved next-month payoffs in 638 months**. Exact next-calendar matching recovered zero. There were 2,496,567 finite lead/current matched pairs and zero discrepancies at tolerance 1e-8. Perturbing all future returns left formation ranks unchanged. Complete formation-only evidence requires additional identified source payoffs; historical database vintages also remain uncertified. This is a data limitation, not permission to call the complete-payoff sensitivity point-in-time.

The reduced run retained all nine architectures: affine and depths 1/2/3/4 at widths 32/64, both neural seeds, and the same fixed optimizer budget for every expert. It saved 72 expert labels per month. The first origin had 63 distinct policies before the two-seed ensemble (rolling 360 duplicated uniform); later origins had 72. Thus the actual seed-fit count is 126+144+144=414. Theory tau=10 was excluded at all three origins for effective history below 36. Exclusions and aliases are recorded in `origins/`; planned candidates are not counted as completed fits.

## Same deep architecture: memory comparison

NN fixed to **two hidden layers of width 32**, seeds 0/1 averaged in holdings, January 1994–December 1995, 24 monthly payoffs, raw scale 1, zero trading cost. The memory rows below are unguarded.

| Method | Annualized Sharpe | Response-one loss |
|---|---:|---:|
| exponential/unguarded | 2.8104 | 0.767055 |
| full | 1.3920 | 0.931473 |
| rolling/unguarded | 2.0599 | 0.836828 |
| taper/unguarded | 3.7774 | 0.804493 |
| theory/unguarded | 1.5076 | 0.898405 |

Every guarded method equals full history because fewer than 60 mature prequential observations exist. This run does not establish market efficacy of the guard. Soft aggregation is saved separately. Sharpe and response-one rankings can differ OOS: calibration is training-only, and no ex-post scale optimization is performed.

![Fixed deep architecture](../private_runs/reduced/report/fixed_L2W32_comparison.png)

## Joint architecture/memory selection

| Method | Annualized Sharpe | Response-one loss |
|---|---:|---:|
| exponential/unguarded | 3.5898 | 0.548818 |
| full | 1.3920 | 0.931473 |
| rolling/unguarded | 2.6517 | 0.651326 |
| taper/unguarded | 3.6103 | 0.479958 |
| theory/unguarded | 2.9729 | 0.574387 |

The warmup baseline is fixed L2W32. Joint theory and exponential choices use the affine control on all 24 common dates; joint taper uses it on 22 dates and L1W32 on two; joint rolling uses it on 21 dates and L1W32 on three. Those joint gains cannot be attributed to memory alone. The fixed-architecture results above and the complete nine-architecture CSV must be considered alongside them.

Adaptive memory **loses on response-one in 10 of 36 fixed-architecture/family comparisons**. Examples: affine rolling increases loss by 0.193227 and reduces Sharpe by 1.638697; four-layer width-32 taper increases loss by 0.021131 and reduces Sharpe by 0.458840. These comparisons are reported, not removed. The existing affine full-history control has Sharpe 4.3331 on this slice, above every unguarded joint memory choice.

Verified legacy gross Sharpes on the same dates are 8.0220 (linear), 10.5589 (Gaussian), and 10.8989 (Matérn-3/2). The new short-budget NN run does not beat them. These are existing seed-0 specifications, not newly fitted deep networks, not a matched optimizer experiment, and not evidence about the full 31-year period.

Paired and simultaneous bootstrap intervals use identical circular 12-month blocks and 2,000 draws. The slice contains only two blocks; intervals are approximate retrospective diagnostics with limited resolution. Fixed-architecture multiplicity adjustment is within each architecture scope, not over all nine scopes together. No new hyperparameter, calendar, seed or training-budget choice was made from these OOS results.

## Costs and risk

All methods retain their raw training scale. On the common slice, joint theory and taper have average gross stock exposure 31.6123 and 38.5514, versus 3.6855 for full history. Mean monthly traded stock notional is 16.1626 and 21.4927, versus 1.2938. Their raw drawdowns are about 65.9% and 42.2%, versus 28.4%. These exposure differences are visible in the accounting CSV; Sharpe alone is incomplete.

At 25 bps per traded stock notional, joint Sharpes become 1.2784 (full), 2.6394 (theory), 3.1909 (taper), 2.2644 (rolling), and 3.1567 (exponential). Turnover includes both trade legs without a factor 1/2. Stock drift uses excess plus same-snapshot cash; the separate short-borrowing sensitivity is 30 bps/year. These are post-fit sensitivities, not cost-optimal policies. Wealth stops if insolvent; no risk cap or silent clipping is introduced.

## Synthetic guard evidence

These are mechanism controls with 24 synthetic stocks and 3 inputs, not additional market observations. Stationary/break paths share innovations within each of two independent seed blocks. Synthetic stress memories do not use the equity n_eff>=36 admissibility filter; their design is recorded in `synthetic/source.json`.

| Scenario / replication | Activation over 180 payoff dates | Guard minus full loss | First new activation after break |
|---|---:|---:|---|
| Stationary / 0 | 0% | 0 | Not applicable |
| Stationary / 1 | 11.11% | +0.002702 | Not applicable |
| Break / 0 | 20.56% | -0.007390 | 11 months |
| Break / 1 | 0% | 0 | None in 90 post-break months |

There are only 120 guard-eligible dates per stream after the 60-score warmup; stationary replication 1 activates on 20/120 of those dates. Its Sharpe falls from 1.3801 to 1.1446. Break replication 0 improves loss after the break by 0.014781, but replication 1 never activates. The controls exhibit both stationary cost and detection delay/failure; two seed blocks cannot establish coverage or a general guarantee.

## Verification and versions

- Initial full test invocation: **124 passed**, comprising 88 inherited tests and 36 initial new tests (`private_runs/all_tests.log`).
- Final current-package checks: **38 passed** in `tests_final_logic.log` plus **4 passed** in `tests_final_integrity.log`, for **42 distinct new tests**. Together with the unchanged inherited suite, 130 distinct tests passed across the recorded invocations; this is not a claim of one 130-test invocation.
- Final real smoke resume: zero additional fits; all checkpoint/log, decision, event and holdings bytes unchanged (`smoke_final/resume_verification.json`). Interrupted/resumed fixture results also match exactly.
- Final versus initial smoke: all stock weights, scores, seeds, IDs, returns and reporting scales agree within 1e-10; maximum absolute difference **7.1054e-15**.
- Reduced-run source archive checksums verified. Core learning modules are unchanged from its executed version. All 35 historical selections and gates replay exactly under the final selector (`reduced/execution_verification.json`).
- The final review made baseline architecture history independent of newly admissible adaptive candidates, handled undefined Sharpe comparisons explicitly, and avoided repeated decompression of previous holdings. The short real pilot remains in guard warmup, so this selector correction does not change its decisions. The synthetic source records paired rather than falsely independent scenario streams.
- Initial, reduced-run and final source versions are archived under `private_runs/code_snapshots/<code_hash>/`. The resume signature intentionally rejects another source version. Figures were rendered from saved results and visually inspected.

Ten fixed Adam updates, eight complete sampled dates per update, are a computational pilot budget. All neural layers train, and calibration optimizes the final scalar over every positive-weight training date. Neither this budget nor its test coverage certifies convergence, a global nonlinear optimum, or OOS improvement.

## Resource record

One Torch compute thread, CPU float64, one Arrow CPU/IO thread, 512-stock autograd blocks, row-contiguous annual memory maps, nice level 10. Every second: process/system CPU, RSS, available RAM, swap, load, disk I/O and free disk. Cooperative abort thresholds: RSS 1,600 MiB or system available below 250 MiB.

Reduced run: **54.31 minutes**, peak sampled RSS **528.875 MiB**, minimum system available **1,002.688 MiB**; no limit failure. Median process CPU was **36.45% of one logical core**, median whole-system CPU **73%**. System swap ranged from **4,786 to 8,063 MiB**; this includes other applications and is not the training process's own swap. The process sampled maximum was 117.7% of one core across preparation/training/reporting; a one-thread training setting does not constrain every helper/import/I/O operation. Free disk never fell below **8.34 GiB**.

Final smoke training peak: **233.141 MiB**; synthetic controls: **238.078 MiB**; raw audit: **1,534.766 MiB**. The initial monthly/column-layout attempt was stopped after 69 fits and zero OOS months when system I/O/swap pressure rose; it has no completed-run claim. It is retained under `reduced_attempt_column_cache/`, status `STOPPED_FOR_IO_REFACTOR`. No unrelated process was stopped.

## Commands and artifacts

Commands were run from the repository root with `PYTHONPATH='non stationarity'`, `PYTHONDONTWRITEBYTECODE=1`, `VECLIB_MAXIMUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. The initial audit called `data_adapter.audit()` inside `ResourceMonitor`; the equivalent reproducible CLI is documented in the README. Development-only benchmarks preceded OOS execution. Subsequent real/fixture commands used `nice -n 10`.

```sh
python3 -m adaptive_memory_equities.cli run --config 'non stationarity/adaptive_memory_equities/configs/reduced.yaml' --run-dir 'non stationarity/adaptive_memory_equities/private_runs/reduced'
python3 -m adaptive_memory_equities.cli synthetic --run-dir 'non stationarity/adaptive_memory_equities/private_runs/synthetic'
python3 -m adaptive_memory_equities.cli smoke --config 'non stationarity/adaptive_memory_equities/configs/smoke.yaml' --run-dir 'non stationarity/adaptive_memory_equities/private_runs/smoke_final'
python3 -m adaptive_memory_equities.cli smoke --config 'non stationarity/adaptive_memory_equities/configs/smoke.yaml' --run-dir 'non stationarity/adaptive_memory_equities/private_runs/smoke_final' --resume
python3 'non stationarity/adaptive_memory_equities/tools/summarize_results.py' --run-dir 'non stationarity/adaptive_memory_equities/private_runs/reduced'
python3 'non stationarity/adaptive_memory_equities/tools/verify_execution.py' --run-dir 'non stationarity/adaptive_memory_equities/private_runs/reduced'
```

Those directories now contain completed evidence: use fresh directory names to rerun. The formation-only CLI attempt is in `private_runs/formation_block/` with its explicit failure JSON. Completed manifests and source hashes, original/generated monthly series, seed payoffs, gate logs, memory curves, architecture comparisons, costs, paired bands and wealth are in the corresponding ignored private run folders. The reduced common report is [here](../private_runs/reduced/report/reduced_common_report.md); the frozen design is in [protocol.md](protocol.md), and the theory limitations in [theory_bridge.md](theory_bridge.md).
