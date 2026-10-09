# Adaptive memory equity portfolios

Isolated research experiment: **130 frozen JKP characteristics → fully trained ReLU NN → stock weights f(Z)/N_t → monthly response-one loss and OOS Sharpe**. No network penalty, weight decay, broker interface, orders, publication or changes to existing results.

The enclosing `non stationarity/` directory follows the user's explicit request. This package imports the existing formation mask and rank transform read-only. Licensed data, monthly holdings, checkpoints and all execution artifacts are under ignored `private_runs/`.

## Evidence and limits

The independent raw audit found **2,504,402 formation stock-months**, **744 months**, **130 features**, and **7,489 unresolved next-month payoffs in 638 months**. Exact next-calendar matching recovered none. Changing every future payoff to missing leaves formation ranks unchanged. Formation-only portfolio evaluation is blocked: `PAYOFF_COMPLETENESS_FAILED`.

Real runs use `retrospective_complete_payoff`, the original filtered sample with its original ranks and N_t. They are **RETROSPECTIVE_ONLY**, not point-in-time backtests. Snapshot revisions and retrospective signal discovery remain uncertified. See [data audit](docs/data_audit.md), [discovery](docs/discovery.md), [frozen protocol](docs/protocol.md), and [theory bridge](docs/theory_bridge.md).

The final completed smoke has **12 actual seed-specific fits, two January refits, three temporal experts, 24 realized monthly payoff dates** (February 1978-January 1980). Architecture: two hidden layers of width 32, seeds 0 and 1 averaged in holdings. Its training invocation peaked at **233.1 MiB** RSS. Real resume preserves every checkpoint, decision, payoff and holdings file without additional fits. All smoke guards fall back exactly to full history because fewer than 60 prequential observations are available.

The reduced comparison completed **414 fits, three refit origins and 35 monthly payoffs**, with 72 expert labels per month. `configs/reduced.yaml` retains the affine control and depths 1/2/3/4 at widths 32/64, both seeds, and two independently fixed candidates per forgetting family. Its calendar was fixed before OOS evaluation: 1993 startup, followed by the first 24 common payoff months in 1994-1995. The 372-month complete-grid comparison and terminal 2020-2024 evaluation remain `NOT_RUN`. The full grid plans 31,104 fits before admissibility/deduplication, not an executed count.

See [completed execution and limitations](docs/execution.md) and the [verified common-period report](private_runs/reduced/report/reduced_common_report.md). For the fixed two-layer width-32 NN, raw annualized Sharpe is 1.3920 for full history, 1.5076 for theory memory, 3.7774 for taper, 2.0599 for rolling and 2.8104 for exponential memory. These are 24-month retrospective, unguarded sensitivities. Joint selection frequently chooses the affine control and must not be interpreted as a pure memory effect. Adaptive memory loses on loss in 10 of 36 fixed-architecture comparisons. The short-budget NN results do not beat the verified legacy gross references.

The 360-fit synthetic controls exhibit stationary cost and delayed/missing activation after a break; they are separate from JKP evidence. Validation comprises 42 current-package tests plus the 88 unchanged inherited tests, executed in multiple recorded invocations. The reduced run took 54.31 minutes, peaked at 528.9 MiB RSS and kept at least 1,002.7 MiB system RAM available. CPU/RAM/swap/I/O logs and source-version checks are saved with each execution.

## Reproduce from the repository root

The existing environment already contains PyTorch, NumPy, pandas, scipy, pyarrow, PyYAML, psutil and pytest. Base dependencies are in the original `requirements.txt`, additions in this package's `requirements-extra.txt`; exact observed versions are recorded in each run's `source.json`.

```sh
export PYTHONPATH="non stationarity${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 -m adaptive_memory_equities.cli audit \
  --config 'non stationarity/adaptive_memory_equities/configs/formation.yaml' \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_audit'

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider \
  'non stationarity/adaptive_memory_equities/tests'

python3 -m adaptive_memory_equities.cli smoke \
  --config 'non stationarity/adaptive_memory_equities/configs/smoke.yaml' \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_smoke'

python3 -m adaptive_memory_equities.cli run \
  --config 'non stationarity/adaptive_memory_equities/configs/reduced.yaml' \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_reduced'

# Same configuration, code and snapshot; continue saved completed fits/months.
python3 -m adaptive_memory_equities.cli run \
  --config 'non stationarity/adaptive_memory_equities/configs/reduced.yaml' \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_reduced' --resume

python3 -m adaptive_memory_equities.cli report \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_reduced'

python3 'non stationarity/adaptive_memory_equities/tools/summarize_results.py' \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_reduced'

python3 'non stationarity/adaptive_memory_equities/tools/verify_execution.py' \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_reduced'

python3 -m adaptive_memory_equities.cli synthetic \
  --run-dir 'non stationarity/adaptive_memory_equities/private_runs/new_synthetic'
```

`benchmark` runs only 1963-1972 development fits. `research.yaml` supplies the full candidate grid and full common-calendar target; it is computationally much larger. To demonstrate the source block, run `smoke --config .../configs/formation.yaml` in a fresh private directory: it fails explicitly on missing historical portfolio payoffs. No missing values are filled, and no stocks are removed after a portfolio decision.

## Statistical and computational contract

Every monthly cross-section has one scalar portfolio loss. The squared residual is computed after summing stock payoffs. All hidden layers train; no random-feature substitution, softmax, demeaning, leverage cap, volatility targeting, dropout or batch normalization. The final affine output permits exact training-only scalar calibration `c_star=mu/M2`, saved with loss before/after. Seed ensembles average contemporaneous weights, never Sharpe ratios.

The exact two-pass gradient computes the complete monthly payoff with frozen parameters, then accumulates stock-block gradients with coefficient `2*omega*(p-1)`. The optimizer updates only after all selected dates. `batch_dates: 0` is the exact full-date reference; the practical fixed budget samples 8 complete dates with replacement from omega, giving an unbiased estimate of the same objective. Ten fixed Adam updates are a pilot budget, not an optimization guarantee. Calibration still uses every positive-weight training date.

Theory weights solve the strictly convex simplex QP with an active-set solver. Tau is a temporal drift tradeoff, never a parameter penalty. Tau zero is exact uniform; candidates with effective history below 36 months are excluded. Calendar ages never compress gaps. Empirical taper, extended taper, rolling and exponential memories are separate families. Matched QP/taper weights and deterministic training parity are tested.

January-close annual cold refits inherit the original pipeline calendar; holdings change monthly. Decisions are saved before reading their next-month returns. The selector scores only mature saved prequential returns. Guard: 60 months, circular 12-month blocks, 2,000 shared bootstrap draws, one-sided simultaneous 95% max statistic; no positive valid LCB means exact full-history fallback. Baseline architecture selection, when scores suffice, is also prequential. Fixed-architecture and joint selection reports coexist. The guard is approximate and is not a stationarity test or time-uniform guarantee.

CPU usage of the process and system, RSS, available memory, swap, disk I/O and free disk space are sampled every second. One CPU training process, one torch thread, float64, 512-stock blocks and monthly views of row-contiguous annual memory maps bound allocations. Limits are 1,600 MiB RSS and a 250 MiB system-available floor, with cooperative abort at work checkpoints. MPS availability does not imply GPU use.

## Saved artifacts

- `source.json`: frozen config/hash, source SHA and tracked-diff hash, hashes of new untracked source, versions/hardware, data fingerprint and complete feature allowlist.
- `fits/`: seed-specific checkpoints and fit diagnostics; keys include origin/cutoff, source/universe/transform, architecture, temporal weights, optimizer and checkpoint rule.
- `origins/`: actual candidate counts, exclusions, deduplication, effective memory, ages, weight distributions and matched-weight checks.
- `decisions/`, `events/`, `holdings/`: immutable monthly decisions, chained payoff ledgers, policy scores, raw weights, reporting scale and actual aggregate holdings. Resume validates hashes and never reselects old decisions.
- `report/`: monthly payoffs, comparisons, separate seeds, depth/width comparisons, gate logs/activation/switches, cost sensitivities, paired bootstrap intervals, wealth and PNG figures generated only from completed runs.
- `memory.jsonl` and `memory.summary.json`: sampled resource record. OS compression/swap is reported separately; a sample maximum is not an exact allocation peak.

`memory.jsonl` appends across resume calls; `memory.summary.json` describes the latest invocation. `execution_verification.json` aggregates all telemetry and verifies saved decisions against the current selector. Historical source versions are preserved in `private_runs/code_snapshots/<code_hash>/`. A different code/config/data signature deliberately refuses resume; recreate the experiment in a fresh directory with the current version, or restore the exact archived sources before resuming its original run.

The optional common-slice reporting tool verifies the saved legacy linear/Gaussian/Matérn gross outputs against source hashes, January-close timing, realization months and raw scale. It imports no old optimizer into the new NN training. Its legacy comparisons are expressly seed-0, gross-only, preexisting outputs; no new kernel fits or net-cost comparison are claimed. The short common slice contains only two 12-month blocks, so bootstrap intervals have limited resolution.

Annualized Sharpe is `sqrt(12)*mean/std(ddof=1)` on concatenated monthly excess returns. The raw training/reporting scale is 1, cash is `1-sum(W)`, and total returns add same-snapshot cash once. Wealth stops at insolvency. Turnover aligns asset entries/exits and prior holdings drifted with total returns where solvent, sums both trade legs without a hidden factor 1/2, and also reports target-weight turnover. Cost/borrowing numbers are illustrative post-fit sensitivities, not net-optimal policies. Previously published kernel results are not silently imported as comparable observations.

Status vocabulary: `IMPLEMENTED`, `TESTED`, `REAL_RUN_COMPLETED`, `RETROSPECTIVE_ONLY`, `PAYOFF_COMPLETENESS_FAILED`, `NOT_RUN`, `EXTERNAL_BLOCKER`. Unit tests and synthetic controls are never market evidence. Complete formation-only evidence needs additional source payoffs; no full-grid or terminal result is claimed from a smoke or reduced run.
