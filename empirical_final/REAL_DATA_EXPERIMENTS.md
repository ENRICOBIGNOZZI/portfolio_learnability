# Five real-data JKP experiments

The new exploratory entry point is `real_data_study.py`. It implements A–E from
the proposed empirical redesign without running simulations. Previous research
artifacts remain historical snapshots; none of their fitted policies enters
this run. The only reused aggregate is the existing French factor data.

Run from the repository root:

```sh
python3 empirical_final/memory_guard.py \
  --out outputs/real_data_ABCDE_20261021/audit/run_memory \
  --limit-mib 600 --minimum-available-mib 256 -- \
  python3 -u -m empirical_final.real_data_study
```

The guard sets BLAS and Arrow thread limits, samples the child process tree once
per second, and stops at 600 MiB RSS or after five consecutive samples with less
than 256 MiB system available memory. It records system swap without modifying
other applications. These are sampled measurements, not OS high-water marks.
Child bytecode caches use the local temporary directory, avoiding iCloud
hydration stalls in this Desktop checkout.
The panel reader uses 8,192-row Arrow batches and the nonlinear map uses
128-stock blocks. Stock positions and managed caches stay in ignored `results/`.

The common test uses January 1994–December 2024 formations, whose payoffs run
February 1994–January 2025. T includes a final 12-month validation interval.
Parameters are selected on that past interval and refitted on the full T-month
history. Bandwidths and feature subsets are frozen before the test. T limits
managed-return estimation history; the common initial dictionary and bandwidth
calibration remain available to all policies.

| Experiment | Implementation |
| --- | --- |
| A | Twenty-four random groups of 32 features; select one prior-validation-Sharpe match within 0.15, maximizing concentration/alignment spread. Report failure if no match exists. Freeze the pair across payoff scenarios. |
| B | Nested identifier-hash training samples N=100,500,1000,3000 and full; T=36,60,120,240,360. Deploy every policy on the full formation universe. Report actual N if fewer than the requested stocks exist. |
| C | Nested predetermined sets of 8,16,32,64,130 characteristics, comparing affine linear and Gaussian RFF policies with chronological ridge tuning. |
| D | Exact effective-complexity targets against T, plus a validation-selected complexity policy. Mark targets exceeding inner-training rank as unavailable; never silently cap them. |
| E | Remove value, profitability, momentum, or three random groups; compare ridge and nested PCR; neutralize stock positions to market or six-factor exposures estimated from the previous 60 observed monthly returns. |

The random E ablations remove 32 characteristics each. They probe broader
perturbations and are not cardinality-matched controls for the smaller named
families; the results do not identify a causal family effect.

The economic operator is the uncentered second moment of monthly managed
payoffs. PCR truncates this operator's directions. Past signal alignment is
reported separately from spectral mass, and incremental direction value is
measured by subsequent quadratic loss, including its cross term.

Net returns use a common scale targeting pretest linear median gross exposure
one. The illustrative default costs are 10 bps per dollar bought or sold and
30 annual bps on short notional, with exact self-financing drift accounting.
They are assumptions rather than observed transaction costs. Invalid/insolvent
paths have no reported net Sharpe. OOS surfaces are descriptive, never selectors.

The source `data/formation_only` retains all stocks selected at formation,
including 7,489 unresolved subsequent returns. Its prior audit verifies ranks
and membership are independent of future payoff availability. The new runner
checks every source-file hash and does not consume `data/clean` ranks.

Three separately labelled missing-TOTAL-return assumptions (-100%, 0%, +100%)
are fitted and evaluated. None is a recovered observation, an identified main
estimate, or a formal bound. The dictionary/current JKP snapshot remains
retrospective; historical vintages are not certified. A matches historical
validation opportunities, and cannot certify equal population Sharpe or a
causal effect of changing characteristics.
Removing characteristics changes the available portfolio policies; it does not
change the underlying financial economy.

The default 512 RFF make this an exploratory approximation, distinct from the
earlier 10,000-feature study. To change features, dates, costs, or code, use new
`--out` and `--private` directories: frozen identities prevent stale-cache reuse.
Generated aggregate tables, PNG figures, report, provenance, and memory logs
are in `outputs/real_data_ABCDE_20261021/` and belong in Git. Licensed panels and
individual stock positions never do.

Generate the additional A/E figures and predefined cross-scenario contrasts
after the main runner completes:

```sh
python3 empirical_final/memory_guard.py \
  --out outputs/real_data_ABCDE_20261021/audit/analysis_memory --limit-mib 500 -- \
  python3 -m empirical_final.real_data_analysis
python3 empirical_final/memory_guard.py \
  --out outputs/real_data_ABCDE_20261021/audit/verification_memory --limit-mib 500 -- \
  python3 -m empirical_final.real_data_verify
```

Numerical/timing tests (small fixtures, not empirical evidence):

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 empirical_final/memory_guard.py \
  --out tmp/real_data_tests_memory --limit-mib 500 -- \
  python3 -m pytest -q -p no:cacheprovider tests/test_real_data_study.py
```
