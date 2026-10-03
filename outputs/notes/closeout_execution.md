# Execution and acceptance record

Historical record of the original smooth-economy close-out. The subsequent
boundary-target extension is recorded in `boundary_target_execution.md`; current
headline figures and combined aggregates supersede the six-configuration exports
described below. The original smooth results are retained in the appendix.

The finished study contains 1,500 independent Monte Carlo paths across six configurations,
10 sample sizes, 360 positive penalties and 21,600 aggregate surface rows.
The main economy has 500 paths; all other cells have 200. Source/code hashes and
software versions are saved alongside the outputs. Raw stock inputs remain private.

## Runtime

- Completed Monte Carlo checkpoint compute: 3227.264 seconds (53.79 minutes).
- Simulation elapsed wall time, including stock/DGP audits, interruptions and resumption: 3421.224 seconds (57.02 minutes).
- Formation-only reconstruction and missing-return audit: 794.495 seconds (13.24 minutes), run concurrently with simulation.
- Final empirical diagnostics/export invocation: 7.054 seconds.
- Final local test suite: 84 passed in 44.10 seconds.

These durations overlap and must not be added as total elapsed time. The simulation
started serially with two BLAS threads, resumed with four one-thread processes, then
finished with two one-thread processes to limit memory pressure. Seeds and completed
replications were preserved. No scientific cell was omitted for speed.

## Exact scientific commands executed

Working directory for these commands: `/tmp/portfolio-schedule-20261002`.
The computation was performed in this local clone to avoid intermittent iCloud file
latency in the shared workspace. Final sources/artifacts are synchronized to that
workspace, and the private Monte Carlo checkpoints are preserved there as well.

```sh
VECLIB_MAXIMUM_THREADS=2 python3 -u simulation_characteristic_factor.py > /tmp/portfolio-simulation-run.log 2>&1
VECLIB_MAXIMUM_THREADS=1 python3 -u simulation_characteristic_factor.py --workers 4 >> /tmp/portfolio-simulation-run.log 2>&1
VECLIB_MAXIMUM_THREADS=1 python3 -u simulation_characteristic_factor.py --workers 2 >> /tmp/portfolio-simulation-run.log 2>&1
VECLIB_MAXIMUM_THREADS=1 python3 -u audit_formation_timing.py --raw /Users/enrico/Desktop/PHD/portfolio/paper_codice/data/raw --destination /Users/enrico/Desktop/PHD/portfolio/paper_codice/data/formation_only > /tmp/portfolio-timing-audit.log 2>&1
VECLIB_MAXIMUM_THREADS=1 python3 empirical_closeout.py > /tmp/portfolio-empirical-closeout.log 2>&1
VECLIB_MAXIMUM_THREADS=1 python3 render_characteristic_factor.py
VECLIB_MAXIMUM_THREADS=1 python3 verify_final_outputs.py > /tmp/portfolio-final-artifact-checks.log
VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests > /tmp/portfolio-final-tests.log 2>&1
```

The first two Monte Carlo launches were interrupted after valid checkpoints to change
resource use; the third resumed and completed all configurations. For a fresh run,
use only the final two-worker compute command, then the rendering command.
A 500-path headline preview was rendered during computation from completed checkpoints;
final figures were subsequently regenerated from the complete six-configuration outputs.

PDF review used this exact loop, followed by image inspection:

```sh
for name in simulation_learnability_law_b150 empirical_E1_managed_payoff_spectrum empirical_E2_row_normalized_heatmap empirical_E3_common_period_window_tradeoff; do pdftoppm -scale-to 1400 -png -singlefile "outputs/figures/$name.pdf" "/tmp/portfolio-final-qa/$name"; done
```

## Acceptance and scientific limits

Simulation criteria 1-12 are met: generated stock returns/characteristics, audited
payoff identity, known optimum and controlled second moment, persistence, r=1
membership, complete Monte Carlo surface, heatmaps with oracle and validation,
rate panels with honest slopes, and finite-J sensitivity. The largest slope
variation is 0.03624 (<0.08 predeclared tolerance); the largest-T regret variation
is 4.525% (<10%). Numerical stock/payoff errors are below 1e-10 in every audit.

Empirical criteria 13-17 are met using the explicitly allowed **isolation** option
for criterion 17: normalized heatmaps, validation diagnostics, identical common
months, concatenated Sharpe, and an audited/documented source limitation.
Formation-only ranks and universes were actually rebuilt and pass the payoff-
availability perturbation check. However, the current snapshot cannot identify
7,489 missing payoffs, and it is not a certified historical vintage. A fully
payoff-complete, pristine real-time empirical reconstruction remains unavailable.
No replacement return is invented; affected figures are visibly labeled sensitivities.

The fitted slopes need not equal the general r=1 envelope for this fixed smooth
target. For b=1.5 they are -0.4777 (ridge), +0.3279 (complexity), -0.6344 (regret),
versus -0.6, +0.4, -0.6. Median validation complexity alone is insufficient:
its mean regret at T=1440 is 5.249 times oracle regret. The empirical validation
has mean relative ex-post regret of 33%-58%, with only 39%-47% of years within
10% of the ex-post oracle. All 15 common-period paired loss intervals include zero.

Four simulation PNGs and five empirical PNGs were visually inspected. The four
main PDF exports were rendered with Poppler and reviewed. The acceptance JSON
checks saved aggregates independently; the CI workflow repeats that verification.
The publication commit SHA is reported in the delivery message rather than embedded
in its own contents (which would make the commit identifier self-referential).
