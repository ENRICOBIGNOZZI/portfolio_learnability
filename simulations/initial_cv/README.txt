INITIAL CV CALIBRATION — RICH6D

Scientific scope
---------------
The canonical output is simulations/outputs/rich6d_initial_cv60_v1.
The previous rich6d_rough_sr3_extended_v2 experiment is read-only reference
material. All original replication seeds, paired return histories, kernel,
rank 4096, population operators and economic parameters are retained.

In each replication and each of the five economies, calibrate a once at T=60.
The predetermined candidates are 71 logarithmically spaced constants from
1e-8 to 1e-1. Forward folds train on observations 1:20, 1:30, 1:40, 1:50 and
validate on the following ten observations. Each fold uses a*n_train**(-0.6).
Minimize pooled response-one squared validation loss; exact ties choose the
largest a. All five selections finish before observation 61 is generated.
At subsequent horizons, refit at the permanently frozen a*T**(-0.6).
No future outcomes, population losses or final Monte Carlo summaries enter CV.

The baseline annualized economic optimum remains 3. The N=300 and N=1200
opportunity sets remain exactly as in the original experiment, so their
analytical optima differ slightly from 3. Changing these would change the DGP.

Reproduction
------------
1. python3 -m simulations.initial_cv.design freezes a NEW protocol and compact
   original-reference checkpoints. Do not rerun this over an existing study.
2. pytest tests/test_initial_cv.py checks direct fold solves, the training-size
   penalty scaling, future-data exclusion and deterministic exact ties.
3. Dispatch rich6d-initial-cv.yml with stage=pilot. It runs original indices
   0:11 on the previously history-qualified macos-26/Python 3.13.3 environment.
4. python3 -m simulations.initial_cv.collect --stage pilot --run-id RUN_ID
   --commit EXACT_HEAD_SHA --download-root /tmp/initial-cv-pilot-RUN_ID
   independently verifies every returned path. Only complete success writes
   pilot_gate.json and production_allocation.json. The gate is correctness
   only: it never changes the scientific design based on selected constants
   or observed performance. The twelve paths count toward the final 300.
5. Commit/push the gate and allocation, then dispatch stage=production.
   Collection uses the same command with stage=production and its own pinned
   run ID, commit and download directory. Exactly indices 12:299 remain.
6. python3 -m simulations.initial_cv.summarize requires all 300 independently
   verified checkpoints and compares CV, the reproduced old fixed constant,
   and the ex-post population oracle. Next run
   python3 -m simulations.initial_cv.audit to independently check the tables.
   Finally run python3 -m simulations.initial_cv.report to write the English
   report and six PDF/PNG figures. Inspect every rendered PDF before final
   publication.

Use OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1,
VECLIB_MAXIMUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1.
The remote workflow installs simulations/requirements.txt and hash-checks
all original frozen numerical inputs. No original input is re-estimated.
Checkpoints contain all fold losses, validation scores, selected constants,
penalty paths, history hashes, population metrics and normal-equation error.
The collector records workflow, source and artifact checksums. Incomplete
runs retain finished checkpoints, but cannot generate a final result report.

Interpretation and uncertainty
------------------------------
The diagnostic oracle selects one common penalty per economy/horizon on the
original predetermined 96-point grid, maximizing mean population Sharpe over
the original 300 fitted paths. It is not an attainable tuning procedure.
It is evaluated only during final postprocessing, never inside calibration.
Its outcome bands and slopes are descriptive and do not adjust for ex-post
selection. It is a Sharpe oracle, not a separate loss-minimizing oracle.

summary.csv contains central 95% replication bands, not confidence intervals
of the mean. Paired CV-minus-fixed and cross-economy tables also give separate
mean confidence intervals. Slopes are slopes of log mean outcomes against
log T, with the original predetermined horizon windows; non-oracle slope
MCSE uses complete cross-horizon covariance and a replication jackknife.
Both empirical and population effective complexity are reported. No fitted
slope alone validates the theorem or identifies maximal source regularity.

All original numerical flags are retained. Additional CV floor/regret checks
use the same 5% criterion, and may fail more often when regret is smaller.
No new higher-rank certification of the CV-selected policies is claimed.
Neither the oracle nor a successful CV improvement resolves the Nyström floor.
Boundary constants remain in the output and are never discarded or retuned.
