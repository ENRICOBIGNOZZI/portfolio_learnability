# Prospective V2 confirmation

The signed configuration is `simulations/config/confirmation_v2.json`, schema
`confirmation-protocol/2.0`, SHA256 identity
`719034dd1fd3a9a1c716c7a5fcd6337b006bce60dbb0725997a82e2cea72639a`.
It was frozen after the cost/projection pilot and before confirmation performance.
An identical copy is in `outputs/confirmation_v2/audit/protocol_freeze.json`.

The machine has 8 GiB physical memory and about 6.4 GiB available disk. The pilot
measured generation times roughly 0.012–0.028 seconds/date at rank 512 and
0.038–0.046 at rank 1024, depending on map/kernel and transient load. Those are
measured serial elapsed costs, not forecasts of parallel wall time. The maximum
replication proposal would require approximately 130,000 serial seconds just
for data generation, before fitting, numerical certification and verification.

The resource decision, recorded before performance, is 100 fresh replications
per loading map at nu=1.5 on T=60,90,120,180,240,360,540,720,1080,1440. The first
50 are coherently extended to 2160,3240,4860,7290. Sensitivities nu=.5 and 2.5 use
50 replications per map on the practical grid only. All cohorts evaluate the
three theory penalties at their actual values. This is a resource-limited
confirmation, not execution of the proposed maximum 500/200/200 targets.

The production rank is 512. Nested ranks 256,512,1024 are compared on the first
50 matched paths; 2048 is optional only if both accuracy and resource evidence
justify it. A region is certified only when the projection floor is below 5%
of its mean regret and abs(mean paired rank difference)+1.96*paired MCSE is at
most 5% of reference mean regret. Unresolved regions retain all observations
and are labeled explicitly. Passing selected regions does not certify the
near-interpolating curve or the entire infinite-dimensional kernel.

Four independent quadrature seeds compare 8192 and 32768 groups on fixed
policies. Spectrum settings are 1024,2048,4096 groups and three seeds, with
windows 30–200,60–400,120–800. Partial/operator approximations must report
residuals and unresolved indices. No closeness to a target slope is an
acceptance gate. The historical slope bands remain descriptive.

Master seed 2026100801 generates distinct anchor, pilot, quadrature, training,
independent-test, bootstrap, fold and spectrum families. Within the economic
generator, state, factor and idiosyncratic innovations are independent streams.
Matching replication indices reuse common random numbers across maps, kernels
and ranks. Therefore environment evaluations are not all independent economic
paths. Replications within a fixed environment are independent.

For each map/kernel, s_ref is the leading population second-moment eigenvalue
from the independent 8192-group pilot at rank 512. It is frozen once and used
in lambda_T=a*s_ref*T^(-b/(b+1)), a=.25,1,4 and b=1+2nu/6, without grid snapping.
This theoretical calibration uses population information; it is not an
implementable selector. Neither s_ref nor windows change with observed loss.

Holdout25 retains its historical split and first-minimum tie rule. Rolling3 uses
floor(.4T), floor(.6T), floor(.8T),T endpoints, with at least two observations
in each training and validation segment. It weights validation loss by segment
length and refits on all T observations. A tie within absolute loss 1e-12 chooses
the largest lambda. A conservative one-SE variant is omitted because no
dependent-data uncertainty rule was prospectively justified. Zero is a separate
benchmark, never silently eligible for either selector; its Sharpe convention
is zero, its loss one and complexity zero.

The ensemble plugin loss oracle and a five-fold oracle use population regret
for calibration, with entire replications assigned to folds. Oracle Sharpe means
Sharpe at the loss-optimal penalty. It is not a Sharpe-optimal oracle. All
bootstrap intervals resample whole economic paths. Mean intervals are
pointwise and conditional on fixed anchors and quadrature.

Each training path is generated through max(T)+720. For each T, only its first
T observations enter fitting. The next 720 give contiguous-future evaluation.
A separately initialized stationary path of length 720 gives independent OOS.
These labels remain distinct; Sharpe is per period. Training fingerprints,
selected coefficients and their actual independent/future payoff arrays are
retained so evaluation can be checked independently.
