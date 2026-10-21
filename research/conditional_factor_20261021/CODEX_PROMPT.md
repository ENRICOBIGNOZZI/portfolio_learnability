# Task: audit and implement the classical conditional one-factor benchmark

Work on ENRICOBIGNOZZI/portfolio_learnability. The user authorizes implementation,
tests, and a bounded pilot, not destruction/replacement of existing simulations.
Read AGENTS.md and the attached AUDIT.md first. Treat this prompt as a scientific
specification to CHECK, not as evidence that its claims are true.

## 0. Scope and execution safety

- Confirm the actual repository root, origin, HEAD, current branch, dirty state,
  worktrees, and live project processes. Record a UTC execution receipt.
- Use the isolated branch/worktree created by run_codex.sh. Preserve main,
  empirical code/data/results, old Rich6D, archived outputs and active workers.
- Do not work in the crypto `portfolio` or Alpha OS repositories. Do not change
  security settings, SSH configuration, credentials, Tailscale routes or services.
- No git reset --hard, git clean, force push, broad process kills, restarts, or
  overwriting existing evidence. Do not install system-wide dependencies.
- Use at most one BLAS thread and one pilot worker initially. Do not launch a
  production grid before the mathematical checks and numerical smoke tests pass.
- Honor ordinary Codex sandbox/approval controls. Never use a bypass flag.
- Commit/push only task-owned scientific files to this isolated branch when
  execution permits. Report a blocked push honestly; do not claim it occurred.
- Separate 'implemented', 'tests passed', 'job running', and 'production done'.

## 1. Economic specification: keep it simple

    F_(t+1) = m(S_t) + sigma_F xi_(0,t+1)
    R_(i,t+1) = beta_i F_(t+1) + sigma_e xi_(i,t+1)

Heterogeneous deterministic fixed beta_i, B=sum_i beta_i^2>0; positive constant
sigma_F,sigma_e; alpha=0. No beta in idiosyncratic noise. No triplets, Gram
normalization of betas, state-dependent noise correction, or cross-sectional
balancing. N is fixed in each economy.

S_t has six continuous coordinates, common to every stock. Generate a stationary
Gaussian AR(1) A_t with |rho|<1 and set S_t=2 Phi(A_t)-1 coordinatewise.
All return shocks are independent, centered, variance one, bounded, independent
of the state process. Start with Uniform(-sqrt(3),sqrt(3)); optionally verify the
standardized truncated-normal version WITHOUT clipping finished returns.

The investor observes Z_i=(asset_id_i,S_t). Keep the full block policy kernel
K((i,s),(j,s'))=1{i=j} k_Matern32(s,s'), fixed lengthscale, unit scalar diagonal.
This is NOT a seven-dimensional Euclidean Matern and NOT the original shared
firm-characteristic policy. Explicitly document that this experiment is a
state-dependent allocation / market-timing benchmark. Do not feed true beta
coefficients, true m, future factors, or analytic optimal weights to the learner.
Oracle evaluations may use these primitives in separately named functions.

## 2. Re-derive and verify, do not merely reproduce

Set D=sigma_e^2+B sigma_F^2. Verify all formulas in AUDIT.md, especially

    mu(s)=beta m(s)
    Sigma_R(s)=sigma_e^2 I + [sigma_F^2+m(s)^2] beta beta'
    W_i^*(s)=N beta_i m(s)/[D+B m(s)^2]
    w_i^*(s)=W_i^*(s)/N
    Sigma_R(s) w^*(s)=mu(s).

Distinguish covariance from second moment everywhere. Check with independent
linear solves, not just a second copy of the same closed-form function.

Write a concise theorem/proposition with proof of E1--E6 for this model and the
explicit common constants/restrictions needed for the stated model class.
N-dependent constants must be labeled; changing N is a different economy.

## 3. RKHS membership and r=1 are mandatory gates

Show H_K is the finite direct sum of N copies of H_k, equivalent as function sets
to N copies of H^(9/2)((-1,1)^6). Derive its norm without silently normalizing by N.
Let f=m/(D+B m^2). Prove m in H^(9/2), bounded on its compact range, implies
f in H^(9/2) by smooth composition. Record

    ||W^*||_(H_K)^2=N^2 B ||f||_(H_k)^2.

State and enforce the class radius, not just membership. Use the exact source
convention in the current paper: W^*=Sigma_H^((r-1)/2)h. For r=1, h=W^*.

CRITICAL: distinguish these three claims:
1. Membership and a valid r=1 source condition.
2. r=1 is the MAXIMAL source order of this particular infinite target.
3. The external dependent-data convergence theorem actually includes r=1.
They require different arguments. No simulation can replace any of them.

Locate the CURRENT companion statistical source and its Lemma A.7 cited by
PORTAFOGLIO (15).pdf. An available older 29-August-2026 version only covers r>1.
Do not pass to r=1 in a formula with 1/(r-1). If the correct endpoint lemma is
unavailable or a gap is found, mark theorem transfer BLOCKED with exact version,
page/line, and missing step. Continue safe DGP algebra/implementation, but do not
assert a certified T^(-3/5) convergence theorem or alter proofs to hide the gap.

## 4. Two targets; preserve the distinction

A. Smooth baseline: choose m(s)=m0+a psi(s), with a bounded smooth nonconstant psi
involving the state coordinates, positive bounded m. This is a valid r=1 class
member. Do not assign its maximal source order from smoothness alone.

B. Optional critical ideal target: use the lacunary series in AUDIT.md,
s0=9/2, Cpsi=-log(1-2^(-s0)),

 psi_inf(s)=(1/(6 Cpsi)) sum_(k=1)^6 sum_(j>=1)
                [2^(-j s0)/j] cos(pi 2^j s_k).

Prove H^s0 membership and failure of every H^(s0+delta), including restriction
from the torus to the cube. Keep m's range strictly inside (0,sqrt(D/B)), so the
map u/(D+B u^2) has a smooth local inverse. Verify non-cancellation of roughness.
For operator maximality independently check the Loewner--Heinz/Douglas range
argument and the NECESSARY inclusion Ran(T_k^((r-1)/2)) subset H^(s0 r), 1<r<=2.
Do not assert equality with all Sobolev spaces on a bounded domain.
If a step remains unsupported, label maximality unverified, not proved.

Numerically use finite J; label it CRITICAL_TARGET_APPROXIMATION, never an exact
r_max=1 target. Preserve the infinite normalizer. Provide a deterministic uniform
truncation bound, J sensitivity and representation/integration error checks.
A tiny sup error alone does not bound higher source norms or ridge inverse error.
The ideal source condition does not force a measured finite-sample slope or prove
a minimax lower bound for this restricted one-factor subclass.

## 5. Implement only additive task-owned modules

Suggested scope: research/conditional_factor_20261021/ initially, then an isolated
simulation module only if integration is justified. Keep the current canonical
simulation system untouched. Functions, type checks, reproducible RNG streams,
clear W versus w naming, explicit parameter validation, no unnecessary framework.

Implement state generation, independent bounded shocks, factor/stock returns,
conditional moments, analytic oracle weights, sample loss and conditional-moment
population evaluation. Implement exact managed-kernel ridge first:

 G_ts=k(S_t,S_s)*(R_(t+1)'R_(s+1))/N^2
 a=(G+T lambda I)^(-1)1
 W_hat_i(s)=(1/N)sum_t a_t R_(i,t+1)k(S_t,s).

This is T by T, not an unnecessarily allocated (NT) by (NT) matrix. Return actual
weights W_hat/N. For larger T, document any approximation and independently check
its error before using it. Never silently train only in the true beta direction;
that would change the learning class/information set.

Evaluate on independent states using exact conditional shock integration:
 first=(w'beta)m(s)
 second=sigma_e^2||w||^2+(sigma_F^2+m(s)^2)(w'beta)^2.
Average first and second separately, then compute signed unconditional marginal
Sharpe. Keep negative-Sharpe and poor seeds. No sign flipping or best-seed filter.
The oracle p=E[B m(S)^2/(D+B m(S)^2)] is generally a numerical integral. Distinguish
its quadrature error from the analytic conditional oracle.

## 6. Required tests and bounded pilot

Run the provided verify_dgp.py and add independent tests:
- optimizer equals direct matrix solve on many independently chosen states;
- conditional normal equation, pricing against random directions, strict SPD;
- correct covariance versus second moment and noise moments;
- W/N normalization, feature-norm and deterministic return bounds;
- exact regret identity equals evaluated loss difference;
- block kernel/managed Gram PSD and dual/primal agreement at small dimensions;
- future-return isolation and independent training/validation/evaluation seeds;
- reproducibility, independent factor and residual innovations;
- state persistence leaves the marginal oracle unchanged;
- rank/truncation sensitivity and preserved failed test receipts.

Start with N=12, T<=96, three seeds, rho=0 and 0.6, smooth and truncated-critical
targets, one worker. This is an implementation pilot, not a rate experiment.
Only after checks pass propose a measured-cost larger design. Chronological
validation must never use the evaluation oracle to choose deployable lambda.
Calibrate any target Sharpe with a separate synthetic-state integration sample,
freeze it before training, and never tune to produce a preferred slope.

## 7. Deliverables and completion report

Produce mathematical_audit.md (with source version ledger), standalone module,
tests, exact commands, timestamped JSON receipts, pilot CSV/JSON preserving every
seed, and a short handoff for the next agent. Cite primary mathematical sources
where needed and distinguish supplied-paper statements from new derivations.

Final report in Italian: commit/branch, files changed, tests actually run and
exit codes, pilot actually completed, theorem endpoint source found or missing,
RKHS membership status, r=1 admissible versus maximal status, remaining numerical
limits, and exact next command. Never say Codex or a remote machine ran something
merely because a prompt was saved or a GitHub workflow ran a Python script.
