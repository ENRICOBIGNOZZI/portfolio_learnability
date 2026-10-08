# Balanced three-factor baseline DGP

Source: supplied `DGP_Baseline_Spec.pdf`, October 2026, Sections 1–9.
`simulations/dgp/balanced.py` implements this economy; `simulations/diagnostics/baseline.py` audits it.
The obsolete prescribed-spectrum economy has been removed. This is the sole canonical simulation DGP.

## Reproduction and conventions

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.diagnostics.baseline
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

The audit writes `simulations/outputs/audit/baseline/dgp_audit.json`, `dgp_spectrum.csv`, and
`dgp_spectrum.png`, and returns a nonzero exit status if any acceptance gate fails.
JSON includes all parameters, analytic constants, seeds, tolerances, individual
checks, library versions and source hashes. `--parameters params.json` and
`--settings settings.json` accept dataclass field overrides. Declare these files
before examining results; do not adjust a fit interval or tolerance to make a
failed spectrum pass. Default audits check all 100,256 generated dates for loading
norm and balance, six randomly preselected dates for dense conditional identities,
and 100,000 periods for factor moments and optimal Sharpe.

All arrays and kernels use float64. Simulation streams for characteristics,
factors and idiosyncratic shocks are distinct `numpy.random.Generator` objects
spawned from an explicit seed. The population quadrature uses a separate seed
and no training or audit path observations. `BalancedFactorDGP.simulate(T)` yields
one date at a time with `state`, `z`, `loadings`, `factors`, `epsilon`, and `returns`.
Each row of time pairs formation information at t with returns at t+1.
`w_star(z, p)` evaluates the scalar policy W*; divide by N for stock weights.
Sharpe is per period, without annualization.

## DGP and derived policy

Defaults are D=6, N=600, G=N/3=200, K_F=3, rho=(0.95,...,0.95),
nu=1.5, ell=1, c_beta=0.04², sigma_eps=0.08 and mu_F=(0.10,0.06,0.03).
N must be divisible by three, |rho_d|<1, and 0<||mu_F||<1.
Strictly positive sigma_eps is necessary for the lower spectral bound.

Initialize independent S[g,d,0] ~ N(0,1), and evolve

    S[g,d,t+1] = rho_d S[g,d,t] + sqrt(1-rho_d²) eta[g,d,t+1].
    U[g,d,t] = Phi(S[g,d,t]).
    Z[g,r,d,t] = 2 ((U[g,d,t] + r/3) mod 1) - 1, r=0,1,2.

The same role phase is used for all coordinates. These are continuous population
ranks, not empirical cross-sectional ranks. Only floating-point endpoint rounding
is guarded with clipping into the open cube. Each stock has the uniform cube law;
stocks in a group are dependent. With u=(z_1+1)/2,

    beta(z) = sqrt(c_beta) [sqrt(2/3) cos(2 pi u),
                            sqrt(2/3) sin(2 pi u), 1/sqrt(3)]'.

Summing exp(2 pi i r/3) and exp(4 pi i r/3) over roles gives zero.
Consequently ||beta(z)||²=c_beta, sum_r beta_gr beta_gr'=c_beta I_3,
and B'B/N=Gamma_beta=(c_beta/3)I_3 exactly.

    F[t+1] ~ N(mu_F, I_3-mu_F mu_F'), epsilon[t+1] ~ N(0,sigma_eps² I_N).
    R[t+1] = B[t] F[t+1] + epsilon[t+1].
    m = B mu_F, V = B(I_3-mu_F mu_F')B' + sigma_eps² I_N.
    S = E[RR' | information_t] = BB' + sigma_eps² I_N.

All next-period shocks are independent of current characteristics and one another.
The factor covariance is checked positive definite before simulation, and E[FF']=I.
For stock weights w(W)=W(Z)/N the unchanged loss is

    Q(W) = E[(1-w'R)² | information_t] = 1-2w'm+w'Sw.
    S w* = m.
    (BB'+sigma_eps² I)^(-1)B = B(B'B+sigma_eps² I_3)^(-1).
    w* = (1/N) B (Gamma_beta + sigma_eps²/N I_3)^(-1) mu_F.
    W*(z) = beta(z)' mu_F / (c_beta/3 + sigma_eps²/N).

The policy coefficients are computed by solving the three-dimensional system
derived here. The policy is never independently specified as a target.

## Conditions E1–E4

E1: the finite-dimensional Gaussian AR state starts in stationarity and has
spectral radius below one, hence is geometrically beta-mixing. A measurable
rank/phase transform cannot increase mixing coefficients. Independent one-step
shocks preserve geometric mixing for (Z_t,R_{t+1}). Any finite gamma>b=1.5 is
therefore admissible; no empirical mixing exponent is estimated.

E2(a): K(z,z')=(1+sqrt(3)||z-z'||/ell) exp(-sqrt(3)||z-z'||/ell), with K(z,z)=1.
For X=(1/N)sum_i R_i K_{Z_i}, ||X||_H²=R'KR/N². Write A=K/N²,
C=A^(1/2) V A^(1/2), d=A^(1/2)m, and B_X²=4(c_beta+sigma_eps²).
Then ||A||<=1/N, E||R||²=N(c_beta+sigma_eps²), and
||V||<=N c_beta/3+sigma_eps². Thus 2||C||/B_X²<=1/2. With tau=1/B_X²,

    log E exp(tau R' A R)
      = -1/2 log det(I-2 tau C) + tau d'(I-2 tau C)^(-1)d
      <= 2 tau tr(C) + 2 tau ||d||²
      = 2 tau tr(A S) <= 1/2.

This uses -log(1-x)<=2x for 0<=x<=1/2 and tr(AS)<=c_beta+sigma_eps².
The audit diagonalizes C and evaluates the full noncentral expression, including
the mean term. This is an exact conditional MGF computation, not a tail-count test.

E2(b): for Y~N(a,v), E[Y⁴]=a⁴+6a²v+3v²<=3(a²+v)², so K4=3.
The audit evaluates exact moments of kernel-section policies, their random linear
combinations, W*, and a mean-zero policy attaining the upper bound. The identically
zero policy satisfies the inequality; its otherwise undefined ratio is reported
as zero by convention.

E3: the trigonometric/constant loading functions have smooth extensions on the
bounded regular cube, hence belong to the Sobolev space H^(nu+D/2)=H^(9/2)
equivalent to the Matérn RKHS. W* is their fixed finite linear combination.
For a uniform parameter class, restrict N to a finite positive set and ell,
c_beta, sigma_eps, and ||mu_F|| to compact nondegenerate ranges, with ||mu_F||
strictly below one, to obtain a common finite RKHS norm bound. No numerical
Nyström norm is claimed to prove this membership.

E4: E[(1-w*'R)v'R | information_t]=v'(m-Sw*)=0 for every admissible v.
The full normal-equation residual and representative RKHS pricing residuals
are checked at each selected date, as is the independent scalar W* representation.

## E5: common-quadrature operator audit

The managed operator is the **uncentered second moment** E[X tensor X], as in
the supplied specification. The idiosyncratic contribution is (sigma_eps²/N)T_K.
Jensen's inequality and ||beta||²=c_beta give the operator sandwich

    (sigma_eps²/N)T_K <= Sigma <= (c_beta+sigma_eps²/N)T_K.

The continuous uniform distribution on the six-dimensional cube and the
Matérn-3/2 kernel give lambda_j(T_K) asymptotic to j^(-3/2); the positive
sandwich transfers this exponent to Sigma. The spectrum is not an input.

One independent sample of M=1024 uniform **base groups** yields 3M=3072
phase-related characteristic nodes. Let phi(z) denote their RKHS coordinates,
obtained from the eigendecomposition of the actual Gram matrix. Define

    T_hat = (1/(3M)) sum_(g,r) phi_gr phi_gr'.
    H_g = (1/3) sum_r phi_gr beta_gr'.
    H_bar = (1/M) sum_g H_g.
    Sigma_hat = (sigma_eps²/N) T_hat
                + (1/(G M)) sum_g H_g H_g'
                + (1-1/G) H_bar H_bar'.

This is the exact managed second moment of an economy whose G independent groups
are drawn from the common empirical group distribution. It accounts for all
within-triplet cross terms and integrates independent factors and idiosyncratic
shocks analytically. Treating the 3M nodes as independent stocks would be wrong.
The same quadrature satisfies the Loewner sandwich by the same Jensen argument.
The audit checks generalized eigenvalues of (Sigma_hat,T_hat) and ordered
eigenvalue ratios against both bounds.

Both log-log slopes use the predeclared inclusive indices 30–200. Directions 1–3
and the spectral tail are excluded in advance. All fit eigenvalues must exceed
1e-12 times their operator's leading eigenvalue. The finite-resolution
compatibility band is -1.5 +/- 0.35; it is not a statistical confidence interval
or a proof of an asymptotic exponent. Slopes, fit R², the entire spectrum, sample
hash and bounds are saved even when compatibility fails. Failure blocks a green
audit; there is no automatic range selection or fallback to prescribed values.

## E6 and Monte Carlo tolerances

For g=c_beta/3,

    q* = g/(g+sigma_eps²/N) ||mu_F||².
    E[R_p* | information_t] = E[(R_p*)² | information_t] = q*.
    Var(R_p* | information_t) = q*-(q*)².
    SR* = sqrt(q*/(1-q*)), Q* = 1-q*.

Since 0<q*<1, optimal Sharpe is finite and positive. The baseline has
q*=0.0142156862745098 and SR* approximately 0.12008616.
The exact conditional optimal-payoff law is constant across time, so optimal
payoffs are iid even though characteristics persist. The audit streams 100,000
full stock-return periods from the DGP, evaluating W*(Z)'R/N each time.
For v=q*-(q*)² and L periods, tolerances are fixed at six analytic standard errors:

    SE(mean) = sqrt(v/L).
    SE(second moment) = sqrt((2v²+4(q*)²v)/L).
    SE(Sharpe) = sqrt((1+(SR*)²/2)/L) [Gaussian delta method].

The factor mean and all nine uncentered factor second moments are separately
checked using six Gaussian standard errors. The covariance of F_i,F_j is C_ij;
Var(F_i F_j)=C_ii C_jj+C_ij²+mu_i² C_jj+mu_j² C_ii+2mu_i mu_j C_ij.
All thresholds are declared before simulation. These finite-sample checks
support, and do not replace, the exact identities above.
