# Conditional one-factor benchmark: mathematical audit

Date: 21 October 2026. Proposed addition to ENRICOBIGNOZZI/portfolio_learnability.
This does not replace the archived Rich6D economy or certify its results.

## Decision and scope

Keep a classical one-factor return process, heterogeneous fixed betas, zero
stock-specific alpha, and additive homoskedastic idiosyncratic noise. The only
state-dependent primitive is the conditional factor premium. The investor
observes asset identity and a common economic state. This is a market-timing
benchmark, not the original common policy over firm-specific characteristics.
The change of representation is explicit and scientifically consequential.

## 1. Exact DGP

Fix N < infinity and a deterministic vector beta in R^N with B=||beta||_2^2>0.
Fix sigma_F>0 and sigma_e>0. Let s0=9/2 and S=(-1,1)^6. Generate a stationary
six-dimensional latent Gaussian AR(1):

    A_t = rho A_(t-1) + sqrt(1-rho^2) eta_t,
    A_0 ~ N(0,I_6), eta_t iid N(0,I_6), |rho| <= rho_bar < 1,
    S_t = 2 Phi(A_t) - 1       (coordinatewise).

S_t is common across all stocks. Its stationary marginal law is uniform on S,
with independent coordinates. Choose m in H^(9/2)(S), with
0 < m_min <= m(s) <= m_max < infinity, and a common H^(9/2) norm bound.

    F_(t+1) = m(S_t) + sigma_F xi_(0,t+1),
    R_(i,t+1) = beta_i F_(t+1) + sigma_e xi_(i,t+1).

All xi are independent across coordinates and dates, independent of the latent
state process, centered, variance one, and bounded by L. Uniform(-sqrt(3),sqrt(3))
is a completely specified baseline. Standardized truncated Gaussian primitives
are another option. Ordinary Gaussian return shocks do NOT satisfy the displayed
almost-sure E2 of PORTAFOGLIO (15).pdf; the Gaussian latent state is harmless.
Use the decision-time filtration containing S through t and past return shocks,
not the next-period shocks.

Define D=sigma_e^2+B sigma_F^2. Conditional moments are

    mu(s) = beta m(s),
    C = Cov(R|S=s) = sigma_e^2 I + sigma_F^2 beta beta',
    Sigma_R(s) = E[RR'|S=s]
               = sigma_e^2 I + [sigma_F^2+m(s)^2] beta beta'.

Sigma_R is an uncentered second moment. Do not substitute C in the response-one
normal equation. No beta multiplies the idiosyncratic noise. No triplets, fixed
cross-sectional beta norms produced by normalization, or covariance correction.

## 2. Representation and exact population optimizer

Use Z_i=(i,S), not S alone. Choose

    K((i,s),(j,s')) = 1{i=j} k(s,s'),

where k is the ordinary stationary Matern-3/2 kernel restricted to S, with fixed
positive lengthscale and unit diagonal. The RKHS is exactly the finite Hilbert
direct sum H_K = direct_sum_(i=1)^N H_k, with squared norm
sum_i ||W_i||_(H_k)^2. As function sets with equivalent norms,
H_k = H^(9/2)(S). The identity coordinate is discrete, NOT an extra Euclidean
coordinate in a seven-dimensional Matern kernel.

The economic payoff convention stays R^p(W)=(1/N) sum_i W_i(S) R_i. Thus actual
stock weights are w_i=W_i/N. Set

    f(s) = m(s) / [D+B m(s)^2],
    W_i^*(s) = N beta_i f(s),
    w_i^*(s) = beta_i f(s).

Direct substitution gives Sigma_R(s) w^*(s)=mu(s). Since Sigma_R(s)>=sigma_e^2 I,
the conditional optimum is unique. The same policy minimizes the unconditional
criterion and satisfies E4 against every policy in H_K. No projection of the
unrestricted conditional optimum onto a smaller policy class is being hidden.

## 3. Membership and the meaning of r=1

Because s0=4.5>6/2, H^s0 on the cube is an algebra and is closed under smooth
scalar composition on bounded ranges. The map g(u)=u/(D+B u^2) is smooth on R,
with denominator at least D>0. Consequently m in H^s0 implies f=g(m) in H^s0,
and W^* belongs to H_K. More precisely,

    ||W^*||_(H_K)^2 = N^2 B ||f||_(H_k)^2.

The class radius must be chosen so that N sqrt(B)||f||_(H_k)<=R. Finite-grid
values or a numerical regression do not certify this infinite-dimensional norm.
Uniform bounds follow from bounded m Sobolev norms, compact parameter ranges,
positive lower variance bounds, a fixed lengthscale, and N fixed.

Using the paper's convention

    W^* = Sigma_H^((r-1)/2) h,  ||h||_(H_K)<=R,

r=1 reduces to h=W^*. Therefore the SOURCE CONDITION AT r=1 is verified.
This does not establish r_max=1. A generic smooth m cannot be declared an exact
critical target. Neither finite Fourier truncation nor a fitted log-log slope
proves maximal source order. Conversely, pointwise smoothness alone does not
identify a higher operator source order on a bounded domain: boundary conditions
and the actual operator matter.

## 4. Verification of E1--E6 for this DGP

E1. The finite-dimensional stationary Gaussian AR process is geometrically
absolutely regular, uniformly for |rho|<=rho_bar<1. Adjoining fresh independent
innovations and taking measurable functions preserves this mixing bound. Hence
any prescribed polynomial exponent gamma>3/2 can be used, with common constants.
There are T financial rows, not NT independent observations.

E2. In the direct-sum feature map,

    X_t = (R_(1,t+1) k_(S_t), ..., R_(N,t+1) k_(S_t))/N,
    ||X_t||_(H_K)^2 = ||R_(t+1)||_2^2/N^2.

One deterministic return-vector bound is
sqrt(B)(m_max+sigma_F L)+sigma_e sqrt(N)L. Squaring and dividing by N^2 bounds
E2. Together with the radius bound, the response-one residual is bounded by
1+R sqrt(kappa), hence satisfies the appropriate conditional Bernstein moment
bound. The unweighted residual need not have conditional mean zero.

E3. The optimizer is the element in Section 2 and belongs to H_K by Section 3.
Strict positivity of the covariance and full support of the state yield unique
identification in the RKHS (up to the usual equality under the state law), so
there is no unresolved minimum-norm choice. The radius is imposed explicitly.

E4. For any policy V with effective weights v,

    E[(1-w^*'R) v'R | F_t] = v'[mu(S_t)-Sigma_R(S_t)w^*(S_t)] = 0.

This also establishes the kernel-weighted filtered innovation with the correct
one-step timing. Do not claim E[residual | realized-return augmented input]=0.

E5. Let T_k=E[k_S tensor k_S] on H_k and T_0=I_N tensor T_k on H_K. For every V,

    sigma_e^2/N^2 * T_0 <= Sigma_H
        <= [sigma_e^2+B(sigma_F^2+m_max^2)]/N^2 * T_0.

Here <= denotes the quadratic-form/Loewner order. Note the factor N^2: T_0 is
not the stock-label-averaged characteristic operator. Uniform state density and
Matern-3/2 in six dimensions give eigenvalues of T_k of order j^(-3/2).
Each is repeated N times in T_0; finite N preserves the exponent, not constants
uniform in increasing N. Min-max comparison gives b=3/2 for Sigma_H.

E6. Put

    h(s)=B m(s)^2/[D+B m(s)^2],  p=E[h(S)].

The optimal payoff has unconditional mean p, second moment p, variance p(1-p),
and signed Sharpe sqrt(p/(1-p)). Positive bounded m, D>0 and B>0 give 0<p<1.
Uniform Sharpe bounds require compact compatible parameter ranges. This is the
Sharpe of the stationary one-period evaluation distribution, not an average of
conditional Sharpes and not a compounded annual Sharpe. The usual displayed
monthly-to-annual marginal convention multiplies by sqrt(12).

Conditional weights are analytic. For nonconstant m, p is an integral, generally
computed numerically with an independent integration error assessment. Do not
advertise numerical quadrature as an exact closed-form unconditional oracle.

## 5. Optional critical infinite target -- separate from the smooth baseline

This is an additional construction, not a consequence of the one-factor model.
Let r0=2^(-s0), Cpsi=-log(1-r0), and on the period-two six-dimensional torus set

    psi_inf(s) = (1/(6 Cpsi)) sum_(k=1)^6 sum_(j>=1)
                  [2^(-j s0)/j] cos(pi 2^j s_k).

Its absolute value is at most one, its mean is zero, and the H^s0 Fourier norm
has tail proportional to sum_j 1/j^2. For any delta>0 the H^(s0+delta) tail is
sum_j 2^(2j delta)/j^2 and diverges. The series is C^4 with periodic derivatives;
restriction to the cube retains the critical threshold via periodic trace/gluing
for a sufficiently small positive delta<1/2. This step must be written out in
any final theorem rather than assumed from a plot.

Choose m=m0+a psi_inf, a!=0, with the whole range strictly inside
(0,sqrt(D/B)). On that interval g'(u)=(D-Bu^2)/(D+Bu^2)^2 is separated from zero.
If g(m) had any extra Sobolev regularity, smooth inverse composition would give
that regularity to m, a contradiction. Thus W^* is spatially critical too.

To convert this into exact OPERATOR source maximality for 1<r<=2, independently
check all of these steps in the final proof:
(a) the Loewner comparison in E5, Loewner--Heinz and Douglas range inclusion give
    Ran(Sigma_H^((r-1)/2)) = Ran(T_0^((r-1)/2));
(b) for the restricted Matern operator, the necessary inclusion is
    Ran(T_k^((r-1)/2)) subset H^(s0 r)(S), as sets of functions;
(c) this inclusion can be obtained using the L2 Mercer operator, its smoothing
    into H^(2s0), and interpolation of its power-range scale;
(d) only an inclusion is needed. Do NOT assert equality of all power spaces with
    unrestricted Sobolev spaces on the cube: boundary conditions can matter.
If a required operator/domain step cannot be established, report maximality as
unverified, without retracting the separately established membership/r=1 result.

For simulation use a fixed finite J and preserve the exact normalization Cpsi:

    ||psi_inf-psi_J||_infinity
      <= 2^[-s0(J+1)] / [(J+1)(1-2^(-s0)) Cpsi].

Every finite J yields a smooth target. This sup-norm error alone is not a
higher-source-norm or ridge-inverse accuracy certificate and does not establish
an asymptotic slope at a fixed numerical resolution. Sparse/additive dependence
of the signal also does not make the entire factor subclass minimax-hard.

## 6. Companion-theorem dependency: an actual unresolved version check

PORTAFOGLIO (15).pdf, Appendix A, invokes Bignozzi et al. (2026), Lemma A.7 for
r=1 and gamma>b. The separately retrieved library version titled
'Paper_finale - 2026-08-29T165753.081.pdf' instead states r in (1,2] and a
mixing condition involving (b-1)/(b(r-1)); its displayed theorem cannot simply
be evaluated at r=1. This is an older version, NOT evidence the current lemma
is false. The latest companion source and endpoint lemma must be located and
verified before certifying the full rate transfer.

The DGP source-condition check is independent of this citation/version check.
Conditional on the valid current endpoint theorem, b=3/2 gives lambda_T of
order T^(-3/5), regret/Sharpe-gap upper rate T^(-3/5), and complexity T^(2/5).
No claim is made that a particular smooth DGP or the known one-factor subclass
must exhibit exactly these slopes or inherit the broad-class minimax lower bound.

## 7. Exact finite-sample implementation

The block kernel still yields a T by T managed Gram matrix:

    G_ts = k(S_t,S_s) (R_(t+1)' R_(s+1))/N^2,
    a = (G+T lambda I)^(-1) 1,
    W_hat_i(s) = (1/N) sum_t a_t R_(i,t+1) k(S_t,s).

Return actual weights W_hat/N. Do not silently restrict the learner to the known
beta direction; that would change the representation and information set.
When evaluating on independent states, integrate factor and idiosyncratic shocks
analytically. For effective weights w(s),

    E[R^p|s]=(w'beta)m(s),
    E[(R^p)^2|s]=sigma_e^2||w||^2+(sigma_F^2+m(s)^2)(w'beta)^2.

Population loss is 1-2 E[first]+E[second]. Signed Sharpe uses E[second]-E[first]^2.
Exact regret identity integrates (w-w^*)'Sigma_R(w-w^*). Use the same independent
states to compare that identity with the loss difference, then vary integration
seeds and resolution to measure numerical sensitivity.

## Source ledger

- User attachment PORTAFOGLIO (15).pdf: pp16--17 E1--E6; pp64--65 source endpoint
  reduction; pp88--89 Matern/Sobolev and finite-truncation caveats. The formulas
  above are a NEW derivation for the proposed benchmark, not quotations from it.
- Companion library document dated 29 August 2026: pp11--13 and 24--26, filtered
  innovation and r>1 theorem. Its date/version limitation is material.
- Bourdaud & Sickel, Composition Operators on Function Spaces with Fractional
  Order of Smoothness, 2010: smooth-composition background.
  https://users.fmi.uni-jena.de/~sickel/japan(v4).pdf
- Karvonen, Santin & Wenzel, General superconvergence for kernel-based
  approximation, 2025: operator power spaces and boundary caveats.
  https://arxiv.org/abs/2505.11435
- Official Codex non-interactive documentation (consulted 21 Oct 2026):
  https://developers.openai.com/codex/non-interactive-mode

No numerical test is a proof of Sobolev membership, mixing, spectral asymptotics,
source maximality, or the validity of an external theorem at its endpoint.
