# Codex implementation brief: theorem-compatible baseline DGP

## Objective
Implement the baseline simulation DGP for **The Law of Portfolio Learnability** exactly as specified below, audit Conditions E1-E6, and prepare the code so Monte Carlo experiments can be run afterward.

Do **not** redesign the theory. Do **not** modify the paper's loss, Sharpe definition, estimator, theorem statements, or RKHS normalization. Do **not** prescribe the managed eigenvalues by hand. The economic return model has exactly **3 factors**.

The accompanying PDF `DGP_Baseline_Spec.pdf` is the mathematical source of truth. If any code design conflicts with the PDF, follow the PDF.

---

## 1. Dimensions and fixed conventions

Use the baseline:

- `D = 6` characteristics.
- `N = 600` stocks, and enforce `N % 3 == 0`.
- `G = N / 3` groups.
- `K_F = 3` economic factors.
- Matérn-3/2 portfolio kernel on `(-1,1)^D`.
- Kernel diagonal normalized to `K(z,z)=1`.
- Baseline length scale `ell = 1.0`.
- Baseline persistence `rho_d = 0.95` for all `d`, but code must accept a vector of six values with absolute value strictly below 1.

Use double precision throughout.

---

## 2. Persistent latent characteristics and rank normalization

For each group `g=1,...,G` and characteristic `d=1,...,D`, simulate a stationary Gaussian AR(1):

\[
S_{g,d,t+1}=\rho_d S_{g,d,t}+\sqrt{1-\rho_d^2}\,\eta_{g,d,t+1},
\qquad \eta_{g,d,t+1}\sim N(0,1).
\]

Initialize from stationarity:

\[
S_{g,d,0}\sim N(0,1).
\]

Map to a continuous **population rank**:

\[
U_{g,d,t}=\Phi(S_{g,d,t})\in(0,1).
\]

Each group creates exactly three stocks, indexed by role `r in {0,1,2}`. Define

\[
U_{g,r,d,t}=\bigl(U_{g,d,t}+r/3\bigr)\bmod 1,
\qquad
Z_{g,r,d,t}=2U_{g,r,d,t}-1.
\]

Important:

- This is the theorem-compatible baseline rank transformation.
- It is continuous and keeps the population characteristic distribution non-discrete.
- Do **not** replace it in the baseline with literal empirical cross-sectional ranks. Empirical ranks may be added later as a robustness specification, clearly labeled as such.
- The same phase `r/3` is added to all six coordinates of a stock.

The process `Z_t` is a deterministic measurable transform of the stable Gaussian AR state.

---

## 3. Smooth balanced factor loadings

Let

\[
u(z)=\frac{z_1+1}{2}.
\]

Fix `c_beta > 0` and define

\[
\beta(z)=\sqrt{c_\beta}
\begin{pmatrix}
\sqrt{2/3}\cos(2\pi u(z))\\
\sqrt{2/3}\sin(2\pi u(z))\\
1/\sqrt3
\end{pmatrix}.
\]

This is the **only** baseline loading map.

The code must verify algebraically/numerically:

\[
\|\beta(z)\|^2=c_\beta \quad \forall z,
\]

and, defining `B_t` as the `N x 3` matrix with row `i = beta(Z_i,t)'`,

\[
\frac1N B_t'B_t=\Gamma_\beta=\frac{c_\beta}{3}I_3
\]

for every simulated date to machine precision.

Reason: the three phase shifts 0, 1/3, 2/3 yield an exact trigonometric orthogonality identity within every three-stock group.

Acceptance tolerance:

- max beta-norm error `< 1e-12`.
- operator/Frobenius norm of balance error `< 1e-12`.

---

## 4. Factor process

Fix a constant vector `mu_F in R^3` satisfying

\[
0<\|\mu_F\|<1.
\]

Generate factors independently over time:

\[
F_{t+1}\sim N\left(\mu_F, I_3-\mu_F\mu_F'\right).
\]

Before simulation, check numerically that `I_3 - mu_F mu_F'` is positive definite.

This normalization implies exactly

\[
E[F_{t+1}]=\mu_F,
\qquad
E[F_{t+1}F_{t+1}']=I_3.
\]

Do not use a state-dependent factor mean in the baseline. Do not reverse-engineer factor premia from a desired portfolio policy.

---

## 5. Idiosyncratic shocks and stock returns

Generate

\[
\varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2I_N)
\]

i.i.d. over time and independent of factors and characteristics.

Returns are

\[
R_{t+1}=B_tF_{t+1}+\varepsilon_{t+1}.
\]

Conditioning on investor information `F_t` that contains current `Z_t` but not the next factor/idiosyncratic innovations,

\[
R_{t+1}\mid\mathcal F_t\sim N(m_t,V_t),
\]

with

\[
m_t=B_t\mu_F,
\]

\[
V_t=B_t(I_3-\mu_F\mu_F')B_t'+\sigma_\varepsilon^2I_N,
\]

and conditional second moment

\[
S_t:=E[R_{t+1}R_{t+1}'\mid\mathcal F_t]
=B_tB_t'+\sigma_\varepsilon^2I_N.
\]

---

## 6. Portfolio loss and exact population policy

For a characteristic policy `W(z)`, define stock weights

\[
w_t(W)=\frac1N\left(W(Z_{1,t}),\ldots,W(Z_{N,t})\right)'
\]

and payoff

\[
R^p_{t+1}(W)=w_t(W)'R_{t+1}.
\]

Conditional loss:

\[
Q_t(W)=E[(1-R^p_{t+1}(W))^2\mid\mathcal F_t]
=1-2w_t(W)'m_t+w_t(W)'S_tw_t(W).
\]

The unconstrained conditional minimizer is

\[
w_t^\star=S_t^{-1}m_t.
\]

Use the identity

\[
(BB'+\sigma^2 I_N)^{-1}B=B(B'B+\sigma^2 I_3)^{-1}
\]

and the exact balance `B_t'B_t/N = Gamma_beta` to derive

\[
w_t^\star
=\frac1N B_t\left(\Gamma_\beta+\frac{\sigma_\varepsilon^2}{N}I_3\right)^{-1}\mu_F.
\]

Therefore the population characteristic policy is

\[
\boxed{
W^\star(z)=\beta(z)'
\left(\Gamma_\beta+\frac{\sigma_\varepsilon^2}{N}I_3\right)^{-1}\mu_F
}
\]

and because `Gamma_beta = c_beta/3 I_3`,

\[
\boxed{
W^\star(z)=\frac{\beta(z)'\mu_F}
{c_\beta/3+\sigma_\varepsilon^2/N}.
}
\]

This formula must be derived in code/docs from the DGP. Never hard-code a target `W_star` independent of this derivation.

Acceptance tests, every audited date:

1. Compute `m_t`, `S_t`, `w_star` from the formula above.
2. Verify `||S_t @ w_star - m_t|| / max(1, ||m_t||) < 1e-11`.
3. Evaluate the scalar function `W_star(Z_i,t)` and verify that `w_star[i] = W_star(Z_i,t)/N` to `<1e-11`.

---

## 7. Audit of paper Conditions E1-E6

### E1: stationarity and beta-mixing

Analytical argument:

- The latent AR state is finite-dimensional stationary Gaussian with spectral radius below one.
- Therefore it is geometrically beta-mixing.
- `Z_t` is a measurable transform of that state, so its beta-mixing coefficients cannot be larger.
- Adding independent next-period Gaussian factors and idiosyncratic shocks preserves geometric mixing.
- Hence `(Z_t,R_{t+1})` is geometrically beta-mixing.
- Therefore for every finite `gamma` there is `C_gamma` such that `beta(k) <= C_gamma k^{-gamma}`.
- With the Matérn choice below, `b=3/2`, so the paper condition `gamma>b` is automatically satisfiable.

No Monte Carlo estimate of `gamma` is needed for the theorem audit.

### E2(a): Gaussian-type managed-feature tails

Define the Gram matrix

\[
(K_t)_{ij}=K(Z_{i,t},Z_{j,t})
\]

and the managed feature

\[
X_t=\frac1N\sum_i R_{i,t+1}K(\cdot,Z_{i,t}).
\]

Identity:

\[
\|X_t\|_{\mathcal H_K}^2
=\frac1{N^2}R_{t+1}'K_tR_{t+1}.
\]

Because `K_t` is PSD and its diagonal is at most `kappa_K=1`,

\[
\|K_t\|_{op}\le N\kappa_K.
\]

Also

\[
E[\|R_{t+1}\|^2\mid\mathcal F_t]=N(c_\beta+\sigma_\varepsilon^2)
\]

and

\[
\|V_t\|_{op}\le Nc_\beta/3+\sigma_\varepsilon^2.
\]

A sufficient uniform constant is

\[
\boxed{B_X^2=4\kappa_K(c_\beta+\sigma_\varepsilon^2).}
\]

Using the exact noncentral Gaussian quadratic-form mgf, verify analytically/numerically that

\[
E\left[
\exp\left(\|X_t\|^2/B_X^2\right)
\mid\mathcal F_t
\right]
\le e^{1/2}<2.
\]

For numerical audit, compute the exact conditional log-mgf from eigenvalues of

\[
A_t^{1/2}V_tA_t^{1/2},\qquad A_t=K_t/N^2,
\]

rather than relying on Monte Carlo tail counts.

### E2(b): fourth moment

For any RKHS policy `V`, conditional on current information, `R^p_{t+1}(V)` is scalar Gaussian. Therefore

\[
E[(R^p(V))^4\mid\mathcal F_t]
\le 3\{E[(R^p(V))^2\mid\mathcal F_t]\}^2.
\]

Thus `K_4=3` exactly as a valid uniform upper bound.

### E3: RKHS membership

The three components of `beta(z)` are trigonometric/constant and therefore smooth on the compact cube. With Matérn-3/2 in `D=6`, the RKHS is equivalent to Sobolev order

\[
s=\nu+D/2=9/2.
\]

Therefore every `beta_k` belongs to the Matérn RKHS. `W_star` is a fixed linear combination of these three components, hence

\[
W^\star\in\mathcal H_K,
\qquad
\|W^\star\|_{\mathcal H_K}<\infty.
\]

For a DGP class rather than one calibration, put compact bounds on parameters to obtain a common RKHS norm upper bound `R`.

### E4: conditional pricing

Since `w_star` solves

\[
S_t w_t^\star=m_t
\]

for every date, for every admissible policy `V` with weight vector `v_t(V)`:

\[
E[(1-R^p(W^\star))R^p(V)\mid\mathcal F_t]
=v_t(V)'(m_t-S_tw_t^\star)=0.
\]

This must be checked numerically at machine precision as part of the DGP audit.

### E5: managed spectrum

Define the population kernel operator

\[
T_K=E[K_Z\otimes K_Z]
\]

and the managed-payoff covariance operator

\[
\Sigma_{\mathcal H_K}=E[X_t\otimes X_t].
\]

The idiosyncratic part contributes exactly

\[
(\sigma_\varepsilon^2/N)T_K.
\]

For the factor part, for any `h in H_K`, use Jensen/Cauchy:

\[
E\left\|
\frac1N\sum_i \beta(Z_i)h(Z_i)
\right\|^2
\le
E\left[\frac1N\sum_i\|\beta(Z_i)\|^2h(Z_i)^2\right]
=c_\beta\langle h,T_Kh\rangle.
\]

Hence the operator sandwich

\[
\boxed{
\frac{\sigma_\varepsilon^2}{N}T_K
\preceq
\Sigma_{\mathcal H_K}
\preceq
\left(c_\beta+\frac{\sigma_\varepsilon^2}{N}\right)T_K.
}
\]

Therefore the positive eigenvalues of the managed operator have the same decay exponent as `T_K`.

For Matérn-3/2 on a six-dimensional compact regular domain with uniform density,

\[
\lambda_j(T_K)\asymp j^{-3/2}.
\]

Therefore

\[
\boxed{\mu_j(\Sigma_{\mathcal H_K})\asymp j^{-3/2},\quad b=3/2.}
\]

Numerical spectrum audit:

- Use one large, independent population/Nyström sample of characteristics, not the Monte Carlo training paths.
- Construct the kernel integral operator approximation and the managed operator approximation using the same quadrature/sample.
- Verify the eigenvalue sandwich numerically for the range where numerical eigenvalues are above tolerance.
- Fit log-eigenvalue versus log-index slope only on a predeclared interior range; do not fit the first few finite-rank-dominated directions or numerical noise floor.
- Expected slope is approximately `-1.5`.

### E6: nondegenerate optimal Sharpe

Let

\[
g=c_\beta/3.
\]

Then

\[
q^\star
=E[R^p(W^\star)\mid\mathcal F_t]
=E[(R^p(W^\star))^2\mid\mathcal F_t]
=\frac{g}{g+\sigma_\varepsilon^2/N}\|\mu_F\|^2.
\]

This is constant over time. Since `0<||mu_F||<1`, `0<q_star<1`. Therefore

\[
(SR^\star)^2=\frac{q^\star}{1-q^\star}.
\]

For a uniform DGP class, restrict `||mu_F||` to a compact interval strictly inside `(0,1)`.

---

## 8. Baseline numerical calibration

Use initially:

- `N = 600`
- `D = 6`
- `rho = [0.95]*6`
- `nu = 1.5`
- `ell = 1.0`
- `c_beta = 0.04**2`
- `sigma_eps = 0.08`
- `mu_F = [0.10, 0.06, 0.03]`

These are calibration values, not theory assumptions.

Compute and save:

- `Gamma_beta`
- analytic `W_star` coefficients
- analytic `q_star`
- analytic `SR_star`
- `B_X_squared = 4*kappa_K*(c_beta + sigma_eps**2)`
- `K4 = 3`
- theoretical `b = 1.5`

---

## 9. Required code structure

Use a modular implementation. Adapt paths to the repository if existing conventions differ, but keep these logical modules:

1. `dgp_balanced_factor.py`
   - parameter dataclass
   - latent AR simulation
   - population-rank transform
   - phase-balanced triplet construction
   - beta loading function
   - factor/idiosyncratic simulation
   - return generation
   - analytic `W_star`, `q_star`, `SR_star`

2. `audit_dgp.py`
   - exact balance tests
   - normal-equation/E4 tests
   - Gaussian quadratic-form E2(a) audit
   - fourth-moment E2(b) audit
   - E5 operator sandwich/spectrum audit
   - E6 analytic versus Monte Carlo audit

3. Outputs:
   - `dgp_audit.json` with every parameter, every analytic constant, and pass/fail status
   - `dgp_spectrum.csv`
   - `dgp_spectrum.png`
   - concise methodology note reproducing the exact equations

All random number generation must use explicit `numpy.random.Generator` instances and deterministic seeds. Do not rely on global RNG state.

---

## 10. Hard acceptance criteria

The DGP implementation is not complete until all are true:

- exact loading norm identity passes
- exact `B_t' B_t / N` identity passes for every audited date
- factor second-moment audit passes
- conditional normal equation passes
- `W_star` coordinate representation passes
- exact Gaussian quadratic-form E2(a) bound passes for audited dates
- E2(b) upper bound `<=3` passes analytically and numerically
- E3 is documented from the smooth closed-form beta map
- E4 residual is machine-zero
- E5 operator sandwich is numerically respected and the interior spectral slope is compatible with `-1.5`
- E6 analytic Sharpe matches long-run Monte Carlo within predeclared MC tolerance

Do not begin tuning the learning algorithm until this DGP audit is green.
