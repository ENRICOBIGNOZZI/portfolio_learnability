
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2


def matern_rff(D, s, M=1000, ell=1.0, seed=42):

    if s <= D/2:
        raise ValueError("Serve s > D/2")

    rng = np.random.default_rng(seed)
    df = 2*s-D

    omega = rng.normal(size=(D, M))
    omega /= np.sqrt(rng.chisquare(df, size=M))[None, :]
    omega *= np.sqrt(df)/ell

    phase = rng.uniform(0, 2*np.pi, M)

    return omega, phase


def phi(Z, omega, phase):

    M = omega.shape[1]

    return np.sqrt(2/M)*np.cos(Z @ omega+phase)


def managed_returns(Z, R, omega, phase):

    T, N, D = Z.shape
    M = omega.shape[1]

    G = np.zeros((T, M))

    for t in range(T):
        G[t] = R[t] @ phi(Z[t], omega, phase)/N

    return G


def fit_W(G, lam):

    T, M = G.shape

    # Pseudoinversa della matrice più piccola.
    # Con lam > 0 coincide teoricamente con l'inversa ridge.
    if T <= M:
        theta = G.T @ (np.linalg.pinv(
            G @ G.T+T*lam*np.eye(T)
        ) @ np.ones(T))
    else:
        theta = np.linalg.pinv(
            G.T @ G+T*lam*np.eye(M)
        ) @ (G.T @ np.ones(T))

    return theta


def get_weights(Z_new, theta, omega, phase):

    N = Z_new.shape[-2]

    return phi(Z_new, omega, phase) @ theta/N


def Z_process(T, N, D, rho):

    S = np.zeros((T, N, D))
    S[0] = np.random.normal(size=(N, D))

    for t in range(1, T):
        S[t] = rho*S[t-1]+np.sqrt(1-rho**2)*np.random.normal(size=(N, D))

    Z = 2*norm.cdf(S)-1

    return Z


def psi(z, M):

    p = np.zeros_like(z, dtype=float)

    for m in range(2, M+1):
        p += np.cos(np.pi*m*z)/(m**3.5*np.log(m+1))

    return p


def alphas(Z, A_alpha):

    return A_alpha*np.sqrt(2)*np.cos(np.pi*np.sum(Z, axis=-1))


def betas(Z, A_beta, M):

    return A_beta*psi(Z, M)


def calibrate(SR_ann, N, sigma_eps, lambda_factor, M, safety=1.01):

    z = np.linspace(-1, 1, 2*M+1, endpoint=False)

    v_psi = np.mean(psi(z, M)**2)
    psi_max = psi(np.array([0.0]), M)[0]

    L = lambda_factor @ lambda_factor

    if L >= 1:
        raise ValueError("Serve lambda_factor @ lambda_factor < 1")

    p_star = SR_ann**2/(12+SR_ann**2)

    C = 1+v_psi*L
    H = (np.sqrt(2)+np.sqrt(3)*psi_max*np.sqrt(L))**2+3*psi_max**2

    den = C-safety*p_star*H

    if den <= 0:
        raise ValueError("Sharpe richiesto non ammissibile")

    A_alpha = np.sqrt(safety*p_star*sigma_eps**2/(N*den))
    A_beta = A_alpha

    v_alpha = A_alpha**2
    v_beta = A_beta**2*v_psi

    kappa_0 = N*(v_alpha+v_beta*L)/p_star

    upper_bound = sigma_eps**2+N*A_alpha**2*H

    if kappa_0 <= upper_bound:
        raise ValueError("Covarianza degli shock non ammissibile")

    return A_alpha, A_beta, kappa_0


def gaussian(shape, cutoff=4):

    G = np.random.normal(size=shape)

    bad = np.abs(G) > cutoff

    while np.any(bad):
        G[bad] = np.random.normal(size=np.sum(bad))
        bad = np.abs(G) > cutoff

    v = 1-2*cutoff*norm.pdf(cutoff)/(2*norm.cdf(cutoff)-1)

    return G/np.sqrt(v)


def factors(lambda_factor, Sigma_factor, T, cutoff=4):

    K = len(lambda_factor)

    F = np.random.multivariate_normal(
        mean=lambda_factor,
        cov=Sigma_factor,
        size=T
    )

    G = F-lambda_factor

    L = lambda_factor @ lambda_factor

    r2 = np.sum(G**2, axis=1)+(G @ lambda_factor)**2/(1-L)

    bad = r2 > cutoff**2

    while np.any(bad):

        F[bad] = np.random.multivariate_normal(
            mean=lambda_factor,
            cov=Sigma_factor,
            size=np.sum(bad)
        )

        G = F-lambda_factor
        r2 = np.sum(G**2, axis=1)+(G @ lambda_factor)**2/(1-L)
        bad = r2 > cutoff**2

    v = chi2.cdf(cutoff**2, K+2)/chi2.cdf(cutoff**2, K)

    return lambda_factor+(F-lambda_factor)/np.sqrt(v)


def noise(mu, B, Sigma_factor, sigma_eps, kappa_0):

    T, N = mu.shape

    h = np.zeros((T, N))

    for t in range(T):

        d = (kappa_0-sigma_eps**2-mu[t]@mu[t])*mu[t] \
            -B[t] @ (Sigma_factor @ (B[t].T @ mu[t]))

        h[t] = d/np.sqrt(mu[t]@d)

    E = sigma_eps*gaussian((T, N))+h*gaussian((T, 1))

    return E, h


def returns(Z, A_alpha, A_beta, lambda_factor, Sigma_factor,
            sigma_eps, kappa_0, M):

    T, N, D = Z.shape

    A = alphas(Z, A_alpha)
    B = betas(Z, A_beta, M)

    mu = A+B @ lambda_factor

    F = factors(lambda_factor, Sigma_factor, T)
    E, h = noise(mu, B, Sigma_factor, sigma_eps, kappa_0)

    R = A+np.sum(B*F[:, None, :], axis=2)+E

    return R, mu, h


def true_weights(Z, A_alpha, A_beta, lambda_factor, kappa_0, M):

    mu = alphas(Z, A_alpha)+betas(Z, A_beta, M) @ lambda_factor

    return mu/kappa_0


K = 3
D = 3
s = 3

b = 2*s/D
C_lambda_reg = 1

lambda_1 = 0.2
lambda_2 = 0.1
lambda_3 = 0.05

ell = 1.0

N_values = [300, 600, 1200]
rho_values = [0, 0.5, 0.95]

trials = 100
TIMESTEPS = 5*np.array([100, 200, 400, 800, 1600])

T = max(TIMESTEPS)
M = 1000

SR_ann = 3
sigma_eps = 0.04

lambda_factor = np.array([lambda_1, lambda_2, lambda_3])

Sigma_factor = np.eye(K)-lambda_factor[:, None]*lambda_factor[None, :]

p_star = SR_ann**2/(12+SR_ann**2)
Q_star = 1-p_star

omega, phase = matern_rff(D, s, M, ell)

results = []

for N in N_values:

    A_alpha, A_beta, kappa_0 = calibrate(
        SR_ann, N, sigma_eps, lambda_factor, M
    )

    print("N =", N)
    print("A_alpha =", A_alpha)
    print("A_beta =", A_beta)
    print("kappa_0 =", kappa_0)

    for rho in rho_values:

        print("rho =", rho)

        for trial in range(trials):

            np.random.seed(42+trial)

            Z = Z_process(T, N, D, rho)

            R, mu, h = returns(
                Z, A_alpha, A_beta, lambda_factor,
                Sigma_factor, sigma_eps, kappa_0, M
            )

            G = managed_returns(Z, R, omega, phase)

            # Verifica E4 senza costruire matrici N x N
            if trial == 0:

                A0 = alphas(Z[0], A_alpha)
                B0 = betas(Z[0], A_beta, M)
                u = mu[0]

                e4 = (sigma_eps**2+u@u)*u \
                    +B0 @ (Sigma_factor @ (B0.T @ u)) \
                    +h[0]*(h[0]@u)-kappa_0*u

                print("Errore E4 =", np.max(np.abs(e4)))

            for t in TIMESTEPS:

                n = t//5

                lambda_reg = C_lambda_reg*n**(-b/(b+1))

                theta = fit_W(G[:n], lambda_reg)

                Rp_hat = G[n:t] @ theta

                w_true = mu[n:t]/kappa_0
                Rp_true = np.sum(w_true*R[n:t], axis=1)

                regret = np.mean((Rp_hat-Rp_true)**2)
                Q_hat = np.mean((1-Rp_hat)**2)

                SR_hat = np.sqrt(12)*np.mean(Rp_hat)/np.std(Rp_hat)

                results.append({
                    "N": N,
                    "rho": rho,
                    "trial": trial+1,
                    "T_train": n,
                    "T_test": t-n,
                    "lambda_reg": lambda_reg,
                    "A_alpha": A_alpha,
                    "A_beta": A_beta,
                    "kappa_0": kappa_0,
                    "Q_star": Q_star,
                    "Q_hat": Q_hat,
                    "regret": regret,
                    "SR_hat": SR_hat
                })


df = pd.DataFrame(results)

df.to_csv("Q_trials.csv", index=False)

summary = df.groupby(
    ["N", "rho", "T_train", "T_test"],
    as_index=False
).agg(
    lambda_reg=("lambda_reg", "first"),
    Q_mean=("Q_hat", "mean"),
    Q_std=("Q_hat", "std"),
    regret_mean=("regret", "mean"),
    regret_std=("regret", "std"),
    SR_mean=("SR_hat", "mean"),
    SR_std=("SR_hat", "std"),
    trials=("trial", "count")
)

summary["Q_se"] = summary["Q_std"]/np.sqrt(summary["trials"])
summary["regret_se"] = summary["regret_std"]/np.sqrt(summary["trials"])
summary["SR_se"] = summary["SR_std"]/np.sqrt(summary["trials"])

summary.to_csv("Q_summary.csv", index=False)

summary[summary["rho"] == 0.5].to_csv("Q_vs_N.csv", index=False)

summary[summary["N"] == 600].to_csv("Q_vs_rho.csv", index=False)
