#!/usr/bin/env python3
"""Independent numerical audit of the conditional one-factor benchmark.

Tests finite-dimensional identities and a small exact-kernel implementation.
It does NOT prove Sobolev membership, a source exponent, or a learning rate.
No network, trading, existing-repository changes, or production jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.special import ndtr
from scipy.spatial.distance import cdist

D_STATE = 6
S_REG = 4.5
SIGMA_F = 0.04
SIGMA_E = 0.08
LENGTHSCALE = 2.0
SHOCK_BOUND = np.sqrt(3.0)


def psi(states: np.ndarray, kind: str, terms: int = 8) -> np.ndarray:
    """Bounded smooth control or finite approximation of a critical series."""
    s = np.asarray(states, dtype=float)
    if s.ndim != 2 or s.shape[1] != D_STATE or not np.all(np.isfinite(s)):
        raise ValueError('states must be finite with shape (T, 6)')
    if kind == 'smooth':
        return np.mean(s, axis=1)
    if kind != 'critical_truncated' or not 1 <= terms <= 20:
        raise ValueError('invalid target kind or truncation level')
    ratio = 2.0 ** (-S_REG)
    normalizer = -np.log1p(-ratio)
    out = np.zeros(len(s))
    for j in range(1, terms + 1):
        out += (ratio**j / j) * np.mean(np.cos(np.pi * 2.0**j * s), axis=1)
    return out / normalizer


def truncation_bound(terms: int) -> float:
    r = 2.0 ** (-S_REG)
    return float(r**(terms + 1) / ((terms + 1) * (1.0 - r) * (-np.log1p(-r))))


def premium(states: np.ndarray, betas: np.ndarray, kind: str) -> np.ndarray:
    # The interval [0.72, 0.88] lies strictly below the inverse-map critical point 1.
    effective_sigma = np.sqrt(SIGMA_F**2 + SIGMA_E**2 / float(betas @ betas))
    return effective_sigma * (0.80 + 0.08 * psi(states, kind))


def stationary_states(t: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    if t < 1 or abs(rho) >= 1:
        raise ValueError('require t >= 1 and |rho| < 1')
    latent = np.empty((t, D_STATE))
    latent[0] = rng.normal(size=D_STATE)
    for j in range(1, t):
        latent[j] = rho * latent[j-1] + np.sqrt(1.0-rho**2)*rng.normal(size=D_STATE)
    return 2.0 * ndtr(latent) - 1.0


def simulate(t: int, rho: float, betas: np.ndarray, kind: str, seed: int):
    rng = np.random.default_rng(seed)
    states = stationary_states(t, rho, rng)
    m = premium(states, betas, kind)
    shocks = rng.uniform(-SHOCK_BOUND, SHOCK_BOUND, size=(t, len(betas) + 1))
    factors = m + SIGMA_F * shocks[:, 0]
    returns = factors[:, None] * betas + SIGMA_E * shocks[:, 1:]
    return states, returns


def oracle_weights(states: np.ndarray, betas: np.ndarray, kind: str) -> np.ndarray:
    """Actual stock weights w = W/N, not the scalar policy W."""
    m = premium(states, betas, kind)
    b2 = float(betas @ betas)
    denominator = SIGMA_E**2 + b2 * (SIGMA_F**2 + m*m)
    return (m / denominator)[:, None] * betas


def conditional_moments(weights: np.ndarray, states: np.ndarray,
                        betas: np.ndarray, kind: str):
    m = premium(states, betas, kind)
    exposure = weights @ betas
    first = exposure * m
    second = SIGMA_E**2 * np.sum(weights**2, axis=1) + (SIGMA_F**2+m*m)*exposure**2
    return first, second


def metrics(weights: np.ndarray, states: np.ndarray, betas: np.ndarray, kind: str):
    first, second = conditional_moments(weights, states, betas, kind)
    mean, moment2 = float(np.mean(first)), float(np.mean(second))
    variance = moment2 - mean*mean
    if variance <= 0:
        raise AssertionError('non-positive integrated variance')
    optimal = oracle_weights(states, betas, kind)
    delta = weights - optimal
    _, regret_rows = conditional_moments(delta, states, betas, kind)
    p, _ = conditional_moments(optimal, states, betas, kind)
    loss = 1.0 - 2.0*mean + moment2
    regret = float(np.mean(regret_rows))
    np.testing.assert_allclose(loss - (1.0 - np.mean(p)), regret, atol=2e-12, rtol=2e-12)
    return {'loss': loss, 'regret': regret, 'monthly_signed_sharpe': mean/np.sqrt(variance),
            'annual_marginal_sharpe': np.sqrt(12.0)*mean/np.sqrt(variance)}


def matern(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.sqrt(3.0) * cdist(x, y) / LENGTHSCALE
    return (1.0+d) * np.exp(-d)


def fit_exact(states: np.ndarray, returns: np.ndarray, ridge: float):
    if ridge <= 0:
        raise ValueError('ridge must be positive')
    t, n = returns.shape
    gram = matern(states, states) * (returns @ returns.T) / n**2
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] < -1e-12:
        raise AssertionError('managed Gram matrix not positive semidefinite')
    a = np.linalg.solve(gram+t*ridge*np.eye(t), np.ones(t))
    residual = float(np.max(np.abs((gram+t*ridge*np.eye(t)) @ a - 1.0)))
    if residual > 2e-10:
        raise AssertionError('ridge linear-system residual too large')
    return a, gram, residual


def predict_weights(states_eval: np.ndarray, states_train: np.ndarray,
                    returns: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    # W_i = sum_t a_t R_it k_t / N; actual weights require a second division by N.
    n = returns.shape[1]
    return (matern(states_eval, states_train) @ (coefficients[:, None] * returns)) / n**2


def run_audit(output: Path) -> dict:
    started = time.monotonic()
    checks: list[str] = []
    n = 12
    betas = np.linspace(0.6, 1.4, n)
    b2 = float(betas @ betas)
    d0 = SIGMA_E**2 + b2 * SIGMA_F**2
    eval_states = np.random.default_rng(91200).uniform(-1.0, 1.0, (4096, D_STATE))
    max_normal_error = 0.0
    for kind in ('smooth', 'critical_truncated'):
        m = premium(eval_states, betas, kind)
        w = oracle_weights(eval_states, betas, kind)
        mu = m[:, None] * betas
        sigma_w = SIGMA_E**2*w + ((SIGMA_F**2+m*m)*(w @ betas))[:, None]*betas
        max_normal_error = max(max_normal_error, float(np.max(np.abs(sigma_w-mu))))
        np.testing.assert_allclose(sigma_w, mu, rtol=2e-13, atol=2e-15)
        checks.append(f'{kind}: conditional normal equation and E4 algebra')
        p1, p2 = conditional_moments(w, eval_states, betas, kind)
        p_explicit = b2*m*m/(d0+b2*m*m)
        np.testing.assert_allclose(p1, p2, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(p1, p_explicit, rtol=2e-13, atol=2e-15)
        p = float(np.mean(p1))
        assert 0.0 < p < 1.0
        oracle = metrics(w, eval_states, betas, kind)
        np.testing.assert_allclose(oracle['monthly_signed_sharpe'], np.sqrt(p/(1-p)), rtol=2e-13)
        checks.append(f'{kind}: oracle mean/second moment, unconditional marginal Sharpe')
        random_w = np.random.default_rng(44).normal(0, .05, w.shape)
        result = metrics(random_w, eval_states, betas, kind)
        assert result['regret'] >= 0
        checks.append(f'{kind}: exact regret identity for arbitrary weights')
        for k in range(12):
            matrix = SIGMA_E**2*np.eye(n)+(SIGMA_F**2+m[k]**2)*np.outer(betas,betas)
            ev = np.linalg.eigvalsh(matrix)
            np.testing.assert_allclose(ev[0], SIGMA_E**2, rtol=1e-12)
            np.testing.assert_allclose(ev[-1], d0+b2*m[k]**2, rtol=1e-12)
        checks.append(f'{kind}: second-moment lower/upper matrix bounds')
        assert np.max(np.abs(psi(eval_states, kind))) <= 1.0 + 1e-14
        assert m.min() > 0 and m.max() < np.sqrt(d0/b2)
        checks.append(f'{kind}: bounded premium and injective composition interval')

    error = float(np.max(np.abs(psi(eval_states,'critical_truncated',8)-psi(eval_states,'critical_truncated',12))))
    assert error <= truncation_bound(8)+1e-14
    checks.append('finite-series uniform tail bound (NOT a maximal-source test)')

    # One large independent conditional sample, not training data.
    rng = np.random.default_rng(611)
    shocks = rng.uniform(-SHOCK_BOUND,SHOCK_BOUND,(60000,n+1))
    state0 = np.zeros((1,D_STATE))
    m0 = float(premium(state0,betas,'smooth')[0])
    r = (m0+SIGMA_F*shocks[:,0])[:,None]*betas+SIGMA_E*shocks[:,1:]
    target_cov = SIGMA_F**2*np.outer(betas,betas)+SIGMA_E**2*np.eye(n)
    assert np.max(np.abs(r.mean(0)-m0*betas)) < 6*np.sqrt(np.max(np.diag(target_cov))/len(r))
    assert np.max(np.abs(np.cov(r,rowvar=False,ddof=0)-target_cov)) < 0.04*np.max(np.diag(target_cov))
    checks.append('empirical conditional mean/covariance within declared Monte Carlo tolerance')

    pilot = []
    for kind in ('smooth','critical_truncated'):
        for rho in (0.0,0.6):
            for seed in (101,102,103):
                states, returns = simulate(96,rho,betas,kind,seed)
                states2, returns2 = simulate(96,rho,betas,kind,seed)
                assert np.array_equal(states,states2) and np.array_equal(returns,returns2)
                m_bound = np.sqrt(d0/b2)*.88
                return_bound = np.abs(betas)*(m_bound+SIGMA_F*SHOCK_BOUND)+SIGMA_E*SHOCK_BOUND
                assert np.all(np.abs(returns) <= return_bound[None,:]+1e-14)
                # Explicit feature-map identity for the block kernel, at a single row.
                feature_norm2 = np.sum(returns**2,axis=1)/n**2
                assert np.all(feature_norm2 <= np.sum(return_bound**2)/n**2 + 1e-14)
                ridge = 1e-4  # A fixed smoke-test penalty, not claimed to be rate-optimal.
                coeff, gram, residual = fit_exact(states,returns,ridge)
                in_weights = predict_weights(states,states,returns,coeff)
                np.testing.assert_allclose(np.sum(in_weights*returns,axis=1),gram @ coeff,rtol=1e-12,atol=1e-13)
                evaluated = predict_weights(eval_states,states,returns,coeff)
                pilot.append({'target':kind,'rho':rho,'seed':seed,'T':96,'N':n,'lambda':ridge,
                              'linear_system_max_error':residual,**metrics(evaluated,eval_states,betas,kind)})
    checks += ['12 seeded exact-kernel smoke fits; no discarded histories',
               'W versus w=W/N normalization and representer payoff identity',
               'reproducibility, return/managed-feature bounds, independent state evaluation']
    report = {
        'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
        'scope':'numerical algebra and implementation only; not a proof of E1-E6, source maximality, or a rate',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_seconds':time.monotonic()-started,'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__,
        'checks':checks,'check_groups':len(checks),'max_normal_equation_error':max_normal_error,
        'critical_series_terms':8,'uniform_tail_bound':truncation_bound(8),'observed_tail_difference':error,
        'evaluation':'4096 independent uniform states, exact conditional shock integration; finite Monte Carlo integration',
        'theorem_endpoint_r1':'must be matched to the current companion lemma; older August theorem excludes endpoint',
        'maximal_source_order':'not inferred from finite truncation or these tests',
        'pilot':pilot,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite previous receipt: {output}')
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = run_audit(args.output)
    print(json.dumps({k:report[k] for k in ('status','check_groups','max_normal_equation_error','runtime_seconds')},indent=2))
    print(f'report={args.output.resolve()}')


if __name__ == '__main__':
    main()
