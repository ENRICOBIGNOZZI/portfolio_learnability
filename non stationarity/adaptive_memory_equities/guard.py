"""Approximate simultaneous block-bootstrap guard, with exact full-history fallback."""
import numpy as np
import pandas as pd


def block_indices(n, replicates=2000, block=12, seed=20261003):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replicates, int(np.ceil(n/block))))
    return ((starts[:,:,None] + np.arange(block)) % n).reshape(replicates,-1)[:,:n]


def select_guard(baseline, candidates, config):
    baseline, candidates = np.asarray(baseline), np.asarray(candidates)
    q = candidates.shape[1] if candidates.ndim == 2 else 0
    fallback = dict(index=None, active=False, reason='INSUFFICIENT_PREQUENTIAL_HISTORY',
                    candidates=q, lcb=[], observations=len(baseline))
    window = config.get('window', 60)
    if len(baseline) < window or q == 0:
        return fallback
    b, c = baseline[-window:], candidates[-window:]
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        return {**fallback, 'reason':'INVALID_PREQUENTIAL_PAYOFF'}
    delta = (1-b[:,None])**2 - (1-c)**2
    indices = block_indices(window, config.get('replicates',2000), config.get('block',12), config.get('seed',20261003))
    means = delta.mean(axis=0)
    boot = delta[indices].mean(axis=1)
    se = boot.std(axis=0, ddof=1)
    valid = se > 1e-12
    # Degenerate series do not create an automatic win, even if their mean is positive.
    lcb = np.full(q, -np.inf)
    critical = None
    if valid.any():
        stat = ((boot[:,valid]-means[valid]) / se[valid]).max(axis=1)
        critical = max(0., float(np.quantile(stat, config.get('level',.95))))
        lcb[valid] = means[valid] - critical * se[valid]
    best = int(np.argmax(lcb))
    active = bool(lcb[best] > 0)
    return dict(index=best if active else None, active=active,
                reason='POSITIVE_SIMULTANEOUS_LCB' if active else 'NO_POSITIVE_SIMULTANEOUS_LCB',
                candidates=q, observations=window, lcb=[float(x) if np.isfinite(x) else None for x in lcb],
                critical=critical, mean_delta=means.tolist(), standard_errors=se.tolist())


def mature_records(records, origin):
    admitted = [r for r in records if pd.Timestamp(r['available_at']) <= pd.Timestamp(origin)]
    admitted.sort(key=lambda r: r['return_date'])
    if any(pd.Timestamp(r['decision_date']) >= pd.Timestamp(r['return_date']) for r in admitted):
        raise ValueError('Invalid prequential decision chronology.')
    return admitted


def soft_weights(losses, temperature=8.):
    losses = np.asarray(losses, float)
    if losses.ndim != 2 or not losses.shape[1]:
        raise ValueError('Expected date by expert losses.')
    scores = losses.mean(axis=0) if len(losses) else np.zeros(losses.shape[1])
    z = -temperature*(scores-scores.min())
    p = np.exp(z)
    return p / p.sum()
