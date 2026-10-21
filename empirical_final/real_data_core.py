"""Past-only algebra for the five JKP experiments (no simulated observations)."""
from __future__ import annotations

import numpy as np


def spectrum(g):
    g = np.asarray(g, dtype=float)
    if g.ndim != 2 or len(g) < 2 or not np.isfinite(g).all():
        raise ValueError('A finite monthly managed-return matrix is required')
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    active = s > max(s[0], 1e-30)*1e-6
    if not active.any():
        raise ValueError('Unidentified managed-return operator')
    return u[:, active], s[active], vt[active]


def fit_path(g, values, kind='ridge'):
    """Ridge ratios, exact effective complexities, or past-only PCR fractions.

    Ratios multiply trace(Sigma)/rank. The numerical penalty is re-estimated
    from the refit history after choosing the ratio on prior validation data.
    PCR uses the uncentered ECONOMIC operator, not characteristic covariance.
    """
    u, s, vt = spectrum(g)
    n, rank = len(g), len(s)
    mu = s*s/n
    projection = u.T @ np.ones(n)
    values = np.asarray(values, float)
    betas, penalties, complexities, valid = [], [], [], []
    for value in values:
        okay = True
        if kind == 'pcr':
            if not 0 < value <= 1:
                raise ValueError('PCR fractions must lie in (0, 1]')
            k = max(1, int(np.ceil(rank*value)))
            while k < rank and abs(mu[k-1]-mu[k]) <= mu[k-1]*1e-8:
                k += 1
            filters = np.zeros(rank)
            filters[:k] = 1
            penalty, complexity = 0., k
        else:
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Positive finite ridge parameters required')
            if kind == 'complexity':
                okay = value < rank
                lo, hi = mu[-1]*1e-12, mu[0]*rank*1e6
                for _ in range(80):
                    mid = np.sqrt(lo*hi)
                    if np.sum(mu/(mu+mid)) > value:
                        lo = mid
                    else:
                        hi = mid
                penalty = np.sqrt(lo*hi)
            elif kind == 'ridge':
                penalty = value*mu.mean()
            else:
                raise ValueError('Unknown spectral filter')
            filters = mu/(mu+penalty)
            complexity = filters.sum()
        betas.append(vt.T @ (projection/s*filters))
        penalties.append(penalty)
        complexities.append(complexity)
        valid.append(okay)
    signal = projection**2/n
    top = max(1, int(np.ceil(rank*.1)))
    shares = mu/mu.sum()
    diagnostics = dict(rank=rank, top_decile_mass=float(shares[:top].sum()),
                       participation_rank=float(1/(shares@shares)),
                       top_decile_signal_share=float(signal[:top].sum()/max(signal.sum(), 1e-30)),
                       historical_opportunity=float(signal.sum()))
    return dict(beta=np.column_stack(betas), penalty=np.array(penalties),
                complexity=np.array(complexities), valid=np.array(valid),
                eigenvalues=mu, signal=signal, diagnostics=diagnostics)


def chronological_fit(g, decision, history, validation, values, kind='ridge'):
    """T is the total available refit history; validation is inside that T."""
    if history <= validation+2 or decision < history:
        raise ValueError('Insufficient chronological history')
    train = np.arange(decision-history, decision-validation)
    valid = np.arange(decision-validation, decision)
    refit = np.arange(decision-history, decision)
    first = fit_path(g[train], values, kind)
    val_returns = g[valid] @ first['beta']
    losses = np.mean((1-val_returns)**2, axis=0)
    final = fit_path(g[refit], values, kind)
    eligible = first['valid'] & final['valid']
    if not eligible.any():
        raise ValueError('No feasible complexity in inner training')
    choice = int(np.argmin(np.where(eligible, losses, np.inf)))
    return final | dict(choice=choice, validation_loss=losses, eligible=eligible,
                        validation_returns=val_returns, calibration_beta=first['beta'][:, choice],
                        train=train, validation=valid, refit=refit)


def match_groups(rows, tolerance=.15):
    """Match prior validation Sharpes; never certify population opportunity."""
    best = None
    for i, left in enumerate(rows):
        for right in rows[i+1:]:
            gap = abs(left['validation_sharpe']-right['validation_sharpe'])
            if not np.isfinite(gap) or gap > tolerance:
                continue
            spread = (abs(left['top_decile_mass']-right['top_decile_mass']) +
                      abs(left['top_decile_signal_share']-right['top_decile_signal_share']))
            key = (spread, -gap, left['variant'], right['variant'])
            if best is None or key > best[0]:
                best = key, (left['variant'], right['variant'])
    return None if best is None else best[1]


def nested_universe(ids, cap, seed):
    """Stable identifier hash: nested N samples independent of future payoffs."""
    ids = np.asarray(ids, dtype=np.int64)
    # SplitMix64, deterministic across Python versions and input row order.
    with np.errstate(over='ignore'):
        h = ids.astype(np.uint64)+np.uint64(seed)+np.uint64(0x9e3779b97f4a7c15)
        h = (h ^ (h >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        h = (h ^ (h >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        h ^= h >> np.uint64(31)
    order = np.lexsort((ids, h))
    return order[:min(cap, len(order))]


def past_stock_betas(ids, observations, factor_rows, prior_months=12.):
    """Trailing observed returns only, ridge towards market beta 1, others 0.

    Missing historical returns contribute no regression observations. The
    fixed prior supplies exposures for new listings without removing them.
    factor_rows[t] and observations[t] refer to the same realized payoff date.
    """
    n = len(ids)
    k = factor_rows.shape[1]
    prior = np.zeros((n, k))
    prior[:, 0] = 1
    if not observations:
        return prior
    factors = np.asarray(factor_rows, float)
    available = np.isfinite(factors).all(axis=1)
    if not available.any():
        return prior
    factors = factors[available]
    scale = np.maximum(factors.std(axis=0), .001)
    x = np.column_stack([np.ones(len(factors)), factors/scale])
    y = np.full((n, len(observations)), np.nan)
    positions = {int(key): j for j, key in enumerate(ids)}
    for t, (old_ids, old_r) in enumerate(observations):
        for key, ret in zip(old_ids, old_r, strict=True):
            j = positions.get(int(key))
            if j is not None and np.isfinite(ret):
                y[j, t] = ret
    y = y[:, available]
    mask = np.isfinite(y)
    width = k+1
    gram = (mask.astype(float) @ (x[:, :, None]*x[:, None, :]).reshape(len(x), -1)).reshape(n, width, width)
    gram += prior_months*np.eye(width)[None]
    rhs = np.where(mask, y, 0.) @ x
    rhs[:, 1] += prior_months*scale[0]
    coefficients = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
    return coefficients[:, 1:]/scale


def projected_statistics(features, known_r, unknown, exposures=None):
    """Known payoff and unknown-payoff basis, keeping formation N fixed."""
    features = np.asarray(features, float)
    if exposures is not None:
        features = features-exposures @ (np.linalg.pinv(exposures) @ features)
    return features.T @ known_r/len(features), features.T @ unknown/len(features)
