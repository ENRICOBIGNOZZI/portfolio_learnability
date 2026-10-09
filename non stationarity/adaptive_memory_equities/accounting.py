"""Self-financing cash convention and explicitly labeled turnover sensitivities."""
import numpy as np


def turnover(new_ids, new_weights, old_ids=None, old_weights=None, old_total_returns=None,
             previous_total_portfolio_return=None):
    current = dict(zip(new_ids, new_weights, strict=True))
    previous = {} if old_ids is None else dict(zip(old_ids, old_weights, strict=True))
    kind = 'target_weight_turnover'
    if old_ids is not None and old_total_returns is not None:
        r = np.asarray(old_total_returns)
        if not np.isfinite(r).all() or previous_total_portfolio_return is None:
            raise ValueError('Drifted turnover requires complete realized total returns.')
        denominator = 1 + previous_total_portfolio_return
        if denominator <= 0:
            return None, 'INSOLVENT'
        previous = dict(zip(old_ids, np.asarray(old_weights)*(1+r)/denominator, strict=True))
        kind = 'drifted_self_financing_turnover'
    return float(sum(abs(current.get(i,0)-previous.get(i,0)) for i in current.keys() | previous.keys())), kind


def wealth(excess, risk_free):
    total = np.asarray(excess) + np.asarray(risk_free)
    if not np.isfinite(total).all():
        raise ValueError('Incomplete accounting returns.')
    values = np.full(len(total), np.nan)
    drawdown = np.full(len(total), np.nan)
    level, peak = 1., 1.
    insolvent = None
    for j, r in enumerate(total):
        if r <= -1:
            insolvent = j
            break
        level *= 1+r
        peak = max(peak,level)
        values[j], drawdown[j] = level, level/peak-1
    return values, drawdown, insolvent


def net_excess(raw, turnover_value, short_notional, cost_bps=0, borrow_annual_bps=0):
    if turnover_value is None:
        return None
    return raw - turnover_value*cost_bps/10000 - short_notional*borrow_annual_bps/120000
