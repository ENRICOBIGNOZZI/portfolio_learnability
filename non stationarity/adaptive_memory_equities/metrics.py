import numpy as np
from .guard import block_indices


def summary(payoffs):
    r = np.asarray(payoffs, float)
    if len(r) < 2 or not np.isfinite(r).all():
        return dict(months=len(r), sharpe=None, reason='INSUFFICIENT_OR_UNIDENTIFIED_PAYOFFS')
    sd = float(r.std(ddof=1))
    return dict(months=len(r), response_one=float(np.mean((1-r)**2)), mean=float(r.mean()),
                volatility=sd, sharpe=float(np.sqrt(12)*r.mean()/sd) if sd > 1e-14 else None,
                reason=None if sd > 1e-14 else 'DEGENERATE_VARIANCE')


def paired_comparisons(wide, baseline, replicates=2000, block=12, seed=20261003):
    """Saved-series retrospective intervals; common indices and simultaneous max errors."""
    if len(wide) < 2*block:
        return [dict(status='NOT_RUN', reason='Less than two blocks of monthly observations')]
    x = wide.to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Comparison requires identical complete dates.')
    b = list(wide.columns).index(baseline)
    indices = block_indices(len(x), replicates, block, seed)
    boot = x[indices]
    rows = []
    for metric in ['loss','sharpe']:
        identified = np.ones(x.shape[1],dtype=bool)
        original_valid = identified.copy()
        if metric == 'loss':
            estimate = ((1-x)**2).mean(axis=0)
            samples = ((1-boot)**2).mean(axis=1)
        else:
            sd = x.std(axis=0,ddof=1)
            bootstrap_sd = boot.std(axis=1,ddof=1)
            original_valid = sd > 1e-14
            identified = original_valid & (bootstrap_sd > 1e-14).all(axis=0)
            estimate = np.divide(np.sqrt(12)*x.mean(axis=0),sd,
                                 out=np.full(x.shape[1],np.nan),where=original_valid)
            samples = np.divide(np.sqrt(12)*boot.mean(axis=1),bootstrap_sd,
                                out=np.full(bootstrap_sd.shape,np.nan),where=bootstrap_sd>1e-14)
        delta = estimate-estimate[b]
        draws = samples-samples[:,b,None]
        se = draws.std(axis=0,ddof=1)
        good = identified & identified[b] & np.isfinite(se) & (se>1e-12)
        critical = np.quantile(np.max(abs((draws[:,good]-delta[good])/se[good]),axis=1),.95) if good.any() else 0
        for j, method in enumerate(wide.columns):
            if j == b:
                continue
            if not (identified[j] and identified[b]):
                rows.append(dict(method=method,baseline=baseline,metric=metric,difference=None,
                    paired_low=None,paired_high=None,simultaneous_low=None,simultaneous_high=None,
                    block=block,replicates=replicates,seed=seed,status='NOT_IDENTIFIED',
                    reason='DEGENERATE_VARIANCE' if not (original_valid[j] and original_valid[b])
                    else 'DEGENERATE_BOOTSTRAP_VARIANCE'))
                continue
            rows.append(dict(method=method,baseline=baseline,metric=metric,difference=float(delta[j]),
                paired_low=float(np.quantile(draws[:,j],.025)),paired_high=float(np.quantile(draws[:,j],.975)),
                simultaneous_low=float(delta[j]-critical*se[j]),simultaneous_high=float(delta[j]+critical*se[j]),
                block=block,replicates=replicates,seed=seed, status='RETROSPECTIVE_ONLY'))
    return rows
