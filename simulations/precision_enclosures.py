"""Numerical enclosures for finite-window slopes of full empirical complexity.

The pointwise Loewner/trace certificate encloses each full-kernel C. Averaging
those bounds first encloses the mean C used by our log-log regression. Its
slope is linear in log(C), so signed regression weights give an enclosure.
These are conditional numerical bounds, separate from sampling intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_slope_enclosure(x, lower, upper):
    x,lower,upper=(np.asarray(v,float) for v in (x,lower,upper))
    if x.ndim!=1 or lower.shape!=x.shape or upper.shape!=x.shape or len(x)<3:
        raise ValueError('Slope bounds require matching vectors and at least three points')
    if not all(np.isfinite(v).all() for v in (x,lower,upper)):
        raise ValueError('Nonfinite input to numerical slope enclosure')
    if np.any(x<=0) or np.any(lower<=0) or np.any(upper<lower):
        raise ValueError('Logarithmic slope bounds require positive, ordered complexity bounds')
    centered=np.log(x)-np.log(x).mean();den=centered@centered
    if den<=0:raise ValueError('Slope grid must contain distinct points')
    weights=centered/den
    lo=weights@np.log(np.where(weights>=0,lower,upper))
    hi=weights@np.log(np.where(weights>=0,upper,lower))
    return float(lo),float(hi)


def complexity_slope_enclosures(config, frame, certificates, spectrum):
    """Enclose full empirical-kernel slopes on every declared fit window.

    Full C has rank at most T_train, which also caps each pointwise upper bound.
    No Sharpe-policy or population-complexity bound is implied by this table.
    """
    rows=[]
    balanced=frame[frame.lambda_multiplier==1].merge(
        certificates, on=['rank','trial','T_train','lambda_multiplier'],validate='one_to_one')
    balanced['full_C_upper']=np.minimum(balanced.complexity+balanced.absolute_C_error_bound,balanced.T_train)
    spectral=spectrum.copy()
    spectral['full_C_upper']=np.minimum(spectral.complexity+spectral.absolute_C_error_bound,spectral.T_train)
    for rank in config['precision_ranks']:
        for cutoff in config['sensitivity_T_min']:
            times=[t for t in config['T_train'] if t>=cutoff]
            if len(times)<3:continue
            part=balanced[(balanced['rank']==rank)&(balanced.T_train>=cutoff)]
            temporal=part.groupby('T_train')[['complexity','full_C_upper']].mean().sort_index()
            if list(temporal.index)!=times:raise ValueError('Incomplete temporal slope grid')
            low=config['precision_frozen_C0']*max(times)**(-2/3)
            high=config['precision_frozen_C0']*min(times)**(-2/3)
            chosen=spectral[(spectral['rank']==rank)&(spectral.lambda_reg>=low*(1-1e-10))&
                            (spectral.lambda_reg<=high*(1+1e-10))]
            penalty=chosen.groupby('lambda_reg')[['complexity','full_C_upper']].mean().sort_index()
            for metric,table,target in [('C_T',temporal,1/3),('C_lambda',penalty,-.5)]:
                x=table.index.to_numpy(float);lower=table.complexity.to_numpy();upper=table.full_C_upper.to_numpy()
                lo,hi=log_slope_enclosure(x,lower,upper)
                rows.append(dict(rank=rank,metric=metric,T_train_min=cutoff,T_train_max=max(times),
                    fit_points=len(x),x_min=float(x.min()),x_max=float(x.max()),expected_exponent=target,
                    approximate_exponent=float(np.polyfit(np.log(x),np.log(lower),1)[0]),
                    full_empirical_exponent_lower=lo,full_empirical_exponent_upper=hi,
                    theoretical_exponent_in_enclosure=bool(lo-1e-12<=target<=hi+1e-12),
                    maximum_mean_C_relative_numerical_uncertainty=float(np.max(upper/lower-1))))
    return pd.DataFrame(rows)
