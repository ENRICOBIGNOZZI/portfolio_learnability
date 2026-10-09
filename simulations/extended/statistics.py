"""Replication distributions and slopes with complete cross-horizon covariance."""
import numpy as np
from scipy.stats import norm

from simulations.extended.design import WINDOWS


def distribution(values,axis=0):
    values=np.asarray(values)
    n=values.shape[axis]
    sd=np.std(values,axis=axis,ddof=1)
    return dict(mean=np.mean(values,axis=axis),median=np.median(values,axis=axis),
                sd=sd,p025=np.percentile(values,2.5,axis=axis),
                p975=np.percentile(values,97.5,axis=axis),mcse=sd/np.sqrt(n))


def gap_slopes(T,replication_gaps):
    T=np.asarray(T,dtype=float)
    gaps=np.asarray(replication_gaps)
    if gaps.ndim!=2 or gaps.shape[1]!=len(T) or len(gaps)<2:
        raise ValueError('Need independent replication rows and paired horizon columns.')
    means=gaps.mean(axis=0)
    if np.any(means<=0):
        raise ValueError('Mean gaps must be positive for logarithmic slopes.')
    rows=[]
    for name,minimum in WINDOWS.items():
        keep=T>=minimum
        if keep.sum()<3:
            rows.append(dict(window=name,T_min=minimum,observations=int(keep.sum()),available=False))
            continue
        x=np.log(T[keep]); weights=(x-x.mean())/np.sum((x-x.mean())**2)
        observed=gaps[:,keep]
        mean=means[keep]
        slope=float(weights@np.log(mean))
        gradient=weights/mean
        covariance_mean=np.cov(observed,rowvar=False,ddof=1)/len(gaps)
        se=float(np.sqrt(max(gradient@covariance_mean@gradient,0.)))
        # Independent leave-one-replication-out jackknife cross-check.
        leave=(observed.sum(axis=0)-observed)/(len(gaps)-1)
        jackknife=np.log(leave)@weights
        jack_se=float(np.sqrt((len(gaps)-1)*np.mean((jackknife-jackknife.mean())**2)))
        rows.append(dict(window=name,T_min=minimum,observations=int(keep.sum()),available=True,
            slope=slope,MCSE_delta=se,CI_low=slope-norm.ppf(.975)*se,
            CI_high=slope+norm.ppf(.975)*se,MCSE_jackknife=jack_se,
            uncertainty='Monte Carlo slope uncertainty; full cross-T covariance, not a rate theorem test'))
    return rows


def deterministic_slopes(T,values):
    T=np.asarray(T,dtype=float);values=np.asarray(values,dtype=float)
    result=[]
    for name,minimum in WINDOWS.items():
        keep=T>=minimum
        if keep.sum()>=3:
            result.append(dict(window=name,T_min=minimum,observations=int(keep.sum()),
                slope=float(np.polyfit(np.log(T[keep]),np.log(values[keep]),1)[0])))
    return result
