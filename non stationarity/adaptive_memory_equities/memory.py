"""Temporal weights. Tau penalizes drift in a surrogate, never network parameters."""
from dataclasses import dataclass
import numpy as np


def uniform(n):
    if n < 1:
        raise ValueError('Empty history.')
    return np.full(n, 1.0 / n)


def theory_weights(d, tau):
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Invalid temporal costs.')
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('Invalid tau.')
    if tau == 0:
        return uniform(len(d)), 1.0 / len(d), 0.0
    # Centered active-set formula avoids j*S2-S1**2 cancellation.
    order = np.argsort(d, kind='stable')
    x = d[order].astype(np.longdouble)
    for j in range(1, len(x) + 1):
        mean = x[:j].mean()
        variance_sum = np.sum((x[:j] - mean)**2)
        slope = mean / (1 / np.longdouble(tau) + variance_sum)
        w = 1 / np.longdouble(j) + slope * (mean - x[:j])
        outside = (j == len(x) or 1 / np.longdouble(j) + slope * (mean - x[j]) <= 0)
        if w[-1] > 0 and outside:
            weights = np.zeros(len(x))
            weights[order[:j]] = np.asarray(w, float)
            weights /= weights.sum()  # only floating point normalization of valid KKT set
            return weights, float(1 / np.longdouble(j) + slope * mean), float(slope)
    raise ArithmeticError('No stable KKT active set found.')


def taper(age, horizon, gamma):
    if horizon <= 0 or not 0 < gamma <= 1:
        raise ValueError('Invalid taper parameters.')
    age = np.asarray(age, float)
    if np.any(age < 1):
        raise ValueError('Forecast lag must be at least one month.')
    w = np.maximum(1 - (age / horizon)**gamma, 0)
    if w.sum() <= 0:
        raise ValueError('EMPTY_TAPER')
    return w / w.sum()


@dataclass(frozen=True)
class Memory:
    family: str
    value: float = 0
    gamma: float = 1

    @property
    def key(self):
        return f'{self.family}:{self.value:g}:{self.gamma:g}'

    def weights(self, age, span):
        age = np.asarray(age, float)
        if (age < 1).any():
            raise ValueError('Forecast lag must be at least one month.')
        if self.family == 'uniform':
            return uniform(len(age))
        if self.family == 'theory':
            return theory_weights((age / span)**self.gamma, self.value)[0]
        if self.family == 'taper':
            return taper(age, self.value, self.gamma)
        if self.family == 'extended_taper':
            return taper(age, span * self.value, self.gamma)
        if self.family == 'rolling':
            w = (age <= self.value).astype(float)
        elif self.family == 'exponential':
            if self.value <= 0:
                raise ValueError('Positive half-life required.')
            w = np.exp2(-(age - age.min()) / self.value)
        else:
            raise ValueError(self.family)
        if w.sum() == 0:
            raise ValueError('Empty memory.')
        return w / w.sum()


def diagnostics(w, age):
    w, age = np.asarray(w), np.asarray(age)
    return dict(n_eff=float(1 / (w @ w)), mean_age=float(w @ age),
                max_active_age=float(age[w > 0].max()), active_dates=int((w > 0).sum()),
                weight_quantiles=np.quantile(w, [0, .25, .5, .75, 1]).tolist())


def matched_horizon(age, span, tau, gamma):
    w, intercept, slope = theory_weights((np.asarray(age) / span)**gamma, tau)
    if slope == 0:
        return w, float('inf')
    return w, span * (intercept / slope)**(1 / gamma)


def grid(config):
    result = [Memory('uniform')]
    for family, entries in config.items():
        if family == 'uniform':
            continue
        for entry in entries:
            value, gamma = entry if isinstance(entry, list) else (entry, 1)
            if family == 'theory' and value == 0:
                continue
            result.append(Memory(family, value, gamma))
    return list(dict.fromkeys(result))
