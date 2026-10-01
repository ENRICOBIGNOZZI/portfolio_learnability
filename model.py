"""Linear, Gaussian and Matern-3/2 direct portfolio learning only."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.spatial.distance import cdist, pdist

KERNELS = ('linear', 'gaussian', 'matern32')
FEATURE_COUNTS = (250, 500, 1000, 2000, 4000, 10000)


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def median_distance(sample):
    sample = np.asarray(sample, float)
    if sample.ndim != 2 or len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError('Need at least two finite characteristic vectors.')
    ell = float(np.median(pdist(sample, metric='euclidean')))
    if not np.isfinite(ell) or ell <= 0:
        raise ValueError('Median distance must be positive.')
    return ell


def exact_kernel(x, y, kernel, ell):
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    if kernel == 'linear':
        return 1.0 + x @ y.T
    if kernel not in KERNELS or not np.isfinite(ell) or ell <= 0:
        raise ValueError('Invalid kernel or bandwidth.')
    q = cdist(x, y, metric='euclidean') / ell
    if kernel == 'gaussian':
        return np.exp(-q*q/2)
    v = np.sqrt(3.0)*q
    return (1+v)*np.exp(-v)


@dataclass
class FeatureBank:
    kernel: str
    dimension: int
    maximum: int = 10000
    ell: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.kernel not in KERNELS or self.dimension < 1 or self.maximum < 1:
            raise ValueError('Invalid feature bank.')
        if not np.isfinite(self.ell) or self.ell <= 0:
            raise ValueError('Bandwidth must be positive.')
        normal_seed, radial_seed, phase_seed = np.random.SeedSequence(self.seed).spawn(3)
        # Rows are frequencies; prefixes are unchanged when maximum changes.
        normal = np.random.default_rng(normal_seed).normal(
            size=(self.maximum, self.dimension))
        radial = np.ones(self.maximum)
        if self.kernel == 'matern32':
            radial = np.sqrt(np.random.default_rng(radial_seed).chisquare(
                3.0, size=self.maximum)/3.0)
        self.frequencies = normal/(self.ell*radial[:, None])
        self.phases = np.random.default_rng(phase_seed).uniform(
            0, 2*np.pi, size=self.maximum)

    def features(self, x, count=None):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError('Invalid characteristic matrix.')
        if self.kernel == 'linear':
            return np.column_stack((np.ones(len(x)), x))
        count = self.maximum if count is None else int(count)
        if not 1 <= count <= self.maximum:
            raise ValueError('Feature count is outside the frozen bank.')
        return np.sqrt(2.0/count)*np.cos(
            x @ self.frequencies[:count].T + self.phases[:count])

    def metadata(self):
        return {'kernel':self.kernel, 'input_dimension':self.dimension,
                'maximum_features':self.maximum, 'ell':self.ell, 'seed':self.seed,
                'frequencies_sha256':array_hash(self.frequencies),
                'phases_sha256':array_hash(self.phases), 'nested_prefixes':True}


def managed_matrix(panels, bank, count=None):
    rows = []
    for panel in panels:
        x, r = np.asarray(panel['x']), np.asarray(panel['r'])
        if len(x) != len(r) or len(r) == 0 or not np.isfinite(r).all():
            raise ValueError('Do not train using an unresolved or misaligned payoff.')
        phi = bank.features(x, count)
        rows.append(phi.T @ r / len(r))
    return np.vstack(rows)


def ridge_path(g, penalties):
    g, penalties = np.asarray(g, float), np.asarray(penalties, float)
    if g.ndim != 2 or len(g) < 2 or not np.isfinite(g).all():
        raise ValueError('Managed payoffs must be a finite matrix.')
    if penalties.ndim != 1 or np.any(~np.isfinite(penalties)) or np.any(penalties <= 0):
        raise ValueError('Penalties must be finite and positive.')
    u, s, vt = np.linalg.svd(g, full_matrices=False)
    projection = u.T @ np.ones(len(g))
    beta = vt.T @ ((s[:, None]*projection[:, None]) /
                   (s[:, None]**2 + len(g)*penalties[None, :]))
    eigenvalues = s*s/len(g)
    complexity = np.sum(eigenvalues[:, None]/
                        (eigenvalues[:, None]+penalties[None, :]), axis=0)
    return beta, eigenvalues, complexity


def complexity_grid(g, count=120):
    s = np.linalg.svd(np.asarray(g, float), compute_uv=False)
    mu = s*s/len(g)
    active = mu[mu > mu.max()*1e-12]
    if not len(active) or count < 2:
        raise ValueError('Degenerate initial managed-payoff spectrum.')
    targets = np.linspace(0.25, 0.995*len(active), count)
    result = []
    for target in targets:
        lower, upper = active.min()*1e-9, active.max()*len(active)*100.0
        for _ in range(90):
            middle = np.sqrt(lower*upper)
            if np.sum(active/(active+middle)) > target:
                lower = middle
            else:
                upper = middle
        result.append(np.sqrt(lower*upper))
    return np.sort(np.asarray(result))


def sharpe(r, axis=0):
    r = np.asarray(r, float)
    if not np.isfinite(r).all():
        raise ValueError('Do not silently skip missing payoff observations.')
    mean, sd = r.mean(axis=axis), r.std(axis=axis, ddof=1)
    return np.divide(np.sqrt(12)*mean, sd, out=np.full_like(mean, np.nan), where=sd > 0)


def annual_splits(dates):
    import pandas as pd
    dates = pd.DatetimeIndex(dates)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Formation dates must be unique and sorted.')
    for year in range(1978, 2025):
        train = np.flatnonzero(dates.year < year-5)
        validation = np.flatnonzero((dates.year >= year-5) & (dates.year < year))
        test = np.flatnonzero(dates.year == year)
        if len(train) != (year-5-1963)*12 or len(validation) != 60 or len(test) != 12:
            raise ValueError('Incomplete annual split.')
        # A label at formation t is available at t+1; first test trade is at month-end.
        first_trade = dates[test[0]]
        if (dates[np.r_[train, validation]] + pd.offsets.MonthEnd(1)).max() > first_trade:
            raise ValueError('A refit label is unavailable at the first test trade.')
        yield year, train, validation, test


def fit_windows(g, dates, penalties=None):
    splits = list(annual_splits(dates))
    penalties = complexity_grid(g[splits[0][1]]) if penalties is None else np.asarray(penalties)
    for year, train, validation, test in splits:
        betas_train, _, _ = ridge_path(g[train], penalties)
        validation_returns = g[validation] @ betas_train
        validation_loss = np.mean((1-validation_returns)**2, axis=0)
        choice = int(np.argmin(validation_loss))
        refit = np.r_[train, validation]
        betas, mu, complexity = ridge_path(g[refit], penalties)
        yield {'year':year, 'train':train, 'validation':validation, 'test':test,
               'penalties':penalties, 'choice':choice, 'beta':betas[:, choice],
               'selected_train_beta':betas_train[:, choice],
               'validation_loss':validation_loss, 'mu':mu, 'complexity':complexity,
               'historical_sharpe':sharpe(g[refit] @ betas, axis=0),
               'test_returns':g[test] @ betas}


def l1_target_turnover(ids, weights, previous=None):
    import pandas as pd
    current = pd.Series(weights, index=ids)
    if not current.index.is_unique:
        raise ValueError('Duplicate stock identifier.')
    if previous is None:
        return float(current.abs().sum()), current
    union = current.index.union(previous.index, sort=False)
    return float((current.reindex(union, fill_value=0)-
                  previous.reindex(union, fill_value=0)).abs().sum()), current
