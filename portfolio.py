"""Direct maximum-Sharpe portfolio learning with ridge regularization."""
import numpy as np
from kernels import STOCK_BATCH_SIZE

def managed_matrix(panels, bank, count=None):
    rows = []
    for panel in panels:
        x, r = np.asarray(panel['x']), np.asarray(panel['r'])
        if len(x) != len(r) or len(r) == 0 or not np.isfinite(r).all():
            raise ValueError('Do not train using an unresolved or misaligned payoff.')
        if bank.kernel == 'linear':
            total = bank.features(x, count).T @ r
        else:
            total = np.zeros(bank.maximum if count is None else count)
            for start in range(0, len(r), STOCK_BATCH_SIZE):
                stop = start+STOCK_BATCH_SIZE
                total += bank.features(x[start:stop], count).T @ r[start:stop]
        rows.append(total / len(r))
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


def dual_path(gram, penalties):
    """Ridge coefficients in time space; one eigendecomposition per history."""
    gram = np.asarray(gram, float)
    n = len(gram)
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    tolerance = max(float(np.max(np.abs(values))),1e-30)*1e-10
    if values.min() < -tolerance:
        raise ValueError('Managed Gram matrix is not positive semidefinite.')
    values = np.maximum(values,0.)
    alpha = vectors @ ((vectors.T @ np.ones(n))[:,None]/
                       (values[:,None]+n*np.asarray(penalties)[None,:]))
    mu = values[::-1]/n
    complexity = np.sum(mu[:,None]/(mu[:,None]+penalties[None,:]),axis=0)
    return alpha, mu, complexity


def fit_windows(g, dates, penalties=None):
    g = np.asarray(g,float)
    if not np.isfinite(g).all():
        raise ValueError('Unresolved managed returns.')
    splits = list(annual_splits(dates))
    penalties = complexity_grid(g[splits[0][1]]) if penalties is None else np.asarray(penalties)
    gram = g @ g.T
    for year, train, validation, test in splits:
        alpha_train, _, _ = dual_path(gram[np.ix_(train,train)],penalties)
        validation_returns = gram[np.ix_(validation,train)] @ alpha_train
        validation_loss = np.mean((1-validation_returns)**2,axis=0)
        choice = int(np.argmin(validation_loss))
        refit = np.r_[train,validation]
        alpha, mu, complexity = dual_path(gram[np.ix_(refit,refit)],penalties)
        yield {'year':year,'train':train,'validation':validation,'test':test,
               'penalties':penalties,'choice':choice,'beta':g[refit].T @ alpha[:,choice],
               'selected_train_beta':g[train].T @ alpha_train[:,choice],
               'validation_loss':validation_loss,'mu':mu,'complexity':complexity,
               'historical_sharpe':sharpe(gram[np.ix_(refit,refit)] @ alpha,axis=0),
               'test_returns':gram[np.ix_(test,refit)] @ alpha}
