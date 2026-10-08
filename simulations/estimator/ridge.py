"""The paper's response-one ridge loss, empirical complexity and honest selection."""
import numpy as np
from scipy.linalg import eigh


def ridge_path(managed, penalties):
    x, lam = np.asarray(managed, dtype=np.float64), np.asarray(penalties, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite managed return observations.')
    if lam.ndim != 1 or not np.isfinite(lam).all() or np.any(lam <= 0):
        raise ValueError('All ridge penalties must be finite and positive.')
    second, mean = x.T@x/len(x), x.mean(axis=0)
    values, vectors = eigh(second)
    tolerance = max(values[-1], 1e-30)*1e-12
    if values[0] < -tolerance:
        raise ValueError('Empirical second moment is not PSD.')
    values = np.maximum(values, 0)
    coefficients = vectors @ ((vectors.T@mean)[:, None]/(values[:, None]+lam))
    complexity = np.sum(values[:, None]/(values[:, None]+lam), axis=0)
    return coefficients, complexity


def validation_select(managed, penalties, validation_fraction=.25):
    """Only earlier training observations and the following validation block enter.

    Signature intentionally accepts neither W*, population moments, nor OOS data.
    Refit on the complete available training history after selection.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must lie inside (0,1).')
    cut = len(managed)-max(2, int(np.ceil(len(managed)*validation_fraction)))
    if cut < 2:
        raise ValueError('Insufficient earlier observations for validation.')
    trained, _ = ridge_path(managed[:cut], penalties)
    validation_loss = np.mean((1-managed[cut:]@trained)**2, axis=0)
    choice = int(np.argmin(validation_loss))
    refitted, complexity = ridge_path(managed, penalties)
    return refitted, complexity, choice, validation_loss, cut


def moment_metrics(coefficients, mean, second):
    a = np.asarray(coefficients)
    expected = mean @ a
    second_payoff = np.sum(a*(second@a), axis=0)
    variance = second_payoff-expected**2
    if np.any(variance < -1e-10):
        raise ValueError('Negative population payoff variance.')
    sr = np.divide(expected, np.sqrt(np.maximum(variance, 0)),
                   out=np.zeros_like(expected), where=variance > 0)
    return 1-2*expected+second_payoff, sr


def sample_metrics(payoffs):
    payoffs = np.asarray(payoffs)
    sd = payoffs.std(axis=0, ddof=1)
    sr = np.divide(payoffs.mean(axis=0), sd, out=np.zeros_like(sd), where=sd > 0)
    return np.mean((1-payoffs)**2, axis=0), sr
