"""Training-only selectors. Evaluation objects cannot enter this interface."""
import hashlib
import numpy as np

from simulations.estimator.ridge import ridge_path


def fit_training(managed, penalties, theory_penalties=(), tie_atol=1e-12):
    x, lam = np.asarray(managed), np.asarray(penalties)
    T = len(x)
    cut = T-max(2, int(np.ceil(.25*T)))
    if cut < 2:
        raise ValueError('Insufficient holdout training data.')
    hold, _ = ridge_path(x[:cut], lam)
    hold_loss = np.mean((1-x[cut:]@hold)**2, axis=0)
    splits = [(int(np.floor(f*T)), int(np.floor(g*T))) for f, g in ((.4, .6), (.6, .8), (.8, 1.))]
    total, observations = np.zeros(len(lam)), 0
    for train_end, val_end in splits:
        if train_end < 2 or val_end-train_end < 2:
            raise ValueError('Rolling folds require at least two training and validation observations.')
        coef, _ = ridge_path(x[:train_end], lam)
        total += np.sum((1-x[train_end:val_end]@coef)**2, axis=0)
        observations += val_end-train_end
    rolling_loss = total/observations
    tied = np.flatnonzero(rolling_loss <= rolling_loss.min()+tie_atol)
    rolling_choice = int(tied[np.argmax(lam[tied])])
    combined = np.concatenate([lam, np.asarray(theory_penalties)])
    coef, complexity = ridge_path(x, combined)
    return {'coefficients': coef, 'complexity': complexity, 'penalties': combined,
            'holdout25_choice': int(np.argmin(hold_loss)), 'rolling3_choice': rolling_choice,
            'holdout25_validation_loss': hold_loss, 'rolling3_validation_loss': rolling_loss,
            'training_sha256': hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            'coefficients_sha256': hashlib.sha256(np.ascontiguousarray(coef).tobytes()).hexdigest(),
            'preprocessing': 'none; fixed characteristic population transform and independent anchors',
            'holdout_cut': cut, 'rolling_splits': splits}
