"""The selector has no access to future observations or population outcomes."""
import numpy as np
from simulations.estimator.ridge import ridge_path

T0 = 60
EXPONENT = -.6
A_GRID = np.geomspace(1e-8, 1e-1, 71)
FOLDS = ((20, 30), (30, 40), (40, 50), (50, 60))


def calibrate_initial(first_sixty):
    x = np.asarray(first_sixty, dtype=float)
    if x.ndim != 2 or x.shape[0] != T0 or not np.isfinite(x).all():
        raise ValueError('Calibration accepts exactly the first sixty finite observations.')
    losses = []
    penalties = []
    for train_end, valid_end in FOLDS:
        lam = A_GRID*train_end**EXPONENT
        coefficients, _ = ridge_path(x[:train_end], lam)
        losses.append(np.mean((1-x[train_end:valid_end]@coefficients)**2, axis=0))
        penalties.append(lam)
    losses = np.asarray(losses)
    weights = np.array([b-a for a, b in FOLDS])
    score = np.average(losses, axis=0, weights=weights)
    # An exact tie is broken deterministically toward greater regularization.
    choice = len(score)-1-int(np.argmin(score[::-1]))
    return dict(selected_index=choice, selected_a=float(A_GRID[choice]),
                fold_losses=losses, mean_validation_loss=score,
                fold_penalties=np.asarray(penalties), boundary=choice in (0, len(A_GRID)-1))
