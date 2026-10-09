import numpy as np
import torch


class PayoffCompletenessError(ValueError):
    pass


def payoff(weights, returns):
    weights, returns = np.asarray(weights), np.asarray(returns)
    if weights.shape != returns.shape or not np.isfinite(weights).all():
        raise ValueError('Invalid holdings.')
    held = weights != 0
    if not np.isfinite(returns[held]).all():
        raise PayoffCompletenessError('PAYOFF_COMPLETENESS_FAILED')
    return float(weights[held] @ returns[held])


def scores(model, z, block_size=512):
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(z[i:i+block_size], dtype=torch.float64)).numpy()
                               for i in range(0, len(z), block_size)])


def holdings(model, z, block_size=512):
    score = scores(model, z, block_size)
    return score, score / len(z)


def ensemble(weight_vectors):
    return np.mean(np.stack(weight_vectors), axis=0)
