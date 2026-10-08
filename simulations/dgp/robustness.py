"""Explicit misspecifications, kept separate from the exact-theory baseline."""
import numpy as np
from scipy.stats import rankdata

from simulations.dgp.balanced import beta


def transform_date(date, parameters, variant):
    if variant == 'baseline':
        return date.z, date.returns
    if variant == 'empirical_ranks':
        # Midranks keep the numerical cube open; finite population support is
        # deliberately a violation of the infinite-rank asymptotic E5 condition.
        z = 2*(rankdata(date.z, axis=0)-.5)/parameters.N-1
        return z, beta(z, parameters)@date.factors+date.epsilon
    if variant == 'heteroskedastic':
        # Bounded state-dependent Gaussian volatility; no residual projection.
        return date.z, date.loadings@date.factors+np.exp(.35*date.z[:, 1])*date.epsilon
    raise ValueError('Unknown robustness variant.')


def condition_status(variant):
    if variant == 'baseline':
        return {f'E{i}': 'exact' for i in range(1, 7)}
    if variant == 'empirical_ranks':
        return {'E1': 'exact: measurable map of stable AR state',
                'E2': 'exact: Gaussian conditional returns and bounded loadings',
                'E3': 'exact: same smooth beta policy',
                'E4': 'exact without ties: equally spaced first-coordinate ranks balance trigonometric loadings',
                'E5': 'violated: fixed-N empirical ranks have finite support',
                'E6': 'exact under rank balance'}
    return {'E1': 'exact: measurable map of stable AR state',
            'E2': 'exact with enlarged bound: bounded conditional Gaussian volatility',
            'E3': 'not established for unrestricted optimum; finite-subspace optimum belongs to RKHS',
            'E4': 'deliberately not imposed: conditional inverse varies with cross-section',
            'E5': 'exact sandwich with bounded positive volatility multipliers',
            'E6': 'numerical finite-subspace reference only, not the baseline analytic optimum'}
