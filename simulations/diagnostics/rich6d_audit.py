"""Analytic and independent-uniform-population gradient moments for rich6d."""
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.dgp.balanced import DGPParameters, w_star, w_star_gradient
from simulations.provenance import file_hash, json_write, require, source_hashes, utc_now


def run(output=Path('simulations/outputs/confirmation_v2/audit')):
    output = Path(output)
    seed, n = 202610080105, 131072
    z = np.random.default_rng(seed).uniform(-1., 1., (n, 6))
    rows = []
    maximum_fd = 0.
    for loading, eta in [('baseline_original', 0.), ('rich6d', .35)]:
        p = DGPParameters(loading_map=loading, eta=eta)
        denominator = p.c_beta/3+p.sigma_eps**2/p.N
        amplitude_squared = (2*p.c_beta/3)*(p.mu_F[0]**2+p.mu_F[1]**2)/denominator**2
        exact = np.full(6, amplitude_squared*np.pi**2*eta**2/4)
        exact[0] = amplitude_squared*np.pi**2*(1+2.5*eta**2)/2
        samples = w_star_gradient(z, p)**2
        means, se = samples.mean(axis=0), samples.std(axis=0, ddof=1)/np.sqrt(n)
        interior = np.clip(z[:1024], -.999, .999)
        gradient = w_star_gradient(interior, p)
        for coordinate in range(6):
            delta = np.eye(6)[coordinate]*1e-6
            fd = (w_star(interior+delta, p)-w_star(interior-delta, p))/2e-6
            error = float(np.max(np.abs(fd-gradient[:, coordinate])))
            maximum_fd = max(maximum_fd, error)
            require(error < 1e-7, 'Analytic coordinate derivative disagrees with finite differences')
            require(abs(means[coordinate]-exact[coordinate]) <= 6*se[coordinate]+1e-12,
                    'Uniform-population derivative moment disagrees with analytic integral')
            rows.append({'loading_map': loading, 'coordinate': coordinate+1,
                         'analytic_squared_derivative_expectation': exact[coordinate],
                         'uniform_cube_MC_mean': means[coordinate], 'MCSE': se[coordinate],
                         'samples': n, 'seed': seed, 'finite_difference_max_error': error})
        if eta:
            require(np.all(exact > 0), 'A rich-map coordinate has no policy effect')
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'rich6d_gradient_moments.csv', index=False)
    report = {'schema': 'rich6d-gradient-audit/2.0', 'passed': True, 'completed_utc': utc_now(),
              'source_sha256': file_hash(__file__), 'scientific_sources': source_hashes(), 'seed': seed, 'samples': n,
              'distribution': 'Uniform on the entire open six-dimensional cube; same one-stock marginal as stationary population ranks',
              'maximum_finite_difference_error': maximum_fd,
              'analytic_argument': 'Conditional on five torus differences, the common phase is uniform. E sine² = E cosine² = 1/2; difference cosines are independent with mean zero and second moment 1/2.',
              'preflight_distinction': 'The earlier preflight finite-difference diagnostic uses the interior cube [-.9,.9]^6; its sample squared gradients are not the full-population expectation.'}
    json_write(output/'rich6d_gradient_audit.json', report)
    return report


if __name__ == '__main__':
    print(run())
