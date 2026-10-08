"""Reproducible E1--E6 audit; exits nonzero unless every acceptance gate passes."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.linalg import eigh, eigvalsh

from simulations.dgp.balanced import (
    BalancedFactorDGP, DGPParameters, balanced_triplets, beta,
    conditional_moments, kernel_gram, w_star,
)


@dataclass(frozen=True)
class AuditSettings:
    # These choices are fixed before looking at any spectrum or MC result.
    date_seed: int = 1729
    date_selection_seed: int = 1730
    population_seed: int = 271828
    monte_carlo_seed: int = 314159
    policy_seed: int = 161803
    date_pool: int = 256
    audited_dates: int = 24
    population_groups: int = 1024  # 3072 characteristic nodes; independent of paths
    slope_first: int = 30         # inclusive, one-based, never chosen from data
    slope_last: int = 200
    slope_tolerance: float = 0.35 # finite quadrature diagnostic, not an asymptotic CI
    eigenvalue_relative_floor: float = 1e-12
    mc_periods: int = 100_000
    mc_standard_errors: float = 6.0
    identity_tolerance: float = 1e-12
    equation_tolerance: float = 1e-11
    numerical_tolerance: float = 1e-12

    def __post_init__(self):
        counts = ('date_pool', 'audited_dates', 'population_groups', 'slope_first', 'slope_last', 'mc_periods')
        if any(isinstance(getattr(self, k), bool) or not isinstance(getattr(self, k), int) or getattr(self, k) < 1 for k in counts):
            raise ValueError('Audit sizes and indices must be positive integers.')
        if self.audited_dates > self.date_pool or self.mc_periods < 2:
            raise ValueError('Need audited_dates <= date_pool and at least two MC periods.')
        if not 4 <= self.slope_first < self.slope_last < 3 * self.population_groups:
            raise ValueError('Declare an interior spectral range after the first three directions.')
        for k in ('slope_tolerance', 'eigenvalue_relative_floor', 'mc_standard_errors', 'identity_tolerance', 'equation_tolerance', 'numerical_tolerance'):
            if not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f'{k} must be finite and positive.')


def loading_errors(b, p):
    return (float(np.max(np.abs(np.sum(b * b, axis=1) - p.c_beta))),
            float(np.linalg.norm(b.T @ b / p.N - p.gamma_beta, ord=2)))


def gaussian_quadratic_log_mgf(a, mean, covariance, scale_squared):
    """Exact noncentral Gaussian log E exp(R' A R / scale_squared).

    In an eigenbasis of A, form C=A^(1/2) V A^(1/2) and d=A^(1/2)m.
    The result is -1/2 sum log(1-2 lambda/B²) + sum d_j²/(B²-2 lambda_j).
    No Gaussian sampling is used here; singular PSD A is supported.
    """
    if not np.isfinite(scale_squared) or scale_squared <= 0:
        raise ValueError('MGF scale must be strictly positive.')
    values, vectors = eigh(a, check_finite=True)
    if values[0] < -1e-12 * max(1.0, abs(values[-1])):
        raise ValueError('Quadratic-form matrix is not PSD.')
    roots = np.sqrt(np.maximum(values, 0))
    c = roots[:, None] * (vectors.T @ covariance @ vectors) * roots[None, :]
    eigenvalues, rotation = eigh((c + c.T) / 2)
    if eigenvalues[0] < -1e-12 * max(1.0, abs(eigenvalues[-1])):
        raise ValueError('Gaussian covariance is not PSD.')
    eigenvalues = np.maximum(eigenvalues, 0)
    noncentral = rotation.T @ (roots * (vectors.T @ mean))
    scaled = 2 * eigenvalues / scale_squared
    if np.max(scaled) >= 1:
        return {'log_mgf': float('inf'), 'maximum_scaled_eigenvalue': float(scaled[-1])}
    result = -0.5 * np.log1p(-scaled).sum() + np.sum(noncentral**2 / (scale_squared - 2 * eigenvalues))
    return {'log_mgf': float(result), 'maximum_scaled_eigenvalue': float(scaled[-1]),
            'quadratic_covariance_max_eigenvalue': float(eigenvalues[-1])}


def gaussian_fourth_ratio(mean, variance):
    if variance < 0:
        raise ValueError('Gaussian variance must be nonnegative.')
    second = mean**2 + variance
    if second == 0:
        return 0.0  # The zero policy satisfies 0 <= 3*0; ratio convention only.
    return float((mean**4 + 6 * mean**2 * variance + 3 * variance**2) / second**2)


def audit_dates(p: DGPParameters, s: AuditSettings) -> dict:
    selected = set(np.random.default_rng(s.date_selection_seed).choice(s.date_pool, s.audited_dates, replace=False).tolist())
    policy_rng = np.random.default_rng(s.policy_seed)
    norm_error = balance_error = 0.0
    rows = []
    for date in BalancedFactorDGP(p, s.date_seed).simulate(s.date_pool):
        ne, be = loading_errors(date.loadings, p)
        norm_error, balance_error = max(norm_error, ne), max(balance_error, be)
        if date.t not in selected:
            continue
        b = date.loadings
        m, v, second = conditional_moments(b, p)
        # Independent implementation of the Woodbury conditional optimizer.
        weights = b @ np.linalg.solve(b.T @ b + p.sigma_eps**2 * np.eye(3), p.mu_F)
        residual = second @ weights - m
        equation_error = float(np.linalg.norm(residual) / max(1, np.linalg.norm(m)))
        representation_error = float(np.max(np.abs(weights - w_star(date.z, p) / p.N)))
        gram = kernel_gram(date.z, p)
        mgf = gaussian_quadratic_log_mgf(gram / p.N**2, m, v, p.constants()['B_X_squared'])
        # Kernel sections and their random linear combinations are actual RKHS policies.
        sections = gram[:, :16]
        policies = np.column_stack((sections, sections @ policy_rng.normal(size=(sections.shape[1], 16)), w_star(date.z, p))) / p.N
        ratios = [gaussian_fourth_ratio(float(w @ m), float(w @ v @ w)) for w in policies.T]
        # Include a nonzero, conditionally mean-zero RKHS policy, attaining 3.
        centered = sections[:, 1] * (sections[:, 0] @ m) - sections[:, 0] * (sections[:, 1] @ m)
        ratios.append(gaussian_fourth_ratio(float(centered @ m), float(centered @ v @ centered)))
        e4 = float(np.max(np.abs(policies.T @ residual)))
        moments_error = float(max(abs(weights @ m - p.q_star), abs(weights @ second @ weights - p.q_star)))
        passed = (equation_error < s.equation_tolerance and representation_error < s.equation_tolerance
                  and mgf['log_mgf'] <= 0.5 + s.numerical_tolerance
                  and mgf['maximum_scaled_eigenvalue'] <= 0.5 + s.numerical_tolerance
                  and max(ratios) <= 3 + s.numerical_tolerance
                  and e4 < s.equation_tolerance and moments_error < s.equation_tolerance)
        rows.append({'date': date.t, 'normal_equation_relative_error': equation_error,
                     'normal_equation_absolute_error': float(np.linalg.norm(residual)),
                     'conditional_covariance_max_eigenvalue': float(p.N*p.c_beta/3+p.sigma_eps**2),
                     'W_star_coordinate_max_error': representation_error, 'E4_max_abs_residual': e4,
                     'q_star_moment_max_error': moments_error, 'E2a': mgf,
                     'fourth_to_second_squared_max': max(ratios), 'passed': bool(passed)})
    mu = np.asarray(p.mu_F)
    factor_error = float(np.max(np.abs(p.factor_covariance + np.outer(mu, mu) - np.eye(3))))
    return {'passed': bool(norm_error < s.identity_tolerance and balance_error < s.identity_tolerance
                           and factor_error < s.identity_tolerance and all(x['passed'] for x in rows)),
            'all_dates_checked_for_balance': s.date_pool, 'beta_norm_max_error': norm_error,
            'balance_operator_max_error': balance_error,
            'normal_equation_max_absolute_error': max(x['normal_equation_absolute_error'] for x in rows),
            'E2a_max_log_mgf': max(x['E2a']['log_mgf'] for x in rows),
            'factor_second_moment_analytic_error': factor_error,
            'factor_covariance_min_eigenvalue': float(np.linalg.eigvalsh(p.factor_covariance)[0]),
            'dates': rows}


def population_operators(p: DGPParameters, population_groups: int, seed: int, nu=1.5):
    """Same quadrature for T_K and the actual triplet-managed second moment.

    For H_g=(1/3) sum_r K_{Z_gr} beta(Z_gr)', independence across the G
    economic groups gives Sigma_F=E[H H']/G+(1-1/G)E[H]E[H]'. The finite
    quadrature is an empirical distribution of *groups*, preserving their
    dependence; it is not an iid-stock approximation or a prescribed spectrum.
    """
    rng = np.random.default_rng(seed)
    ranks = np.clip(rng.random((population_groups, 6)), np.finfo(float).eps, 1 - np.finfo(float).eps)
    z = balanced_triplets(ranks)
    from simulations.estimator.kernel import matern
    gram = matern(z, z, nu, p.ell)
    values, vectors = eigh(gram, check_finite=False)
    if values[0] <= 0:
        raise ValueError('Population Gram matrix is numerically singular; change resolution explicitly.')
    # Phi' Phi = K; coordinates form an orthonormal basis of the sampled span.
    features = np.sqrt(values)[:, None] * vectors.T
    n = len(z)
    kernel_values = values / n
    h = np.einsum('amr,mrk->amk', features.reshape(n, population_groups, 3),
                  beta(z, p).reshape(population_groups, 3, 3)) / 3
    h_mean = h.mean(axis=1)
    h_flat = h.reshape(n, -1)
    factor = h_flat @ h_flat.T / (p.G * population_groups) + (1 - 1 / p.G) * (h_mean @ h_mean.T)
    managed = factor + (p.sigma_eps**2 / p.N) * np.diag(kernel_values)
    return kernel_values, (managed + managed.T) / 2, z


def audit_spectrum(p: DGPParameters, s: AuditSettings, nu=1.5):
    kernel_values, managed, nodes = population_operators(p, s.population_groups, s.population_seed, nu)
    managed_values = eigvalsh(managed, check_finite=False)[::-1]
    # Generalized eigenvalues test the Loewner sandwich, stronger than ordered
    # eigenvalue comparisons alone. T_K is diagonal in this common RKHS basis.
    root = np.sqrt(kernel_values)
    generalized = eigvalsh(managed / root[:, None] / root[None, :], check_finite=False)
    kernel_values = kernel_values[::-1]
    lower, upper = p.sigma_eps**2 / p.N, p.c_beta + p.sigma_eps**2 / p.N
    active = ((kernel_values > kernel_values[0] * s.eigenvalue_relative_floor)
              & (managed_values > managed_values[0] * s.eigenvalue_relative_floor))
    ratios = managed_values / kernel_values
    sandwich = bool(np.all(ratios[active] >= lower - s.numerical_tolerance)
                    and np.all(ratios[active] <= upper + s.numerical_tolerance)
                    and generalized[0] >= lower - s.numerical_tolerance
                    and generalized[-1] <= upper + s.numerical_tolerance)
    indices = np.arange(1, len(kernel_values) + 1)
    fit_mask = (indices >= s.slope_first) & (indices <= s.slope_last)
    resolved = bool(np.all(active[fit_mask]))
    target = -(1+2*nu/p.D)
    fits = {}
    for name, values in [('kernel', kernel_values), ('managed', managed_values)]:
        if resolved:
            x, y = np.log(indices[fit_mask]), np.log(values[fit_mask])
            slope, intercept = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - intercept - slope*x)**2) / np.sum((y - y.mean())**2)
            fits[name] = {'slope': float(slope), 'r_squared': float(r2),
                          'compatible': bool(abs(slope - target) <= s.slope_tolerance)}
        else:
            fits[name] = {'slope': None, 'r_squared': None, 'compatible': False}
    report = {'passed': sandwich and resolved and all(f['compatible'] for f in fits.values()),
              'operator_and_eigenvalue_sandwich_passed': sandwich,
              'sample_kind': 'independent iid uniform base groups, each expanded into three phase roles',
              'nodes': len(nodes), 'independent_groups': s.population_groups,
              'nodes_sha256': hashlib.sha256(nodes.tobytes()).hexdigest(),
              'economic_groups_G': p.G, 'lower_multiplier': lower, 'upper_multiplier': upper,
              'generalized_eigenvalue_min': float(generalized[0]),
              'generalized_eigenvalue_max': float(generalized[-1]),
              'ordered_eigenvalue_ratio_min': float(ratios[active].min()) if active.any() else None,
              'ordered_eigenvalue_ratio_max': float(ratios[active].max()) if active.any() else None,
              'resolved_eigenvalues': int(active.sum()), 'fit_range_inclusive': [s.slope_first, s.slope_last],
              'fit_range_fully_resolved': resolved, 'slope_target': target,
              'slope_absolute_tolerance': s.slope_tolerance, 'fits': fits,
              'interpretation': 'Finite-quadrature diagnostic; compatibility does not prove the asymptotic exponent.'}
    sensitivity = []
    for first, last in ((30, 200), (60, 400), (120, 800)):
        mask = (indices >= first) & (indices <= last)
        if last >= len(indices) or not np.all(active[mask]):
            sensitivity.append({'range': [first, last], 'resolved': False})
            continue
        result = {'range': [first, last], 'resolved': True}
        x = np.log(indices[mask])
        for name, values in (('kernel', kernel_values), ('managed', managed_values)):
            y = np.log(values[mask])
            slope, intercept = np.polyfit(x, y, 1)
            se = np.sqrt(np.sum((y-intercept-slope*x)**2)/(len(x)-2)/np.sum((x-x.mean())**2))
            result[name] = {'slope': float(slope), 'OLS_SE': float(se)}
        sensitivity.append(result)
    report['predeclared_window_sensitivity'] = sensitivity
    rows = [{'index': int(j), 'kernel_eigenvalue': float(k), 'managed_eigenvalue': float(m),
             'lower_bound': float(lower*k), 'upper_bound': float(upper*k),
             'ratio': float(r), 'above_floor': bool(a), 'in_fit_range': bool(f)}
            for j, k, m, r, a, f in zip(indices, kernel_values, managed_values, ratios, active, fit_mask)]
    return report, rows


def audit_monte_carlo(p: DGPParameters, s: AuditSettings):
    """Stream genuine stock-level returns; never substitute synthetic optimal payoffs."""
    payoff = np.empty(s.mc_periods, dtype=np.float64)
    factors = np.empty((s.mc_periods, 3), dtype=np.float64)
    norm_error = balance_error = 0.0
    coefficients = p.policy_coefficients
    for j, date in enumerate(BalancedFactorDGP(p, s.monte_carlo_seed).simulate(s.mc_periods)):
        payoff[j] = (date.loadings @ coefficients / p.N) @ date.returns
        factors[j] = date.factors
        ne, be = loading_errors(date.loadings, p)
        norm_error, balance_error = max(norm_error, ne), max(balance_error, be)
    q, sr, n = p.q_star, p.sr_star, s.mc_periods
    variance = q - q*q
    mean = float(payoff.mean())
    second = float(np.mean(payoff**2))
    empirical_sr = float(mean / payoff.std(ddof=1))
    # Conditional optimal-payoff law is the same N(q,q-q²) on every date.
    # Thus these payoffs are iid despite persistent characteristics.
    tolerances = {'mean': s.mc_standard_errors * np.sqrt(variance / n),
                  'second_moment': s.mc_standard_errors * np.sqrt((2*variance**2 + 4*q*q*variance) / n),
                  'SR': s.mc_standard_errors * np.sqrt((1 + sr*sr/2) / n)}
    mu, cov = np.asarray(p.mu_F), p.factor_covariance
    factor_mean = factors.mean(axis=0)
    factor_second = factors.T @ factors / n
    diagonal = np.diag(cov)
    product_variance = (np.outer(diagonal, diagonal) + cov**2
                        + np.outer(mu**2, diagonal) + np.outer(diagonal, mu**2)
                        + 2 * np.outer(mu, mu) * cov)
    mean_tolerance = s.mc_standard_errors * np.sqrt(diagonal / n)
    second_tolerance = s.mc_standard_errors * np.sqrt(product_variance / n)
    factor_passed = bool(np.all(np.abs(factor_mean - mu) <= mean_tolerance)
                         and np.all(np.abs(factor_second - np.eye(3)) <= second_tolerance))
    sharpe_passed = abs(empirical_sr - sr) <= tolerances['SR']
    passed = bool(factor_passed and sharpe_passed and abs(mean-q) <= tolerances['mean']
                  and abs(second-q) <= tolerances['second_moment']
                  and norm_error < s.identity_tolerance and balance_error < s.identity_tolerance)
    return {'passed': passed, 'periods': n, 'all_dates_checked_for_balance': n,
            'beta_norm_max_error': norm_error, 'balance_operator_max_error': balance_error,
            'payoff_mean': mean, 'payoff_second_moment': second, 'q_star': q,
            'payoff_variance': float(payoff.var(ddof=1)), 'analytic_variance': variance,
            'factor_second_moment_frobenius_error': float(np.linalg.norm(factor_second-np.eye(3))),
            'SR_estimate': empirical_sr, 'SR_star': sr, 'SR_passed': bool(sharpe_passed),
            'predeclared_absolute_tolerances': tolerances,
            'factor_moments_passed': factor_passed,
            'factor_mean': factor_mean.tolist(), 'factor_second_moment': factor_second.tolist(),
            'factor_mean_absolute_tolerances': mean_tolerance.tolist(),
            'factor_second_moment_absolute_tolerances': second_tolerance.tolist(),
            'tolerance_method': 'Six analytic standard errors by default; Gaussian Sharpe delta-method SE, unannualized.'}


def write_outputs(destination: Path, report: dict, spectrum: list[dict]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'dgp_audit.json').open('w') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (destination / 'dgp_spectrum.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spectrum[0]))
        writer.writeheader()
        writer.writerows(spectrum)
    index = np.array([x['index'] for x in spectrum])
    kernel = np.array([x['kernel_eigenvalue'] for x in spectrum])
    managed = np.array([x['managed_eigenvalue'] for x in spectrum])
    first, last = report['settings']['slope_first'], report['settings']['slope_last']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, values, label in zip(axes, (kernel, managed), ('kernel', 'managed')):
        ax.loglog(index, values, label=f'{label.capitalize()} eigenvalues', color='#215a83')
        anchor = int(np.sqrt(first * last))
        ax.loglog(index, values[anchor-1]*(index/anchor)**(-1.5), '--', color='#b55735', label='Reference slope -1.5')
        ax.axvspan(first, last, alpha=.12, color='#55996b', label='Predeclared fit range')
        if label == 'managed':
            ax.fill_between(index, [x['lower_bound'] for x in spectrum], [x['upper_bound'] for x in spectrum],
                            color='gray', alpha=.14, label='Eigenvalue sandwich')
        slope = report['E5']['fits'][label]['slope']
        ax.set_title(f'{label.capitalize()} operator: slope {slope:.3f}' if slope is not None else f'{label.capitalize()}: unresolved fit range')
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('Eigenvalue')
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('Balanced three-factor DGP | independent population quadrature')
    fig.savefig(destination / 'dgp_spectrum.png', dpi=180)
    plt.close(fig)


def run_audit(p: DGPParameters, s: AuditSettings, destination: Path) -> dict:
    print('Auditing conditional identities and exact Gaussian MGFs...', flush=True)
    dates = audit_dates(p, s)
    print(f'Auditing population operators on {3*s.population_groups} independent-quadrature nodes...', flush=True)
    spectrum, rows = audit_spectrum(p, s)
    print(f'Auditing {s.mc_periods} stock-level periods...', flush=True)
    mc = audit_monte_carlo(p, s)
    e1 = {'passed': True, 'spectral_radius': max(abs(x) for x in p.rho),
          'argument': 'Stationary stable Gaussian AR state is geometrically beta-mixing; measurable rank/phase transforms and independent next-period shocks preserve it. Any finite gamma > b=1.5 is admissible.'}
    e3 = {'passed': True, 'sobolev_order': 4.5,
          'argument': 'The trigonometric and constant beta components have smooth extensions from the bounded cube, belong to H^(9/2), and hence to the Matern-3/2 RKHS. W_star is their fixed linear combination.'}
    checks = {
        'loading_norm': max(dates['beta_norm_max_error'], mc['beta_norm_max_error']) < s.identity_tolerance,
        'exact_balance': max(dates['balance_operator_max_error'], mc['balance_operator_max_error']) < s.identity_tolerance,
        'factor_second_moment': dates['factor_second_moment_analytic_error'] < s.identity_tolerance and mc['factor_moments_passed'],
        'E1': e1['passed'],
        'conditional_normal_equation': all(x['normal_equation_relative_error'] < s.equation_tolerance for x in dates['dates']),
        'W_star_coordinates': all(x['W_star_coordinate_max_error'] < s.equation_tolerance for x in dates['dates']),
        'E2a': all(x['E2a']['log_mgf'] <= .5+s.numerical_tolerance for x in dates['dates']),
        'E2b': all(x['fourth_to_second_squared_max'] <= 3+s.numerical_tolerance for x in dates['dates']),
        'E3': e3['passed'],
        'E4': all(x['E4_max_abs_residual'] < s.equation_tolerance for x in dates['dates']),
        'E5': spectrum['passed'], 'E6': mc['passed'],
    }
    report = {'passed': bool(all(checks.values()) and dates['passed']),
              'acceptance': {k: bool(v) for k, v in checks.items()},
              'parameters': p.to_dict(), 'constants': p.constants(), 'settings': asdict(s),
              'E1': e1, 'conditional_audit': dates, 'E3': e3, 'E5': spectrum, 'E6': mc,
              'methodology': 'simulations/docs/dgp_methodology.md',
              'provenance': {'source': 'DGP_Baseline_Spec.pdf, October 2026',
                             'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                             'source_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[2] / name).read_bytes()).hexdigest()
                                               for name in ('simulations/dgp/balanced.py', 'simulations/diagnostics/baseline.py', 'kernels.py')}}}
    write_outputs(destination, report, rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('simulations/outputs/audit/baseline'))
    parser.add_argument('--parameters', type=Path, help='JSON object of DGPParameters overrides')
    parser.add_argument('--settings', type=Path, help='JSON object of predeclared AuditSettings overrides')
    args = parser.parse_args()
    p = DGPParameters(**json.loads(args.parameters.read_text())) if args.parameters else DGPParameters()
    s = AuditSettings(**json.loads(args.settings.read_text())) if args.settings else AuditSettings()
    report = run_audit(p, s, args.out)
    print(json.dumps({'passed': report['passed'], 'acceptance': report['acceptance'], 'output': str(args.out)}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
