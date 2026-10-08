"""Numerical checks for explicitly labeled misspecification experiments."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from simulations.config.design import environments
from simulations.dgp.balanced import BalancedFactorDGP, beta, w_star
from simulations.dgp.robustness import transform_date, condition_status
from simulations.diagnostics.baseline import gaussian_quadratic_log_mgf, gaussian_fourth_ratio
from simulations.estimator.kernel import matern


def audit_variant(env, states=24):
    p = env.parameters()
    rows = []
    for date in BalancedFactorDGP(p, 202610079+env.code).simulate(states):
        z, _ = transform_date(date, p, env.variant)
        b = beta(z, p)
        variances = p.sigma_eps**2*(np.exp(.7*z[:, 1]) if env.variant=='heteroskedastic' else np.ones(p.N))
        mean = b@p.mu_F
        second = b@b.T+np.diag(variances)
        covariance = second-np.outer(mean, mean)
        gram = matern(z, z, env.nu, p.ell)
        bound = 4*(p.c_beta+p.sigma_eps**2*(np.exp(.7) if env.variant=='heteroskedastic' else 1))
        mgf = gaussian_quadratic_log_mgf(gram/p.N**2, mean, covariance, bound)
        baseline_weights = w_star(z, p)/p.N
        residual = float(np.linalg.norm(second@baseline_weights-mean))
        policies = gram[:, :12]/p.N
        ratios = [gaussian_fourth_ratio(float(w@mean), float(w@covariance@w)) for w in policies.T]
        rows.append({'date': date.t, 'balance_error': float(np.linalg.norm(b.T@b/p.N-p.gamma_beta)),
                     'beta_norm_error': float(np.max(np.abs(np.sum(b*b, axis=1)-p.c_beta))),
                     'baseline_policy_normal_equation_error': residual,
                     'max_fourth_moment_ratio': max(ratios), 'B_X_squared': float(bound), **mgf})
    passed = all(row['balance_error']<1e-12 and row['beta_norm_error']<1e-12
                 and row['max_fourth_moment_ratio']<=3+1e-12 and row['log_mgf']<=.5+1e-12 for row in rows)
    if env.variant!='heteroskedastic':
        passed = passed and all(row['baseline_policy_normal_equation_error']<1e-11 for row in rows)
    else:
        # Detect the intended failure instead of accidentally labeling E4 exact.
        passed = passed and max(row['baseline_policy_normal_equation_error'] for row in rows)>1e-8
    return {'passed': bool(passed), 'baseline_theorem_claim': env.variant=='baseline',
            'conditions': condition_status(env.variant), 'states': rows,
            'reference': 'finite-subspace population optimum' if env.variant=='heteroskedastic' else 'balanced analytic optimum',
            'environment_hash': hashlib.sha256(json.dumps(asdict(env), sort_keys=True).encode()).hexdigest()}


def main():
    for env in environments():
        if env.variant=='baseline':
            continue
        report = audit_variant(env)
        path = Path('simulations/outputs/paper/data')/env.name/'environment_audit.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n')
        print(env.name, report['passed'], flush=True)
        if not report['passed']:
            raise SystemExit(1)


if __name__=='__main__':
    main()
