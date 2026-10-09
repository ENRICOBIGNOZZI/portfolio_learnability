"""Finite-operator checks; no claim of infinite-spectrum resolution."""
import numpy as np
from scipy.linalg import eigh

from simulations.provenance import json_write


def validate(context, output):
    E = context['E']
    S, m, H = context['second'], context['mean'], context['rank_second']
    values, vectors = eigh(H)
    residual = np.linalg.norm(H@vectors-vectors*values,axis=0)/np.linalg.norm(H,ord='fro')
    report = {
        'embedding_orthonormality_max_error': float(np.abs(E.T@E-np.eye(512)).max()),
        'maximum_eigenpair_residual_relative_to_operator_Frobenius_norm': float(residual.max()),
        'spectrum_minimum': float(values[0]), 'spectrum_maximum': float(values[-1]),
        'production_payoff_covariance_minimum_eigenvalue': float(eigh(S-np.outer(m,m),eigvals_only=True)[0]),
        'scope': 'Finite rank-1024 operator only; residuals do not bound quadrature error or the omitted infinite-spectrum tail.'}
    assert report['embedding_orthonormality_max_error'] < 1e-10
    assert report['maximum_eigenpair_residual_relative_to_operator_Frobenius_norm'] < 1e-10
    assert report['production_payoff_covariance_minimum_eigenvalue'] > 0
    json_write(output/'operator_validation.json',report)
    return report
