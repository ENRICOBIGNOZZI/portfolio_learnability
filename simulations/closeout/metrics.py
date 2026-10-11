"""Monthly payoff moments and exact policy-level risk decompositions.

The legacy moment_metrics API and its frozen source hash remain untouched.
Negative errors beyond the explicit tolerance raise; no sign or root is chosen.
"""
import numpy as np

ROUNDING_TOLERANCE = 1e-10


def payoff_moments(coefficients, mean, second):
    a = np.asarray(coefficients, dtype=float)
    return np.asarray(mean) @ a, np.sum(a * (np.asarray(second) @ a), axis=0)


def financial_decomposition(u, v, q, tolerance=ROUNDING_TOLERANCE):
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    if not 0 < q < 1 or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('Finite moments and analytic q in (0,1) required.')
    variance = v-u*u
    d = v-u*u/q
    if np.any(variance < -tolerance) or np.any(d < -tolerance) or np.any(v < -tolerance):
        raise ValueError('Invalid second moment or directional error beyond rounding tolerance.')
    # Raw d and variance are retained, including roundoff-sized negatives.
    sr = np.divide(u, np.sqrt(np.maximum(variance, 0)),
                   out=np.full_like(u, np.nan), where=variance > 0)
    c = u/q
    radial = q*(c-1)**2
    normalized = np.divide(d, c*c*q*(1-q), out=np.full_like(u, np.nan), where=c > 0)
    positive = (u > 0) & (v > 0)
    ray = np.where(positive, q-np.divide(u*u, v, out=np.zeros_like(v), where=v > 0), np.nan)
    scale = np.where(positive, np.divide((v-u)**2, v, out=np.zeros_like(v), where=v > 0), np.nan)
    regret = q-2*u+v
    np.testing.assert_allclose(regret, radial+d, atol=tolerance, rtol=1e-9)
    np.testing.assert_allclose(regret[positive], (ray+scale)[positive], atol=tolerance, rtol=1e-9)
    active = positive & (variance > 0)
    np.testing.assert_allclose(sr[active], np.sqrt(q/(1-q))/np.sqrt(1+normalized[active]), atol=tolerance, rtol=1e-8)
    np.testing.assert_allclose(ray[active], 1/(1+sr[active]**2)-(1-q), atol=tolerance, rtol=1e-8)
    status = np.where(variance <= 0, 'zero_variance', np.where(u > 0, 'positive_mean', np.where(u < 0, 'negative_mean', 'zero_mean')))
    return dict(u=u, v=v, q=np.full_like(u,q), c=c, d=d, radial_gap=radial,
                normalized_directional_error=normalized, ray_gap=ray, scale_gap=scale,
                regret=regret, monthly_SR=sr, annual_SR=np.sqrt(12)*sr,
                monthly_gap=np.sqrt(q/(1-q))-sr, annual_gap=np.sqrt(12)*(np.sqrt(q/(1-q))-sr), variance=variance, moment_status=status)


def ridge_decomposition(coefficients, mean, second, q, penalties, eigensystem=None):
    """Projection + within-space bias + displacement + signed cross term.

    beta_P = S^-1 m; beta_lambda = (S+lambda I)^-1 m.
    Orthogonality of beta_P gives floor=q-m'S^-1m, whereas the
    cross term 2(delta)'S(beta_lambda-beta_P) need not vanish.
    """
    a = np.asarray(coefficients, float)
    if a.ndim == 1:
        a = a[:,None]
    values, vectors = np.linalg.eigh(second) if eigensystem is None else eigensystem
    if np.any(values <= 0):
        raise ValueError('Population operator must be positive definite; no jitter.')
    projected = vectors.T @ mean
    population = vectors @ (projected[:,None]/(values[:,None]+np.asarray(penalties)))
    optimum = vectors @ (projected/values)
    floor = q - projected @ (projected/values)
    bias_delta = population-optimum[:,None]
    displacement = a-population
    bias = np.sum(bias_delta*(second@bias_delta), axis=0)
    norm = np.sum(displacement*(second@displacement), axis=0)
    cross = 2*np.sum(displacement*(second@bias_delta), axis=0)
    u,v = payoff_moments(a,mean,second)
    np.testing.assert_allclose(q-2*u+v, floor+bias+norm+cross, atol=1e-9, rtol=1e-8)
    return dict(projection_floor=np.full(a.shape[1],floor), regularization_bias_within_P=bias,
                estimation_displacement=norm, signed_cross_term=cross), population, optimum


def normal_residual(x, coefficients, penalties):
    a=np.asarray(coefficients)
    lhs=x.T@(x@a)/len(x)+a*np.asarray(penalties)
    rhs=x.mean(axis=0)[:,None]
    residual=lhs-rhs
    denominator=np.linalg.norm(lhs,axis=0)+np.linalg.norm(rhs,axis=0)
    relative=np.divide(np.linalg.norm(residual,axis=0),denominator,out=np.zeros_like(denominator),where=denominator>0)
    return float(np.max(np.abs(residual))), float(np.max(relative))


def identified_components(regret, monthly_sr, q, tolerance=ROUNDING_TOLERANCE):
    """Recover only quantities uniquely identified by loss and signed Sharpe.

    No payoff moment, scale coefficient, d, or quadratic root is reconstructed.
    For positive mean and variance, SR²=u²/(v-u²) identifies u²/v.
    The ray/scale split and normalized angular error therefore follow exactly.
    Nonpositive signed SR is retained with quantities marked inapplicable.
    """
    regret,sr=np.broadcast_arrays(np.asarray(regret,float),np.asarray(monthly_sr,float))
    active=sr>0
    ray=np.where(active,q-sr*sr/(1+sr*sr),np.nan)
    scale=np.where(active,regret-ray,np.nan)
    direction=np.divide(q/(1-q),sr*sr,out=np.full_like(sr,np.nan),where=active)-1
    if np.any(ray[active]<-tolerance) or np.any(scale[active]<-tolerance) or np.any(direction[active]<-tolerance):
        raise ValueError('Identified component negative beyond declared rounding tolerance.')
    return dict(ray_gap=ray,scale_gap=scale,normalized_directional_error=direction,
                component_applicable=active)
