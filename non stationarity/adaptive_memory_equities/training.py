"""Date-level response-one training with exact two-pass block gradients."""
import time
import numpy as np
import torch
from .calendar import eligible
from .networks import Policy
from .portfolio import PayoffCompletenessError, scores


def validate_weights(w, n):
    w = np.asarray(w, float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('Temporal weights must lie on the date simplex.')
    return w


def panel_tensors(panel, origin=None):
    if origin is not None and not eligible(panel, origin):
        raise ValueError('Payoff not available at cutoff.')
    if not np.isfinite(panel.forward_excess_returns).all():
        raise PayoffCompletenessError('PAYOFF_COMPLETENESS_FAILED: training portfolio unidentified')
    return (torch.tensor(panel.Z, dtype=torch.float64),
            torch.tensor(panel.forward_excess_returns, dtype=torch.float64))


def two_pass_gradient(model, panels, coefficients, block_size=512, origin=None, monitor=None):
    """Coefficients may be the exact omega or unbiased date-sampling weights.

    Each month's complete payoff is computed first. All block/month gradients
    accumulate with frozen theta; the caller performs one optimizer update.
    """
    model.zero_grad(set_to_none=True)
    loss = 0.
    for index, weight in enumerate(coefficients):
        if weight == 0:
            continue
        if monitor:
            monitor.check()
        x, r = panel_tensors(panels[index], origin)
        n = len(r)
        with torch.no_grad():
            p = sum(torch.dot(model(x[j:j+block_size]), r[j:j+block_size]) / n
                    for j in range(0, n, block_size))
        coefficient = float(2 * weight * (p - 1))
        loss += float(weight * (1 - p)**2)
        for j in range(0, n, block_size):
            (coefficient * torch.dot(model(x[j:j+block_size]), r[j:j+block_size]) / n).backward()
    return loss


def training_payoffs(model, panels, block_size=512, origin=None, monitor=None):
    result = []
    for panel in panels:
        if monitor:
            monitor.check()
        if origin is not None and not eligible(panel, origin):
            raise ValueError('Unavailable calibration payoff.')
        if not np.isfinite(panel.forward_excess_returns).all():
            raise PayoffCompletenessError('PAYOFF_COMPLETENESS_FAILED')
        result.append(float(scores(model, panel.Z, block_size) @ panel.forward_excess_returns / len(panel.Z)))
    return np.array(result)


def calibrate(model, payoffs, omega):
    omega = validate_weights(omega, len(payoffs))
    mu = float(omega @ payoffs)
    m2 = float(omega @ payoffs**2)
    if not np.isfinite(m2) or m2 <= np.finfo(float).tiny:
        raise ArithmeticError('DEGENERATE_SCALAR_CALIBRATION')
    scalar = mu / m2
    before = float(omega @ (1-payoffs)**2)
    after = float(omega @ (1-scalar*payoffs)**2)
    if not np.isfinite(scalar) or after > before + 1e-10:
        raise ArithmeticError('Scalar calibration failed.')
    model.scale_output(scalar)
    return dict(c_star=scalar, loss_before=before, loss_after=after, weighted_mean=mu,
                weighted_second_moment=m2)


def fit(panels, omega, architecture, seed, optimizer_config, origin=None, monitor=None):
    started = time.perf_counter()
    omega = validate_weights(omega, len(panels))
    torch.set_num_threads(optimizer_config.get('threads', 1))
    torch.use_deterministic_algorithms(True)
    model = Policy(panels[0].Z.shape[1], architecture['depth'], architecture['width'], seed)
    if optimizer_config.get('weight_decay') != 0:
        raise ValueError('No weight decay allowed.')
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config['learning_rate'],
                                 weight_decay=0.0, betas=(.9,.999), eps=1e-8)
    losses = []
    rng = np.random.default_rng(seed)
    batch_dates = optimizer_config.get('batch_dates', 0)
    block = optimizer_config.get('stock_block', 512)
    for _ in range(optimizer_config['steps']):
        if batch_dates:
            # q_s=omega_s: unbiased estimator of sum_s omega_s loss_s.
            # Repeated draws retain multiplicity; complete cross-sections stay intact.
            draws = rng.choice(len(panels), batch_dates, replace=True, p=omega)
            coefficients = np.bincount(draws, minlength=len(panels)) / batch_dates
        else:
            coefficients = omega
        loss = two_pass_gradient(model, panels, coefficients, block, origin, monitor)
        if not np.isfinite(loss):
            raise ArithmeticError('NONFINITE_TRAINING_LOSS')
        optimizer.step()
        losses.append(loss)
    # Exact training-wide calibration, including all months with positive weight.
    active = np.flatnonzero(omega)
    active_panels = _Subset(panels, active)
    p = training_payoffs(model, active_panels, block, origin, monitor)
    calibration = calibrate(model, p, omega[active])
    return model, dict(**calibration, seed=seed, parameters=model.parameter_count,
        optimization_losses=losses, steps=len(losses), date_batch=batch_dates,
        gradient='exact full dates' if not batch_dates else 'unbiased date sampling q=omega',
        weight_decay=0.0, checkpoint_rule='final_fixed_step', cold_start=True,
        training_dates=len(panels), positive_weight_dates=len(active), seconds=time.perf_counter()-started)


class _Subset:
    def __init__(self, panels, indices):
        self.panels, self.indices = panels, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.panels[self.indices[i]]
