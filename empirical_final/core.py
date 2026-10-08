"""Accounting and training-only spectral diagnostics, without resampling."""
import numpy as np
from scipy.optimize import brentq


def execution_cost(target, drift, trade_rate):
    """Cost / pre-trade NAV when target weights refer to post-cost NAV.

    Both purchases and sales are charged. Solve c = tau |(1-c)w - d|_1.
    The cash residual then finances positions and execution costs exactly.
    """
    target, drift = np.asarray(target, float), np.asarray(drift, float)
    if target.shape != drift.shape or not np.isfinite(target).all() or not np.isfinite(drift).all():
        raise ValueError('Aligned finite stock weights required')
    if not np.isfinite(trade_rate) or trade_rate < 0 or trade_rate*np.abs(target).sum() >= 1:
        raise ValueError('Invalid execution rate or noncontractive leverage')
    equation = lambda c: c-trade_rate*np.abs((1-c)*target-drift).sum()
    if equation(1) <= 0:
        raise ValueError('Execution costs exhaust wealth')
    cost = 0. if trade_rate == 0 else brentq(equation, 0., 1., xtol=1e-15)
    trades = (1-cost)*target-drift
    error = abs(cost-trade_rate*np.abs(trades).sum())
    cash_before = 1-drift.sum()
    cash_after = cash_before-trades.sum()-cost
    error = max(error, abs(cash_after-(1-cost)*(1-target.sum())))
    return cost, float(np.abs(trades).sum()), float(error)


def accounting_step(target, stock_excess, rf, drift, trade_rate=0., borrow_rate=0.):
    """Exact one-period self-financing sensitivity, with end-period borrow fees.

    No stock-specific observed cost is inferred. Funding earns/pays the same
    frozen cash rate as the baseline. Rates are explicit scenario parameters.
    """
    target, stock_excess = np.asarray(target,float), np.asarray(stock_excess,float)
    held=target!=0
    if target.shape != stock_excess.shape or not np.isfinite(stock_excess[held]).all():
        raise ValueError('Unknown returns cannot be imputed in accounting')
    if not np.isfinite(rf) or not np.isfinite(borrow_rate) or borrow_rate < 0:
        raise ValueError('Invalid cash or borrowing rate')
    cost, turnover, residual = execution_cost(target,drift,trade_rate)
    short = float(np.maximum(-target,0).sum())
    borrow = borrow_rate/12*short
    gross_excess = float(target[held]@stock_excess[held])
    holding_growth = 1+rf+gross_excess-borrow
    if holding_growth <= 0:
        raise ValueError('Portfolio insolvency; stop rather than conceal it')
    total = (1-cost)*holding_growth-1
    next_drift = np.zeros_like(target)
    next_drift[held] = target[held]*(1+rf+stock_excess[held])/holding_growth
    return dict(total_return=total, excess_return=total-rf,
                gross_excess_return=gross_excess, turnover=turnover,
                trading_fee=cost, borrowing_fee=(1-cost)*borrow,
                trading_return_drag=cost*(1+rf+gross_excess),
                short_notional=short, next_drift=next_drift,
                accounting_residual=residual)


def performance(excess, total):
    excess, total = np.asarray(excess,float), np.asarray(total,float)
    if not np.isfinite(excess).all() or not np.isfinite(total).all() or np.any(total <= -1):
        raise ValueError('Finite solvent observations required')
    wealth=np.cumprod(1+total)
    drawdown=wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1
    volatility=float(excess.std(ddof=1)*np.sqrt(12))
    return dict(annual_excess_return=float(12*excess.mean()), annual_volatility=volatility,
                sharpe=float(12*excess.mean()/volatility) if volatility else np.nan,
                maximum_drawdown=float(drawdown.min()))


def spectral_policy(g, penalty, fractions=(.1,.5,.9,1.)):
    """Nested ridge-filtered policies from the uncentered training operator.

    No OOS argument exists. Near-equal eigenvalues crossing a boundary are
    retained together; the full endpoint uses every computed direction.
    """
    g=np.asarray(g,float); t=len(g)
    if g.ndim != 2 or not np.isfinite(g).all() or penalty <= 0:
        raise ValueError('Finite managed payoffs and positive ridge required')
    gram=g@g.T
    values,u=np.linalg.eigh((gram+gram.T)/2)
    order=np.argsort(values)[::-1];values=values[order];u=u[:,order]
    if values[-1] < -max(values[0],1e-30)*1e-10:
        raise ValueError('Operator is not positive semidefinite')
    values=np.maximum(values,0.)
    active=int(np.sum(values>values[0]*1e-12))
    if active == 0: raise ValueError('Unidentified managed-payoff operator')
    rhs=u.T@np.ones(t); betas=[];counts=[];gaps=[]
    for fraction in fractions:
        count=max(1,int(np.ceil(active*fraction)))
        while count < active and abs(values[count-1]-values[count]) <= 1e-8*max(values[count-1],1e-30):
            count+=1
        use=t if fraction == 1 else count
        betas.append(g.T@(u[:,:use]@(rhs[:use]/(values[:use]+t*penalty))))
        counts.append(count)
        gaps.append(float((values[count-1]-values[count])/values[count-1]) if count<active else None)
    mu=values/t
    complexity=float(np.sum(mu/(mu+penalty)))
    # Verify the same positive eigenpairs directly in feature space, without a P x P allocation.
    residuals=[]
    for j in sorted({0,active//2,active-1}):
        v=g.T@u[:,j]/np.sqrt(values[j])
        residuals.append(float(np.linalg.norm(g.T@(g@v)/t-mu[j]*v)/max(mu[j],1e-30)))
    return dict(beta=np.column_stack(betas),eigenvalues=mu,rank=active,
                counts=counts,boundary_relative_gaps=gaps,complexity=complexity,
                feature_operator_relative_residual=max(residuals))


def incremental_loss(previous,added):
    """Exact response-one loss change, including the OOS cross term."""
    previous,added=np.asarray(previous,float),np.asarray(added,float)
    direct=float(np.mean((1-previous-added)**2)-np.mean((1-previous)**2))
    cross=float(-2*np.mean((1-previous)*added))
    square=float(np.mean(added**2))
    if not np.isclose(direct,cross+square,atol=1e-12,rtol=1e-10):
        raise ValueError('Incremental loss identity failed')
    return direct,cross,square
