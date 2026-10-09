"""The three Section VI experiments, using the canonical real-stock managed cache.

Run with single BLAS threads: python3 -m empirical_final.three_experiments
The protocol in the output directory is frozen before estimation. No synthetic
returns, stock-data transformations, feature selection, or new trading rules.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from portfolio import fit_windows, dual_path, sharpe
from local_learnability import rolling_split
from empirical_final.run import ROOT, INPUTS, bind, clean_manifest, load_managed, public, read_json
from empirical_final.core import incremental_loss
from data_pipeline import digest

OUT = ROOT / 'outputs/empirical_three_experiments_20261009'
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
FRACTIONS = (.1, .5, .9, 1.)
BLOCKS = ((1978, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024))
REPS = 5000
SEED = 20261009


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / 'tables' / (name + '.csv'), index=False)
    return frame


def block_counts(n, length=12, seed=SEED, replicates=REPS):
    """Circular moving-block bootstrap; one set of draws for all paired policies."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replicates, int(np.ceil(n / length))))
    indices = ((starts[..., None] + np.arange(length)) % n).reshape(replicates, -1)[:, :n]
    counts = np.zeros((replicates, n), dtype=float)
    np.add.at(counts, (np.arange(replicates)[:, None], indices), 1)
    return counts


def boot_metrics(r, counts):
    r = np.asarray(r, float)
    if r.ndim == 1:
        r = r[:, None]
    n = len(r)
    means = counts @ r / n
    var = np.maximum((counts @ (r*r) - n*means*means)/(n-1), 0)
    return dict(sharpe=np.sqrt(12)*means/np.sqrt(var),
                loss=counts @ ((1-r)**2)/n, mean=12*means,
                volatility=np.sqrt(12*var))


def interval(x):
    return np.quantile(x, [.025, .975], axis=0)


def hac_regression(y, factors, lags=12):
    """OLS descriptive spanning with Bartlett/Newey-West covariance (HC1 scale)."""
    y, factors = np.asarray(y, float), np.asarray(factors, float)
    x = np.column_stack([np.ones(len(y)), factors])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Incomplete factor regression')
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y-x@beta
    scores = x*resid[:, None]
    meat = scores.T@scores
    for lag in range(1, min(lags, len(y)-1)+1):
        gamma = scores[lag:].T@scores[:-lag]
        meat += (1-lag/(lags+1))*(gamma+gamma.T)
    bread = np.linalg.inv(x.T@x)
    cov = bread@meat@bread*len(y)/(len(y)-x.shape[1])
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se, 1-np.var(resid)/np.var(y)


def concentration(mu):
    mu = np.asarray(mu)
    p = mu[mu > mu.max()*1e-12]
    share = p/p.sum()
    return dict(rank=len(p), top10_share=float(share[:10].sum()),
                participation_rank=float(1/np.sum(share**2)))


def spectral_groups(g, penalty):
    """Past-only signed eigenportfolio baskets and disjoint ridge contributions.

    Diagnostic baskets have training RMS 10%/sqrt(12). Policy contributions
    retain their fitted magnitude. Signs are irrelevant to the ridge policy.
    Near-tied eigenvalues are not split across rank boundaries.
    """
    n = len(g)
    gram = g@g.T
    values, u = np.linalg.eigh((gram+gram.T)/2)
    values, u = np.maximum(values[::-1], 0), u[:, ::-1]
    rank = int(np.sum(values > values[0]*1e-12))
    rhs = u.T@np.ones(n)
    signs = np.sign(rhs[:rank])
    for j in np.flatnonzero(np.abs(rhs[:rank]) < 1e-12):
        v = g.T@u[:, j]
        signs[j] = 1 if v[np.argmax(np.abs(v))] >= 0 else -1
    diagnostic, contribution, meta = [], [], []
    first = 0
    for fraction in FRACTIONS:
        last = max(1, int(np.ceil(rank*fraction)))
        while last < rank and abs(values[last-1]-values[last]) <= 1e-8*values[last-1]:
            last += 1
        use = n if fraction == 1 else last
        coeff = g.T@(u[:, first:use]@(rhs[first:use]/(values[first:use]+n*penalty)))
        diagnostic_coeff = g.T@(u[:, first:last]@(
            np.sqrt(n)*signs[first:last]/values[first:last]))
        diagnostic_coeff *= .1/np.sqrt(12*(last-first))
        contribution.append(coeff)
        diagnostic.append(diagnostic_coeff)
        mu = values[first:last]/n
        meta.append(dict(first_rank=first+1, last_rank=last, rank=rank,
                         second_moment_share=float(values[first:last].sum()/values.sum()),
                         active_complexity=float(np.sum(mu/(mu+penalty))),
                         fitted_mean=float(np.mean(g@coeff)),
                         coefficient_norm=float(np.linalg.norm(coeff)),
                         diagnostic_coefficient_norm=float(np.linalg.norm(diagnostic_coeff))))
        first = last
    d, c = np.column_stack(diagnostic), np.column_stack(contribution)
    # Orthogonality refers to training second moments, not OOS covariances.
    np.testing.assert_allclose((g@d).T@(g@d)/n, np.eye(4)*.1**2/12, rtol=1e-6, atol=1e-10)
    return d, c, meta


def expanding(g, dates, grid, factors, kappa, reference):
    paths, annual, selected, spectral, group_meta, factor_annual = [], [], [], [], [], []
    previous_coeff = None
    max_error = 0.
    for result in fit_windows(g, dates, grid):
        year, choice = result['year'], result['choice']
        h = np.r_[result['train'], result['validation']]
        o = result['test']
        rdates = dates[o]+pd.offsets.MonthEnd(1)
        train_dates = dates[h]+pd.offsets.MonthEnd(1)
        assert train_dates.max() <= dates[o].min() < rdates.min()
        raw = result['test_returns']
        old = reference.loc[rdates, 'raw_excess_return'].to_numpy()
        error = float(np.max(np.abs(raw[:, choice]-old)))
        max_error = max(max_error, error)
        np.testing.assert_allclose(raw[:, choice], old, rtol=2e-6, atol=2e-7)
        for j, penalty in enumerate(grid):
            annual.append(dict(decision_year=year, candidate=j, penalty=penalty,
                               C=result['complexity'][j], is_sharpe=result['historical_sharpe'][j],
                               selected=j == choice, validation_loss=result['validation_loss'][j]))
        paths.append(pd.DataFrame(raw, columns=[f'lambda_{i:03}' for i in range(len(grid))]).assign(
            decision_year=year, return_date=rdates))
        for i, date in enumerate(rdates):
            selected.append(dict(decision_year=year, return_date=date, candidate=choice,
                                 penalty=grid[choice], C=result['complexity'][choice], raw_return=raw[i, choice]))
        d, c, meta = spectral_groups(np.asarray(g[h]), grid[choice])
        contributions, diagnostics = np.asarray(g[o])@c, np.asarray(g[o])@d
        np.testing.assert_allclose(contributions.sum(axis=1), raw[:, choice], rtol=2e-6, atol=2e-7)
        train_factors = factors.reindex(train_dates)[FACTORS].to_numpy()
        test_factors = factors.loc[rdates, FACTORS].to_numpy()
        available = np.isfinite(train_factors).all(axis=1)
        xtrain = np.column_stack([np.ones(available.sum()), train_factors[available]])
        # Only slopes are hedged: retain the unpriced expected payoff (alpha).
        fits = {}
        for kind, coefficients in [('diagnostic', d), ('contribution', kappa*c)]:
            train_returns = np.asarray(g[h])@coefficients
            fits[kind] = np.linalg.lstsq(xtrain, train_returns[available], rcond=None)[0][1:]
            for j in range(4):
                for f, b in zip(FACTORS, fits[kind][:, j]):
                    factor_annual.append(dict(decision_year=year, group=j+1, kind=kind, factor=f, loading=b))
        for j in range(4):
            m = meta[j]
            cosine = np.nan if previous_coeff is None else float(c[:, j]@previous_coeff[:, j]/(
                np.linalg.norm(c[:, j])*np.linalg.norm(previous_coeff[:, j])))
            group_meta.append(dict(decision_year=year, group=j+1, penalty=grid[choice],
                                   refit_coefficient_cosine=cosine, **m))
            for i, date in enumerate(rdates):
                diagnostic = diagnostics[i, j]
                contribution = contributions[i, j]
                spectral.append(dict(decision_year=year, return_date=date, group=j+1,
                    diagnostic_return=diagnostic, raw_contribution=contribution,
                    scaled_contribution=kappa*contribution,
                    raw_nested_return=contributions[i, :j+1].sum(),
                    diagnostic_hedged=diagnostic-test_factors[i]@fits['diagnostic'][:, j],
                    contribution_hedged=kappa*contribution-test_factors[i]@fits['contribution'][:, j]))
        previous_coeff = c
        if year % 5 == 0 or year == 2024:
            print(f'Expanding/spectral {year}; baseline max error {max_error:.2g}', flush=True)
    return (save('e1_monthly_paths', pd.concat(paths, ignore_index=True)), save('e1_annual_paths', annual),
            save('e1_selected_monthly', selected), save('e3_monthly', spectral),
            save('e3_annual_groups', group_meta), save('e3_training_factor_loadings', factor_annual), max_error)


def summarize_e1(paths, annual, selected):
    curve, policies, comparisons = [], [], []
    cols = [c for c in paths if c.startswith('lambda_')]
    for lo, hi in BLOCKS:
        label = f'{lo}-{hi}'
        p = paths[paths.decision_year.between(lo, hi)]
        a = annual[annual.decision_year.between(lo, hi)]
        s = selected[selected.decision_year.between(lo, hi)]
        r = p[cols].to_numpy()
        counts = block_counts(len(p))
        all_returns = np.column_stack([r, s.raw_return])
        boot = boot_metrics(all_returns, counts)['sharpe']
        ci = interval(boot)
        sr = sharpe(r)
        peak = int(np.argmax(sr))
        for j in range(len(cols)):
            aa = a[a.candidate == j]
            curve.append(dict(period=label, candidate=j, penalty=aa.penalty.iloc[0],
                mean_C=aa.C.mean(), is_sharpe=aa.is_sharpe.mean(), oos_sharpe=sr[j],
                sr_low=ci[0, j], sr_high=ci[1, j], descriptive_peak=j == peak,
                months=len(p), first_return=str(p.return_date.min().date()), last_return=str(p.return_date.max().date())))
        policies.append(dict(period=label, months=len(s), mean_C=s.C.mean(),
            min_C=s.C.min(), max_C=s.C.max(), sharpe=float(sharpe(s.raw_return)),
            sr_low=ci[0, -1], sr_high=ci[1, -1], raw_loss=float(np.mean((1-s.raw_return)**2)),
            descriptive_peak_C=float(a[a.candidate == peak].C.mean()), descriptive_peak_sharpe=sr[peak],
            peak_at_grid_boundary=peak in (0, len(cols)-1)))
        # Predeclared endpoints, not significance tests of an ex-post optimum.
        delta = boot[:, 0]-boot[:, -2]
        low, high = interval(delta)
        comparisons.append(dict(period=label, contrast='highest minus lowest complexity',
            delta_sharpe=sr[0]-sr[-1], low=low, high=high))
    save('e1_curves', curve); save('e1_selected_summary', policies); save('e1_endpoint_contrasts', comparisons)


def history_experiment(g, dates, grid):
    ret_dates = dates+pd.offsets.MonthEnd(1)
    gram = g@g.T
    years = [y for y in range(1978, 2025) if all(rolling_split(ret_dates, y, t, 20, 'january-close')[0]
                                              is not None for t in (60, 120, 240, 360))]
    annual, monthly = [], []
    for year in years:
        for t in (60, 120, 240, 360):
            split, _ = rolling_split(ret_dates, year, t, 20, 'january-close')
            h, i, v, o = (split[k] for k in ('history', 'inner', 'validation', 'test'))
            alpha, _, _ = dual_path(gram[np.ix_(i, i)], grid)
            losses = np.mean((1-gram[np.ix_(v, i)]@alpha)**2, axis=0)
            choice = int(np.argmin(losses))
            alpha, mu, c = dual_path(gram[np.ix_(h, h)], grid[[choice]])
            r = (gram[np.ix_(o, h)]@alpha)[:, 0]
            annual.append(dict(decision_year=year, T=t, V=20, inner_months=len(i),
                first_training_return=str(ret_dates[h[0]].date()), last_training_return=str(ret_dates[h[-1]].date()),
                first_test_return=str(ret_dates[o[0]].date()), last_test_return=str(ret_dates[o[-1]].date()),
                penalty=grid[choice], candidate=choice, C=c[0], validation_loss=losses[choice],
                grid_boundary=choice in (0, len(grid)-1), **concentration(mu)))
            for date, value in zip(ret_dates[o], r):
                monthly.append(dict(decision_year=year, return_date=date, T=t, raw_return=value))
        if year % 5 == 0 or year == years[-1]:
            print(f'Common-history comparison {year}', flush=True)
    return save('e2_annual', annual), save('e2_monthly', monthly)


def summarize_e2(annual, monthly):
    periods = [('all', annual.decision_year.min(), 2024)] + [
        (f'{max(lo, annual.decision_year.min())}-{hi}', max(lo, annual.decision_year.min()), hi)
        for lo, hi in BLOCKS if hi >= annual.decision_year.min()]
    summary, contrasts, concentration_rows = [], [], []
    for label, lo, hi in periods:
        a = annual[annual.decision_year.between(lo, hi)]
        m = monthly[monthly.decision_year.between(lo, hi)]
        wide = m.pivot(index='return_date', columns='T', values='raw_return').sort_index()
        assert wide.notna().all().all()
        boot = boot_metrics(wide.to_numpy(), block_counts(len(wide)))
        ac = a.pivot(index='decision_year', columns='T', values='C').sort_index()
        draws = block_counts(len(ac), 3)@ac.to_numpy()/len(ac)
        for j, t in enumerate(wide.columns):
            aa = a[a['T'] == t]
            r = wide[t].to_numpy()
            sr_ci, loss_ci, c_ci = interval(boot['sharpe'][:, j]), interval(boot['loss'][:, j]), interval(draws[:, j])
            summary.append(dict(period=label, T=t, months=len(r), mean_C=aa.C.mean(), C_low=c_ci[0], C_high=c_ci[1],
                median_lambda=aa.penalty.median(), mean_top10_share=aa.top10_share.mean(),
                mean_participation_rank=aa.participation_rank.mean(), sharpe=float(sharpe(r)),
                sr_low=sr_ci[0], sr_high=sr_ci[1], loss=float(np.mean((1-r)**2)),
                loss_low=loss_ci[0], loss_high=loss_ci[1], grid_boundary_fraction=aa.grid_boundary.mean()))
        for j in range(1, 4):
            for base in sorted({0, j-1}):
                dc, ds, dl = draws[:, j]-draws[:, base], boot['sharpe'][:, j]-boot['sharpe'][:, base], boot['loss'][:, j]-boot['loss'][:, base]
                cc, sc, lc = interval(dc), interval(ds), interval(dl)
                contrasts.append(dict(period=label, T=wide.columns[j], reference_T=wide.columns[base],
                    delta_C=float((ac.iloc[:, j]-ac.iloc[:, base]).mean()), C_low=cc[0], C_high=cc[1],
                    delta_sharpe=float(sharpe(wide.iloc[:, j])-sharpe(wide.iloc[:, base])), sr_low=sc[0], sr_high=sc[1],
                    delta_loss=float(np.mean((1-wide.iloc[:, j])**2-(1-wide.iloc[:, base])**2)), loss_low=lc[0], loss_high=lc[1],
                    fraction_years_C_increases=float(np.mean(ac.iloc[:, j] > ac.iloc[:, base]))))
    # Relate each within-year history contrast to a pre-test local concentration measure.
    wide = annual.pivot(index='decision_year', columns='T', values='C')
    local = annual[annual['T'] == 60].set_index('decision_year').top10_share
    for year in wide.index:
        concentration_rows.append(dict(decision_year=year, top10_share_T60=local.loc[year],
            delta_C_360_60=wide.loc[year, 360]-wide.loc[year, 60],
            within_year_logT_slope=float(np.polyfit(np.log([60, 120, 240, 360]), wide.loc[year].to_numpy(), 1)[0])))
    cr = save('e2_local_comparison', concentration_rows)
    slope_boot = block_counts(len(cr), 3)@cr.within_year_logT_slope.to_numpy()/len(cr)
    low, high = interval(slope_boot)
    save('e2_history_slope', [dict(mean_slope=cr.within_year_logT_slope.mean(), low=low, high=high,
        descriptive_concentration_correlation=cr.top10_share_T60.corr(cr.delta_C_360_60),
        definition='within-year C slope on log(T), annual circular 3-year bootstrap')])
    save('e2_summary', summary); save('e2_paired_contrasts', contrasts)


def summarize_e3(monthly, meta, factors, kappa):
    regression, summaries, incremental, covariance = [], [], [], []
    for label, lo, hi in [('all', 1978, 2024)] + [(f'{lo}-{hi}', lo, hi) for lo, hi in BLOCKS]:
        m = monthly[monthly.decision_year.between(lo, hi)]
        a = meta[meta.decision_year.between(lo, hi)]
        wide = m.pivot(index='return_date', columns='group', values='raw_nested_return').sort_index()
        contributions = m.pivot(index='return_date', columns='group', values='raw_contribution').loc[wide.index]
        f = factors.loc[wide.index, FACTORS].to_numpy()
        counts = block_counts(len(wide))
        boot = boot_metrics(wide.to_numpy(), counts)
        previous = np.zeros(len(wide))
        for j, group in enumerate(wide.columns):
            mm = m[m.group == group].sort_values('return_date')
            aa = a[a.group == group]
            raw = wide[group].to_numpy()
            addition = raw-previous
            delta, cross, square = incremental_loss(previous, addition)
            delta_boot = counts@(((1-raw)**2-(1-previous)**2))/len(raw)
            ci = interval(delta_boot)
            sr_ci = interval(boot['sharpe'][:, j])
            loss_ci = interval(boot['loss'][:, j])
            dmean_ci = interval(counts@mm.diagnostic_return.to_numpy()/len(mm)*12)
            cmean_ci = interval(counts@mm.scaled_contribution.to_numpy()/len(mm)*12)
            dh_ci = interval(counts@mm.diagnostic_hedged.to_numpy()/len(mm)*12)
            ch_ci = interval(counts@mm.contribution_hedged.to_numpy()/len(mm)*12)
            summaries.append(dict(period=label, group=group, months=len(mm),
                mean_spectral_share=aa.second_moment_share.mean(), mean_active_C=aa.active_complexity.mean(),
                training_mean_raw=aa.fitted_mean.mean(), contribution_annual_mean=12*mm.scaled_contribution.mean(),
                contribution_mean_low=cmean_ci[0], contribution_mean_high=cmean_ci[1],
                diagnostic_annual_mean=12*mm.diagnostic_return.mean(), diagnostic_mean_low=dmean_ci[0], diagnostic_mean_high=dmean_ci[1],
                contribution_volatility=np.sqrt(12)*mm.scaled_contribution.std(ddof=1),
                diagnostic_volatility=np.sqrt(12)*mm.diagnostic_return.std(ddof=1),
                diagnostic_sharpe=float(sharpe(mm.diagnostic_return)),
                contribution_hedged_annual_mean=12*mm.contribution_hedged.mean(),
                diagnostic_hedged_annual_mean=12*mm.diagnostic_hedged.mean(),
                contribution_hedged_low=ch_ci[0], contribution_hedged_high=ch_ci[1],
                diagnostic_hedged_low=dh_ci[0], diagnostic_hedged_high=dh_ci[1],
                mean_refit_cosine=aa.refit_coefficient_cosine.mean(),
                median_coefficient_norm=aa.coefficient_norm.median(),
                median_diagnostic_coefficient_norm=aa.diagnostic_coefficient_norm.median()))
            incremental.append(dict(period=label, group=group, rank_fraction=FRACTIONS[j],
                sharpe=float(sharpe(raw)), sr_low=sr_ci[0], sr_high=sr_ci[1],
                loss=float(np.mean((1-raw)**2)), loss_low=loss_ci[0], loss_high=loss_ci[1],
                delta_loss=delta, delta_low=ci[0], delta_high=ci[1], cross_loss_term=cross, own_square_term=square,
                annual_volatility=kappa*np.std(raw, ddof=1)*np.sqrt(12),
                added_variance=12*kappa*kappa*np.var(addition, ddof=1),
                twice_covariance=24*kappa*kappa*np.cov(previous, addition, ddof=1)[0, 1],
                delta_variance=12*kappa*kappa*(np.var(raw, ddof=1)-np.var(previous, ddof=1))))
            for kind, y in [('diagnostic', mm.diagnostic_return.to_numpy()),
                            ('contribution', mm.scaled_contribution.to_numpy()), ('nested', kappa*raw),
                            ('diagnostic_hedged', mm.diagnostic_hedged.to_numpy()),
                            ('contribution_hedged', mm.contribution_hedged.to_numpy())]:
                beta, se, r2 = hac_regression(y, f)
                for factor, value, error in zip(['alpha_monthly']+FACTORS, beta, se):
                    regression.append(dict(period=label, group=group, kind=kind, factor=factor,
                        coefficient=value, se_hac12=error, low=value-1.96*error, high=value+1.96*error,
                        r_squared=r2, months=len(y)))
            previous = raw
        cov = contributions.cov()*kappa*kappa*12
        for i in cov.index:
            for j in cov.columns:
                covariance.append(dict(period=label, group_i=i, group_j=j, annual_covariance=cov.loc[i, j]))
    save('e3_group_summary', summaries); save('e3_nested_summary', incremental)
    save('e3_factor_regressions', regression); save('e3_oos_covariance', covariance)


def uncertainty_sensitivity():
    """Block-length checks of the same three experiments, without retuning policies."""
    read = lambda name: pd.read_csv(OUT/'tables'/f'{name}.csv')
    history = read('e2_monthly').pivot(index='return_date', columns='T', values='raw_return')
    nested = read('e3_monthly').pivot(index='return_date', columns='group', values='raw_nested_return')
    selected = read('e1_selected_monthly')
    rows = []
    for length in (6, 12, 24):
        for lo, hi in BLOCKS:
            r = selected[selected.decision_year.between(lo, hi)].raw_return.to_numpy()
            ci = interval(boot_metrics(r, block_counts(len(r), length))['sharpe'][:, 0])
            rows.append(dict(experiment=1, contrast=f'selected SR {lo}-{hi}', block_length=length,
                             estimate=float(sharpe(r)), low=ci[0], high=ci[1]))
        boot = boot_metrics(history.to_numpy(), block_counts(len(history), length))
        for j, base in [(2, 0), (3, 2)]:
            for metric in ('sharpe', 'loss'):
                ci = interval(boot[metric][:, j]-boot[metric][:, base])
                observed = sharpe(history.to_numpy()) if metric == 'sharpe' else np.mean((1-history.to_numpy())**2, axis=0)
                rows.append(dict(experiment=2, contrast=f'{metric} T{history.columns[j]}-T{history.columns[base]}',
                    block_length=length, estimate=observed[j]-observed[base], low=ci[0], high=ci[1]))
        counts = block_counts(len(nested), length)
        for j in range(1, 4):
            differences = (1-nested.iloc[:, j])**2-(1-nested.iloc[:, j-1])**2
            ci = interval(counts@differences.to_numpy()/len(nested))
            rows.append(dict(experiment=3, contrast=f'incremental loss group {j+1}', block_length=length,
                             estimate=differences.mean(), low=ci[0], high=ci[1]))
    ac = read('e2_annual').pivot(index='decision_year', columns='T', values='C')
    for length in (2, 3, 5):
        difference = ac[360]-ac[60]
        ci = interval(block_counts(len(ac), length)@difference.to_numpy()/len(ac))
        rows.append(dict(experiment=2, contrast='annual mean C360-C60 (year blocks)', block_length=length,
                         estimate=difference.mean(), low=ci[0], high=ci[1]))
    save('bootstrap_block_sensitivity', rows)


def main():
    started = time.monotonic()
    protocol = read_json(OUT/'audit/protocol.json')
    assert protocol['e2_validation_months'] == 20 and protocol['bootstrap']['replicates'] == REPS
    clean = clean_manifest()
    folder, manifest = public('gaussian')
    g, dates = load_managed('gaussian', clean)
    grid = np.array(manifest['lambda_grid'])
    kappa = read_json(ROOT/'results/final/public/linear/seed_0/calibration.json')['scale']
    factors = pd.read_csv(bind(OUT/'tables/french_factors.csv'), parse_dates=['return_date']).set_index('return_date')
    reference = pd.read_csv(folder/'monthly.csv', parse_dates=['return_date']).set_index('return_date')
    p, a, s, m, meta, _, error = expanding(g, dates, grid, factors, kappa, reference)
    summarize_e1(p, a, s)
    ha, hm = history_experiment(g, dates, grid)
    summarize_e2(ha, hm)
    summarize_e3(m, meta, factors, kappa)
    uncertainty_sensitivity()
    audit = dict(passed=True, elapsed_seconds=time.monotonic()-started, inputs=INPUTS,
        source_sha256=digest(Path(__file__)),
        shared_sources={name:digest(ROOT/name) for name in ('portfolio.py', 'kernels.py', 'local_learnability.py',
            'data_pipeline.py', 'empirical_final/core.py', 'empirical_final/run.py')},
        baseline_max_absolute_error=error,
        expanding_refits=int(a.decision_year.nunique()), history_refits=len(ha),
        common_history_first_year=int(ha.decision_year.min()), common_history_last_year=int(ha.decision_year.max()),
        kappa=kappa, bootstrap_replicates=REPS, simulations_run=False,
        checks=['canonical managed-cache provenance', 'past-only payoff timing', 'full Gaussian baseline reconstruction',
                'spectral group sum equals selected ridge', 'training diagnostic second-moment orthogonality',
                'paired complete common history dates', 'incremental loss includes OOS cross term'],
        tables={p.name:digest(p) for p in sorted((OUT/'tables').glob('*.csv'))})
    (OUT/'audit/execution.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('inputs','tables')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
