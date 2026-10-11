"""Paired inference, rolling diagnostics and delivery for the final 2x2 study."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data_pipeline import digest
from empirical_final.core import performance
from empirical_final.final_theory_regularization import (OUT,ROOT,METHODS,SEED,REPLICATES,
    frozen,load_inputs,penalty,slope)
from empirical_final.theory_guided_lambda import save,write_json
from empirical_final.three_experiments import block_counts,boot_metrics
from portfolio import sharpe

PERIODS=(('1978-2024',1978,2024),('1979-2024',1979,2024),('1984-2024',1984,2024),
         ('2020-2024',2020,2024),('1979-1989',1979,1989),('1990-1999',1990,1999),
         ('2000-2009',2000,2009),('2010-2019',2010,2019))
PAIRS=(('B','A'),('C','A'),('D','B'),('D','C'),('D','joint_cv'),('B','joint_cv'),('D','annual_cv'))


def summarize(out):
    audit=frozen(out);assert json.loads((out/'audit/costs.json').read_text())['passed']
    accounts=pd.read_csv(out/'tables/monthly_accounts.csv',parse_dates=['return_date'])
    annual=pd.read_csv(out/'tables/annual_policies.csv')
    summary=[];annual_summary=[];cumulative=[];bootstrap=[];interactions=[]
    for (method,scenario),d in accounts.groupby(['method','scenario']):
        d=d.sort_values('return_date');cum=np.cumprod(1+d.total_return.to_numpy())
        running_drawdown=cum/np.maximum.accumulate(np.r_[1.,cum])[1:]-1
        policy=annual[annual.method.eq(method)].set_index('year')
        for i,(_,row) in enumerate(d.iterrows()):
            prefix=d.iloc[:i+1]
            cumulative.append(dict(method=method,scenario=scenario,return_date=row.return_date,
                wealth=float(cum[i]),drawdown=float(cum[i]/max(1,float(cum[:i+1].max()))-1),
                expanding_annual_mean=float(12*prefix.excess_return.mean()),
                expanding_annual_volatility=float(np.sqrt(12)*prefix.excess_return.std(ddof=1)) if i else np.nan,
                expanding_sharpe=float(sharpe(prefix.excess_return)) if i else np.nan,
                expanding_mean_leverage=float(prefix.gross_exposure.mean()),
                expanding_mean_monthly_turnover=float(prefix.turnover.mean()),
                expanding_mean_complexity=float(prefix.year.map(policy.complexity).mean()),
                latest_penalty=float(policy.loc[row.year].penalty),
                expanding_maximum_drawdown=float(running_drawdown[:i+1].min()),
                expanding_raw_unit_quadratic_loss=float(np.mean((1-prefix.excess_return/prefix.kappa)**2))))
        for year,q in d.groupby('year'):
            row=annual[annual.method.eq(method)&annual.year.eq(year)].iloc[0]
            annual_summary.append(dict(method=method,scenario=scenario,year=year,months=len(q),
                **performance(q.excess_return,q.total_return),
                raw_unit_quadratic_loss=float(np.mean((1-q.excess_return/q.kappa)**2)),
                mean_monthly_turnover=float(q.turnover.mean()),mean_leverage=float(q.gross_exposure.mean()),
                mean_net_exposure=float(q.net_exposure.mean()),mean_short_notional=float(q.short_notional.mean()),
                annual_trading_fee=float(q.trading_fee.sum()),annual_borrowing_fee=float(q.borrowing_fee.sum()),
                complexity=row.complexity,penalty=row.penalty,multiplier=row.multiplier))
        for label,lo,hi in PERIODS:
            q=d[d.year.between(lo,hi)];assert len(q)==(hi-lo+1)*12
            a=annual[annual.method.eq(method)&annual.year.between(lo,hi)]
            summary.append(dict(period=label,method=method,scenario=scenario,months=len(q),
                first_return=q.return_date.min(),last_return=q.return_date.max(),
                **performance(q.excess_return,q.total_return),
                raw_unit_quadratic_loss=float(np.mean((1-q.excess_return/q.kappa)**2)),
                mean_monthly_turnover=float(q.turnover.mean()),mean_leverage=float(q.gross_exposure.mean()),
                mean_net_exposure=float(q.net_exposure.mean()),mean_short_notional=float(q.short_notional.mean()),
                annualized_trading_fee=float(12*q.trading_fee.mean()),annualized_borrowing_fee=float(12*q.borrowing_fee.mean()),
                mean_complexity=float(a.complexity.mean()),lambda_first=float(a.penalty.iloc[0]),lambda_last=float(a.penalty.iloc[-1])))
    # Every draw samples the same months for every method, gross and net.
    for label,lo,hi in PERIODS:
        q=accounts[accounts.year.between(lo,hi)];n=(hi-lo+1)*12
        for length in (6,12,24):
            counts=block_counts(n,length,SEED+lo,REPLICATES)
            for scenario in ('gross','trade25_borrow30'):
                d=q[q.scenario.eq(scenario)]
                wide=d.pivot(index='return_date',columns='method',values='excess_return')[list(METHODS)]
                assert len(wide)==n and not wide.isna().any().any()
                raw=wide.to_numpy()/audit['kappa'];metrics=boot_metrics(raw,counts)
                estimates=dict(sharpe=np.asarray(sharpe(raw)),loss=np.mean((1-raw)**2,axis=0))
                for right,left in PAIRS:
                    i=METHODS.index(right);j=METHODS.index(left)
                    for metric in ('sharpe','loss'):
                        diff=metrics[metric][:,i]-metrics[metric][:,j];lower,upper=np.quantile(diff,[.025,.975])
                        bootstrap.append(dict(period=label,scenario=scenario,comparison=f'{right}-{left}',metric=metric,
                            block_months=length,replicates=REPLICATES,seed=SEED+lo,
                            estimate=float(estimates[metric][i]-estimates[metric][j]),ci_lower=float(lower),ci_upper=float(upper),
                            bootstrap_fraction_above_zero=float(np.mean(diff>0)),months=n,
                            uncertainty='Paired, conditional on fitted strategies; percentile 95%; no multiplicity correction'))
                for metric in ('sharpe','loss'):
                    a,b,c,d=[METHODS.index(m) for m in ('A','B','C','D')]
                    diff=(metrics[metric][:,d]-metrics[metric][:,c])-(metrics[metric][:,b]-metrics[metric][:,a])
                    lower,upper=np.quantile(diff,[.025,.975])
                    interactions.append(dict(period=label,scenario=scenario,metric=metric,block_months=length,
                        estimate=float((estimates[metric][d]-estimates[metric][c])-(estimates[metric][b]-estimates[metric][a])),
                        ci_lower=float(lower),ci_upper=float(upper),interpretation='Descriptive fitted-strategy difference-in-differences; no causal effect'))
        print('Paired bootstrap',label,flush=True)
    for name,rows in (('performance_summary',summary),('annual_performance',annual_summary),('cumulative_performance',cumulative),
                      ('paired_bootstrap',bootstrap),('interactions',interactions)):
        save(out/'tables'/f'{name}.csv',rows)
    extra=pd.read_csv(out/'tables/exclusion_sensitivity_monthly.csv')
    original=pd.read_csv(out/'tables/monthly_payoffs.csv')
    sensitivity=[]
    for label,lo,hi in PERIODS:
        for method in ('C','D'):
            q=extra[extra.year.between(lo,hi)&extra.method.eq(method)]
            r=original[original.year.between(lo,hi)&original.method.eq(method)]
            sensitivity.append(dict(period=label,method=method,admissible_gross_sharpe=float(sharpe(r.raw_return)),
                extrapolated_gross_sharpe=float(sharpe(q.raw_return)),
                changed_selection_years=int(q[q.changed_selection].year.nunique()),
                outside_assumption_years=int(q[q.outside_assumptions].year.nunique()),
                warning='Positive-b extrapolation beyond b>1 theorem; gross sensitivity only, no recommended algorithm or net claim'))
    save(out/'tables/exclusion_sensitivity_summary.csv',sensitivity)
    searches=[]
    for method in METHODS:
        searches.append(dict(method=method,initial_lambda_candidates=(1000 if method in ('C','D') else 200 if method in ('A','B','fixed_lambda') else 120 if method in ('annual_cv','one_anchor') else 600),
            subsequent_annual_lambda_grid_candidates=(120 if method=='annual_cv' else 600 if method=='joint_cv' else 0),
            subsequent_annual_length_candidates=3 if method in ('C','D') else 5 if method=='joint_cv' else 1,
            subsequent_decisions=46,notes='Initial count is candidate-policy evaluations; A-D grids nested original 200, CV grids original 120. One-anchor performs a scalar spectral root, no CV grid.'))
    save(out/'tables/search_budget.csv',searches)
    constants=pd.read_csv(out/'tables/frozen_constants.csv').set_index('candidate')
    counterfactual=[]
    base=constants.loc[2]
    for r in annual[annual.method.isin(['A','B','C','D'])].itertuples():
        c=constants.loc[r.candidate]
        fixed_rep=penalty(c.C0,r.T,c.b);fixed_base=penalty(base.C0,r.T,base.b)
        counterfactual.append(dict(year=r.year,method=r.method,T=r.T,multiplier=r.multiplier,
            selected_lambda=r.penalty,counterfactual_baseline_fixed_b_lambda=fixed_base,
            counterfactual_selected_representation_fixed_b_lambda=fixed_rep,
            sample_size_factor=fixed_base/base.initial_lambda,representation_factor=fixed_rep/fixed_base,
            b_update_factor=r.penalty/fixed_rep,
            log_sample_size_contribution=r.log_sample_size_contribution,
            log_representation_contribution=r.log_representation_contribution,
            log_b_update_contribution=r.log_b_update_contribution))
    save(out/'tables/lambda_decomposition.csv',counterfactual)
    write_json(out/'audit/summary.json',dict(passed=True,bootstrap_replicates=REPLICATES,blocks=[6,12,24],
        actual_monthly_excess_sharpe=True,common_dates=True,state_carried=True,analysis_source_sha256=digest(Path(__file__))))


def rolling(out):
    """Small time-Gram eigensystems only; never fit a monthly portfolio."""
    started=time.monotonic();frozen(out);_,gs,dates,_=load_inputs(out)
    g=gs[2];gram=g@g.T;cal=pd.read_csv(out/'tables/frozen_constants.csv').iloc[2]
    anchor_T=252;mu=np.linalg.eigvalsh(gram[:anchor_T,:anchor_T])[::-1]/anchor_T
    anchor_lambda=penalty(cal.C0,anchor_T,slope(mu)['b'])
    ranges=((.1,.6),(.1,.5),(.2,.6),(.1,.7));rows=[];constants=[];checks=[]
    for W in (180,240):
        anchor=np.linalg.eigvalsh(gram[anchor_T-W:anchor_T,anchor_T-W:anchor_T])[::-1]/W
        for low,high in ranges:
            fit=slope(anchor,(low,high));b=fit['b']
            if not fit['admissible']:raise ValueError('Rolling pretest anchor inadmissible')
            cw=anchor_lambda*W**(b/(b+1))
            constants.append(dict(W=W,rank_fraction_low=low,rank_fraction_high=high,C0_W=float(cw),anchor_b=b,
                expanding_anchor_T=anchor_T,anchor_lambda=anchor_lambda,
                anchor_first_formation=dates[anchor_T-W],anchor_last_formation=dates[anchor_T-1],
                last_known_anchor_payoff=dates[anchor_T-1]+pd.offsets.MonthEnd(1)))
        for n in range(anchor_T,len(g)+1):
            matrix=gram[n-W:n,n-W:n];eigen=np.linalg.eigvalsh((matrix+matrix.T)/2)
            assert eigen.min()>=-max(eigen.max(),1e-30)*1e-10
            mu=np.maximum(eigen[::-1],0)/W
            for k,(low,high) in enumerate(ranges):
                fit=slope(mu,(low,high));cw=constants[-4+k]['C0_W']
                lam=penalty(cw,W,fit['b']) if fit['admissible'] else np.nan
                comp=float(np.sum(mu/(mu+lam))) if fit['admissible'] else np.nan
                rows.append(dict(W=W,decision_date=dates[n-1]+pd.offsets.MonthEnd(1),
                    first_formation=dates[n-W],last_formation=dates[n-1],last_known_payoff=dates[n-1]+pd.offsets.MonthEnd(1),
                    rank_fraction_low=low,rank_fraction_high=high,C0_W=cw,penalty=lam,complexity=comp,**fit))
            if n%60==0:print('Rolling spectral diagnostic',W,n,flush=True)
    table=pd.DataFrame(rows);summary=[]
    for (W,low,high),d in table.groupby(['W','rank_fraction_low','rank_fraction_high']):
        lb=np.log(d.penalty);lc=np.log(d.complexity)
        summary.append(dict(W=W,rank_fraction_low=low,rank_fraction_high=high,months=len(d),
            b_mean=float(d.b.mean()),b_std=float(d.b.std(ddof=1)),b_monthly_change_std=float(d.b.diff().std(ddof=1)),
            median_abs_b_monthly_change=float(d.b.diff().abs().median()),inadmissible_fraction=float((~d.admissible).mean()),
            median_abs_log_lambda_monthly_change=float(lb.diff().abs().median()),
            p95_abs_log_lambda_monthly_change=float(lb.diff().abs().quantile(.95)),
            median_abs_log_complexity_monthly_change=float(lc.diff().abs().median()),
            penalty_min=float(d.penalty.min()),penalty_max=float(d.penalty.max()),
            complexity_min=float(d.complexity.min()),complexity_max=float(d.complexity.max())))
    for W in (180,240):
        d=table[table.W.eq(W)].pivot(index='decision_date',columns=['rank_fraction_low','rank_fraction_high'],values='penalty')
        logspread=np.log(d).max(axis=1)-np.log(d).min(axis=1)
        primary=next(r for r in summary if r['W']==W and r['rank_fraction_low']==.1 and r['rank_fraction_high']==.6)
        compelling=primary['median_abs_log_lambda_monthly_change']>=.05 and float(logspread.median())<=primary['median_abs_log_lambda_monthly_change'] and primary['inadmissible_fraction']<=.01
        checks.append(dict(W=W,median_cross_rank_log_lambda_spread=float(logspread.median()),
            primary_monthly_log_change=primary['median_abs_log_lambda_monthly_change'],compelling_gate_passed=compelling))
    save(out/'tables/rolling_constants.csv',constants);save(out/'tables/rolling_monthly_diagnostics.csv',table)
    save(out/'tables/rolling_diagnostic_summary.csv',summary);save(out/'tables/rolling_rank_sensitivity.csv',checks)
    write_json(out/'audit/rolling.json',dict(passed=True,windows=[180,240],baseline_only=True,
        pretest='Both fixed windows translated at January 1984 close to expanding B penalty; no post-anchor recalibration',
        no_portfolio_refits=True,compelling_gate_passed=all(c['compelling_gate_passed'] for c in checks),
        full_monthly_backtest_executed=False,elapsed_seconds=time.monotonic()-started))


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42})
    annual=pd.read_csv(out/'tables/annual_policies.csv');accounts=pd.read_csv(out/'tables/monthly_accounts.csv',parse_dates=['return_date'])
    colours={'A':'#778da9','B':'#244c85','C':'#b87129','D':'#12806b','joint_cv':'#6c3f85'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for method in ('A','B','C','D'):
        d=annual[annual.method.eq(method)].sort_values('year')
        for ax,col,title in ((axes[0,0],'schedule_b','Exponent used in schedule'),(axes[0,1],'penalty','Annual penalty'),(axes[1,0],'complexity','Effective complexity')):
            ax.plot(d.year,d[col],label=method,color=colours[method],lw=1.5);ax.set_title(title);ax.set_xlabel('Decision year');ax.grid(alpha=.2)
    axes[0,1].set_yscale('log');axes[0,0].legend(ncol=4,frameon=False)
    d=annual[annual.method.eq('D')]
    for col,label,color in (('log_sample_size_contribution','Increasing T','#244c85'),('log_b_update_contribution','Updating b','#12806b'),('log_representation_contribution','Changing length','#b87129')):
        axes[1,1].plot(d.year,d[col],label=label,color=color)
    axes[1,1].plot(d.year,d.log_lambda_relative_initial,ls=':',color='black',label='Total')
    axes[1,1].set_title('D: exact log-penalty decomposition');axes[1,1].set_xlabel('Decision year');axes[1,1].legend(frameon=False,fontsize=8);axes[1,1].grid(alpha=.2)
    fig.suptitle('Theory-guided regularization | frozen initial constants',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(out/'figures'/f'regularization_diagnostic.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');wealth_rows=[]
    for ax,scenario,title in zip(axes,('gross','trade25_borrow30'),('Gross wealth','Net wealth: 25 bp trades + 30 bp/year shorts')):
        for method in ('A','B','C','D','joint_cv'):
            d=accounts[accounts.scenario.eq(scenario)&accounts.method.eq(method)&accounts.year.ge(1979)].sort_values('return_date')
            wealth=np.cumprod(1+d.total_return.to_numpy())
            ax.plot(d.return_date,wealth,label=method,color=colours[method],lw=1.4)
            wealth_rows.extend(dict(method=method,scenario=scenario,return_date=date,wealth=float(w)) for date,w in zip(d.return_date,wealth))
        ax.set_yscale('log');ax.set_title(title,fontsize=11);ax.set_ylabel('Wealth, initial value 1 (log scale)');ax.grid(alpha=.2);ax.legend(frameon=False,ncol=3,fontsize=8)
    fig.suptitle('February 1979-January 2025 | identical fixed pretest exposure scale',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(out/'figures'/f'performance_comparison.{ext}',dpi=180)
    plt.close(fig);save(out/'tables/performance_figure_data.csv',wealth_rows)
    rolling=pd.read_csv(out/'tables/rolling_monthly_diagnostics.csv',parse_dates=['decision_date'],float_precision='round_trip')
    primary_rows=rolling[rolling.rank_fraction_low.eq(.1)&rolling.rank_fraction_high.eq(.6)]
    assert len(primary_rows)==986, 'Both rolling primary-band curves must contain 493 decisions'
    fig,axes=plt.subplots(3,1,figsize=(10,7),layout='constrained')
    for W,color in ((180,'#244c85'),(240,'#12806b')):
        d=rolling[rolling.W.eq(W)&rolling.rank_fraction_low.eq(.1)&rolling.rank_fraction_high.eq(.6)]
        for ax,col,title in zip(axes,('b','penalty','complexity'),('Rolling exponent','Rolling theoretical penalty','Rolling effective complexity')):
            ax.plot(d.decision_date,d[col],label=f'{W} months',color=color,lw=1);ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend(frameon=False);axes[1].set_yscale('log');fig.suptitle('Monthly spectral diagnostic | translated pretest constants, no portfolios')
    for ext in ('png','pdf'):fig.savefig(out/'figures'/f'rolling_diagnostic.{ext}',dpi=180)
    plt.close(fig)
    write_json(out/'audit/figure_inputs.json',dict(input_hashes={p:digest(out/'tables'/p) for p in (
        'annual_policies.csv','monthly_accounts.csv','performance_figure_data.csv','rolling_monthly_diagnostics.csv')},
        figures={p.name:digest(p) for p in (out/'figures').iterdir()},
        all_curves_exported=True,rolling_primary_rows=len(primary_rows),wealth='Primary period rebased to 1; net state carried from 1978'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('summary','rolling','figures','all'),default='all')
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase in ('summary','all'):summarize(args.output)
        if args.phase in ('rolling','all'):rolling(args.output)
        if args.phase in ('figures','all'):figures(args.output)


if __name__=='__main__':main()
