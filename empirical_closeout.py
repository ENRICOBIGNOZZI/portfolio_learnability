"""Ex-post empirical diagnostics; no exported diagnostic feeds portfolio selection."""
from __future__ import annotations
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from data_pipeline import write_json,digest
from local_learnability import bin_observations,T_MAIN,T_GRID

SEED=20261003
BOOTSTRAPS=5000
BLOCK=12


def validation_diagnostics(paths):
    groups=['T_months','decision_year']
    # Stable ex-post tie rule mirrors validation: largest lambda among exact ties.
    oracle=paths.sort_values(['oos_loss','lambda'],ascending=[True,False]).drop_duplicates(groups)
    oracle=oracle[groups+['oos_loss','complexity','relative_complexity','lambda']].rename(columns={
        'oos_loss':'oracle_loss','complexity':'oracle_C','relative_complexity':'oracle_C_over_T','lambda':'oracle_lambda'})
    selected=paths.loc[paths.selected].copy()
    assert not selected.duplicated(groups).any()
    yearly=selected.merge(oracle,on=groups,validate='one_to_one')
    yearly['validation_regret']=yearly.oos_loss-yearly.oracle_loss
    yearly['relative_validation_regret']=yearly.validation_regret/yearly.oracle_loss
    yearly['absolute_C_over_T_gap']=abs(yearly.relative_complexity-yearly.oracle_C_over_T)
    rows=[]
    for t,d in yearly.groupby('T_months'):
        rows.append(dict(T_months=t,years=len(d),mean_validation_regret=d.validation_regret.mean(),
            mean_relative_validation_regret=d.relative_validation_regret.mean(),
            mean_absolute_C_over_T_gap=d.absolute_C_over_T_gap.mean(),
            spearman_C_over_T=spearmanr(d.relative_complexity,d.oracle_C_over_T).statistic,
            within_1_percent=float((d.relative_validation_regret<=.01).mean()),
            within_5_percent=float((d.relative_validation_regret<=.05).mean()),
            within_10_percent=float((d.relative_validation_regret<=.10).mean()),
            fraction_near_one=float((d.relative_complexity>=.99).mean()),
            fraction_near_zero=float((d.relative_complexity<=.01).mean()),
            fraction_exact_zero_lambda=float((d['lambda']==0).mean())))
    return yearly,pd.DataFrame(rows)


def common_period(monthly,selected):
    wide=monthly.pivot(index='return_date',columns='T_months',values='raw_excess_return').sort_index()
    wide.index=pd.to_datetime(wide.index)
    wide=wide.dropna()
    expected=pd.date_range('1994-01-31','2024-12-31',freq='ME')
    if not wide.index.equals(expected) or list(wide.columns)!=list(T_GRID):
        raise ValueError('Common period differs from the predeclared January 1994-December 2024.')
    summary=[]
    for t in T_GRID:
        r=wide[t];s=selected[(selected.T_months==t)&selected.decision_year.between(1994,2024)]
        assert len(s)==31
        summary.append(dict(T_months=t,start=wide.index.min(),end=wide.index.max(),months=len(r),
            response_one_loss=float(np.mean((1-r)**2)),monthly_mean=r.mean(),monthly_volatility=r.std(ddof=1),
            annualized_sharpe=float(np.sqrt(12)*r.mean()/r.std(ddof=1)),mean_C=s.complexity.mean(),
            mean_C_over_T=s.relative_complexity.mean(),fraction_near_one=float((s.relative_complexity>=.99).mean()),
            turnover_available=False))
    return wide,pd.DataFrame(summary)


def paired_bootstrap(wide,replicates=BOOTSTRAPS,block=BLOCK):
    rng=np.random.default_rng(SEED);n=len(wide)
    starts=rng.integers(0,n,size=(replicates,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(replicates,-1)[:,:n]
    loss=(1-wide.to_numpy())**2
    boot=loss[indices].mean(axis=1)
    rows=[]
    for a,b in itertools.combinations(range(wide.shape[1]),2):
        difference=boot[:,a]-boot[:,b]
        lo,hi=np.quantile(difference,[.025,.975])
        rows.append(dict(T_a=int(wide.columns[a]),T_b=int(wide.columns[b]),
            loss_a_minus_b=float(loss[:,a].mean()-loss[:,b].mean()),ci_low=lo,ci_high=hi,
            bootstrap_se=float(difference.std(ddof=1)),block_months=block,replicates=replicates,seed=SEED))
    intervals=np.quantile(boot,[.025,.975],axis=0)
    return pd.DataFrame(rows),intervals


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
        'font.size':11,'axes.titlesize':13,'axes.labelsize':12,'axes.edgecolor':'#727c89',
        'axes.linewidth':.65,'pdf.fonttype':42,'savefig.facecolor':'white'})
    return plt


def save(fig,output,stem,pdf=False):
    fig.savefig(output/(stem+'.png'),dpi=300)
    if pdf:fig.savefig(output/(stem+'.pdf'))


def heatmap(paths,yearly,output,metric):
    from matplotlib.colors import Normalize,LogNorm
    plt=style();years=np.arange(1978,2025)
    minima=paths.groupby(['T_months','decision_year']).oos_loss.transform('min')
    d=paths.copy()
    d['plot_value']=d.oos_loss-minima if metric=='delta' else np.log(d.oos_loss/minima) if metric=='log_ratio' else d.oos_loss
    main=d[d.T_months.isin(T_MAIN)]
    norm=LogNorm(main.plot_value.min(),main.plot_value.max()) if metric=='raw' else Normalize(0,main.plot_value.max())
    fig,axes=plt.subplots(1,3,figsize=(12.8,7.2),sharey=True)
    fig.subplots_adjust(left=.075,right=.855,bottom=.20,top=.80,wspace=.14)
    title={'delta':'Local learnability: loss above the year-specific optimum',
        'log_ratio':'Local learnability: loss relative to the year-specific optimum',
        'raw':'Local learnability: absolute next-year loss'}[metric]
    fig.text(.075,.95,title,fontsize=19,weight='bold')
    fig.text(.075,.90,'Matérn-3/2 | Complete-payoff sample sensitivity; current JKP snapshot',fontsize=12,color='#80552c')
    cells=[]
    for ax,t in zip(axes,T_MAIN):
        frame=d[d.T_months==t].copy();frame['oos_loss']=frame.plot_value
        edges,z,nearest=bin_observations(frame,60,years)
        cells.append(nearest[['T_months','decision_year','candidate','relative_complexity','plot_value','bin','bin_center']])
        mesh=ax.pcolormesh(edges,np.r_[years-.5,years[-1]+.5],np.ma.masked_invalid(z),cmap='cividis',norm=norm,rasterized=True)
        s=yearly[yearly.T_months==t]
        ax.scatter(s.relative_complexity,s.decision_year,s=20,facecolor='white',edgecolor='black',lw=.7,clip_on=False,zorder=3)
        ax.set(xlim=(0,1),ylim=(2024.5,1977.5),title=f'T = {t} months',xticks=np.linspace(0,1,5))
        if s.decision_year.min()>1978:
            ax.text(.5,(1978+s.decision_year.min()-1)/2,'Insufficient history',ha='center',color='#68707d')
    axes[0].set_ylabel('Decision year')
    fig.text(.45,.12,r'Relative empirical effective complexity $\widehat{\mathcal{C}}_T/T$',ha='center',fontsize=14)
    cax=fig.add_axes([.88,.20,.018,.60]);cb=fig.colorbar(mesh,cax=cax)
    cb.set_label({'delta':r'$Q_{y,T}(\lambda)-\min_{\lambda}Q_{y,T}$','log_ratio':r'$\log[Q_{y,T}(\lambda)/\min_{\lambda}Q_{y,T}]$','raw':'OOS response-one loss (log color scale)'}[metric],labelpad=12)
    fig.text(.075,.027,'White circles: chronological-validation selection, with no OOS tuning. Each cell is an observed path point.\n'
        'Row normalization is ex post and diagnostic only. Future-payoff availability affects the inherited stock sample.',fontsize=9.5,color='#4e5661',linespacing=1.5)
    stem='empirical_E2_row_normalized_heatmap' if metric=='delta' else 'empirical_appendix_'+metric+'_heatmap'
    save(fig,output,stem,pdf=metric=='delta');plt.close(fig)
    return pd.concat(cells,ignore_index=True)


def spectrum(cache,output,tables):
    plt=style();g=np.load(Path(cache)/'matern32_managed.npy')
    manifest=json.loads((Path(cache)/'manifest.json').read_text())
    # Cache rows run Jan 1963-Dec 2024 formation; January 2024 cutoff uses payoffs through Dec 2023.
    formations=pd.DatetimeIndex(manifest['dates'])
    assert len(formations)==len(g) and formations.is_monotonic_increasing and not formations.has_duplicates
    assert digest(Path(cache)/'matern32_managed.npy')==manifest['managed_file_sha256']
    returns=formations+pd.offsets.MonthEnd(1)
    rows=[];fig,ax=plt.subplots(figsize=(9.6,6.2));fig.subplots_adjust(left=.11,right=.96,bottom=.20,top=.78)
    fig.text(.11,.94,'E1  |  Managed-payoff economic spectrum',fontsize=19,weight='bold')
    fig.text(.11,.886,'January 2024 information cutoff | Complete-payoff sample sensitivity',fontsize=11.7,color='#80552c')
    for t,color in zip(T_MAIN,['#245b7d','#34918a','#c18933']):
        x=g[(returns<='2023-12-31')][-t:];assert len(x)==t
        mu=np.maximum(np.linalg.eigvalsh(x@x.T/t)[::-1],0)
        ax.loglog(np.arange(1,t+1),mu,lw=1.8,label=f'T = {t}',color=color)
        rows.extend(dict(T_months=t,rank=k+1,second_moment_eigenvalue=m,cutoff='2023-12-31') for k,m in enumerate(mu))
    ax.set(xlabel='Eigenvalue rank',ylabel='Managed-payoff second-moment eigenvalue')
    ax.grid(which='major',color='#dce2e8',lw=.6);ax.legend(frameon=False)
    fig.text(.11,.038,'Fixed 10,000-feature Matérn representation; uncentered second moments, with normalization 1/T.\n'
        'This retrospective spectrum describes payoff geometry; it does not establish a stationary rate law.',fontsize=10,color='#4e5661',linespacing=1.45)
    save(fig,output,'empirical_E1_managed_payoff_spectrum',pdf=True);plt.close(fig)
    pd.DataFrame(rows).to_csv(tables/'managed_payoff_spectrum_common_cutoff.csv',index=False)


def tradeoff(summary,intervals,output):
    plt=style();fig,axes=plt.subplots(1,3,figsize=(12.8,5.5))
    fig.subplots_adjust(left=.072,right=.97,bottom=.22,top=.75,wspace=.36)
    fig.text(.072,.94,'E3  |  More history versus older regimes',fontsize=19,weight='bold')
    fig.text(.072,.878,'Common OOS months: January 1994-December 2024 | Complete-payoff sample sensitivity',fontsize=12,color='#80552c')
    x=summary.T_months
    axes[0].plot(x,summary.response_one_loss,'o-',color='#245b7d',lw=1.6)
    axes[0].vlines(x,intervals[0],intervals[1],color='#245b7d',lw=1.3)
    axes[0].set_ylabel('Monthly response-one loss')
    axes[1].plot(x,summary.annualized_sharpe,'o-',color='#34918a',lw=1.6);axes[1].set_ylabel('Concatenated annualized Sharpe')
    axes[2].plot(x,summary.mean_C_over_T,'o-',color='#b58030',lw=1.6);axes[2].set_ylabel('Mean selected C/T');axes[2].set_ylim(0,1)
    for ax in axes:
        ax.set_xlabel('Historical window T (months)');ax.set_xticks([60,120,180,240,360]);ax.grid(color='#dce2e8',lw=.6)
    fig.text(.072,.037,'All windows use the same 372 monthly payoffs. Loss bars: 95% circular 12-month block-bootstrap intervals.\n'
        'Sharpe uses the concatenated return series. Costs and turnover are unavailable; this is not a stationary asymptotic-rate test.',fontsize=9.5,color='#4e5661',linespacing=1.5)
    save(fig,output,'empirical_E3_common_period_window_tradeoff',pdf=True);plt.close(fig)


def methodology(out,summary,common):
    audit=json.loads((out/'tables'/'formation_timing_audit.json').read_text())
    def markdown(d):
        columns=list(d.columns)
        return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
    quality=markdown(summary[['T_months','years','mean_validation_regret','mean_relative_validation_regret','spearman_C_over_T','within_10_percent','fraction_near_one']])
    performance=markdown(common[['T_months','months','response_one_loss','monthly_mean','monthly_volatility','annualized_sharpe','mean_C_over_T']])
    text=f"""# Empirical close-out: retrospective sample sensitivities

## Status of the evidence

The proposed three figure roles are E1 (managed-payoff spectrum), E2 (row-normalized
year-by-C/T learnability), and E3 (common-period window tradeoff). **All three current
figures are explicitly labeled complete-payoff sample sensitivities. None is presented
as pristine real-time investment evidence.** Rebuilding the formation universe is
possible; identifying every realized payoff from the current snapshot is not.
The missing-payoff limitation is isolated rather than silently filled or reclassified.

The private formation-only panel now contains {audit['formation_stock_months']:,} stock-months
in {audit['formation_months']} months, before any future-payoff filter. Ranks are constructed
from formation metadata and characteristics only, retaining unknown returns as NaN.
Changing all future payoffs to missing leaves identities and ranks exactly unchanged
(maximum perturbation error {audit['future_availability_perturbation_rank_error']}).
The maximum characteristic-rank change versus the earlier complete-payoff construction
is {audit['max_characteristic_rank_change_vs_complete_payoff_sample']:.8f} on the [-0.5,0.5] scale.
This measures the cross-sectional rank effect, not the resulting portfolio-performance bias.

There are {audit['original_missing_payoffs']:,} missing forward payoffs. The audit matches each
security to its **exact next calendar month** in the frozen raw source and tries the
source's current excess return. It recovers {audit['recovered_exact_next_calendar_current_excess']};
{audit['unresolved_payoffs']:,} remain unidentified across {audit['months_with_unresolved_payoffs']} formation
months. No nearest-date substitution, zero return, sample reweighting, or undocumented
delisting imputation is applied. Calculating complete managed returns for this
formation-only panel would require additional source evidence or an explicit missing-return
assumption. The original fitted cache is therefore retained only for the labeled
sensitivity diagnostics, avoiding a misleading claim that reranking alone solves missing
payoffs. These numbers do not prove the missing payoffs are zero or economically irrelevant.

The previous CRSP checks concerned **finite** JKP forward returns lacking a next-month
characteristic row; they do not resolve these {audit['unresolved_payoffs']:,} missing outcomes.
No new WRDS fetch was used in this close-out. Private rebuilt files are in
`data/formation_only`; `formation_timing_by_month.csv` publishes aggregate counts only.

## Characteristic selection and timing

The 130-variable list was recomputed from the 153 frozen candidates using only
1963-1972 formation metadata and characteristic coverage, with alphabetical ties.
It exactly matches the stored list: {audit['characteristic_coverage_observations']:,} reference
observations, no forward-return availability or OOS outcome in selection.
This meets the pre-OOS coverage-freeze convention. It does not reconstruct historical
vintages: candidate discovery, revised accounting values and current JKP snapshot
availability cannot be certified as known in real time. The list is a retrospective
research specification, clearly separate from outcome-dependent feature selection.

Annual fitting continues to use the already audited January decision chronology:
T payoffs through December y-1; first T-V train, last V=min(60,floor(T/3)) validate;
refit all T, then evaluate January-December y. The 1,201-point lambda paths include
zero and use inner-training information only. Representation settings remain frozen
from the initial 1963-1972 sample. Existing future-payoff perturbation checks show
that later portfolio payoffs cannot change earlier fitted grids or selections.
That estimator-level check does not cure stock-source selection or vintage limitations.

## E1: managed economic spectrum

The three curves use the same December 31, 2023 payoff cutoff for T=60,120,240.
They diagonalize the uncentered second-moment operator G'G/T via GG'/T.
Formation rows run through November 2023. Neither centered covariance nor a spectrum
of raw characteristics is substituted for the managed-payoff operator.

## E2: row-normalized heatmaps and selection quality

For each year and T, the ex-post oracle minimizes OOS response-one loss across the
entire saved lambda path; ties choose largest lambda. This diagnostic never selects
the portfolio. Default color is DeltaQ=Q-min(Q); the appendix includes natural
log(Q/min(Q)) and the original raw loss, each separately labeled. The natural-log
version reveals smaller within-year differences more clearly, while DeltaQ retains
an immediately interpretable loss scale. All paths here have positive minima.

The 60 equal-width C/T bins select the observed path point nearest the bin center
within the bin. Empty cells remain blank. Minima are computed on the **full** paths
before binning, not from displayed cells. Saved cell tables identify candidate indices.
Validation overlays are white circles with black borders, without connecting lines.
Near-one means C/T >=0.99; near-zero means <=0.01, fixed before inspecting frequencies.
The full table includes absolute C/T gaps and fractions within 1%, 5%, and 10% of oracle.

{quality}

Validation is not consistently interior and does not track the ex-post optimum reliably.
Its mean relative regret ranges from approximately 33% to 58%; Spearman correlations
range from approximately -0.18 to +0.25. Ex-post selection among 1,201 candidates using
only 12 OOS returns also makes the oracle optimistic as a regime diagnostic.
These are diagnostics of a noisy tuning problem, not a claim of implementable oracle gains.

## E3: exact common-period comparison

Every row uses exactly the same 372 monthly payoffs, January 1994-December 2024.
Response-one loss is the mean of (1-r)^2; volatility uses the sample standard deviation
(ddof=1), and annualized Sharpe is sqrt(12)*mean(r)/std(r). Raw portfolio payoffs are
unscaled excess returns from the response-one estimator, without a cash overlay,
transaction costs or an additional leverage normalization. Their magnitudes should
not be interpreted as returns on a standard unlevered stock index. These values are
not averages of 31 annual Sharpes. Mean C and C/T use the 31 annual selections.

{performance}

Uncertainty uses 5,000 circular moving-block bootstrap replicates with 12-month blocks
and seed 20261003. The **same monthly indices** resample all six windows in each
replicate. Pairwise percentile 95% intervals for all 15 loss differences are saved;
each includes zero in this sample, so the point-estimate ranking is not a statistically
clear winner under this procedure. Intervals describe uncertainty of the realized
performance series, conditional on already fitted policies; the bootstrap does not
rerun training and is not a proof of stationarity across 1994-2024.
The question is more observations versus older, potentially less relevant regimes,
not the stationary asymptotic T-law. Turnover is unavailable because local annual
stock weights were not retained by this pipeline; it is not set to zero.

## Reproduction and proposed placement

Run `VECLIB_MAXIMUM_THREADS=1 python3 audit_formation_timing.py --raw data/raw --destination data/formation_only`
then `VECLIB_MAXIMUM_THREADS=1 python3 empirical_closeout.py`.
No fitting is repeated for these diagnostics; their input paths and cache hashes
are pinned in `manifest_empirical_closeout.json`. The source-timing audit took
{audit['elapsed_seconds']:.1f} seconds on the recorded local run.

Use E1-E3 as the proposed three empirical figure roles **with their sensitivity status
visible**. The existing absolute-C and selected-path figures and the new raw/log-loss
heatmaps belong in an appendix. Gaussian cache replication remains an available
pipeline option, not an additional completed result of this close-out.
"""
    (out/'notes'/'empirical_closeout_methodology.md').write_text(text)


def run(args):
    started=time.perf_counter();out=Path(args.output);tables=out/'tables';figs=out/'figures'
    paths=pd.read_parquet(tables/'year_lambda_complexity_path_matern32.parquet')
    monthly=pd.read_csv(tables/'selected_monthly_payoffs_matern32.csv')
    yearly,summary=validation_diagnostics(paths)
    yearly.to_csv(tables/'validation_vs_oracle_by_year.csv',index=False)
    summary.to_csv(tables/'validation_vs_oracle_summary.csv',index=False)
    wide,common=common_period(monthly,yearly)
    bootstrap,intervals=paired_bootstrap(wide)
    common.to_csv(tables/'common_period_window_comparison.csv',index=False)
    bootstrap.to_csv(tables/'common_period_paired_loss_bootstrap.csv',index=False)
    for metric in ('delta','log_ratio','raw'):
        cells=heatmap(paths,yearly,figs,metric)
        cells.to_csv(tables/f'empirical_heatmap_cells_{metric}.csv',index=False)
    spectrum(args.cache,figs,tables);tradeoff(common,intervals,figs)
    write_json(out/'manifest_empirical_closeout.json',dict(seed=SEED,bootstrap_replicates=BOOTSTRAPS,
        block_months=BLOCK,common_months=len(wide),near_one_threshold=.99,near_zero_threshold=.01,
        inputs={str(p):digest(p) for p in [tables/'year_lambda_complexity_path_matern32.parquet',tables/'selected_monthly_payoffs_matern32.csv',Path(args.cache)/'matern32_managed.npy']},
        source_sha256=digest(__file__),elapsed_seconds=time.perf_counter()-started,
        evidence_status='Complete-payoff sample sensitivity, not pristine real-time OOS. See formation_timing_audit.json.'))
    methodology(out,summary,common)
    print(summary.to_string(index=False));print(common.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='outputs')
    parser.add_argument('--cache',default='results/schedule_cache');run(parser.parse_args())
