"""Rolling local-economy learnability maps with strictly chronological validation."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from complexity_schedule import cache_managed
from data_pipeline import digest, write_json
from portfolio import dual_path, sharpe

T_GRID=(60,84,120,180,240,360)
T_MAIN=(60,120,240)


def rolling_split(return_dates, year, months, validation_months=None, timing='calendar'):
    dates=pd.DatetimeIndex(return_dates)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Return dates must be ordered and unique.')
    if months < 3 or timing not in ('calendar','january-close'):
        raise ValueError('Invalid window or timing.')
    v=min(60,months//3) if validation_months is None else int(validation_months)
    if not 1 <= v < months:
        raise ValueError('Validation length must lie strictly inside each rolling window.')
    cutoff=pd.Timestamp(year-1,12,31) if timing=='calendar' else pd.Timestamp(year,1,31)
    expected_history=pd.date_range(end=cutoff,periods=months,freq='ME')
    expected_test=pd.date_range(start=cutoff+pd.offsets.MonthEnd(1),periods=12,freq='ME')
    h=dates.get_indexer(expected_history);test=dates.get_indexer(expected_test)
    if (h<0).any():return None,'insufficient trailing history'
    if (test<0).any():return None,'incomplete 12-month OOS block'
    assert dates[h].max()<=cutoff<dates[test].min()
    return {'year':year,'T':months,'V':v,'cutoff':cutoff,
        'history':h,'inner':h[:-v],'validation':h[-v:],'test':test},None


def local_lambda_grid(inner_gram, count=1200):
    """Grid depends exclusively on inner training, never validation or test payoffs."""
    if count < 60:raise ValueError('Use at least 60 positive lambda candidates.')
    n=len(inner_gram)
    mu=np.linalg.eigvalsh((inner_gram+inner_gram.T)/2)/n
    active=mu[mu>max(float(mu.max()),1e-30)*1e-12]
    if len(active)==0:raise ValueError('Degenerate inner-training spectrum.')
    return np.r_[0.,np.geomspace(active.min()*1e-6,active.max()*1e6,count)]


def fit_local(gram, split, count=1200):
    h,i,v,o=(split[n] for n in ('history','inner','validation','test'))
    inner_gram=gram[np.ix_(i,i)]
    penalties=local_lambda_grid(inner_gram,count)
    a,_,_=dual_path(inner_gram,penalties)
    validation_returns=gram[np.ix_(v,i)]@a
    losses=np.mean((1-validation_returns)**2,axis=0)
    choice=int(np.flatnonzero(losses==losses.min())[-1])
    full,mu,complexity=dual_path(gram[np.ix_(h,h)],penalties)
    # Only now touch held-out payoffs; no OOS quantity selects lambda.
    returns=gram[np.ix_(o,h)]@full
    return {'lambda':penalties,'validation_loss':losses,'choice':choice,
        'complexity':complexity,'relative_complexity':complexity/len(h),
        'oos_loss':np.mean((1-returns)**2,axis=0),'oos_sharpe':sharpe(returns,axis=0),
        'returns':returns,'selected_alpha':full[:,choice],
        'rank':int(np.sum(mu>max(float(mu.max()),1e-30)*1e-12))}


def bin_observations(frame, bins=60, years=None):
    """Nearest observed path point INSIDE each bin; unoccupied bins remain blank."""
    years=sorted(frame.decision_year.unique()) if years is None else list(years)
    edges=np.linspace(0.,1.,bins+1);centers=(edges[:-1]+edges[1:])/2
    d=frame.copy()
    x=d.relative_complexity.to_numpy()
    if (~np.isfinite(x)).any() or (x < -1e-10).any() or (x>1+1e-10).any():
        raise ValueError('Relative empirical complexity must be in [0,1].')
    d['bin']=np.minimum(np.floor(np.clip(x,0,1)*bins).astype(int),bins-1)
    d['distance_to_bin_center']=np.abs(x-centers[d.bin.to_numpy()])
    nearest=d.sort_values(['distance_to_bin_center','candidate']).drop_duplicates(['decision_year','bin'])
    z=np.full((len(years),bins),np.nan);index={int(y):i for i,y in enumerate(years)}
    for r in nearest.itertuples():z[index[r.decision_year],r.bin]=r.oos_loss
    nearest=nearest.assign(bin_left=edges[nearest.bin],bin_right=edges[nearest.bin+1],
                           bin_center=centers[nearest.bin])
    return edges,z,nearest


def render_figures(paths, selected, kernel, output, bins, start_year, end_year, timing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix','font.size':11,'axes.labelsize':12,
        'axes.titlesize':13,'axes.edgecolor':'#777e89','axes.linewidth':.7,
        'pdf.fonttype':42,'savefig.facecolor':'white'})
    out=output/'figures';out.mkdir(parents=True,exist_ok=True)
    label='Matérn-3/2' if kernel=='matern32' else 'Gaussian'
    main=paths[paths.T_months.isin(T_MAIN)]
    if not np.isfinite(main.oos_loss).all() or main.oos_loss.min()<=0:
        raise ValueError('A log color scale requires positive finite OOS losses.')
    norm=LogNorm(float(main.oos_loss.min()),float(main.oos_loss.max()))
    cmap=plt.get_cmap('cividis').copy();cmap.set_bad((1,1,1,0))
    years=np.arange(start_year,end_year+1);cells=[]
    for relative in (True,False):
        fig,axes=plt.subplots(1,3,figsize=(12.8,7.1),sharey=True)
        fig.subplots_adjust(left=.075,right=.865,bottom=.19,top=.80,wspace=.15)
        fig.text(.075,.947,'Local learnability map: next-year out-of-sample loss',fontsize=19,weight='bold')
        fig.text(.075,.902,f'{label} | Each row is a local economy defined by a trailing T-month history.',fontsize=12,color='#4e5661')
        for ax,t in zip(axes,T_MAIN):
            d=paths[paths.T_months==t];s=selected[selected.T_months==t]
            edges,z,nearest=bin_observations(d,bins,years)
            if relative:cells.append(nearest)
            xe=edges if relative else edges*t
            mesh=ax.pcolormesh(xe,np.r_[years-.5,years[-1]+.5],np.ma.masked_invalid(z),
                              cmap=cmap,norm=norm,rasterized=True,shading='flat')
            x=s.relative_complexity if relative else s.complexity
            ax.plot(x,s.decision_year,color='white',lw=.55,alpha=.7,zorder=3)
            ax.scatter(x,s.decision_year,facecolor='white',edgecolor='#111111',s=20,linewidth=.75,zorder=4,clip_on=False)
            ax.set_ylim(end_year+.5,start_year-.5);ax.set_xlim(0,1 if relative else t)
            title=f'T = {t} months'
            title+=f'  |  V = {int(s.V_months.iloc[0])}' if len(s) else '  |  unavailable'
            ax.set_title(title,pad=10)
            ax.set_xticks(np.linspace(0,1 if relative else t,5))
            ax.tick_params(axis='both',labelsize=10)
            if len(s):
                first=int(s.decision_year.min())
                if first>start_year:
                    ax.text(.5,(start_year+first-1)/2,'Insufficient history',ha='center',va='center',
                            transform=ax.get_yaxis_transform(),color='#68707d',fontsize=10)
        axes[0].set_ylabel('Decision year')
        xlabel=r'Relative effective complexity $\widehat{\mathcal C}_T/T$' if relative else r'Absolute effective complexity $\widehat{\mathcal C}_T$'
        fig.text(.47,.108,xlabel.replace(r'\mathcal C',r'\mathcal{C}'),ha='center',fontsize=14)
        cax=fig.add_axes([.89,.19,.017,.61]);cb=fig.colorbar(mesh,cax=cax)
        cb.set_label('Next-year OOS response-one loss',fontsize=12,labelpad=12)
        note='White circles: lambda chosen by inner chronological validation, then refitted on all T months.'
        period='January-December OOS payoffs; information cutoff is December 31 of the previous year.' if timing=='calendar' else 'February-January OOS payoffs; information cutoff is the January formation close.'
        fig.text(.075,.025,note+'\n'+period+' Blank cells contain no observed path point.',fontsize=9.3,color='#4e5661',linespacing=1.45)
        stem=f'heatmap_loss_by_year_{"complexity" if relative else "abs_complexity"}_{kernel}_T60_T120_T240'
        fig.savefig(out/(stem+'.png'),dpi=300);fig.savefig(out/(stem+'.pdf'));plt.close(fig)
    for relative in (False,True):
        fig,ax=plt.subplots(figsize=(10.2,6.1));fig.subplots_adjust(left=.10,right=.97,bottom=.19,top=.77)
        fig.text(.10,.94,'Validation-selected '+('relative complexity' if relative else 'effective complexity'),fontsize=19,weight='bold')
        fig.text(.10,.888,f'{label} | Chronological validation inside each trailing local window',fontsize=12,color='#4e5661')
        styles=['-','--','-.',':','-','--'];markers=['o','s','^','D','v','P']
        colors=['#143d59','#355f8d','#287d8e','#568c4e','#b38a23','#a55b42']
        for j,t in enumerate(T_GRID):
            s=selected[selected.T_months==t]
            ax.plot(s.decision_year,s.relative_complexity if relative else s.complexity,
                color=colors[j],ls=styles[j],marker=markers[j],ms=2.7,lw=1.1,label=f'T = {t}',clip_on=False)
        ax.set_xlabel('Decision year');ax.set_ylabel('Selected C/T' if relative else 'Selected effective complexity C')
        ax.set_ylim(bottom=0,top=1.03 if relative else None)
        ax.grid(color='#dfe3e8',lw=.6);ax.set_axisbelow(True)
        fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.09,.85),
                   ncol=6,frameon=False,fontsize=10,columnspacing=1.1)
        fig.text(.10,.045,'Each point uses a full T-month refit at the lambda minimizing inner validation response-one loss.\n'
                 'No OOS payoff enters the selection. Lines start only when the entire trailing history is available.',
                 fontsize=10,color='#4e5661',linespacing=1.4)
        stem=f'selected_{"relative_" if relative else ""}complexity_over_time_{kernel}'
        fig.savefig(out/(stem+'.png'),dpi=300);fig.savefig(out/(stem+'.pdf'));plt.close(fig)
    return pd.concat(cells,ignore_index=True),{'loss_vmin':norm.vmin,'loss_vmax':norm.vmax,'colormap':'cividis','normalization':'log, identical across both heatmap figures'}


def methodology(kernel,selected,skipped,count,bins,timing):
    coverage=selected.groupby('T_months').agg(first_year=('decision_year','min'),last_year=('decision_year','max'),
        years=('decision_year','count'),validation_months=('V_months','first')).reset_index()
    rows='\n'.join(f'| {r.T_months} | {r.validation_months} | {r.first_year}-{r.last_year} | {r.years} |' for r in coverage.itertuples())
    chronology=('Decision year y starts after the December 31 close of y-1. The trailing history contains '
        'T **payoffs** through that December, corresponding to formations through November y-1. '
        'The OOS block is January-December y (formations December y-1 through November y). '
        'For example, y=2000 and T=60 use January 1995-December 1999 payoffs; '
        'default inner training ends April 1998, validation is May 1998-December 1999, '
        'and OOS is January-December 2000.') if timing=='calendar' else (
        'This optional repository convention refits at the January close of y, using payoffs through January y '
        '(formations through December y-1), and tests February y-January y+1. It uses information after '
        'the beginning of calendar year y; it is not the strict before-year-y convention.')
    return f'''# Local learnability heatmaps: {kernel}

T is the length of trailing monthly history treated as locally informative for the current regime.
It is not the total history of a globally stationary economy, nor a count of stock-month rows.

## Chronology and validation

{chronology}
At the information cutoff we assume month-end inputs can be used for portfolio formation;
execution delay and trading costs are not modeled. Every fit uses exactly T consecutive months.
Within that history the earliest T-V months train the candidate policy and the latest V months
validate it. By default V=min(60,floor(T/3)); `--validation-months` supplies a fixed override.
The candidate minimizing mean `(1 - raw portfolio excess payoff)^2` on validation is selected
(exact ties use the largest lambda). It is then refitted on **all T months**, and the overlay
uses the resulting full-window complexity. No OOS loss or Sharpe participates in selection.
This rolling-validation design replaces the fixed-C0 schedule for this experiment only.

## Estimation and grid

The existing fixed feature bank and managed payoffs are reused: 130 characteristics,
10,000 fixed RFFs and seed zero; representation inputs were fixed using 1963-1972.
Write G for the T-by-feature managed-payoff matrix and M=GG'. The repository solves
`alpha=(M+T*lambda*I)^(-1)1`. Complexity is `sum(mu/(mu+lambda))`, with mu the
eigenvalues of M/T, exactly equivalent to `tr[M(M+T*lambda*I)^(-1)]`.
The grid has {count} positive log-spaced penalties from `mu_min*1e-6` to `mu_max*1e6`,
using only the **inner-training** spectrum, plus exact lambda=0. All {count+1} candidates
are eligible for validation selection. Zero uses a minimum-norm pseudoinverse; numerical
rank uses a relative 1e-12 cutoff. Thus C/T is at most one. OOS loss uses the 12 held-out
monthly raw excess payoffs with response one; neither a ridge penalty nor kappa is added.
Annual OOS Sharpe is a secondary, noisy statistic computed from those same 12 payoffs.

## Heatmap construction

The main panels use T=60,120,240. Their common relative grid has {bins} equal-width bins
on [0,1]. Each occupied year-bin uses the **observed** path point nearest its center,
among points inside that bin. Empty bins and unavailable years stay transparent; no
interpolation or extrapolation is performed. A saved cells table identifies every point
used. The absolute-complexity companion rescales these same bin edges by each panel's T;
its horizontal limits consequently differ. Both figures share the same full-path OOS-loss
color range and logarithmic normalization. Cividis has monotonic luminance for grayscale
printing; white circles with black borders identify validation selections.

## Coverage

| T (months) | V (months) | Decision years | Annual fits |
|---:|---:|---|---:|
{rows}

{len(skipped)} requested year-window combinations are unavailable; reasons and available
history are recorded in the skipped-windows CSV. Plots retain the requested year axis.
The summary gives arithmetic means across each T's **available years**; different coverage
means those rows are not a matched-period causal comparison. Mean annual Sharpe is not
the Sharpe of the concatenated monthly history.

## Provenance and limits

The manifest records source/cache/code hashes, numerical settings and perturbation checks.
Changing OOS and later payoffs must leave earlier grids, validation selections and fitted
coefficients unchanged; Gram and primal implementations are independently compared in tests.
The inherited sample excludes missing future stock payoffs before ranking. Its dependence
on future payoff availability, the retrospective characteristic dictionary and the current
JKP snapshot remain source-level limitations. The overlay introduces no additional future
information. Local windows do not by themselves establish a stationary local regime.

Reproduce with `python3 local_learnability.py --kernel {kernel} --timing {timing}`.
Use `--kernel gaussian` for the Gaussian representation (its cache is built when absent).
Use a fresh `--output` directory for a new run; source data and private caches are excluded
from Git. The published path parquet contains portfolio-level aggregates only.
'''


def run(args):
    if args.bins<2:raise ValueError('Use at least two heatmap bins.')
    if args.start_year<1978 or args.end_year<args.start_year:raise ValueError('Use decision years from 1978 onward.')
    if args.validation_months is not None and not 1<=args.validation_months<min(T_GRID):
        raise ValueError('A fixed V must lie between 1 and 59 for this T grid.')
    output=Path(args.output);tables=output/'tables';figures=output/'figures';notes=output/'notes'
    for d in (tables,figures,notes):d.mkdir(parents=True,exist_ok=True)
    manifest_path=output/f'manifest_local_learnability_{args.kernel}.json'
    stem=f'year_lambda_complexity_path_{args.kernel}'
    if manifest_path.exists() or (tables/(stem+'.parquet')).exists():raise FileExistsError('Use fresh outputs; do not mix runs.')
    cache=args.cache or ('results/schedule_cache' if args.kernel=='matern32' else 'results/gaussian_cache')
    g,formations,source=cache_managed(args.clean,cache,kernel=args.kernel)
    dates=formations+pd.offsets.MonthEnd(1);gram=g@g.T
    frames,selected,skipped,monthly,records=[],[],[],[],[]
    for t in T_GRID:
        for year in range(args.start_year,args.end_year+1):
            sp,reason=rolling_split(dates,year,t,args.validation_months,args.timing)
            if sp is None:
                cutoff=pd.Timestamp(year-1,12,31) if args.timing=='calendar' else pd.Timestamp(year,1,31)
                skipped.append({'decision_year':year,'T_months':t,'reason':reason,'available_history_months':int((dates<=cutoff).sum())})
                continue
            result=fit_local(gram,sp,args.lambda_count);choice=result['choice'];n=len(result['lambda'])
            common={'kernel':args.kernel,'decision_year':year,'T_months':t,'V_months':sp['V'],
                'information_cutoff':str(sp['cutoff'].date()),'history_start':str(dates[sp['history'][0]].date()),
                'history_end':str(dates[sp['history'][-1]].date()),'inner_train_end':str(dates[sp['inner'][-1]].date()),
                'validation_start':str(dates[sp['validation'][0]].date()),'validation_end':str(dates[sp['validation'][-1]].date()),
                'oos_start':str(dates[sp['test'][0]].date()),'oos_end':str(dates[sp['test'][-1]].date()),'numerical_rank':result['rank']}
            f=pd.DataFrame({**common,'candidate':np.arange(n),**{k:result[k] for k in
                ['lambda','complexity','relative_complexity','validation_loss','oos_loss','oos_sharpe']},'selected':np.arange(n)==choice})
            frames.append(f);selected.append(f.iloc[choice].to_dict());records.append((sp,result))
            for j,index in enumerate(sp['test']):
                monthly.append({'kernel':args.kernel,'decision_year':year,'T_months':t,'return_date':dates[index],
                                'lambda':result['lambda'][choice],'raw_excess_return':result['returns'][j,choice]})
        print('Finished T =',t,flush=True)
    if not frames:raise ValueError('No complete rolling windows available.')
    paths=pd.concat(frames,ignore_index=True);chosen=pd.DataFrame(selected)
    omitted=pd.DataFrame(skipped,columns=['decision_year','T_months','reason','available_history_months'])
    if not np.isfinite(paths[['lambda','complexity','relative_complexity','validation_loss','oos_loss']]).all().all():raise ValueError('Nonfinite path metrics.')
    summary=chosen.groupby('T_months').agg(first_year=('decision_year','min'),last_year=('decision_year','max'),n_years=('decision_year','count'),
        mean_selected_C=('complexity','mean'),mean_selected_C_over_T=('relative_complexity','mean'),
        mean_oos_loss=('oos_loss','mean'),mean_oos_sharpe=('oos_sharpe','mean')).reset_index()
    paths.to_parquet(tables/(stem+'.parquet'),index=False,compression='zstd')
    chosen.to_csv(tables/f'selected_complexity_by_year_{args.kernel}.csv',index=False)
    summary.to_csv(tables/f'selected_complexity_summary_{args.kernel}.csv',index=False)
    omitted.to_csv(tables/f'skipped_windows_{args.kernel}.csv',index=False)
    pd.DataFrame(monthly).to_csv(tables/f'selected_monthly_payoffs_{args.kernel}.csv',index=False)
    # Adversarial future perturbations on actual first/last available windows of every T.
    audit=[]
    for t in T_GRID:
        cases=[r for r in records if r[0]['T']==t]
        if not cases:continue
        for sp,before in (cases[0],cases[-1]):
            changed=g.copy();changed[dates>sp['cutoff']]*=-7
            after=fit_local(changed@changed.T,sp,args.lambda_count)
            for key in ['lambda','validation_loss','complexity','selected_alpha']:
                np.testing.assert_allclose(before[key],after[key],rtol=1e-12,atol=1e-12)
            assert before['choice']==after['choice']
            audit.append({'T':t,'decision_year':sp['year'],'future_payoff_perturbation':'passed'})
    cells,color=render_figures(paths,chosen,args.kernel,output,args.bins,args.start_year,args.end_year,args.timing)
    cells.to_csv(tables/f'heatmap_cells_{args.kernel}.csv',index=False)
    (notes/f'heatmap_methodology_{args.kernel}.md').write_text(methodology(args.kernel,chosen,omitted,args.lambda_count,args.bins,args.timing))
    public_files=list(figures.glob(f'*{args.kernel}*'))+list(tables.glob(f'*{args.kernel}*'))+list(notes.glob(f'*{args.kernel}*'))
    write_json(manifest_path,{'status':'complete','kernel':args.kernel,'T_grid':T_GRID,'T_main':T_MAIN,'timing':args.timing,
        'validation_months_override':args.validation_months,'positive_lambda_count':args.lambda_count,'includes_exact_zero':True,
        'source':source,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_checksums':{n:digest(Path(n)) for n in ['local_learnability.py','portfolio.py','complexity_schedule.py','kernels.py','data_pipeline.py']},
        'fit_count':len(chosen),'path_rows':len(paths),'skipped_windows':skipped,'future_perturbation_audit':audit,'color_scale':color,
        'source_caveat':'Inherited future-payoff-availability filter; current snapshot and retrospective dictionary; no added future information in validation.',
        'outputs':{str(p.relative_to(output)):digest(p) for p in public_files}})
    print(summary.to_string(index=False),flush=True)
    print(f'COMPLETE: figures={figures}; data={tables}; methodology={notes}; manifest={manifest_path}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel',choices=['matern32','gaussian'],default='matern32')
    p.add_argument('--clean',default='data/clean');p.add_argument('--cache')
    p.add_argument('--output',default='outputs');p.add_argument('--start-year',type=int,default=1978)
    p.add_argument('--end-year',type=int,default=2024);p.add_argument('--validation-months',type=int)
    p.add_argument('--lambda-count',type=int,default=1200);p.add_argument('--bins',type=int,default=60)
    p.add_argument('--timing',choices=['calendar','january-close'],default='calendar')
    run(p.parse_args())
