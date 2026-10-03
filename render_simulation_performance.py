"""Four reference-style performance figures for the simulated boundary economy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator,FuncFormatter
from empirical_closeout import style,save
from data_pipeline import write_json,digest

STEMS=['simulation_dgp_01_loss_complexity','simulation_dgp_02_sharpe_complexity',
       'simulation_dgp_03_loss_heatmap','simulation_dgp_04_selected_complexity_lambda']
OVERVIEW='simulation_dgp_four_panel_overview'


def curves(ax,surface,metric,error,colors,sharpe=False,legend=True):
    for t,color in colors.items():
        d=surface[surface['T']==t].sort_values('population_C')
        x=d.population_C.to_numpy();y=d[metric].to_numpy();se=d[error].to_numpy()
        ax.plot(x,y,color=color,lw=1.6,label=str(t),zorder=3)
        ax.fill_between(x,y-1.96*se,y+1.96*se,color=color,alpha=.055,lw=0)
    ax.set_xscale('log')
    if sharpe:
        optimum=np.sqrt(12*.16/.84)
        ax.axhline(optimum,color='#75808c',lw=1,ls='--')
        ax.set_ylim(min(0,float((surface[metric]-1.96*surface[error]).min())),optimum*1.06)
        ax.set_ylabel('Population Sharpe (annualized monthly)')
    else:
        ax.set_yscale('log');ax.axhline(.84,color='#75808c',lw=1,ls='--')
        ax.set_ylabel(r'Population loss $Q=E[(1-\widehat{\theta}^{\prime}F)^2]$')
    ax.set_xlim(surface.population_C.min(),surface.population_C.max())
    ax.set_xlabel(r'Population effective complexity $C(\lambda)$')
    ax.grid(which='major',color='#dce2e8',lw=.6)
    if legend:
        ax.legend(title='Training months T',loc='upper left',bbox_to_anchor=(1.02,1),
                  frameon=False,fontsize=10,title_fontsize=10.5)


def heatmap(ax,fig,surface,selection,compact=False):
    ts=sorted(surface['T'].unique());matrix=[];oracle=[]
    for t in ts:
        d=surface[surface['T']==t].sort_values('population_C')
        matrix.append(d.mean_population_loss.to_numpy())
        oracle.append(float(d.loc[d.mean_population_loss.idxmin(),'population_C']))
    c=d.population_C.to_numpy();edges=np.r_[c[0],np.sqrt(c[:-1]*c[1:]),c[-1]]
    z=np.array(matrix)
    mesh=ax.pcolormesh(edges,np.arange(len(ts)+1)-.5,z,cmap='coolwarm',
                      norm=LogNorm(z.min(),z.max()),rasterized=True)
    ax.plot(oracle,np.arange(len(ts)),'o-',color='black',ms=4,lw=1.3,label='MC loss oracle')
    ax.scatter(selection.C_median,np.arange(len(ts)),s=29,facecolor='white',
               edgecolor='black',lw=.8,label='Median validation',zorder=4)
    ax.set(xscale='log',xlim=(c[0],c[-1]),ylim=(len(ts)-.5,-.5),
           yticks=np.arange(len(ts)),yticklabels=ts,ylabel='Training months T',
           xlabel=r'Population effective complexity $C(\lambda)$')
    if not compact:
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=9.8)
    else:
        ax.legend(loc='upper left',fontsize=8.6,framealpha=.94)
    cb=fig.colorbar(mesh,ax=ax,pad=.018,fraction=.047)
    cb.set_label('Population loss Q (log color scale)',fontsize=10)
    cb.set_ticks([x for x in [.85,1,1.5,2] if z.min()<=x<=z.max()])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))


def selection_path(ax,selection):
    blue='#245b7d';red='#b94541';t=selection['T'].to_numpy()
    right=ax.twinx()
    handles=[]
    for axis,label,color,marker in [(ax,'C',blue,'o'),(right,'lambda',red,'s')]:
        y=selection[label+'_median'].to_numpy()
        line,=axis.plot(t,y,marker+'-',color=color,lw=1.7,ms=5,
                       label='Selected complexity' if label=='C' else 'Selected regularization')
        axis.fill_between(t,selection[label+'_q25'].to_numpy(),selection[label+'_q75'].to_numpy(),
                          color=color,alpha=.10,lw=0)
        axis.set_yscale('log');axis.tick_params(axis='y',colors=color)
        axis.spines['left' if label=='C' else 'right'].set_color(color)
        handles.append(line)
    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
    ax.set_xscale('log');ax.set_xlim(t.min()*.92,t.max()*1.08)
    ticks=[60,120,240,360,720,1440];ax.set_xticks(ticks);ax.set_xticklabels(ticks)
    ax.set_xlabel('Training months T')
    ax.set_ylabel(r'Selected complexity $C(\widehat{\lambda}_T)$',color=blue)
    right.set_ylabel(r'Selected regularization $\widehat{\lambda}_T$',color=red)
    ax.grid(which='major',color='#dce2e8',lw=.6)
    return handles


def main(output='outputs'):
    out=Path(output);sim=out/'simulations';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    surface_path=sim/'characteristic_factor_performance_surface.parquet'
    selection_file=sim/'characteristic_factor_performance_selection.csv'
    audit_file=sim/'characteristic_factor_performance_audit.json'
    surface=pd.read_parquet(surface_path);selection=pd.read_csv(selection_file).sort_values('T')
    audit=json.loads(audit_file.read_text())
    assert audit['all_replicates_matched_existing_ridge_risk'] and audit['population_variance_positive']
    assert surface.R.eq(500).all() and len(surface)==3600
    original=pd.read_parquet(sim/'characteristic_factor_full_surface.parquet')
    original=original[(original.economy=='R1')&(original.b==1.5)&(original.J==2000)]
    matched=surface.merge(original[['T','lambda','mean_regret']],on=['T','lambda'],validate='one_to_one')
    np.testing.assert_allclose(matched.mean_population_loss,.84+matched.mean_regret,rtol=1e-11)
    plt=style();ts=sorted(surface['T'].unique())
    colors=dict(zip(ts,plt.get_cmap('coolwarm')(np.linspace(.05,.95,len(ts)))))
    titles=['Population loss versus portfolio complexity',
            'Population Sharpe versus portfolio complexity',
            'Population loss across sample size and complexity',
            'Validation-selected complexity and regularization']
    footers=[
        'Exact population loss, averaged over 500 fitted paths per T; shaded bands are pointwise 95% Monte Carlo intervals.\n'
        'Dashed line: attainable loss floor Q* = 0.84. Every evaluated penalty is shown; no imposed curve shape.',
        'Mean of the population Sharpe ratios of 500 fitted policies; bands are pointwise 95% Monte Carlo intervals.\n'
        'Annualization uses sqrt(12) times monthly Sharpe, not the Sharpe of an AR-dependent annual sum.\n'
        'Dashed line: attainable population maximum. Vanishing exposure does not imply a zero Sharpe ratio.',
        'Cells display actual mean population loss on the full sampled complexity range; colors use a common log scale.\n'
        'Black: ex-post loss oracle. White: median choice from chronological validation. No interpolation or row normalization.',
        'Points: medians across 500 paths. Shading: interquartile ranges, describing tuning dispersion rather than confidence intervals.\n'
        'Lambda is selected on the last V = min(60, floor(T/3)) observations, then the policy is refitted on all T.\n'
        'Population loss and Sharpe never enter the validation choice.'
    ]
    for i,(stem,title,footer) in enumerate(zip(STEMS,titles,footers)):
        fig,ax=plt.subplots(figsize=(10.8,7.6))
        fig.subplots_adjust(left=.12 if i==3 else .105,right=.80 if i<2 else .84,
                            top=.77,bottom=.21)
        fig.text(.105,.947,title,fontsize=20,weight='bold')
        fig.text(.105,.889,'Boundary target | b = 1.5 | 2,000 directions | 500 independent paths per T',
                 fontsize=12,color='#4e5661')
        if i==0:curves(ax,surface,'mean_population_loss','loss_mcse',colors)
        elif i==1:curves(ax,surface,'mean_population_sharpe','sharpe_mcse',colors,sharpe=True)
        elif i==2:heatmap(ax,fig,surface,selection)
        else:
            handles=selection_path(ax,selection)
            fig.legend(handles,[h.get_label() for h in handles],loc='center',
                       bbox_to_anchor=(.49,.822),ncol=2,frameon=False,fontsize=11)
        fig.text(.105,.045,footer,fontsize=10,color='#4e5661',linespacing=1.5)
        save(fig,figs,stem,pdf=True);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15.5,12.8))
    fig.subplots_adjust(left=.075,right=.91,bottom=.16,top=.82,wspace=.43,hspace=.43)
    fig.text(.075,.963,'Portfolio complexity and performance in the simulated economy',fontsize=22,weight='bold')
    fig.text(.075,.926,'Boundary target | b = 1.5 | J = 2,000 | 500 independent fits per T | Exact population evaluation',
             fontsize=12.5,color='#4e5661')
    handles=[Line2D([],[],color=color,lw=2,label=str(t)) for t,color in colors.items()]
    fig.legend(handles=handles,title='Training months T (panels a-b)',loc='center',
               bbox_to_anchor=(.5,.867),ncol=10,frameon=False,fontsize=10,title_fontsize=11)
    curves(axes[0,0],surface,'mean_population_loss','loss_mcse',colors,legend=False)
    curves(axes[0,1],surface,'mean_population_sharpe','sharpe_mcse',colors,sharpe=True,legend=False)
    heatmap(axes[1,0],fig,surface,selection,compact=True)
    selection_path(axes[1,1],selection)
    for ax,title in zip(axes.flat,['(a) Population loss','(b) Population Sharpe',
                                  '(c) Population loss heatmap','(d) Validation-selected C and lambda']):
        ax.set_title(title,pad=12)
    fig.text(.075,.037,
        'Evaluation uses a fresh stationary population draw: these are simulation results, not historical JKP returns or realized next-year payoffs.\n'
        'Loss is Q = 0.84 + regret. Sharpe is sqrt(12) times the monthly ratio, averaged across policies. Dashed lines: population optima.\n'
        'Curve bands: pointwise 95% Monte Carlo intervals. Panel (d): medians and interquartile ranges; blue is C, red is lambda.\n'
        'Chronological validation selects lambda without population loss or Sharpe. All 360 positive penalties and the full complexity range are retained.',
        fontsize=10.5,color='#4e5661',linespacing=1.5)
    save(fig,figs,OVERVIEW);plt.close(fig)
    (out/'notes'/'simulation_performance_figures.md').write_text(
        '# Four DGP performance figures\n\n'
        'The four PDFs/PNGs reproduce the reference figure roles for the simulated R1 boundary economy. '
        'Training sample size T replaces calendar year; the DGP is stationary. '
        'There are 500 fitted paths per T, nested across T within each independent replication.\n\n'
        'For each fitted coefficient vector beta, m = beta.T S theta*, s2 = beta.T S beta, '
        'Q = 1 - 2m + s2 = 0.84 + regret, and SR = sqrt(12) m / sqrt(s2 - m^2). '
        'The average of these policy-specific ratios is reported, not a ratio of averaged moments. '
        'The sqrt(12) convention does not equal the Sharpe of a twelve-month sum under AR dependence. '
        'Evaluation integrates a fresh stationary population draw; it is not conditional on the last training observation '
        'and is not an observed next-year return sequence.\n\n'
        'The selected-lambda panel uses exactly the existing chronological validation choices. '
        'Population moments do not affect tuning. C and lambda medians and their interquartile ranges '
        'are computed separately from their actual replication-level values. They summarize selection dispersion, '
        'not the parameters of one synthetic median policy. The validation window is capped at 60 observations; '
        'the main simulation note documents its substantial regret relative to the oracle.\n\n'
        'Every recomputed fit is checked against the saved original regret and empirical-complexity arrays. '
        f"The maximum loss identity discrepancy is {audit['max_loss_identity_error']:.3e}. "
        'The population Sharpe bound is sqrt(12*0.16/0.84). '
        'No curve shape, downturn, or zero Sharpe endpoint is imposed. '
        'All 360 positive penalties remain visible; the smallest is near the unregularized finite-rank limit, not exactly zero.\n\n'
        'Reproduce calculation: VECLIB_MAXIMUM_THREADS=1 python3 simulation_performance.py --workers 2.\n'
        'Reproduce figures from public aggregates: VECLIB_MAXIMUM_THREADS=1 python3 render_simulation_performance.py.\n')
    paths=[surface_path,selection_file,audit_file,sim/'characteristic_factor_full_surface.parquet']
    exports=[figs/(stem+suffix) for stem in STEMS for suffix in ['.png','.pdf']]+[figs/(OVERVIEW+'.png')]
    write_json(out/'manifest_simulation_performance.json',dict(source_sha256=digest(__file__),
        inputs={str(p.relative_to(out)):digest(p) for p in paths},
        outputs={str(p.relative_to(out)):digest(p) for p in exports},
        figure_count=4,overview_png=True,full_complexity_range=True,
        population_loss_matches_original_regret=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='outputs')
    args=parser.parse_args();main(args.output)
