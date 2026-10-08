"""Render aggregate real-data results into a new, immutable publication revision."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/final_empirical_20261008'
COLORS={'linear':'#61666D','gaussian':'#176B9A','matern32':'#BB542D'}
LABELS={'linear':'Linear','gaussian':'Gaussian','matern32':r'Matérn-3/2'}

def render(out=DEFAULT,revision='v1'):
    out=Path(out).resolve()
    pub=out/'publication'/revision
    pub.mkdir(parents=True,exist_ok=False)
    figs=pub/'figures';figs.mkdir()
    inputs={}
    def read(name):
        path=out/'tables'/name
        inputs[str(path.relative_to(out))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    monthly=read('monthly_accounting.csv');perf=read('performance_scenarios.csv')
    annual=read('annual_complexity.csv');spectra=read('managed_spectra.csv')
    nested=read('spectral_contributions.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#30343A',
        'grid.color':'#DDE1E5','grid.linewidth':.6,'pdf.fonttype':42,'savefig.facecolor':'white'})
    def finish(fig,name):
        fig.savefig(figs/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(figs/(name+'.png'),dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.2),sharex=True,sharey='row',layout='constrained')
    for col,(scenario,title) in enumerate([('gross','Gross'),('trade25_borrow30','Cost sensitivity: 25 bps trades + 30 bps/year shorts')]):
        for kernel in COLORS:
            f=monthly[(monthly.kernel==kernel)&(monthly.scenario==scenario)]
            dates=pd.to_datetime(f.return_date)
            dates=pd.DatetimeIndex([pd.Timestamp('1978-01-31'),*dates])
            wealth=np.r_[1,np.cumprod(1+f.total_return.to_numpy())]
            dd=wealth/np.maximum.accumulate(wealth)-1
            axes[0,col].plot(dates,wealth,color=COLORS[kernel],label=LABELS[kernel],lw=1.5)
            axes[1,col].plot(dates,dd,color=COLORS[kernel],lw=1.1)
        axes[0,col].set_title(title,fontsize=10)
        axes[0,col].set_yscale('log');axes[1,col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1,col].set_xlabel('Payoff date')
    axes[0,0].set_ylabel('Total wealth (initial NAV = 1; log scale)')
    axes[1,0].set_ylabel('Drawdown from running peak')
    axes[0,0].legend(loc='upper left',frameon=False)
    for ax in axes.flat:ax.grid(True,alpha=.7)
    finish(fig,'E1_wealth_drawdown')

    fig,axes=plt.subplots(1,3,figsize=(10,3.5),sharey=True,layout='constrained')
    for ax,year in zip(axes,[1978,2000,2024]):
        for kernel in ('gaussian','matern32'):
            f=spectra[(spectra.kernel==kernel)&(spectra.decision_year==year)]
            f=f[f.eigenvalue>f.eigenvalue.max()*1e-12]
            ax.loglog(f['rank'],f.eigenvalue,color=COLORS[kernel],label=LABELS[kernel],lw=1.6)
        ax.set_title(f'January {year}');ax.set_xlabel('Managed-payoff eigenvalue rank');ax.grid(True,alpha=.6)
    axes[0].set_ylabel(r'Eigenvalue of $G^\prime G/T$ (uncentered)')
    axes[-1].legend(frameon=False,loc='upper right')
    finish(fig,'E2_managed_spectra')

    fig,axes=plt.subplots(1,2,figsize=(9.5,3.7),layout='constrained')
    for kernel in ('gaussian','matern32'):
        f=nested[nested.kernel==kernel].sort_values('rank_fraction')
        axes[0].plot(range(4),f.raw_oos_loss,'o-',color=COLORS[kernel],label=LABELS[kernel])
        # First-stage improvement is relative to the null portfolio (loss 1),
        # omitted here so that later incremental changes remain legible.
        offset=-.14 if kernel=='gaussian' else .14
        axes[1].bar(np.arange(1,4)+offset,f.incremental_loss_change.iloc[1:],width=.27,color=COLORS[kernel],label=LABELS[kernel])
    axes[0].set_xticks(range(4),['10%','50%','90%','Full']);axes[0].set_ylabel('OOS response-one loss (raw payoff)')
    axes[0].set_xlabel('Cumulative fraction of positive training rank')
    axes[1].set_xticks([1,2,3],['10% → 50%','50% → 90%','90% → Full'])
    axes[1].set_ylabel('Change in OOS loss (negative = improvement)')
    axes[1].set_xlabel('Additional spectral directions');axes[1].axhline(0,color='#404040',lw=.8)
    axes[0].legend(frameon=False);axes[0].set_title('Nested policies');axes[1].set_title('Incremental value, including cross terms')
    for ax in axes:ax.grid(axis='y',alpha=.6);ax.set_axisbelow(True)
    finish(fig,'E2_nested_value')

    fig,axes=plt.subplots(2,1,figsize=(9.3,6),sharex=True,layout='constrained')
    for kernel in ('gaussian','matern32'):
        f=annual[annual.kernel==kernel]
        for ax,col in zip(axes,['C_over_T','C']):
            ax.plot(f.decision_year,f[col],'-o',ms=2.5,lw=1.3,color=COLORS[kernel],label=LABELS[kernel])
    axes[0].set_ylabel('Relative effective complexity C / T');axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel('Absolute effective complexity C');axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel('January decision year');axes[0].legend(frameon=False)
    for ax in axes:ax.grid(True,alpha=.6)
    finish(fig,'E3_measured_complexity')

    fig,axes=plt.subplots(1,2,figsize=(9.3,3.5),layout='constrained')
    for kernel in COLORS:
        f=annual[annual.kernel==kernel]
        axes[0].plot(f.decision_year,f.descriptive_b,color=COLORS[kernel],label=LABELS[kernel])
        axes[1].plot(f.decision_year,f.top10_eigenvalue_share,color=COLORS[kernel])
    axes[0].set_ylabel('Descriptive decay slope b');axes[1].set_ylabel('Eigenvalue mass share\nin the top 10 directions')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(frameon=False)
    for ax in axes:ax.set_xlabel('January decision year');ax.grid(True,alpha=.6)
    finish(fig,'A_spectral_diagnostics')

    rows=[('Gross annual excess return (\\%)','gross','annual_excess_return',100),
          ('Gross annual volatility (\\%)','gross','annual_volatility',100),
          ('Gross Sharpe','gross','sharpe',1),
          ('Sharpe: trading costs only','trade25','sharpe',1),
          ('Sharpe: trading + borrowing costs','trade25_borrow30','sharpe',1),
          ('Cost-sensitivity annual excess return (\\%)','trade25_borrow30','annual_excess_return',100),
          ('Cost-sensitivity annual volatility (\\%)','trade25_borrow30','annual_volatility',100),
          ('Gross maximum drawdown (\\%)','gross','maximum_drawdown',100),
          ('Cost-sensitivity maximum drawdown (\\%)','trade25_borrow30','maximum_drawdown',100),
          ('Annual traded notional / NAV, gross path','gross','annualized_turnover_two_sided',1),
          ('Annual traded notional / NAV, cost path','trade25_borrow30','annualized_turnover_two_sided',1),
          ('Mean gross exposure / NAV','gross','mean_gross_exposure',1),
          ('Mean net exposure / NAV','gross','mean_net_exposure',1),
          ('Mean short notional / NAV','gross','mean_short_notional',1),
          ('Annual trading fees (\\% of NAV)','trade25_borrow30','annual_trading_fees',100),
          ('Annual borrowing fees (\\% of NAV)','trade25_borrow30','annual_borrowing_fees',100)]
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r' & Linear & Gaussian & Mat\'ern-3/2 \\',r'\midrule']
    for label,scenario,field,scale in rows:
        vals=[float(perf[(perf.kernel==k)&(perf.scenario==scenario)].iloc[0][field])*scale for k in COLORS]
        lines.append(label+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (pub/'performance_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Policy & Rank fraction & OOS loss & $\Delta$ loss & Sharpe & $\Delta$ Sharpe \\',r'\midrule']
    for kernel in COLORS:
        for _,r in nested[nested.kernel==kernel].iterrows():
            label={'linear':'Linear','gaussian':'Gaussian','matern32':r"Mat\'ern-3/2"}[kernel]
            frac='Full' if r.rank_fraction==1 else f'{r.rank_fraction:.0%}'.replace('%',r'\%')
            delta='---' if r.rank_fraction==.1 else f'{r.sharpe-r.previous_sharpe:+.3f}'
            lines.append(f'{label} & {frac} & {r.raw_oos_loss:.5f} & {r.incremental_loss_change:+.5f} & {r.sharpe:.3f} & {delta}'+r' \\')
        lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (pub/'spectral_table.tex').write_text('\n'.join(lines)+'\n')
    templates=ROOT/'empirical_final/templates'
    for path in sorted(templates.glob('*.tex')):
        (pub/path.name).write_text(path.read_text())
    main=(ROOT/'paper/main.tex').read_text()
    main='\n'.join(line for line in main.splitlines() if r'\simoutput' not in line)+'\n'
    rel=Path(__import__('os').path.relpath(pub,ROOT/'paper')).as_posix()
    main=main.replace(r'\begin{document}',rf'\newcommand{{\empout}}{{{rel}}}'+'\n'+r'\begin{document}')
    main=main.replace(r'\input{empirics}',r'\input{\empout/empirical_section.tex}')
    main=main.replace(r'\input{complexity_schedule/empirics_schedule}','')
    main=main.replace(r'\end{document}',r'\input{\empout/empirical_appendix.tex}'+'\n'+r'\end{document}')
    (pub/'manuscript_real_data.tex').write_text(main)
    manifest={'inputs':inputs,'revision':revision,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'templates':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in templates.glob('*.tex')},
              'generated':{str(p.relative_to(pub)):hashlib.sha256(p.read_bytes()).hexdigest() for p in pub.rglob('*') if p.is_file()}}
    (pub/'render_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pub)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--revision',default='v1')
    args=parser.parse_args();render(args.out,args.revision)
