"""Five main figures, plus conditional neural and sensitivity diagnostics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from empirical_final.compact_common import OUT, setup, save, PROTOCOL

COLORS={'linear':'#5e6268','gaussian':'#236d9f','matern32':'#b46a36'}
LABELS={'linear':'Linear','gaussian':'Gaussian','matern32':'Matérn-3/2'}


def read(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')


def finish(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=240,bbox_inches='tight')
    plt.close(fig)


def main():
    setup()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white','grid.alpha':.3})
    accounts=read('monthly_accounting')
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained',sharey=True)
    wealth_rows=[]
    for ax,sc,title in zip(axs,['gross','trade25_borrow30'],['Gross','25 bps trades + 30 bps/year short fee']):
        for policy in COLORS:
            m=accounts[(accounts.policy==policy)&(accounts.scenario==sc)].sort_values('return_date')
            dates=pd.to_datetime(m.return_date);w=np.cumprod(1+m.total_return)
            ax.plot(dates,w,color=COLORS[policy],label=LABELS[policy],lw=1.8)
            wealth_rows.append(pd.DataFrame(dict(policy=policy,scenario=sc,return_date=dates,total_wealth=w)))
        ax.set_yscale('log');ax.set_title(title,fontsize=11);ax.grid(True)
        ax.set_xlabel('Realization date')
    axs[0].set_ylabel('Total wealth, initial NAV = 1 (log scale)');axs[0].legend(frameon=False,fontsize=11)
    finish(fig,'fig01_kernel_wealth');save('figure01_wealth',pd.concat(wealth_rows))

    s=read('managed_spectra').query("decision_year==2024 and policy in ['gaussian','matern32']").copy()
    s=s[s.eigenvalue> s.groupby('policy').eigenvalue.transform('max')*1e-12]
    s['normalized_eigenvalue']=s.eigenvalue/s.groupby('policy').eigenvalue.transform('sum')
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.1),layout='constrained')
    for p in ['gaussian','matern32']:
        f=s[s.policy.eq(p)]
        for ax,col in zip(axs,['eigenvalue','normalized_eigenvalue']):
            ax.loglog(f['rank'],f[col],lw=1.8,color=COLORS[p],label=LABELS[p]);ax.grid(True);ax.set_xlabel('Positive spectral rank')
    axs[0].set_ylabel('Uncentered payoff second moment');axs[1].set_ylabel('Share of total second-moment mass')
    axs[0].legend(frameon=False,fontsize=11);axs[0].set_title('Raw representation units');axs[1].set_title('Mass-normalized diagnostic')
    finish(fig,'fig02_managed_spectrum');save('figure02_spectrum',s)

    curve=read('single_window_complexity').sort_values('C')
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
    axs[0].plot(curve.C,curve.historical_sharpe,color='#70777e',lw=2)
    axs[1].plot(curve.C,curve.oos_sharpe,color=COLORS['gaussian'],lw=2)
    peak=curve.loc[curve.oos_sharpe.idxmax()];selected=curve[curve.selected].iloc[0]
    axs[1].scatter(peak.C,peak.oos_sharpe,color='#b46a36',marker='*',s=150,label='Retrospective grid maximum',zorder=5)
    axs[1].scatter(selected.C,selected.oos_sharpe,color='black',marker='D',s=50,label='Prior-validation selection',zorder=5)
    axs[0].set_ylabel('Historical Sharpe');axs[1].set_ylabel('Subsequent Sharpe');axs[1].set_xlabel('Effective complexity, fixed representation')
    axs[0].set_title('January 2024 refit: history through January 2024')
    axs[1].set_title('February 2024–January 2025: 12 subsequent months');axs[1].legend(frameon=False,fontsize=11)
    for ax in axs:ax.grid(True)
    finish(fig,'fig03_gaussian_complexity')

    curves=read('nine_panel_curves');summary=read('nine_panel_summary').set_index('period')
    plt.rcParams.update({'font.size':14})
    fig,axs=plt.subplots(3,3,figsize=(11,11),sharex=True,sharey=True,layout='constrained')
    for ax,(lo,hi) in zip(axs.flat,PROTOCOL['blocks']):
        label=f'{lo}-{hi}';f=curves[curves.period.eq(label)].sort_values('mean_C');s=summary.loc[label]
        ax.fill_between(f.mean_C,f.sr_low,f.sr_high,color=COLORS['gaussian'],alpha=.16,lw=0)
        ax.plot(f.mean_C,f.oos_sharpe,color=COLORS['gaussian'],lw=2)
        peak=f[f.retrospective_maximum].iloc[0]
        ax.scatter(peak.mean_C,peak.oos_sharpe,color='#b46a36',marker='*',s=140,zorder=5)
        ax.scatter(s.selected_mean_C,s.selected_sharpe,color='black',marker='D',s=45,zorder=5)
        ax.set_title(f'{lo}–{hi} · n = {int(s.months)}',fontsize=16,pad=9)
        ax.grid(True);ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=13)
    for ax in axs[2]:ax.set_xlabel('Mean effective complexity',fontsize=13)
    for ax in axs[:,0]:ax.set_ylabel('OOS Sharpe',fontsize=13)
    # Same scales, without extending any curve beyond its observed support.
    axs[0,0].set_xlim(0,curves.mean_C.max()*1.03)
    axs[0,0].set_ylim(curves.sr_low.min()-.1,curves.sr_high.max()+.1)
    handles=[plt.Line2D([0],[0],color=COLORS['gaussian'],lw=2,label='Fixed-λ path'),
             plt.Line2D([0],[0],color='#b46a36',marker='*',lw=0,ms=12,label='Retrospective maximum'),
             plt.Line2D([0],[0],color='black',marker='D',lw=0,ms=7,label='Annual selected policy')]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,fontsize=13)
    finish(fig,'fig04_gaussian_nine_panels')

    e=read('economic_exposure_summary')
    names=['Value','Profitability','Investment','Momentum','Recent return','Size','Liquidity','Issuance','Debt financing']
    fig,axs=plt.subplots(1,2,figsize=(10,5.1),layout='constrained',sharey=True)
    for ax,col,title in zip(axs,['long_minus_short','signed_notional'],['Long-minus-short rank profile','Signed notional rank exposure']):
        values=e.pivot(index='family',columns='group',values=col).loc[names].to_numpy()
        v=np.abs(values).max();im=ax.imshow(values,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto')
        ax.set_xticks(range(4),['0–10%','10–50%','50–90%','90–100%'],fontsize=10)
        ax.set_yticks(range(len(names)),names,fontsize=12);ax.set_title(title,fontsize=12)
        ax.set_xlabel('Positive-rank group',fontsize=11)
        for i in range(len(names)):
            for j in range(4):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=11,
                color='white' if abs(values[i,j])>.6*v else '#202020')
        fig.colorbar(im,ax=ax,shrink=.8,pad=.03)
    finish(fig,'fig05_economic_holdings')
    plt.rcParams.update({'font.size':11})
    if (OUT/'tables/neural_spectra_seed_0.csv').exists():
        n=read('neural_spectra_seed_0').query('decision_year==2024')
        fig,axs=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
        for ax,col,title in zip(axs,['eigenvalue','normalized_eigenvalue'],['Raw hidden-feature units','Mass-normalized diagnostic']):
            ax.semilogy(n['rank'],n[col],color='#607b3d',lw=2);ax.set_xlabel('Frozen readout rank');ax.set_title(title);ax.grid(True)
            ax.set_ylabel(col.replace('_',' '))
        finish(fig,'appendix_neural_spectrum')
        logs=read('neural_optimization_seed_0')
        fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
        for year in [1978,2000,2024]:
            f=logs[(logs.decision_year==year)&(logs.stage.astype(str)=='refit')]
            ax.plot(f.step,f.objective,'o-',label=str(year))
        ax.set_xlabel('Hidden-layer optimization step');ax.set_ylabel('Full-history penalized training criterion');ax.legend(frameon=False);ax.grid(True)
        finish(fig,'appendix_neural_optimization')
    print('Five main PDF/PNG figures rendered',flush=True)


if __name__=='__main__':main()
