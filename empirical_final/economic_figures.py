"""Economic holdings and accounting figures; no ex-post strategy selection."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from empirical_final.economic_holdings import OUT,FAMILIES
from empirical_final.economic_financing import EXTRA,EXTRA_PAIRS

GROUPS=['0-10%','10-50%','50-90%','90-100%']
BLUE,RED,GREY='#195777','#ad4945','#687582'


def read(name):
    f=pd.read_csv(OUT/'tables'/f'{name}.csv')
    extra=OUT/'tables'/f'financing_{name}.csv'
    return pd.concat([f,pd.read_csv(extra)],ignore_index=True) if extra.exists() else f


def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)


def heat(ax,data,limit=None,format='.1f',fontsize=8):
    values=np.asarray(data)
    lim=max(float(np.max(np.abs(values))),1e-8) if limit is None else limit
    im=ax.imshow(values,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value=values[i,j]
            ax.text(j,i,f'{value:{format}}',ha='center',va='center',fontsize=fontsize,
                    color='white' if abs(value)>.7*lim else '#20262b')
    return im


def main():
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
        'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.16,'pdf.fonttype':42,'savefig.facecolor':'white'})
    exposure=read('family_exposures');families=list(FAMILIES)+[EXTRA]
    whole=exposure[exposure.period.eq('all')]
    balance=read('group_holdings_summary').query("period=='all'").sort_values('group')
    fig=plt.figure(figsize=(7,6.4),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.25,1])
    ax=fig.add_subplot(gs[0,:]);matrix=whole.pivot(index='group',columns='family',values='long_minus_short')[families]*100
    im=heat(ax,matrix,fontsize=7)
    ax.set_xticks(range(len(families)),families,rotation=40,ha='right');ax.set_yticks(range(4),GROUPS)
    ax.set_title('A. Long-minus-short characteristic ranks')
    fig.colorbar(im,ax=ax,shrink=.8,label='Average input percentile-point difference')
    ax=fig.add_subplot(gs[1,0]);x=np.arange(4)
    ax.bar(x-.17,100*balance.mean_long,width=.34,color=BLUE,label='Long')
    ax.bar(x+.17,-100*balance.mean_short,width=.34,color=RED,label='Short')
    ax.axhline(0,color=GREY,lw=.7);ax.set_xticks(x,GROUPS)
    ax.set(ylabel='Average signed notional (% of NAV)',xlabel='Spectral rank group',title='B. Actual component holdings')
    ax.legend(fontsize=8)
    ax=fig.add_subplot(gs[1,1])
    ax.bar(x-.17,100*balance.annual_long_payoff,width=.34,color=BLUE,label='Long leg')
    ax.bar(x+.17,100*balance.annual_short_payoff,width=.34,color=RED,label='Short leg')
    ax.scatter(x,100*balance.annual_mean,marker='D',color='#151e26',label='Combined',zorder=5)
    ax.axhline(0,color=GREY,lw=.7);ax.set_xticks(x,GROUPS)
    ax.set(ylabel='Annual excess payoff (%)',xlabel='Spectral rank group',title='C. Long and short payoffs')
    ax.legend(fontsize=8)
    fig.suptitle('Economic strategies in the actual ridge-selected holdings\nFebruary 1978-January 2025 | positive means more of the attribute',fontsize=11)
    save(fig,'EC1_holdings_and_characteristics')

    period=exposure[~exposure.period.eq('all')];limit=float(np.max(np.abs(period.long_minus_short))*100)
    stab=read('economic_stability')
    fig,axes=plt.subplots(2,2,figsize=(7,8.5),layout='constrained')
    for j,ax in enumerate(axes.flat,1):
        d=period[period.group.eq(j)].pivot(index='period',columns='family',values='long_minus_short')[families].T*100
        im=heat(ax,d,limit=limit,fontsize=7)
        ax.set_xticks(range(len(d.columns)),d.columns,rotation=65,ha='right',fontsize=8)
        ax.set_yticks(range(len(d)),d.index,fontsize=8)
        cosine=stab[(stab.group==j)&(stab.period_a=='1978-1989')&(stab.period_b=='2020-2024')].cosine.iloc[0]
        ax.set_title(f'Ranks {GROUPS[j-1]} | core-style cosine {cosine:.2f}')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Long-minus-short percentile-point difference')
    fig.suptitle('Economic strategies across decades\nCommon scale; cosine compares first and last core-family profiles',fontsize=11)
    save(fig,'EC2_economic_stability')

    profiles=read('joint_weight_profiles').query("period=='all' and group==2")
    attributes=read('cell_return_attribution').query("period=='all' and group==2")
    pairs=[('Value','Profitability'),*EXTRA_PAIRS,('Momentum','Recent return')]
    pair_names=[a+' x '+b for a,b in pairs]
    joint=attributes[attributes.partition.isin(pair_names)]
    profiles=profiles[profiles.partition.isin(pair_names)]
    fig,axes=plt.subplots(4,3,figsize=(7,8.5),layout='constrained')
    limits=[max(abs(profiles.weight_density)),max(abs(profiles.nonadditive_density)),max(abs(joint.annual_mean*100))]
    for j,(a,b) in enumerate(pairs):
        name=a+' x '+b;p=profiles[profiles.partition==name].sort_values('cell');r=joint[joint.partition==name].sort_values('cell')
        for i,values in enumerate([p.weight_density,p.nonadditive_density,r.annual_mean*100]):
            ax=axes[j,i]
            heat(ax,np.asarray(values).reshape(3,3),limit=limits[i],format='.2f',fontsize=8)
            ax.set_xticks(range(3),['Low','Mid','High'],fontsize=8)
            ax.set_yticks(range(3),['Low','Mid','High'],fontsize=8)
            ax.set_xlabel(b,fontsize=8);ax.set_ylabel(a if i==0 else '',fontsize=8)
            if j==0:ax.set_title(['Signed weight density','Nonadditive component','Annual payoff (%)'][i],fontsize=8)
    fig.suptitle('Ranks 10-50%: joint characteristic profiles\nAll cells retained; each column uses a common color scale.',fontsize=11)
    save(fig,'EC3_group2_joint_strategies')

    one=attributes[~attributes.partition.str.contains(' x ')]
    fig,axes=plt.subplots(3,1,figsize=(7,8),sharex=True,layout='constrained')
    colors=['#286883','#9aa4ab','#a7433e'];x=np.arange(len(families));width=.24
    for cell in range(3):
        d=one[one.cell.eq(cell)].set_index('partition').loc[families]
        for i,(name,scale,label) in enumerate([
            ('annual_mean',100,'Annual excess payoff (%)'),
            ('allocated_incremental_loss',1,'Response-one loss change'),
            ('allocated_incremental_variance',10000,'Annual variance change (pp squared)')]):
            values=d[name]*scale
            axes[i].bar(x+(cell-1)*width,values,width=width,color=colors[cell],label=['Low tercile','Middle tercile','High tercile'][cell])
            if i<2:
                low,high=('mean_low','mean_high') if i==0 else ('loss_low','loss_high')
                axes[i].errorbar(x+(cell-1)*width,values,yerr=np.vstack([values-d[low]*scale,d[high]*scale-values]),
                                fmt='none',ecolor='#333333',lw=.7,capsize=1.5)
            axes[i].set_ylabel(label,fontsize=8)
    for ax in axes:ax.axhline(0,color=GREY,lw=.7)
    axes[0].legend(ncol=3,fontsize=8)
    axes[-1].set_xticks(x,families,rotation=40,ha='right')
    axes[0].set_title('A. Sources of mean payoff',loc='left')
    axes[1].set_title('B. Incremental loss beyond ranks 0-10% (negative improves)',loc='left')
    axes[2].set_title('C. Variance allocation including cross-covariances',loc='left')
    fig.suptitle('Ranks 10-50%: incremental financial value\nEach family is a separate view: do not add across families.\nPointwise paired 95% block intervals.',fontsize=11)
    save(fig,'EC4_group2_incremental_attribution')


if __name__=='__main__':main()
