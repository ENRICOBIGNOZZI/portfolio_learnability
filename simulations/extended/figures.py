"""Ten publication figures from verified 300-replication distributions only."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogLocator,LogFormatterSciNotation,NullFormatter

from simulations.extended.design import OUTPUT,ENVIRONMENTS,parameters
from simulations.extended.statistics import distribution

COLORS=['#244b73','#b55f39','#3e806b','#80629b','#ae8a2e']
LABELS={'baseline':r'$N=600$','N300':r'$N=300$','N1200':r'$N=1200$',
        'rho000':r'$\rho=0$','rho075':r'$\rho=0.75$'}
N_NAMES=['N300','baseline','N1200']
RHO_NAMES=['rho000','rho075','baseline']


class Figures:
    def __init__(self):
        self.protocol=json.loads((OUTPUT/'protocol.json').read_text())
        verified=json.loads((OUTPUT/'summary_verification.json').read_text())
        if verified['replications_per_environment']!=300 or verified['run_hash']!=self.protocol['run_hash']:
            raise ValueError('Final figures require verified complete production.')
        self.data={};self.pop={}
        for name in ENVIRONMENTS:
            with np.load(OUTPUT/'distributions'/f'{name}.npz') as z:
                self.data[name]={k:z[k] for k in z.files}
                assert int(z['replications'])==300
            with np.load(OUTPUT/'population'/f'{name}_theory.npz') as z:
                self.pop[name]={k:z[k] for k in z.files}
        final_resolution=OUTPUT/'production_resolution_final.csv'
        self.resolution=pd.read_csv(final_resolution if final_resolution.exists() else OUTPUT/'production_resolution.csv')
        self.ranks=pd.read_csv(OUTPUT/'rank_audit.csv')
        self.folder=OUTPUT/'figures';self.folder.mkdir(exist_ok=True)
        self.captions=[]
        self.set_style()

    @staticmethod
    def set_style():
        plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
            'mathtext.fontset':'dejavuserif','font.size':9,'axes.labelsize':10,
            'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
            'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
            'lines.linewidth':1.6,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42})

    def save(self,fig,name,caption):
        fig.savefig(self.folder/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(self.folder/(name+'.png'),bbox_inches='tight',dpi=300)
        plt.close(fig)
        self.captions.append((name,caption))

    def palette(self,names):
        return [COLORS[0] if name=='baseline' else
                COLORS[1] if name in ('N300','rho075') else COLORS[2] for name in names]

    def axes(self,n):
        fig,axes=plt.subplots(1,n,figsize=(3.5*n,3.0),layout='constrained')
        axes=np.atleast_1d(axes)
        for ax in axes:
            ax.grid(True,which='major',color='#dddddd',lw=.45,alpha=.6)
            ax.set_axisbelow(True)
        return fig,axes

    def label(self,name,kind):
        if kind=='rho' and name=='baseline':
            return r'$\rho=0.95$'
        return LABELS[name]

    def distribution_line(self,ax,x,y,color,label=None,marker=True,logy=False):
        s=distribution(y)
        ax.plot(x,s['mean'],color=color,label=label,marker='o' if marker else None,ms=3)
        ax.fill_between(x,s['p025'],s['p975'],color=color,alpha=.11,lw=0)
        if logy:
            ax.set_yscale('log')
        return s

    def time_axis(self,ax):
        ax.set_xscale('log');ax.set_xlabel(r'Training history $T$')
        maximum=max(float(np.max(line.get_xdata())) for line in ax.lines)
        ticks=[t for t in [60,240,720] if t<=maximum]
        if maximum>=4860:
            ticks.append(2160)
        if maximum>720:
            ticks.append(maximum)
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())

    def shared_legend(self,fig,axes,names):
        handles,labels=axes[0].get_legend_handles_labels()
        fig.set_size_inches(7.0,3.5)
        fig.legend(handles,labels,loc='outside lower center',frameon=False,
                   ncol=3 if len(handles)>4 else len(handles))

    def quantitative_log_ticks(self,ax):
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10,labelOnlyBase=False,
            minor_thresholds=(np.inf,np.inf)))
        ax.yaxis.set_minor_formatter(NullFormatter())

    def unresolved_time(self,ax,names):
        rows=self.resolution[self.resolution.environment.isin(names)]
        failed=rows[~(rows.floor_pass & rows.paired_pass)]
        if len(failed):
            start=float(failed['T'].min())
            stop=(max(float(self.data[n]['T'].max()) for n in names) if len(names)==1
                  else float(max(self.protocol['robustness_T'])))
            ax.axvspan(start,stop*1.06,color='#777777',alpha=.07,zorder=0)

    def spectrum(self):
        fig,axes=self.axes(3)
        values=self.pop['baseline']['eigenvalues'][::-1]
        j=np.arange(1,len(values)+1)
        axes[0].loglog(j,values,color=COLORS[0],label='Managed-payoff spectrum')
        axes[0].loglog(j,values[63]*(j/64.)**(-1.5),'--',color='#777777',label=r'$j^{-1.5}$ order reference')
        axes[0].set(xlabel='Eigenvalue rank',ylabel=r'$\mu_j$',title='(a) Economic spectrum')
        for color,T in zip(COLORS,[60,240,720,1440,self.protocol['baseline_T'][-1]]):
            lam=self.protocol['a']['baseline']*float(T)**(-.6)
            h=values/(values+lam)
            axes[1].semilogx(j,h,color=color,label=fr'$T={T}$')
            axes[2].semilogx(j,np.cumsum(h),color=color,label=fr'$T={T}$')
        axes[1].set(xlabel='Eigenvalue rank',ylabel=r'$\mu_j/(\mu_j+\lambda_T)$',title='(b) Shrinkage filters')
        axes[2].set(xlabel='Eigenvalue rank',ylabel='Cumulative effective complexity',title='(c) Complexity accumulation')
        axes[0].legend(frameon=False);axes[1].legend(frameon=False,ncol=1)
        self.save(fig,'Figure0_economic_spectrum',
            'Population managed-payoff eigenvalues, shrinkage filters and cumulative complexity use one fixed independently evaluated spectrum. The polynomial reference is anchored at rank 64 and is an order comparison, not a fitted spectral law. Curves are unsmoothed. Basis/quadrature and component-operator comparisons are in the spectral audit tables; the finite rank cannot establish an infinite-dimensional tail exponent.')

    def learnability(self,names,kind,filename):
        fig,axes=self.axes(2)
        for color,name in zip(self.palette(names),names):
            d=self.data[name]
            keep=np.ones(len(d['T']),dtype=bool) if len(names)==1 else np.isin(d['T'],self.protocol['robustness_T'])
            T=d['T'][keep];label=self.label(name,kind) if len(names)>1 else 'Theory-scaled policy'
            sr=self.distribution_line(axes[0],T,d['annual_SR'][:,keep,-1],color,label)
            gap=self.distribution_line(axes[1],T,d['annual_gap'][:,keep,-1],color,label,logy=True)
            optimum=parameters(name).sr_star*np.sqrt(12)
            if kind!='rho' or name==names[0]:
                axes[0].axhline(optimum,color=color if kind!='rho' else '#555555',ls=':',lw=1,
                               label=(fr'$SR^\star={optimum:.3f}$' if kind!='rho' else r'Common $SR^\star=3$'))
            if len(names)==1:
                axes[1].plot(T,gap['mean'][0]*(T/T[0])**(-.6),'--',color='#666666',label=r'$T^{-0.6}$ reference')
        for ax in axes:
            self.time_axis(ax);self.unresolved_time(ax,names)
            if len(names)==1:
                ax.legend(frameon=False)
        if len(names)>1:
            self.shared_legend(fig,axes,names)
        axes[0].set(ylabel='Annualized population Sharpe',title='(a) Sharpe recovery')
        axes[1].set(ylabel=r'Annualized gap $SR^\star-SR(\widehat W)$',title='(b) Gap to the economic optimum')
        self.save(fig,filename,
            'Means and central 95% replication-percentile bands across 300 independent economic paths. Histories are nested across T and innovations are paired across environments. References are analytical economic optima. Grey high-T regions fail at least one checked 5% regret-based numerical condition in at least one displayed economy. A highest-rank backward comparison is not a higher-rank certificate. Slope estimates for every predeclared window and their full-covariance Monte Carlo uncertainty are reported separately.')

    def main_complexity(self):
        fig,axes=self.axes(3)
        p=self.pop['baseline'];T=p['T'];C=p['complexity']
        axes[0].loglog(T,p['penalties'],color=COLORS[0],label=r'$aT^{-0.6}$')
        axes[1].loglog(T,C,color=COLORS[0],label='Population complexity')
        axes[1].loglog(T,C[0]*(T/T[0])**.4,'--',color='#777777',label=r'$T^{0.4}$ reference')
        axes[2].loglog(T,C/T,color=COLORS[0],label='Population relative complexity')
        axes[2].loglog(T,(C[0]/T[0])*(T/T[0])**(-.6),'--',color='#777777',label=r'$T^{-0.6}$ reference')
        for k,(title,y) in enumerate([('(a) Prescribed regularization',r'$\lambda_T$'),
             ('(b) Effective complexity',r'$\mathcal{C}(\lambda_T)$'),('(c) Relative complexity',r'$\mathcal{C}(\lambda_T)/T$')]):
            axes[k].set(title=title,ylabel=y);self.time_axis(axes[k]);axes[k].legend(frameon=False)
            self.quantitative_log_ticks(axes[k])
        self.save(fig,'Figure2_complexity',
            'Prescribed penalties and deterministic population complexity along the fixed theory-scaled path. There are no Monte Carlo bands or point markers on deterministic curves. Power references are anchored at the first horizon. Exact local logarithmic elasticities, finite-difference verification, all predeclared window slopes and rank/seed sensitivity are tabulated; a window regression slope and a local elasticity are distinct quantities.')

    def robustness_complexity(self,names,kind,filename):
        fig,axes=self.axes(2)
        for color,name in zip(self.palette(names),names):
            d=self.data[name];keep=np.isin(d['T'],self.protocol['robustness_T'])
            T=d['T'][keep];p=self.pop[name];label=self.label(name,kind)
            for k,metric in enumerate(['empirical_complexity','relative_complexity']):
                self.distribution_line(axes[k],T,d[metric][:,keep,-1],color,label,logy=True)
                if kind!='rho' or name==names[0]:
                    y=p['complexity'][keep]/(T if k else 1)
                    axes[k].plot(T,y,ls='--',color=color if kind!='rho' else '#222222',lw=1.1,
                        label=('Population '+label if kind!='rho' else 'Common population reference'))
        for ax in axes:
            self.time_axis(ax);self.quantitative_log_ticks(ax)
        self.shared_legend(fig,axes,names)
        axes[0].set(title='(a) Empirical effective complexity',ylabel=r'$\widehat{\mathcal{C}}_T(\lambda_T)$')
        axes[1].set(title='(b) Relative empirical complexity',ylabel=r'$\widehat{\mathcal{C}}_T(\lambda_T)/T$')
        self.save(fig,filename,
            'Empirical complexity means and central 95% replication-percentile bands use 300 independent paths per economy. Dashed curves are deterministic population quantities. The rho comparison shows a single common population reference because stationary marginal operators coincide exactly. Population elasticities are reported separately from empirical variation; no effective-sample-size correction is imposed.')

    def path(self,ax,name,T,color,label,band=True):
        d=self.data[name];p=self.pop[name]
        t=int(np.flatnonzero(d['T']==T)[0]);x=p['diagnostic_grid_complexity']
        s=distribution(d['annual_SR'][:,t,:-1]);order=np.argsort(x)
        ax.plot(x[order],s['mean'][order],color=color,label=label)
        if band:
            ax.fill_between(x[order],s['p025'][order],s['p975'][order],color=color,alpha=.075,lw=0)
        ax.scatter(p['complexity'][t],d['annual_SR'][:,t,-1].mean(),marker='*',s=62,color=color,
                   edgecolors='white',linewidths=.4,zorder=5)
        audit=self.ranks[(self.ranks.environment==name)&(self.ranks['T']==T)&
                         (self.ranks.higher_rank==self.protocol['rank'])&(~self.ranks.exact_theory_choice)]
        if len(audit):
            audit=audit.sort_values('penalty_index')
            unresolved=(audit.higher_projection_floor.to_numpy()>.05*audit.mean_regret_higher.to_numpy()) | (audit.paired_upper_bound.to_numpy()>.05*audit.mean_regret_higher.to_numpy())
            # Dotted grey overprint identifies diagnostic penalty regions that
            # fail the same checked numerical conditions; no smoothing.
            ax.plot(x,np.where(unresolved,s['mean'],np.nan),':',color='#222222',lw=1.1)
        ax.set_xscale('log');ax.set_xlabel(r'Population complexity $\mathcal{C}(\lambda)$')
        ax.set_ylabel('Annualized population Sharpe')

    def main_path(self):
        fig,axes=self.axes(1);ax=axes[0]
        fig.set_size_inches(7.2,3.6)
        for color,T in zip(COLORS,self.protocol['main_path_T']):
            self.path(ax,'baseline',T,color,fr'$T={T}$')
        ax.axhline(3,color='#444444',ls='--',lw=1,label=r'$SR^\star=3$')
        ax.legend(frameon=False,ncol=3)
        self.save(fig,'Figure3_sharpe_complexity',
            'Full predetermined 96-penalty diagnostic paths at the predeclared horizons. Lines are mean population Sharpe and bands are central 95% replication percentiles. Stars identify the exact theory-scaled penalty, which need not coincide with a grid penalty or the ex-post maximum. The coordinate is deterministic population complexity. Dotted overprints mark failed checked numerical conditions; these paths are diagnostics, not a data-driven selection rule. An additional large-T path is included only if the frozen numerical feasibility decision supports it.')

    def robustness_path(self,names,kind,filename):
        fig,axes=self.axes(2)
        for panel,(ax,T) in enumerate(zip(axes,self.protocol['robustness_path_T'])):
            for color,name in zip(self.palette(names),names):
                self.path(ax,name,T,color,self.label(name,kind))
                if kind!='rho' or name==names[0]:
                    optimum=parameters(name).sr_star*np.sqrt(12)
                    ax.axhline(optimum,color=color if kind!='rho' else '#333333',ls='--',lw=.9,
                        label=fr'$SR^\star={optimum:.3f}$')
            ax.set_title(fr'({chr(97+panel)}) $T={T}$')
        self.shared_legend(fig,axes,names)
        self.save(fig,filename,
            'Predetermined early and late horizons compare all 96 diagnostic penalties across the specified economies. Means and central 95% replication bands use 300 independent paths; stars are the exact prescribed theory-scaled choices. Population optima remain explicit. The rho comparison uses the same population-complexity coordinate across environments. Dotted overprints identify numerical failures in the paired-rank/floor checks. No maximum along these ex-post curves is used to choose the reported strategy.')

    def render(self):
        self.spectrum()
        self.learnability(['baseline'],'main','Figure1_learnability')
        self.main_complexity();self.main_path()
        self.learnability(N_NAMES,'N','FigureR1_N_learnability')
        self.learnability(RHO_NAMES,'rho','FigureR2_rho_learnability')
        self.robustness_complexity(N_NAMES,'N','FigureR3_N_complexity')
        self.robustness_complexity(RHO_NAMES,'rho','FigureR4_rho_complexity')
        self.robustness_path(N_NAMES,'N','FigureR5_N_sharpe_complexity')
        self.robustness_path(RHO_NAMES,'rho','FigureR6_rho_sharpe_complexity')
        (OUTPUT/'figure_captions.txt').write_text('\n\n'.join(name+'\n'+caption for name,caption in self.captions)+'\n')
        assert len(self.captions)==10

if __name__=='__main__':
    Figures().render()
