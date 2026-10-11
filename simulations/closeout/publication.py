"""Four principal figures and generated LaTeX; extra diagnostics in appendix."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from simulations.closeout.results import DEST,SPECS
from simulations.extended.design import ENVIRONMENTS,parameters
from simulations.extended.statistics import distribution
from simulations.provenance import json_write,file_hash

COLORS=('#126e82','#bc6840','#8c5aa9','#55884b')
LABELS=('Initial CV','Historical fixed','Oracle Q','Oracle SR')


def configure():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelsize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})


def plot_series(ax,times,values,j,band=True):
    d=distribution(values)
    ax.plot(times,d['mean'],color=COLORS[j],label=LABELS[j],marker='o',ms=3,lw=1.4,
            ls='--' if j>1 else '-')
    if band: ax.fill_between(times,d['p025'],d['p975'],color=COLORS[j],alpha=.09,lw=0)
    ax.grid(alpha=.2,lw=.5)


def save(fig,name):
    folder=DEST/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),bbox_inches='tight',dpi=180)
    plt.close(fig)


def figures():
    configure()
    with np.load(DEST/'distributions/baseline.npz') as z: data={k:z[k] for k in z.files}
    times=data['T'];q=parameters().q_star
    with np.load(DEST/'spectrum_energy.npz') as z:
        spectrum=z['eigenvalues'];energy=z['target_energy'];floor=float(z['projection_floor'])
    j=np.arange(1,len(spectrum)+1)
    fig,ax=plt.subplots(1,3,figsize=(12,3.65),layout='constrained')
    ax[0].loglog(j,spectrum,color=COLORS[0],lw=1.5)
    ax[0].set(xlabel='Economic eigen-direction j',ylabel='Eigenvalue',title='(a) Numerical economic spectrum')
    for T,line in [(60,'-'),(720,'--'),(4860,':')]:
        pos=int(np.flatnonzero(times==T)[0]);lam=np.median(data['penalties'][:,pos,0])
        ax[1].semilogx(j,spectrum/(spectrum+lam),label=f'T={T}',ls=line,lw=1.4)
    ax[1].set(xlabel='Economic eigen-direction j',ylabel='Ridge filter',title='(b) Filters at median CV penalties');ax[1].legend(frameon=False)
    ax[2].semilogx(j,np.cumsum(energy)/q,color=COLORS[0])
    ax[2].axhline(1-floor/q,color='gray',ls=':',label='Subspace total')
    ax[2].set(xlabel='Economic eigen-direction j',ylabel='Cumulative target energy / q',title='(c) Target in the same directions',ylim=(0,1.02));ax[2].legend(frameon=False)
    for a in ax:a.grid(alpha=.2,lw=.5)
    save(fig,'figure_0_spectrum')
    pd.DataFrame(dict(direction=j,eigenvalue=spectrum,target_energy=energy,cumulative_target_energy=np.cumsum(energy))).to_csv(DEST/'spectrum_energy.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for k,(metric,title,label) in enumerate([('annual_SR','(a) Sharpe recovery','Annualized population Sharpe'),('annual_gap','(b) Gap to economic optimum','Annualized Sharpe gap')]):
        for s in range(4):plot_series(ax[k],times,data[metric][:,:,s],s)
        ax[k].set(xlabel='Training history T',ylabel=label,title=title,xscale='log')
    ax[0].axhline(np.sqrt(12)*parameters().sr_star,color='black',ls='--',lw=1,label='Economic optimum')
    ax[0].axhline(np.sqrt(12*(q-floor)/(1-q+floor)),color='gray',ls=':',lw=1,label='Numerical subspace optimum')
    floor_fail=floor>.05*data['regret'][:,:,0].mean(axis=0)
    ax[0].scatter(times[floor_fail],data['annual_SR'][:,floor_fail,0].mean(axis=0),marker='x',s=30,color='black',zorder=5,label='CV floor exceeds 5%')
    ax[1].set_yscale('log')
    sigma=np.linspace(0,1,100)
    for s in range(4): ax[2].plot(sigma,sigma*data['annual_SR'][:,-1,s].mean(),color=COLORS[s],ls='--' if s>1 else '-',label=LABELS[s])
    ax[2].plot(sigma,sigma*np.sqrt(12)*parameters().sr_star,color='black',ls=':',label='Economic optimum')
    ax[2].set(xlabel='Annual volatility target',ylabel='Annual expected payoff',title='(c) Final population frontier')
    handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'figure_1_recovery')
    fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for k,(metric,title,label) in enumerate([('empirical_complexity','(a) Empirical complexity','Empirical effective complexity'),('population_complexity','(b) Numerical population complexity','Population effective complexity'),('relative_complexity','(c) Complexity relative to history','Empirical complexity / T')]):
        for s in range(4):plot_series(ax[k],times,data[metric][:,:,s],s)
        ax[k].set(xlabel='Training history T',ylabel=label,title=title,xscale='log',yscale='log')
    fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    save(fig,'figure_2_complexity')
    fig,ax=plt.subplots(figsize=(7,4.7),layout='constrained')
    for s in range(4):
        C=data['empirical_complexity'][:,:,s].mean(axis=0);SR=data['annual_SR'][:,:,s].mean(axis=0)
        ax.plot(C,SR,color=COLORS[s],marker='o',ms=4,label=LABELS[s],ls='--' if s>1 else '-')
        ax.annotate('T=60',(C[0],SR[0]),fontsize=8,xytext=(3,-11),textcoords='offset points',color=COLORS[s])
        ax.annotate('T=4860',(C[-1],SR[-1]),fontsize=8,xytext=(3,4),textcoords='offset points',color=COLORS[s])
    ax.axhline(np.sqrt(12)*parameters().sr_star,color='black',ls=':',label='Economic optimum')
    ax.set(xlabel='Mean empirical effective complexity',ylabel='Mean annualized population Sharpe',xscale='log',title='Sharpe against complexity: all archived horizons')
    ax.grid(alpha=.2,lw=.5);ax.legend(frameon=False,ncol=2,loc='lower right')
    save(fig,'figure_3_sharpe_complexity')
    # Additional figures stay in the appendix, never become a fifth main figure.
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,name in zip(axes.flat,list(ENVIRONMENTS)[1:]):
        with np.load(DEST/'distributions'/f'{name}.npz') as z:
            for s in range(4):plot_series(ax,z['T'],z['annual_SR'][:,:,s],s)
            ax.axhline(np.sqrt(12)*parameters(name).sr_star,color='black',ls=':',lw=1)
        ax.set(title=name,xlabel='Training history T',ylabel='Annualized population Sharpe',xscale='log')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    save(fig,'appendix_economies')
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for k,(metric,title) in enumerate([('normalized_regret',r'(a) $T^{0.6}$ response-one regret'),('normalized_annual_gap',r'(b) $T^{0.6}$ annual Sharpe gap')]):
        for s in range(2):plot_series(ax[k],times,data[metric][:,:,s],s)
        ax[k].set(title=title,xlabel='Training history T',ylabel='Normalized outcome',xscale='log')
    fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    save(fig,'appendix_normalized')


def table(frame,columns,labels,float_format='%.5f'):
    return frame[columns].rename(columns=dict(zip(columns,labels))).to_latex(index=False,escape=False,float_format=lambda x:float_format%x,na_rep='--')


def latex():
    folder=DEST/'paper';folder.mkdir(exist_ok=True)
    summary=pd.read_csv(DEST/'summary.csv');slopes=pd.read_csv(DEST/'slopes_all_windows.csv')
    final=summary[(summary.environment=='baseline')&(summary['T']==4860)&summary.metric.isin(['annual_SR','regret','empirical_complexity'])].pivot(index='specification',columns='metric',values='mean').reindex(SPECS).reset_index()
    final['specification']=LABELS
    (folder/'final_table.tex').write_text(table(final,['specification','annual_SR','regret','empirical_complexity'],['Specification','Annual SR','Regret','Empirical C']))
    rate=slopes[(slopes.environment=='baseline')&(slopes.specification=='initial_cv')&(slopes.grid=='full')&slopes.quantity.isin(['regret','annual_gap'])]
    rate=rate[['quantity','window','observations','slope','MCSE_delta','MCSE_jackknife']].copy()
    rate['quantity']=rate.quantity.map({'regret':'Regret','annual_gap':'Sharpe gap'})
    rate['window']=rate.window.str.replace('_',r'\_',regex=False)
    (folder/'slope_table.tex').write_text(table(rate,list(rate.columns),['Quantity','Window','Horizons','Slope','MCSE','Jackknife']))
    endpoint=summary[(summary.specification=='initial_cv')&summary.metric.isin(['annual_SR','regret'])].copy()
    endpoint=endpoint[endpoint['T']==endpoint.groupby('environment')['T'].transform('max')]
    endpoint=endpoint.pivot(index=['environment','T'],columns='metric',values='mean').reset_index()
    (folder/'economy_table.tex').write_text(table(endpoint,['environment','T','annual_SR','regret'],['Economy','T','Annual SR','Regret']))
    numerical='The paired CV rank and quadrature audit remains unresolved; its scope and failed cells are tabulated in the appendix.'
    section=r'''\section{Simulations: a constant calibrated once}
\label{sec:simulations}
We use the frozen Rich6D economic experiment: a six-dimensional Mat\'ern-$3/2$ representation, Fourier truncation $M=128$, and 300 independent histories in each of five paired economies. The baseline has $N=600$ and $\rho=0.95$; alternatives change $N$ to 300 or 1200, or persistence to 0 or 0.75. Analytic economic optima are specific to $N$. Sharpe ratios are marginal monthly population quantities multiplied by $\sqrt{12}$; response-one loss is not annualized.

For each economy and history, initial cross-validation selects one constant from the unchanged 71-point grid $[10^{-8},10^{-1}]$ using observations 1--60 only. Four forward folds train on 20, 30, 40, and 50 observations and validate on the subsequent ten. Each fold uses $\lambda=a n_{\rm train}^{-0.6}$ and observation-weighted response-one squared loss; exact ties favor larger $a$. The selected constant is then frozen:
\[
\lambda_{k,T}=\widehat a_{k,60}T^{-0.6}.
\]
Later observations refit the policy, with no further calibration. The historical constant is a comparison. Oracle Q minimizes mean regret and oracle SR maximizes mean Sharpe on the original 96-penalty grid, choosing one common penalty per economy and horizon. Both use population information and are diagnostics only.

\begin{table}[htbp]\centering\small
\caption{Baseline endpoint at $T=4860$, all 300 histories.}
\input{\simoutput/paper/final_table.tex}
\end{table}

Initial CV reduces baseline mean endpoint regret from 0.022205 to 0.011296 (paired change $-0.010909$, 95\% Monte Carlo mean interval $[-0.011399,-0.010419]$). Annualized Sharpe changes from 2.934774 to 2.932497 (paired change $-0.002277$, interval $[-0.005079,0.000525]$). Empirical complexity rises from 14.956 to 143.782. Thus the response-one improvement does not establish a final Sharpe improvement. Oracle Q and oracle SR yield distinct endpoint penalties and outcomes; they do not tune the strategy.

The financial components uniquely identified from each policy's signed Sharpe and regret explain this mean discrepancy. At $T=4860$, mean positive-ray loss is 0.011119 under initial CV versus 0.010733 historically, while mean scale loss falls from 0.011472 to 0.000177. These add to the respective mean regrets. Thus improved scaling more than offsets worse directional fit in the response-one objective. These quantities require no choice of a payoff-moment root; actual monthly moments and the remaining radial decomposition are recovered separately on the audit subset.

The full-grid descriptive slopes of log mean regret and log mean Sharpe gap are $-0.437820$ and $-0.398648$, respectively. The unchanged windows $T\ge360$, $T\ge720$, and $T\ge1440$, full cross-horizon covariance MCSE, jackknife checks, and normalized-outcome quantiles are reported in the appendix. No horizon or replication was removed to approach $-0.6$. The asymptotic statement is an $O_{\mathbb P}$ upper bound, not an equality for a finite Monte Carlo mean.

The ideal infinite Fourier target, the simulated $M=128$ economy, and the fixed-dimensional Nystr\"om estimator are distinct objects. The attached derivation supports maximal source order $r=1$ for the ideal target and preserves the kernel upper rate under a frozen selection on a finite positive grid, conditional on the manuscript assumptions. It does not certify the numerical approximation. A fixed subspace with positive representation error cannot have regret to the full economic optimum converging to zero. At the baseline endpoint the rank-4096 representation floor is about 15.7\% of mean CV regret, exceeding the unchanged 5\% criterion. NUMERICAL

'''.replace('NUMERICAL',numerical)
    captions=[('figure_0_spectrum','Economic spectrum, ridge filters at median initial-CV penalties, and target energy measured in the same numerical eigen-directions. The displayed operator is the frozen rank-4096 approximation; it is not an infinite-dimensional spectrum.'),
      ('figure_1_recovery','Baseline population Sharpe recovery, gap to the analytic economic optimum, and final capital-allocation frontiers. Curves include every archived horizon. Shaded bands are the central 95 percent of replication outcomes, not confidence intervals of the mean; oracle bands are descriptive and not selection adjusted. Crosses mark CV projection floor/mean regret exceeding 5 percent. The CV audit remains scope-limited; rank and quadrature uncertainty are additional.'),
      ('figure_2_complexity','Empirical and numerical population effective complexity, and empirical complexity divided by history length. Population complexity is evaluated at each actual selected penalty before aggregation; it is not complexity at the mean penalty. Bands are replication percentiles.'),
      ('figure_3_sharpe_complexity','Mean population Sharpe against mean empirical complexity across the full baseline horizon grid. Solid paths implement frozen initial CV or the historical rule; dashed oracle paths use population information. The horizontal line is the true economic optimum.')]
    for name,caption in captions:
        section+=r'\begin{figure}[p]\centering'+'\n'+r'\includegraphics[width=\textwidth]{\simoutput/figures/'+name+r'.pdf}'+'\n'+r'\caption{'+caption+r'}\end{figure}'+'\n'
    (folder/'simulation_section.tex').write_text(section.replace('response-one squared loss;', 'response-one loss;'))
    appendix=r'''\section{Simulation diagnostics and numerical scope}
\label{app:simulations}
All diagnostics added during closeout are analyses after observing the archived experiment. The economic specification, folds, grid, pairing, initial calibration time, exponent and all original outcomes are immutable. Full per-history data, normalized outcomes, distribution summaries and reproduction commands accompany the generated tables.

\begin{table}[htbp]\centering\small
\caption{Initial-CV descriptive baseline slopes on the established windows. The slope is the regression of log Monte Carlo mean, not the mean of per-history slopes. Full covariance preserves complete history dependence across horizons.}
\input{\simoutput/paper/slope_table.tex}
\end{table}
\begin{table}[htbp]\centering\small
\caption{Initial-CV endpoints in the five economies, all 300 histories. The common comparison grid ends at $T=3240$; baseline continues to 4860.}
\input{\simoutput/paper/economy_table.tex}
\end{table}

For each policy, let $u=\E Y$, $v=\E Y^2$ and $q=1-Q(W^\star)$. These are monthly moments. Set $c=u/q$, $d=v-u^2/q$. Direct calculation verifies
\[
Q(W)-Q(W^\star)=q(c-1)^2+d,\qquad
\SR(W)=\frac{u}{\sqrt{v-u^2}}.
\]
For $c>0$, the scale-invariant directional error is $d/[c^2q(1-q)]$. For $u,v>0$, the positive-ray identity is
\[
\underbrace{q-u^2/v}_{\text{ray gap}}+
\underbrace{(v-u)^2/v}_{\text{scale gap}}=Q(W)-Q(W^\star).
\]
Null or negative means and zero variances are recorded separately, without changing policy orientation. Roundoff-sized negative directional errors are retained; larger negative errors fail verification. Nonlinear identities are checked per policy rather than on aggregate moments.

For the same numerical subspace and actual penalty, let $\beta_P=S^{-1}m$, $\beta_\lambda=(S+\lambda I)^{-1}m$, and $\Delta=\widehat\beta-\beta_\lambda$. Then
\begin{align*}
Q(\widehat\beta)-Q(W^\star)
={}&q-m'S^{-1}m
+(\beta_\lambda-\beta_P)'S(\beta_\lambda-\beta_P)\\
&+\Delta'S\Delta+2\Delta'S(\beta_\lambda-\beta_P).
\end{align*}
The signed cross term is preserved. The displacement norm is not automatically a variance, particularly with a data-selected penalty. Recovered-moment tables explicitly state the number of histories and horizons re-estimated; they must not be read as decompositions of all 1500 archived histories.

\input{\simoutput/paper/recovered_diagnostics.tex}

The numerical protocol separates A (the archived rank-4096 constant held fixed at both ranks) from B (initial CV repeated only on the identical first 60 observations at each rank). It declares twelve representative original production indices and separate stress indices selected by the smallest initial constant. These differ from the historical rank-pilot seeds. Full and short audits have separate labels. The original 5\% projection-floor criterion, paired absolute mean plus 1.96 MCSE relative to mean rank-4096 regret, and independent-quadrature comparison remain unchanged. Mean, second moment, Sharpe-gap and normalized-directional sensitivity thresholds are also 5\%. No subset result certifies an untested horizon or provides a forward certificate for rank 8192.

\input{\simoutput/paper/numerical_scope.tex}

The infinite Fourier tail satisfies $\|\psi_\infty-\psi_M\|_\infty\le c_\psi/[4M^4\log(M+1)]$, about $6.6991\times10^{-9}$ at $M=128$. For fixed policies this implies uniform loading and moment perturbation bounds. The exported operator perturbation ratios use every actually selected penalty. Small pointwise error alone does not establish stability of a ridge inverse or an initial argmin; paired Fourier diagnostics and any remaining gaps are explicitly separated. The finite truncation does not identify a maximal source order from its differentiability.

\begin{figure}[p]\centering
\includegraphics[width=\textwidth]{\simoutput/figures/appendix_economies.pdf}
\caption{All four alternative economies on the common grid, with their analytic economic optima. Bands show replication outcomes. Economies differing only in persistence share the marginal population operator but can select different penalties and complexities.}
\end{figure}
\begin{figure}[p]\centering
\includegraphics[width=\textwidth]{\simoutput/figures/appendix_normalized.pdf}
\caption{Normalized baseline regret and Sharpe gap on the complete grid. Means and replication-percentile bands are finite descriptive distributions; they do not identify an asymptotic rate in probability.}
\end{figure}
'''
    (folder/'simulation_appendix.tex').write_text(appendix.replace(r'\begin{figure}[p]',r'\begin{figure}[htbp]'))
    # Dynamic, scope-qualified diagnostics are refreshed after numerical evaluation.
    diagnostics(folder)


def diagnostics(folder):
    files=sorted((DEST/'audit_fits').glob('*.csv'))
    if files:
        rows=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
        # Prefer full audits where a history also has a short record.
        rows=rows.sort_values('sample_scope').drop_duplicates(['environment','index','T','audit'],keep='first')
        rows=rows[rows.audit.isin(['A','C'])].copy()
        end=rows[(rows.environment=='baseline')&(rows['T']==rows['T'].max())]
        columns=['ray_gap','scale_gap','projection_floor','regularization_bias_within_P','estimation_displacement','signed_cross_term']
        means=end.groupby('audit')[columns].mean().reset_index()
        count=end.groupby('audit')['index'].nunique().to_dict()
        (folder/'decomposition_table.tex').write_text(table(means,['audit',*columns],['Policy','Ray','Scale','Floor','Bias','Displacement','Cross']))
        text=(r'\begin{table}[htbp]\centering\scriptsize'+'\n'+r'\caption{Recovered baseline endpoint moments. A uses frozen initial CV; C uses the historical constant. Scope: '+', '.join(f'{label}: {n} histories' for label,n in count.items())+r'; descriptive paired audit histories only. Ray plus scale and floor plus bias plus displacement plus signed cross reconstruct regret.}'+'\n'+r'\resizebox{\textwidth}{!}{\input{\simoutput/paper/decomposition_table.tex}}'+'\n'+r'\end{table}'+'\n')
        moment_columns=['u','v','c','d','radial_gap','normalized_directional_error','monthly_SR']
        measured=end.groupby('audit')[moment_columns].mean().reset_index()
        (folder/'recovered_moment_table.tex').write_text(table(measured,['audit',*moment_columns],['Policy','u','v','c','d','Radial','Normalized direction','Monthly SR'],float_format='%.6f'))
        text=r'\begin{table}[htbp]\centering\scriptsize'+'\n'+r'\caption{Actual recovered monthly moments and radial/angular components, baseline T=4860. Scope is the same explicitly counted audit histories as the following table; these values are not inferred by choosing roots of Q/SR.}'+'\n'+r'\resizebox{\textwidth}{!}{\input{\simoutput/paper/recovered_moment_table.tex}}'+'\n'+r'\end{table}'+'\n'+text
    else:text='No recovered policy moments are available; this quantitative decomposition is not run.\n'
    components_path=DEST/'component_summary.csv'
    if components_path.exists():
        components=pd.read_csv(components_path)
        selected=components[(components.environment=='baseline')&(components['T']==4860)&components.specification.isin(['initial_cv','old_fixed'])]
        full=selected.pivot(index='specification',columns='metric',values='mean').reset_index()
        cols=['specification','ray_gap','scale_gap','projection_floor','regularization_bias_within_P','estimation_displacement_plus_cross']
        full['specification']=full.specification.map({'initial_cv':'Initial CV','old_fixed':'Historical'})
        (folder/'identified_components_table.tex').write_text(table(full,cols,['Specification','Ray','Scale','Floor','Bias','Displacement + cross']))
        text=r'\begin{table}[htbp]\centering\scriptsize'+'\n'+r'\caption{Baseline endpoint financial components, all 300 histories. Ray/scale quantities are uniquely identified from positive signed Sharpe and regret for each policy. Population-ridge bias uses each actual penalty in the same numerical operator. The last column keeps displacement and signed cross combined; it is not variance. Their separate values require recovered coefficients.}'+'\n'+r'\resizebox{\textwidth}{!}{\input{\simoutput/paper/identified_components_table.tex}}'+'\n'+r'\end{table}'+'\n'+text
    (folder/'recovered_diagnostics.tex').write_text(text)
    scope_path=DEST/'numerical_status.json'
    if scope_path.exists():
        status=json.loads(scope_path.read_text())
        scope=status['latex_scope']
    else:scope='The new audit is pending. No numerical certificate is asserted.'
    comparison_path=DEST/'audit_paired_comparisons.csv'
    if comparison_path.exists():
        compared=pd.read_csv(comparison_path)
        rank=compared[(compared.comparison=='rank8192_minus4096')&compared.audit.isin(['A','B'])&(compared.cohort=='representative')].copy()
        rank=rank[rank['T']==rank.groupby('environment')['T'].transform('max')]
        endpoint=rank.pivot(index=['environment','T','audit','histories'],columns='metric',values='paired_mean').reset_index()
        columns=['environment','T','audit','histories','regret','u','v','annual_gap','normalized_directional_error']
        (folder/'rank_endpoint_table.tex').write_text(table(endpoint,columns,['Economy','T','Audit','n',r'$\Delta Q$',r'$\Delta u$',r'$\Delta v$',r'$\Delta$ SR gap',r'$\Delta$ direction'],float_format='%.3e'))
        scope+=r'\begin{table}[htbp]\centering\scriptsize'+'\n'+r'\caption{Paired rank8192 minus4096 endpoint changes on the recovered audit histories. A holds the archived constant fixed; B reselects once on the initial60 at each rank. These are point differences. Where n=1 a replication MCSE is not estimable; the complete representative/stress endpoint audit remains unresolved.}'+'\n'+r'\resizebox{\textwidth}{!}{\input{\simoutput/paper/rank_endpoint_table.tex}}'+'\n'+r'\end{table}'+'\n'
    (folder/'numerical_scope.tex').write_text(scope+'\n')


def run():
    figures();latex()
    json_write(DEST/'publication_manifest.json',dict(main_figures=4,appendix_figures=2,empirics_changed=False,theorems_changed=False,
        source_summary_sha256=file_hash(DEST/'summary.csv'),source_slopes_sha256=file_hash(DEST/'slopes_all_windows.csv')))

if __name__=='__main__':run()
