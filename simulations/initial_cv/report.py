"""English report and publication plots for the frozen initial-CV experiment."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from simulations.extended.design import ENVIRONMENTS, parameters
from simulations.extended.figures import Figures
from simulations.extended.statistics import distribution
from simulations.initial_cv.design import OUTPUT, protocol
from simulations.initial_cv.summarize import SPECIFICATIONS
from simulations.provenance import json_write, utc_now, file_hash

LABELS=['Theory + initial CV','Original fixed constant','Ex-post population oracle']
COLORS=['#244b73','#b55f39','#3e806b']
TITLES={'baseline':'Baseline: N=600, rho=0.95','N300':'N=300, rho=0.95',
        'N1200':'N=1200, rho=0.95','rho000':'N=600, rho=0','rho075':'N=600, rho=0.75'}


def run():
    frozen=protocol()
    verified=json.loads((OUTPUT/'summary_verification.json').read_text())
    arithmetic=json.loads((OUTPUT/'arithmetic_audit.json').read_text())
    if not arithmetic['passed'] or arithmetic['source_summary_sha256']!=file_hash(OUTPUT/'summary.csv'):
        raise ValueError('Independently audit the final tables first.')
    if not verified['passed'] or verified['replications_per_economy']!=300:
        raise ValueError('Final reports require complete verified production.')
    summary=pd.read_csv(OUTPUT/'summary.csv')
    slopes=pd.read_csv(OUTPUT/'slopes_all_windows.csv')
    constants=pd.read_csv(OUTPUT/'selected_constant_summary.csv')
    selected=pd.read_csv(OUTPUT/'selected_constants.csv')
    warnings=pd.read_csv(OUTPUT/'numerical_warnings.csv')
    paired=pd.read_csv(OUTPUT/'paired_cv_vs_fixed.csv')
    T_final=max(frozen['design']['baseline_T'])
    end=summary[(summary.environment=='baseline')&(summary['T']==T_final)&(summary.metric=='annual_SR')].set_index('specification')
    difference=paired[(paired.environment=='baseline')&(paired['T']==T_final)&(paired.metric=='annual_SR')].iloc[0]
    direction=('improves' if difference.mean_CI_low>0 else 'reduces' if difference.mean_CI_high<0 else 'does not show a clear change in')
    cv=end.loc['initial_cv'];fixed=end.loc['old_fixed'];oracle=end.loc['population_oracle']
    opening=(f'BASELINE FINDING AT T={T_final}: Initial CV {direction} mean population Sharpe relative to the old fixed constant. '
        f'Annualized means: CV={cv["mean"]:.6f}, fixed={fixed["mean"]:.6f}, diagnostic oracle={oracle["mean"]:.6f}, economic optimum=3. '
        f'Paired CV-minus-fixed mean={difference["mean"]:.6f}, 95% mean interval=[{difference.mean_CI_low:.6f}, {difference.mean_CI_high:.6f}]. '
        f'The central 95% of CV replication outcomes is [{cv.p025:.6f}, {cv.p975:.6f}]. '
        f'The remaining mean economic Sharpe gap is {3-cv["mean"]:.6f}; the oracle-minus-CV difference is {oracle["mean"]-cv["mean"]:.6f}. '
        'These comparisons describe this frozen experiment; they do not establish a theorem or resolve rank approximation.')
    start=summary[(summary.environment=='baseline')&(summary['T']==60)&(summary.metric=='annual_SR')].set_index('specification')
    start_difference=paired[(paired.environment=='baseline')&(paired['T']==60)&(paired.metric=='annual_SR')].iloc[0]
    early=(f'EARLY VERSUS LATE RECOVERY: At T=60, mean annualized Sharpe is {start.loc["initial_cv","mean"]:.6f} with initial CV versus {start.loc["old_fixed","mean"]:.6f} with the old constant, '
        f'a paired improvement of {start_difference["mean"]:.6f} (95% mean interval [{start_difference.mean_CI_low:.6f}, {start_difference.mean_CI_high:.6f}]). '
        'In the baseline the positive paired mean difference remains clear through T=2160, but its interval includes zero at T=3240 and T=4860. '
        'A single initial calibration therefore substantially improves short-history recovery in this experiment, but does not deliver a persistent final-horizon Sharpe advantage or match the diagnostic oracle.')
    full_slope=slopes[(slopes.environment=='baseline')&(slopes.specification=='initial_cv')&(slopes.quantity=='annual_gap')&(slopes.grid=='full')&(slopes.window=='full')].iloc[0]
    late_slope=slopes[(slopes.environment=='baseline')&(slopes.specification=='initial_cv')&(slopes.quantity=='annual_gap')&(slopes.grid=='full')&(slopes.window=='T_ge_1440')].iloc[0]
    rate=(f'BASELINE GAP SLOPES: Initial CV gives {full_slope.slope:.6f} over all thirteen horizons and {late_slope.slope:.6f} for T>=1440, versus the -0.6 theoretical reference. '
        'The full-window slope also reflects the much smaller starting gap under CV; it does not by itself rank the procedures at every horizon or validate an asymptotic rate.')
    with np.load(OUTPUT/'distributions/baseline.npz') as z:
        median_penalties=np.median(z['penalties'][:,-1,:],axis=0)
    complexity=summary[(summary.environment=='baseline')&(summary['T']==T_final)&(summary.metric=='empirical_complexity')].set_index('specification')
    regret=summary[(summary.environment=='baseline')&(summary['T']==T_final)&(summary.metric=='regret')].set_index('specification')
    diagnosis=(f'END-HORIZON DIAGNOSIS: The median CV penalty is {median_penalties[0]:.8g}, compared with {median_penalties[1]:.8g} for the old rule and {median_penalties[2]:.8g} for the common-grid Sharpe oracle. '
        f'Mean empirical effective complexity is {complexity.loc["initial_cv","mean"]:.3f}, {complexity.loc["old_fixed","mean"]:.3f}, and {complexity.loc["population_oracle","mean"]:.3f}, respectively. '
        f'The standard deviation of CV population Sharpe is {cv["sd"]/fixed["sd"]:.2f} times the old-rule standard deviation. '
        f'CV nevertheless reduces mean response-one regret from {regret.loc["old_fixed","mean"]:.6f} to {regret.loc["initial_cv","mean"]:.6f}. '
        'This distinction matters: CV targets prediction loss, while Sharpe is invariant to portfolio scale. The median CV penalty is below the diagnostic Sharpe optimum. '
        'This is an ex-post diagnosis only; neither the selected constants nor the theoretical exponent has been revised.')
    findings=[opening,early,rate,diagnosis]
    for name in ENVIRONMENTS:
        T=max(frozen['design']['baseline_T'] if name=='baseline' else frozen['design']['robustness_T'])
        row=paired[(paired.environment==name)&(paired['T']==T)&(paired.metric=='annual_SR')].iloc[0]
        alpha=constants[constants.environment==name].iloc[0]
        warning=warnings[(warnings.environment==name)&(warnings['T']==T)&(warnings.specification=='initial_cv')].iloc[0]
        findings.append(f'{name}, T={T}: paired mean Sharpe change {row["mean"]:+.6f} '
            f'(95% mean interval {row.mean_CI_low:+.6f}, {row.mean_CI_high:+.6f}); '
            f'median initial a={alpha["median"]:.6g}, central 95% a=[{alpha.p025:.6g}, {alpha.p975:.6g}], '
            f'boundary selections {int(alpha.lower_boundary_count)} lower / {int(alpha.upper_boundary_count)} upper out of 300. '
            f'Projection floor / mean CV regret={100*warning.floor_fraction:.2f}%. '
            f'Nyström subspace Sharpe optimum={warning.subspace_annual_SR:.6f}; '
            f'representation floor accounts for {100*warning.fraction_mean_Sharpe_gap_numerical:.2f}% of the mean CV economic Sharpe gap.')
    (OUTPUT/'findings.txt').write_text('\n\n'.join(findings)+'\n')
    lines=['INITIAL CV CALIBRATION: FINAL RICH6D SIMULATION',*findings,
        'All five economies have 300 independent replication indices, paired across environments and nested across horizons. The baseline analytical annualized Sharpe remains 3; the original N-specific opportunity sets are preserved.',
        'Each replication selects a once, separately in each economy, using observations 1-60 only. Expanding training prefixes 1-20,1-30,1-40,1-50 predict the next ten observations. The common predetermined 71-point logarithmic grid runs from 1e-8 to 1e-1. Every fold uses lambda=a*n_train^(-0.6); validation minimizes the observation-weighted response-one squared loss. Exact ties favor larger a. All five selections occur before observation 61 is generated. The selected a is frozen forever and the exponent is never optimized.',
        'The first twelve original production-seed paths passed the implementation gate before remaining production began; these same verified paths count toward the 300. No performance-dependent change to grid or folds was made.',
        'COMPARISONS: Original fixed-a policies are recomputed on every path and match the previous saved results. The population oracle chooses one common penalty per economy/horizon by maximizing mean population Sharpe on the original fixed 96-point grid. It is diagnostic and inaccessible to the selector. Oracle bands and slopes are descriptive and do not adjust for ex-post grid selection.',
        'RATE REFERENCES: The theoretical decay exponent remains -0.6. The reference slope is -0.6 for the Sharpe gap and +0.4 for effective complexity (hence -0.6 for complexity/T), subject to the theoretical assumptions and finite-sample qualifications. Complexity scaling partly reflects the imposed penalty rule and is not an independent test of optimal tuning.',
        'UNCERTAINTY: All plotted bands are the central 95% of the learned-policy outcomes across replications, not confidence intervals of the mean. Bands condition on the frozen Nyström basis and population operator; they do not include numerical integration or rank-approximation uncertainty. Mean MCSE and paired mean confidence intervals are tabulated separately. Non-oracle slope MCSE uses the full cross-T covariance and is checked by a leave-one-path jackknife.',
        'SELECTED CONSTANTS',constants.to_string(index=False),
        'END-HORIZON POPULATION SHARPE AND SHARPE GAP']
    final=[]
    for name in ENVIRONMENTS:
        T=max(frozen['design']['baseline_T'] if name=='baseline' else frozen['design']['robustness_T'])
        table=summary[(summary.environment==name)&(summary['T']==T)&summary.metric.isin(['annual_SR','annual_gap'])]
        final.append(table)
    lines.append(pd.concat(final).to_string(index=False))
    lines.extend(['PAIRED CV MINUS ORIGINAL FIXED: END-HORIZON SHARPE',
        pd.concat([paired[(paired.environment==n)&(paired['T']==max(frozen['design']['baseline_T'] if n=='baseline' else frozen['design']['robustness_T']))&(paired.metric=='annual_SR')] for n in ENVIRONMENTS]).to_string(index=False),
        'ALL PREDECLARED LEARNING AND COMPLEXITY SLOPES',slopes.to_string(index=False),
        'NUMERICAL WARNINGS',warnings.to_string(index=False),
        'Original Nyström warnings are retained. CV-specific floor/regret ratios can worsen when statistical regret becomes smaller. A new higher-rank comparison for the CV-selected policies has not been performed. Numerical certification does not follow from changing the regularization rule, more replications, or an attractive fitted slope.',
        'The initial sixty-observation calibration is noisy; boundary selections and the entire distribution of a are retained. Later data change the fitted policy and prescribed penalty, never the selected constant. No calibration is selected using final Monte Carlo outcomes. These finite-window results do not establish the source condition or prove the asymptotic learnability theorem.'])
    (OUTPUT/'critical_report.txt').write_text('\n\n'.join(lines)+'\n')
    Figures.set_style()
    folder=OUTPUT/'figures';folder.mkdir(exist_ok=True)
    captions=[]
    def save(fig,name,caption):
        fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(folder/(name+'.png'),bbox_inches='tight',dpi=300)
        plt.close(fig)
        captions.append(name+'\n'+caption)
    for name in ENVIRONMENTS:
        with np.load(OUTPUT/'distributions'/f'{name}.npz') as z:
            data={k:z[k] for k in z.files}
        times=data['T']
        fig,axes=plt.subplots(1,3,figsize=(10.2,3.6),layout='constrained')
        fig.suptitle(TITLES[name],fontsize=11)
        for k,(metric,ylabel,title) in enumerate([
            ('annual_SR','Annualized population Sharpe','(a) Sharpe recovery'),
            ('annual_gap','Annualized Sharpe gap','(b) Gap from economic optimum'),
            ('empirical_complexity','Empirical effective complexity','(c) Effective complexity')]):
            ax=axes[k]
            for j,label in enumerate(LABELS):
                stats=distribution(data[metric][:,:,j])
                ax.plot(times,stats['mean'],color=COLORS[j],label=label,marker='o',ms=2.5)
                ax.fill_between(times,stats['p025'],stats['p975'],color=COLORS[j],alpha=.10,lw=0)
            ax.set(xlabel='Training history T',ylabel=ylabel,title=title,xscale='log')
            if k>0:
                ax.set_yscale('log')
            ax.set_xticks([60,240,720,int(times[-1])]);ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(alpha=.22,lw=.5)
            original=warnings[(warnings.environment==name)&(warnings.specification=='old_fixed')]
            bad=original[~(original.original_floor_pass&original.original_paired_pass)]
            if len(bad):
                ax.axvspan(bad['T'].min(),times[-1]*1.04,color='#666666',alpha=.08,zorder=0)
        axes[0].axhline(np.sqrt(12)*parameters(name).sr_star,color='black',ls='--',lw=1,label='Economic optimum')
        subspace_sr=warnings[warnings.environment==name].subspace_annual_SR.iloc[0]
        axes[0].axhline(subspace_sr,color='#666666',ls=':',lw=1,label='Nyström subspace optimum')
        cv_warning=warnings[(warnings.environment==name)&(warnings.specification=='initial_cv')&(~warnings.floor_pass)]
        if len(cv_warning):
            t_index=[int(np.flatnonzero(times==T)[0]) for T in cv_warning['T']]
            axes[0].scatter(times[t_index],data['annual_SR'][:,t_index,0].mean(axis=0),
                marker='x',color='black',s=24,label='CV floor exceeds 5%',zorder=5)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
        save(fig,f'comparison_{name}',TITLES[name]+'. Means and central 95% replication-percentile bands across 300 paths. CV selects a once within the first 60 observations and freezes it thereafter. The oracle is an ex-post population diagnostic. Horizontal lines distinguish the economic optimum from the frozen Nyström subspace optimum. Grey areas retain original checked numerical failures; crosses identify CV mean-regret floor failures. All CV policies lack a new higher-rank certificate.')
    fig,axes=plt.subplots(2,3,figsize=(9.4,5.3),layout='constrained')
    for ax,name in zip(axes.flat,ENVIRONMENTS):
        values=selected[selected.environment==name].selected_a
        ax.hist(np.log10(values),bins=np.linspace(-8.05,-.95,72),color=COLORS[0],alpha=.85)
        ax.axvline(np.log10(frozen['design']['a_old'][name]),color=COLORS[1],ls='--',label='Original a')
        ax.set(title=TITLES[name],xlabel=r'$\log_{10}(\widehat a_{60})$',ylabel='Replications')
        ax.set_xticks([-8,-6,-4,-2,-1]);ax.legend(frameon=False)
    axes.flat[-1].set_visible(False)
    save(fig,'initial_constants','Distribution of the one-time initial CV constants across 300 paired replication indices per economy. The full predetermined grid and boundary selections are retained; dashed lines mark each original fixed constant. No later horizon changes the selected a.')
    (OUTPUT/'figure_captions.txt').write_text('\n\n'.join(captions)+'\n')
    json_write(OUTPUT/'report_status.json',dict(completed_utc=utc_now(),run_hash=frozen['run_hash'],figures=6,replications_per_economy=300,
        numerical_limitations_retained=True,visual_review_required=True,
        source_summary_sha256=file_hash(OUTPUT/'summary.csv')))


if __name__=='__main__':
    run()
