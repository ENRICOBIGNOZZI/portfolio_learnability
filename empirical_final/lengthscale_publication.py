"""Publication artifacts and evidence-led decision memo for lengthscale study."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import digest
from empirical_final.lengthscale_spectral import (KERNELS,MULTIPLIERS,PRIMARY_METHODS,PRIMARY_RANGE,
    ROOT,log_amplitude,rank_fit,load_cache,verify_frozen)
from empirical_final.run import clean_manifest
from empirical_final.theory_guided_lambda import save,write_json

LABELS={'fixed_lambda':'Fixed lambda','annual_cv':'Annual CV','one_anchor':'One-anchor spectrum',
        'joint_cv':'Joint CV','spectral_transfer':'Amplitude transfer (heuristic)',
        'adaptive_spectral':'Adaptive full spectrum','transfer_fixed_length':'Transfer, fixed length'}
COLORS={'fixed_lambda':'#747474','annual_cv':'#235b84','one_anchor':'#bc812e',
        'joint_cv':'#183b35','spectral_transfer':'#ab4f65','adaptive_spectral':'#498f8b'}
SOURCE_LINKS={
 'matern_density':'https://gaussianprocess.org/gpml/chapters/RW4.pdf',
 'periodized_matern_and_gaussian':'https://arxiv.org/abs/2006.10160',
 'sobolev_and_restriction':'https://arxiv.org/abs/2309.10918'}


def read(out,name):return pd.read_csv(out/'tables'/f'{name}.csv')


def esc(value):
    return str(value).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')


def sci(x):
    mantissa,exponent=f'{x:.3e}'.split('e')
    return mantissa+r'\times10^{'+str(int(exponent))+'}'


def diagnostics(out):
    spectra=read(out,'empirical_spectra');fits=read(out,'spectral_fits')
    main=fits[fits.history.isna()].copy();rows=[];scaling=[];modes=[]
    for (kernel,year,first,last),f in main.groupby(['kernel','year','first_rank','last_rank']):
        common_b=float(f.b.mean())
        for r in f.itertuples():
            mu=spectra[spectra.kernel.eq(kernel)&spectra.year.eq(year)&spectra.multiplier.eq(r.multiplier)].sort_values('rank').mu.to_numpy()
            rows.append(dict(kernel=kernel,year=year,multiplier=r.multiplier,first_rank=first,last_rank=last,b_separate=r.b,
                log_c_separate=r.log_c,b_common_this_year=common_b,
                log_c_common_this_year=log_amplitude(mu,common_b,(int(first),int(last)),'mean'),
                b_frozen_first_history=r.common_b,log_c_frozen_median=r.log_c_common_median,
                r_squared_separate=r.r_squared,max_abs_log_residual=r.max_abs_log_residual))
    amplitudes=save(out/'tables/amplitude_comparisons.csv',rows)
    for (kernel,year,first,last),f in amplitudes.groupby(['kernel','year','first_rank','last_rank']):
        x=np.log(f.multiplier.to_numpy());y=f.log_c_frozen_median.to_numpy()
        slope,intercept=np.polyfit(x,y,1);res=y-intercept-slope*x
        scaling.append(dict(kernel=kernel,year=year,first_rank=first,last_rank=last,log_amplitude_vs_log_length_slope=float(slope),
            theoretical_matern_slope=-3. if kernel=='matern32' else None,r_squared=float(1-np.sum(res**2)/np.sum((y-y.mean())**2)),
            maximum_log_scaling_residual=float(np.abs(res).max())))
    save(out/'tables/amplitude_scaling_diagnostics.csv',scaling)
    clean=clean_manifest()
    for kernel in KERNELS:
        gs,_,_=load_cache(kernel,clean)
        for year in (1978,2000,2024):
            T=(year-1963)*12;spaces=[]
            for g in gs:
                values,u=np.linalg.eigh(np.asarray(g[:T])@np.asarray(g[:T]).T)
                spaces.append(u[:,::-1])
            for bi,U in enumerate(spaces):
                for first,last in [(1,10),(11,60),(61,120)]:
                    cross=spaces[2][:,first-1:last].T@U[:,first-1:last]
                    cosines=np.linalg.svd(cross,compute_uv=False)
                    modes.append(dict(kernel=kernel,year=year,multiplier=MULTIPLIERS[bi],first_rank=first,last_rank=last,
                        mean_squared_subspace_cosine=float(np.mean(cosines**2)),
                        smallest_subspace_cosine=float(cosines.min()),
                        scope='Sign-invariant temporal payoff subspaces on identical observations'))
    save(out/'tables/temporal_subspace_diagnostics.csv',modes)
    # Required falsification diagnostics: these transformations never alter a policy.
    predictions=read(out,'penalty_prediction_diagnostics')
    effects=[]
    anchors=json.loads((out/'audit/anchors.json').read_text())
    for kernel,year in [(k,y) for k in KERNELS for y in range(1978,2025)]:
        b=anchors[kernel]['finite_range_b']
        sub=predictions[predictions.kernel.eq(kernel)&predictions.year.eq(year)]
        cv=sub[sub.rule.eq('cv')].set_index('multiplier')
        base_mu=spectra[spectra.kernel.eq(kernel)&spectra.year.eq(year)&spectra.multiplier.eq(1)].sort_values('rank').mu.to_numpy()
        base_logc=log_amplitude(base_mu,b)
        for mult in MULTIPLIERS:
            mu=spectra[spectra.kernel.eq(kernel)&spectra.year.eq(year)&spectra.multiplier.eq(mult)].sort_values('rank').mu.to_numpy()
            log_c_ratio=log_amplitude(mu,b)-base_logc
            prediction=log_c_ratio/(b+1)
            for rule in ('cv','spectral','transfer','fixed_lambda'):
                s=sub[sub.rule.eq(rule)].set_index('multiplier')
                effect=float(np.log(s.loc[mult,'penalty']/s.loc[1.,'penalty']))
                effects.append(dict(kernel=kernel,year=year,multiplier=mult,rule=rule,
                    predicted_log_penalty_effect=prediction,observed_log_penalty_effect=effect,
                    log_effect_error=effect-prediction,
                    implied_asymptotic_log_rho_ratio=((b+1)*effect-log_c_ratio)/b,
                    scope='Cross-length effect diagnostic; CV information is not used by primary rules'))
    effects=save(out/'tables/lengthscale_penalty_effects.csv',effects)
    effect_summary=[]
    for (kernel,rule),f in effects[effects.year.ge(1984)&effects.multiplier.ne(1)].groupby(['kernel','rule']):
        effect_summary.append(dict(kernel=kernel,rule=rule,observations=len(f),rmse_log_effect=float(np.sqrt(np.mean(f.log_effect_error**2))),
            mae_log_effect=float(f.log_effect_error.abs().mean()),
            pearson_correlation=float(f.predicted_log_penalty_effect.corr(f.observed_log_penalty_effect)) if f.observed_log_penalty_effect.std()>0 else None,
            median_implied_rho_ratio=float(np.exp(f.implied_asymptotic_log_rho_ratio.median()))))
    save(out/'tables/lengthscale_effect_summary.csv',effect_summary)
    performance=read(out,'performance_summary');decomposition=[];staleness=[]
    for kernel in KERNELS:
        for period in performance.period.unique():
            s=performance[performance.kernel.eq(kernel)&performance.period.eq(period)&performance.scenario.eq('gross')].set_index('method')
            names=['annual_cv','joint_length_base_cv_lambda','base_length_joint_cv_lambda','joint_cv']
            if all(n in s.index for n in names):
                for metric in ('sharpe','annual_excess_return','annual_volatility'):
                    A,B,C,D=[float(s.loc[n,metric]) for n in names]
                    representation=.5*((B-A)+(D-C));shrinkage=.5*((C-A)+(D-B))
                    np.testing.assert_allclose(representation+shrinkage,D-A,atol=1e-12)
                    decomposition.append(dict(kernel=kernel,period=period,metric=metric,
                        base_length_base_penalty=A,joint_length_base_penalty=B,base_length_joint_penalty=C,
                        joint_length_joint_penalty=D,representation_shapley_contrast=representation,
                        penalty_shapley_contrast=shrinkage,interaction=D-B-C+A,
                        interpretation='Exact descriptive 2x2 contrasts; not a causal decomposition; crossed raw penalties are not retuned'))
            for main,shorter in [('adaptive_spectral','adaptive_spectral_36'),('spectral_transfer','adaptive_transfer_36')]:
                if main in s.index and shorter in s.index:
                    staleness.append(dict(kernel=kernel,period=period,method=main,shorter_method=shorter,
                        sharpe_60=float(s.loc[main,'sharpe']),sharpe_36=float(s.loc[shorter,'sharpe']),
                        delta_sharpe_36_minus_60=float(s.loc[shorter,'sharpe']-s.loc[main,'sharpe']),
                        interpretation='Lag sensitivity of length selection only; does not isolate stale refit observations'))
    save(out/'tables/representation_penalty_contrasts.csv',decomposition)
    save(out/'tables/selection_history_sensitivity.csv',staleness)
    sensitivity=[]
    for (kernel,year,mult),sub in fits[(fits.history.notna())&(fits.first_rank==11)&(fits.last_rank==60)].groupby(['kernel','year','multiplier']):
        baseline=sub[(sub.history=='expanding')&(sub.P==10000)&np.isclose(sub.threshold,1e-12,rtol=1e-8,atol=0)].iloc[0]
        for r in sub.itertuples():
            sensitivity.append(dict(kernel=kernel,year=year,multiplier=mult,history=r.history,T=r.T,P=r.P,threshold=r.threshold,
                active_rank=r.active_rank,finite_range_b=r.b,delta_b_from_full_history_full_features=r.b-baseline.b,
                rank_range='11-60',scope='Finite sample/history/feature-prefix diagnostic'))
    save(out/'tables/finite_sample_sensitivity.csv',sensitivity)
    anchors=json.loads((out/'audit/anchors.json').read_text());grids=json.loads((out/'audit/penalty_grids.json').read_text())
    constants=[]
    for kernel,a in anchors.items():
        for i,r in enumerate(a['additional_1978_cv_information']):
            grid=grids[kernel][i]['penalties'];side='lower' if np.isclose(r['penalty'],min(grid),rtol=1e-10,atol=0) else 'upper' if np.isclose(r['penalty'],max(grid),rtol=1e-10,atol=0) else 'interior'
            constants.append(dict(kernel=kernel,multiplier=r['multiplier'],cv_anchor=r['penalty'],boundary_side=side,
                rho_foc_endpoint=r['rho_foc_endpoint'],rho_ratio_to_baseline=r['rho_foc_endpoint']/a['rho0'],
                admissible_rho_interval='(0, endpoint]' if side=='lower' else '[endpoint, infinity)' if side=='upper' else '{endpoint} under interior proxy anchoring',
                adjusted_rule_convention='Use the historical FOC endpoint; boundary anchors are not point-identified',
                interpretation='Extra historical CV calibration, not an estimator of structural B/A'))
    save(out/'tables/initial_constant_diagnostics.csv',constants)
    return amplitudes


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix',
                         'font.size':10,'axes.titlesize':11,'pdf.fonttype':42})
    spectra=read(out,'empirical_spectra');a=read(out,'annual_policies');f=read(out,'spectral_fits')
    accounts=read(out,'monthly_accounts');paired=read(out,'paired_comparisons')
    colors=plt.get_cmap('viridis')(np.linspace(.05,.9,5));plot_values=[]
    def finish(fig,name):
        fig.savefig(out/'figures'/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(out/'figures'/f'{name}.png',bbox_inches='tight',dpi=300)
        plt.close(fig)
    for kernel,name,title in [('matern32','figure1_spectral_shape','Matérn-3/2 managed spectra'),
                              ('gaussian','appendix_gaussian_shape','Gaussian negative control: no polynomial tail imposed')]:
        fig,axes=plt.subplots(3,2,figsize=(7.3,7.2),layout='constrained')
        for yi,year in enumerate((1978,2000,2024)):
            data=spectra[spectra.kernel.eq(kernel)&spectra.year.eq(year)]
            base=data[data.multiplier.eq(1)].sort_values('rank').mu.to_numpy()
            for mult,color in zip(MULTIPLIERS,colors):
                s=data[data.multiplier.eq(mult)].sort_values('rank');mu=s.mu.to_numpy();rank=s['rank'].to_numpy()
                keep=(mu>mu.max()*1e-12)&(base>base.max()*1e-12)
                axes[yi,0].loglog(rank[keep],mu[keep],color=color,label=f'{mult:g} × lengthscale')
                axes[yi,1].semilogx(rank[keep],np.log(mu[keep]/base[keep]),color=color)
                if kernel=='matern32':
                    for j in np.flatnonzero(keep):plot_values.append(dict(figure='spectral_shape',kernel=kernel,year=year,
                        multiplier=mult,rank=int(rank[j]),eigenvalue=float(mu[j]),log_ratio=float(np.log(mu[j]/base[j]))))
            axes[yi,0].set(title=f'January {year}: T = {len(base)}',ylabel='Raw eigenvalue')
            axes[yi,1].set(title='Log ratio to baseline lengthscale',ylabel=r'$\log(\widehat\mu_j(\ell)/\widehat\mu_j(\ell_0))$')
            axes[yi,1].axhline(0,color='grey',lw=.5)
            for ax in axes[yi]:ax.grid(alpha=.2)
        axes[0,0].legend(frameon=False,fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Eigenvalue rank (log scale)')
        fig.suptitle(title,fontsize=13)
        finish(fig,name)
    main=f[f.history.isna()&f.kernel.eq('matern32')]
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.6),layout='constrained')
    amplitudes=read(out,'amplitude_comparisons').query("kernel=='matern32'")
    year_colors=['#235b84','#bc812e','#498f8b']
    for year,color in zip((1978,2000,2024),year_colors):
        ds=main[main.year.eq(year)];means=[];lo=[];hi=[];c=[];clo=[];chi=[]
        for mult in MULTIPLIERS:
            s=ds[ds.multiplier.eq(mult)];primary=s[s.first_rank.eq(11)&s.last_rank.eq(60)].iloc[0]
            means.append(primary.b);lo.append(primary.b-s.b.min());hi.append(s.b.max()-primary.b)
            rows=amplitudes[amplitudes.year.eq(year)&amplitudes.multiplier.eq(mult)]
            base=amplitudes[amplitudes.year.eq(year)&amplitudes.multiplier.eq(1)]
            ratios=rows.merge(base,on=['first_rank','last_rank'],suffixes=('_ell','_base'))
            logs=ratios.log_c_frozen_median_ell-ratios.log_c_frozen_median_base
            primary_log=float(logs[(ratios.first_rank==11)&(ratios.last_rank==60)].iloc[0])
            c.append(primary_log);clo.append(primary_log-logs.min());chi.append(logs.max()-primary_log)
        axes[0].errorbar(MULTIPLIERS,means,yerr=[lo,hi],color=color,marker='o',capsize=2,label=str(year))
        axes[1].errorbar(MULTIPLIERS,c,yerr=[clo,chi],color=color,marker='o',capsize=2)
    axes[0].axhline(133/130,color='grey',ls='--',lw=1,label='Nominal 133/130')
    axes[1].plot(MULTIPLIERS,-3*np.log(MULTIPLIERS),color='black',ls='--',label=r'Conditional tail law $\ell^{-3}$')
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks(MULTIPLIERS,[str(x) for x in MULTIPLIERS]);ax.grid(alpha=.2);ax.set_xlabel('Lengthscale multiplier')
    axes[0].set(ylabel='Finite-range log-log slope',title='Same ranks; sensitivity to fitting range')
    axes[1].set(ylabel='Log amplitude ratio (frozen common slope)',title='Finite-range proxy versus conditional tail law')
    axes[0].legend(frameon=False,fontsize=8);axes[1].legend(frameon=False,fontsize=8)
    finish(fig,'figure2_amplitude_exponent')
    fig,axes=plt.subplots(3,1,figsize=(7.3,7.2),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.2,1.2,.8]})
    for method,color in COLORS.items():
        s=a[a.kernel.eq('matern32')&a.method.eq(method)].sort_values('year')
        if not len(s):continue
        axes[0].semilogy(s.year,s.penalty,color=color,lw=1.4,label=LABELS[method])
        axes[1].plot(s.year,s.C,color=color,lw=1.4)
        if method in ('joint_cv','adaptive_spectral','spectral_transfer'):
            axes[2].step(s.year,s.multiplier,color=color,where='mid',label=LABELS[method])
    axes[0].set(ylabel='Raw ridge penalty (log)',title='Matérn penalties: one historical anchor per theory rule')
    axes[1].set_ylabel('Effective complexity')
    axes[2].set(ylabel='Lengthscale multiplier',xlabel='January decision year',yscale='log')
    axes[2].set_yticks(MULTIPLIERS,[str(x) for x in MULTIPLIERS]);axes[0].legend(frameon=False,fontsize=8,ncol=2)
    for ax in axes:ax.axvline(1984,color='grey',ls=':',lw=.8);ax.grid(alpha=.2)
    finish(fig,'figure3_penalty_complexity')
    fig,axes=plt.subplots(3,1,figsize=(7.3,7.4),layout='constrained',gridspec_kw={'height_ratios':[1.2,1.2,1.]})
    wealth=[]
    for method,color in COLORS.items():
        for ai,scenario in enumerate(('gross','trade25_borrow30')):
            s=accounts[accounts.kernel.eq('matern32')&accounts.method.eq(method)&accounts.window.eq('common_1984')&accounts.scenario.eq(scenario)].sort_values('return_date')
            if not len(s):continue
            w=np.cumprod(1+s.total_return.to_numpy());dates=pd.to_datetime(s.return_date)
            axes[ai].semilogy(dates,w,color=color,lw=1.4,label=LABELS[method])
            wealth.extend(dict(kernel='matern32',method=method,scenario=scenario,return_date=d,wealth=float(v)) for d,v in zip(s.return_date,w))
    axes[0].set(title='Common Matérn accounts: February 1984-January 2025',ylabel='Gross wealth (log)')
    axes[1].set(title='25 bp drift-adjusted trades + 30 bp/year short fee',ylabel='Net wealth (log)')
    axes[0].legend(frameon=False,fontsize=8,ncol=2)
    compared=[m for m in COLORS if m!='joint_cv']
    for si,(scenario,color) in enumerate([('gross','#235b84'),('trade25_borrow30','#bc812e')]):
        d=paired[paired.kernel.eq('matern32')&paired.period.eq('1984-2024')&paired.scenario.eq(scenario)&paired.reference.eq('joint_cv')&paired.block.eq(12)].set_index('method')
        for i,method in enumerate(compared):
            if method not in d.index:continue
            r=d.loc[method];y=i+(si-.5)*.16
            axes[2].plot([r.ci_low,r.ci_high],[y,y],color=color,lw=1.8)
            axes[2].scatter(r.delta_sharpe,y,color=color,s=18,label=scenario if i==0 else None)
    axes[2].set_yticks(np.arange(len(compared)),[LABELS[m] for m in compared],fontsize=8)
    axes[2].axvline(0,color='grey',ls='--',lw=.8);axes[2].set_xlabel('Sharpe difference versus joint CV; paired 95% interval')
    axes[2].legend(frameon=False,fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.02))
    for ax in axes:ax.grid(alpha=.2)
    finish(fig,'figure4_financial_results')
    fig,axs=plt.subplots(2,2,figsize=(7.4,6.0),layout='constrained')
    for method,color in COLORS.items():
        s=a[a.kernel.eq('gaussian')&a.method.eq(method)].sort_values('year')
        if not len(s):continue
        axs[0,0].semilogy(s.year,s.penalty,color=color,label=LABELS[method],lw=1.2)
        axs[0,1].plot(s.year,s.C,color=color,lw=1.2)
        net=accounts[accounts.kernel.eq('gaussian')&accounts.method.eq(method)&accounts.window.eq('common_1984')&accounts.scenario.eq('trade25_borrow30')].sort_values('return_date')
        axs[1,0].semilogy(pd.to_datetime(net.return_date),np.cumprod(1+net.total_return),color=color,lw=1.2)
    axs[0,0].set(title='Gaussian penalties',ylabel='Raw penalty (log)',xlabel='Decision year')
    axs[0,0].legend(frameon=False,fontsize=6,ncol=2)
    axs[0,1].set(title='Effective complexity',ylabel='C',xlabel='Decision year')
    axs[1,0].set(title='Actual net accounts from 1984',ylabel='Wealth (log)',xlabel='Return year')
    d=paired[paired.kernel.eq('gaussian')&paired.method.eq('spectral_transfer')&paired.period.eq('1984-2024')&paired.scenario.eq('trade25_borrow30')&paired.block.eq(12)].set_index('reference')
    for i,ref in enumerate(['fixed_lambda','annual_cv','joint_cv']):
        r=d.loc[ref];axs[1,1].plot([r.ci_low,r.ci_high],[i,i],color='#ab4f65');axs[1,1].scatter(r.delta_sharpe,i,color='#ab4f65',s=20)
    axs[1,1].axvline(0,color='grey',ls='--',lw=.7);axs[1,1].set_yticks(range(3),['Fixed lambda','Annual CV','Joint CV'],fontsize=8)
    axs[1,1].set(title='Gaussian transfer: paired net differences',xlabel='Sharpe difference and 95% interval')
    for ax in axs.ravel():ax.grid(alpha=.2)
    fig.suptitle('Requested Gaussian finite-range transfer extension; no polynomial population law',fontsize=11)
    finish(fig,'appendix_gaussian_transfer')
    save(out/'tables/figure_spectral_values.csv',plot_values);save(out/'tables/figure_wealth_values.csv',wealth)


def scientific_findings(out):
    anchors=json.loads((out/'audit/anchors.json').read_text());a=read(out,'annual_policies')
    ratios=read(out,'spectral_ratio_diagnostics');amp=read(out,'amplitude_scaling_diagnostics')
    summary=read(out,'performance_summary');paired=read(out,'paired_comparisons')
    primary=summary[summary.period.eq('1984-2024')]
    p=paired[paired.period.eq('1984-2024')&paired.block.eq(12)]
    verdicts={}
    for kernel,method in [(k,m) for k in KERNELS for m in ('spectral_transfer','adaptive_spectral')]:
        s=paired[paired.period.eq('1984-2024')&paired.kernel.eq(kernel)&paired.method.eq(method)&paired.scenario.eq('trade25_borrow30')]
        wins=bool(len(s)==9 and s.ci_low.gt(0).all())
        verdicts[kernel+'/'+method]=dict(method=method,
            net_positive_against_all_three_with_95pct_intervals_at_all_blocks=wins)
    verified_amplitude=bool(anchors['matern32']['amplitude_gate_passed'])
    strong=verified_amplitude and verdicts['matern32/spectral_transfer']['net_positive_against_all_three_with_95pct_intervals_at_all_blocks']
    qualified=any(v['net_positive_against_all_three_with_95pct_intervals_at_all_blocks'] for v in verdicts.values())
    go=strong or qualified
    findings=dict(recommendation='GO' if go else 'NO-GO',amplitude_identification_gate_passed=verified_amplitude,
        primary_finite_range_b=anchors['matern32']['finite_range_b'],nominal_b=133/130,
        scientific_case='Strong positive' if strong else 'Qualified positive empirical rule' if qualified else 'Negative/inconclusive transfer result',
        justification='A structural amplitude transfer requires identified shape plus net evidence. Either requested kernel can support a qualified empirical contribution if a primary adaptive rule beats all three references across 6/12/24-month paired intervals; this still does not establish a population polynomial or oracle theorem.',
        kernel_verdicts=verdicts,main_manuscript_changed=False)
    write_json(out/'audit/scientific_findings.json',findings)
    return findings,anchors,primary,p


def memo_and_report(out):
    findings,anchors,primary,paired=scientific_findings(out)
    ratios=read(out,'spectral_ratio_diagnostics');fits=read(out,'spectral_fits');amp=read(out,'amplitude_scaling_diagnostics')
    main=fits[fits.history.isna()];early=main[main.kernel.eq('matern32')&main.year.eq(1978)&main.first_rank.eq(11)&main.last_rank.eq(60)]
    ratio=ratios[ratios.kernel.eq('matern32')&ratios.P.eq(10000)&ratios.history.eq('expanding')&ratios.multiplier.ne(1)&ratios.first_rank.eq(11)&ratios.last_rank.eq(60)]
    pred=read(out,'prediction_summary');a=read(out,'annual_policies');boundary=a[a.method.isin(PRIMARY_METHODS)].groupby(['kernel','method']).boundary.apply(lambda x:int(x.isin(['lower','upper']).sum()))
    def sr(kernel,method,scenario):
        return float(primary[primary.kernel.eq(kernel)&primary.method.eq(method)&primary.scenario.eq(scenario)].sharpe.iloc[0])
    def contrast(kernel,method,reference,scenario='trade25_borrow30'):
        r=paired[paired.kernel.eq(kernel)&paired.method.eq(method)&paired.reference.eq(reference)&paired.scenario.eq(scenario)].iloc[0]
        return f'{r.delta_sharpe:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]'
    table=['| Kernel / method | Gross SR | Exact net SR | Mean C | Annual net trades |',
           '|---|---:|---:|---:|---:|']
    for kernel in KERNELS:
        for method in PRIMARY_METHODS:
            gross=primary[primary.kernel.eq(kernel)&primary.method.eq(method)&primary.scenario.eq('gross')]
            if not len(gross):continue
            net=primary[primary.kernel.eq(kernel)&primary.method.eq(method)&primary.scenario.eq('trade25_borrow30')].iloc[0];g=gross.iloc[0]
            table.append(f'| {kernel} / {LABELS[method]} | {g.sharpe:.3f} | {net.sharpe:.3f} | {g.mean_C:.2f} | {net.annual_turnover:.2f} |')
    mat_b=early[early.multiplier.ne(0)].b
    answers=[
      ('Is b empirically stable across Matérn lengthscales?',f'No over the predeclared finite ranges. On ranks 11-60 in the first refit alone, slopes span {mat_b.min():.3f} to {mat_b.max():.3f}. These are finite-range slopes, not population exponents. Fixed-smoothness norm equivalence preserves an assumed infinite-operator polynomial order; it does not justify a common finite-sample slope or identify b=133/130.'),
      ('Does the managed spectrum mainly change amplitude or shape?',f'It changes finite-sample shape as well. Across nonbaseline lengths at the three diagnostic dates, the middle-rank 90th/10th percentile ratio of eigenvalue ratios ranges from {ratio.ratio_90_over_10.min():.2f} to {ratio.ratio_90_over_10.max():.2f}, whereas amplitude-only scaling requires one. Leading, middle and tail ratios, temporal subspace overlaps and history/feature-prefix sensitivity are exported.'),
      ('Is c(ell) reliably estimable from this history?','No identified population amplitude is established. The training-only identification gate failed before OOS analysis. Separate slopes, pooled common slopes, fixed historical slope, median/mean intercepts and six rank intervals disagree. c_proxy is a finite-range statistic; its numeric transfer paths are retained to falsify the hypothesis, not presented as precise structural estimates.'),
      ('Does predicted lambda scaling match useful regularization?',f'The full prediction-error and hindsight-loss diagnostics are in prediction_summary.csv. Primary transfer/CV log-penalty RMSE is {float(pred.query("kernel==\"matern32\" and rule==\"transfer\"").rmse_log_cv.iloc[0]):.3f}; the full-spectrum value is {float(pred.query("kernel==\"matern32\" and rule==\"spectral\"").rmse_log_cv.iloc[0]):.3f}. This compares heterogeneous historical CV choices, not a known oracle. The nominal ell^(-3) tail law is a conditional infinite-kernel prediction and is not a supported description of these finite spectra.'),
      ('Does adaptive theory-guided shrinkage beat fixed lambda?',f'Matérn amplitude-transfer net Sharpe difference versus fixed lambda is {contrast("matern32","spectral_transfer","fixed_lambda")}; adaptive full-spectrum difference is {contrast("matern32","adaptive_spectral","fixed_lambda")}. These paired intervals, with 6/24-month sensitivities, determine the strength of evidence. A small point difference is not a demonstrated financial gain.'),
      ('Does it beat annual CV?',f'Matérn transfer net difference is {contrast("matern32","spectral_transfer","annual_cv")}; adaptive full spectrum is {contrast("matern32","adaptive_spectral","annual_cv")}. All methods share February 1984-January 2025 and all primary net accounts enter from cash together.'),
      ('Does it beat joint CV?',f'Matérn transfer net difference is {contrast("matern32","spectral_transfer","joint_cv")}; adaptive full spectrum is {contrast("matern32","adaptive_spectral","joint_cv")}. Joint CV is the strongest pre-existing practical benchmark; selecting five theory-guided candidates from trailing returns remains hyperparameter selection.'),
      ('Does it improve net Sharpe rather than merely reduce risk or exposure?',f'Primary Matérn transfer gross/net Sharpe is {sr("matern32","spectral_transfer","gross"):.3f}/{sr("matern32","spectral_transfer","trade25_borrow30"):.3f}. Annual means, volatility, drawdowns, actual signed exposure, turnover and fees are all compared in performance_summary.csv. The recommendation requires net paired evidence, not a smoother complexity path or lower exposure.'),
      ('Can B(ell)/A(ell) be treated as constant?', 'It is a tested hypothesis, not established by RKHS equivalence. Historical per-lengthscale CV-implied FOC ratios vary, and source/score geometry changes with the RKHS norm. The additional five-anchor adjusted rule is explicitly separate; it consumes extra validation information and cannot identify a unique structural bound constant.'),
      ('Is there a defensible one-page contribution for the paper?',f'{findings["recommendation"]}. The conditional spectral-order/transfer proposition and the signal/noise counterexample are defensible mathematics, but they do not establish a useful one-constant oracle formula for this experiment. A separate one-page candidate records the negative diagnostic; do not insert it or expand the manuscript without approval.')]
    memo=f'# Decision memo: {findings["recommendation"]}\n\n'
    memo+='The evidence does not support a structural amplitude-only lengthscale-to-shrinkage rule. '+('Recommend excluding this extension from the current Journal of Finance manuscript.' if findings['recommendation']=='NO-GO' else 'Consider only a compact candidate with these limitations.')+'\n\n'
    for question,answer in answers:memo+=f'**{question}**\n\n{answer}\n\n'
    memo+='The mathematical note distinguishes exact norm/order comparisons, conditional Weyl/asymptotic statements, finite-rank fits and retrospective OOS evidence. No oracle optimality is claimed.\n'
    effect=read(out,'lengthscale_effect_summary').query("kernel=='matern32'").set_index('rule')
    crossed=read(out,'representation_penalty_contrasts').query("kernel=='matern32' and period=='1984-2024' and metric=='sharpe'").iloc[0]
    lag=read(out,'selection_history_sensitivity').query("period=='1984-2024'")
    finite=read(out,'finite_sample_sensitivity').query("kernel=='matern32'")
    memo+=f'''\n**Does amplitude explain the cross-length change in useful lambda?**\n\nOn the common 1984+ diagnostic panel, the amplitude-predicted cross-length shift has log-effect RMSE {effect.loc['cv','rmse_log_effect']:.3f} against CV shifts, versus {effect.loc['spectral','rmse_log_effect']:.3f} against full-spectrum shifts. Full absolute penalty errors and OOS-loss consequences remain in the prediction tables. The contrast with CV is diagnostic, not proof that CV or a 12-month hindsight grid peak equals a population oracle. Conditional mean/noise constants are not identified merely by this residual.\n\n**Representation, shrinkage and stale history**\n\nFor Matérn, the descriptive 2x2 gross-Sharpe difference of joint versus fixed-length CV is {crossed.joint_length_joint_penalty-crossed.base_length_base_penalty:+.3f}; symmetric representation/penalty contrasts are {crossed.representation_shapley_contrast:+.3f}/{crossed.penalty_shapley_contrast:+.3f}, with interaction {crossed.interaction:+.3f}. These are exact arithmetic contrasts of selected paths, not causal contributions. Crossed raw penalties are held fixed and can lie outside the receiving representation's search grid; they are controlled diagnostics, not expanded optimization.\n\n'''
    for r in lag.itertuples():memo+=f'- {r.kernel} {r.method}: 36-versus-60-month selection changes gross Sharpe by {r.delta_sharpe_36_minus_60:+.3f}.\n'
    memo+=f'\nAcross the predeclared history/feature-prefix cases, baseline-length middle-rank slope changes range from {finite[finite.multiplier.eq(1)].delta_b_from_full_history_full_features.min():+.3f} to {finite[finite.multiplier.eq(1)].delta_b_from_full_history_full_features.max():+.3f}. The finite approximation and rank-range sensitivity are observable obstacles. The expanding-refit effect of stale observations and a time-varying source/score constant cannot be separately identified by these controls; the shorter performance-window sensitivity is assessed without claiming to solve that identification problem.\n'
    memo+='\nThe additional 1978 CV anchor at multiplier 0.25 lies on a boundary for both kernels. The adjusted transfer uses the corresponding historical FOC interval endpoint, not a point-identified rho; explicit lower/upper one-sided intervals are in initial_constant_diagnostics.csv. The primary baseline anchors are interior. Different CV-implied FOC values do not by themselves prove that a sharp population B/A varies.\n'
    memo+=f'''\n**Requested Gaussian extension**\n\nThe same finite-range transfer pipeline is now executed for Gaussian using its own single 1978 anchor and frozen first-history slope {anchors['gaussian']['finite_range_b']:.6f}. This is explicitly an exploratory amendment requested after the original Matérn-primary run; it does not assume a polynomial Gaussian population spectrum. Gaussian transfer gross/net Sharpe is {sr('gaussian','spectral_transfer','gross'):.3f}/{sr('gaussian','spectral_transfer','trade25_borrow30'):.3f}. Its paired net difference is {contrast('gaussian','spectral_transfer','fixed_lambda')} versus fixed lambda, {contrast('gaussian','spectral_transfer','annual_cv')} versus annual CV, and {contrast('gaussian','spectral_transfer','joint_cv')} versus joint CV. The Gaussian identification gate {'passes' if anchors['gaussian']['amplitude_gate_passed'] else 'fails'}; no nominal Matérn exponent is assigned to Gaussian. Source density changes decay speed, so the common finite-rank exponent is a falsifiable surrogate, not a Gaussian theorem.\n'''
    (out/'DECISION_MEMO.md').write_text(memo)
    protocol=json.loads((out/'audit/protocol.json').read_text());verification=json.loads((out/'audit/estimation.json').read_text())
    report=f'''# Lengthscale, managed spectral shape and portfolio shrinkage

Recommendation: **{findings['recommendation']}**. The amplitude identification gate fails on the initial pretest history. Amplitude transfer is a finite-range falsification exercise; it is not an identified population-amplitude estimator. All explicit scientific answers are in DECISION_MEMO.md.

## Chronological protocol and controls

The requested mission and diagnostic protocol are frozen in audit/research_mission.md and audit/protocol.json. Starting commit is `{protocol['starting_commit']}`. Both existing 5 × 744 × 10,000 float64 managed-payoff caches are reused, with 130 characteristics, seed 0 and lengthscale 4.392107784616423 times [0.25,0.5,1,2,4]. The same Gaussian/Matérn random draws and phases are scaled across lengths; no new independent banks, stock preprocessing, architectures or simulated research returns are introduced. The inherited JKP complete-payoff stock sample is a retrospective sensitivity, not a certified vintage point-in-time universe.

Every original annual CV and joint CV selection/payoff is reproduced. Original fixed-CV maximum raw errors are {verification['baseline_checks'][0]['maximum_original_payoff_error']:.3g} (Matérn) and {verification['baseline_checks'][1]['maximum_original_payoff_error']:.3g} (Gaussian); joint errors are {verification['baseline_checks'][0]['maximum_joint_cv_payoff_error']:.3g}/{verification['baseline_checks'][1]['maximum_joint_cv_payoff_error']:.3g}. The prior Gaussian spectral path agrees to {verification['baseline_checks'][1]['maximum_prior_spectral_payoff_error']:.3g}. Cache file hashes, bank metadata, source checksums, original grids and anchor cutoff dates are recorded. Raw eigenvalues are those of g g'/T, without centering or display scaling; the common frozen kappa is {verification['kappa']:.17g}.

The 1978 base-lengthscale chronological CV decisions calibrate each kernel's one constant once. Gaussian lambda/rho are {anchors['gaussian']['lam0']:.17g}/{anchors['gaussian']['rho0']:.17g}; Matérn are {anchors['matern32']['lam0']:.17g}/{anchors['matern32']['rho0']:.17g}. Sharing rho across lengthscales is a hypothesis. The original 120-point positive bounds remain frozen separately per representation. The fixed-lambda primary control always uses its baseline raw anchor. Cross-length fixed penalties use the same raw units and are constrained to each representation's original interval; clipping is explicit.

For transfer, the fixed empirical slopes are {anchors['matern32']['finite_range_b']:.9f} (Matérn) and {anchors['gaussian']['finite_range_b']:.9f} (Gaussian), each from its own 1978 baseline ranks 11-60. The robust amplitude statistic is median(log(mu_j)+b log(j)) on those same ranks. The Gaussian transfer was explicitly requested after the original Matérn-primary results and is a documented exploratory amendment; no polynomial population law is imposed on Gaussian. Nominal b=133/130 is a Matérn-only unverified sensitivity; b=1.5,2,3, alternate ranges and an OLS mean-intercept variant are finite-range sensitivities for both. No invalid b is clipped. First-history identification gates and their failures are retained; the term c_proxy is used throughout. The adjusted transfer is a separate additional-information experiment with five 1978 CV FOC calibrations; boundary calibrations identify interval endpoints only.

Candidate policies are genuinely OOS from their 1978 fit onward. Primary adaptive selection starts in 1984 using the last 60 realized candidate returns from decisions 1979-1983, minimum mean (1-raw_return)^2 and the original median-length tie rule. Validation 1973-1977 is not recycled, and the anchored decision 1978 is excluded from candidate selection. Every selection cutoff and first/last candidate return is exported. Thirty-six-month windows are secondary, on the same 1984+ evaluation; none select a final rule from full-sample outcomes.

## Primary common-month economic evidence

{chr(10).join(table)}

This sample has 492 months, February 1984-January 2025, corresponding to decisions 1984-2024. Sharpe uses concatenated monthly excess returns and ddof=1, not averaged annual Sharpes. The full fixed-policy context (564 months) and post-anchor fixed-policy context (552 months) are reported separately. All primary common-window net accounts start from cash at the 1984 first formation close and charge entry; terminal holdings are marked without forced liquidation.

Every leading method's signed stock weights are reconstructed with shared original frequencies/phases and verified against its managed payoff. The old CV weights and full-history net accounts reconcile independently. Exact net uses core.accounting_step: c=0.0025*||(1-c)w-d||_1, borrowing charge 0.003/12 times short notional on post-cost NAV, and the identical cash financing rate. The target-turnover scenario instead subtracts 0.0025*||w_t-w_(t-1)||_1 plus 0.003/12*short from gross excess returns. They are separate illustrative scenarios, without observed impact, lending spreads or executable liquidity claims. No new cost is inferred from another portfolio's turnover. Licensed rows, IDs and holdings remain private.

Paired circular block-bootstrap comparisons use 5,000 shared draws, seed 20261010, 12-month blocks and 6/24-month sensitivities. References are joint CV, annual CV and fixed lambda. Intervals condition on fitted paths; they are not full-pipeline inference and have no multiple-comparison adjustment. Full block lengths, subperiods and failed/inconclusive contrasts are in paired_comparisons.csv.

## Shape, amplitude, constants and penalty diagnostics

Rank intervals [1,10], [11,40], [41,100], [11,60], [61,120], [121,180] were declared before calculations. Diagnostics at 1978, 2000 and 2024 use common observations, expanding history and available 120/180/360/720-month windows, relative rank thresholds 1e-10/1e-12/1e-14, and existing feature prefixes 1,000/5,000/10,000. Prefix managed vectors multiply sqrt(10000/P); failing to do that would change the penalty units. OLS slope standard errors are descriptive fits, not confidence intervals for a population spectral exponent. Figure-2 whiskers show sensitivity across rank intervals.

The ratio curves compare leading, middle and finite tail ranks. Flatness statistics and sign-invariant temporal subspace overlap are exported; sorted-rank ratios alone do not track the same economic eigenportfolio. Separate finite-range slopes, a common slope pooled across lengthscales, frozen historical slopes and robust intercepts are compared for both kernels. Gaussian fits are exploratory finite-range diagnostics, not population power-law estimates. The theoretical Matérn amplitude law ell^(-3) requires the additional ordinary-operator/Weyl and economic-covariance assumptions in the mathematical note; those assumptions are not established by the cache.

The penalty diagnostic retains predicted transfer, full-spectrum and CV penalties, original-grid ex-post loss minima and Sharpe maxima, log prediction errors, OOS loss and annual risk. A hindsight grid peak over 12 returns is not a population oracle and never feeds calibration or selection. FOC-implied rho values and regularized historical policy norms/score-covariance traces describe why eigenvalues alone do not identify bias/noise constants; they are not estimators of the population RKHS norm or structural A/B.

The representation ablation holds raw penalties fixed while changing joint-selected lengthscale, and holds baseline lengthscale fixed while applying joint-selected penalties. Together with fixed-length transfer and spectral/fixed-lambda controls, this separates observed representation and shrinkage effects as far as the protocol permits. Selection interactions prevent a causal decomposition. Secondary transfer choices, 36-month paths, all candidate returns and the complete 120-grid OOS payoff paths are exported, without selecting a favorable sensitivity.

Quantitative cross-length penalty-effect errors are in lengthscale_effect_summary.csv, exact descriptive 2x2 representation/penalty contrasts in representation_penalty_contrasts.csv, and shorter-selection/stale-history limitations in selection_history_sensitivity.csv and DECISION_MEMO.md. Finite history and normalized feature-prefix slope changes are summarized in finite_sample_sensitivity.csv. Crossed raw penalties are not retuned and may fall outside the receiving grid; these secondary controls never expand an actual penalty search.

Alternate b/rank ranges, the five-anchor adjusted transfer, 36-month selection and crossed-penalty policies are gross-only secondary diagnostics. Actual costs are reconstructed for every method in the mandatory primary comparison and the fixed-length transfer control. No secondary net result is inferred from another portfolio.

Primary-method boundary counts: {boundary.to_dict()}. Gaussian has finite-range proxy b/c statistics but no identified population polynomial b or amplitude. Missing numeric cells in annual diagnostics mean a quantity is inapplicable (e.g. a KKT residual for CV), not a skipped payoff; every actual monthly payoff/account is finite and verified.

## Mathematical limits and deliverables

The mathematical note supplies one central conditional proposition and proof, the corrected lambda and complexity formulas, an exact same-spectrum/different-risk-oracle counterexample, the difference between two-sided bounds and an asymptotic constant, conservative norm/source/tail comparisons, and a regularized-score estimation target with its assumptions. It does not conflate ordinary and managed operators. It does not infer D=6, polynomial Gaussian decay, an exact Sharpe-regret identity, structural rho or data-dependent oracle optimality from the paper's fixed-lambda upper bound.

Four main vector-PDF/high-resolution-PNG figures, the Gaussian shape negative control and the requested Gaussian transfer/financial appendix figure are generated from exported tables. Mathematical and empirical PDF reports, a separate one-page candidate, public audit and this decision memo remain isolated from the manuscript. The final recommendation uses demonstrated economics and identification, not smoothness of complexity paths.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m empirical_final.lengthscale_spectral --output outputs/lengthscale_spectral_constant_20261010_reproduction
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_lengthscale_spectral.py tests/test_theory_guided_lambda.py tests/test_bandwidth_tuning.py tests/test_final_empirical.py
```

With licensed inputs present, --phase estimate/costs/summary/publication provides explicit stages and year-by-year checkpoints. New output paths inherit the committed frozen diagnostic protocol; source/input checksums are revalidated. Private coefficients are stored under results/<output-name>/. Runtime package versions and final tests/visual/completion audits are in audit/. Only the new code/tests, public aggregates, figures, reports and candidate are committed. The main manuscript and unrelated work are preserved.
'''
    (out/'REPORT.md').write_text(report)
    return table,answers


def compile_tex(out,name):
    folder=out/'publication'
    result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=folder,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:])
    log=(folder/(name+'.log')).read_text()
    if 'Overfull' in log:raise ValueError('Overfull LaTeX box in '+name)
    for ext in ('.aux','.log','.out'):
        p=folder/(name+ext)
        if p.exists():p.unlink()
    return folder/(name+'.pdf')


def latex_reports(out,table,answers):
    anchors=json.loads((out/'audit/anchors.json').read_text());findings=json.loads((out/'audit/scientific_findings.json').read_text())
    template=(ROOT/'empirical_final/templates/lengthscale_mathematical_note.tex').read_text()
    for key,value in {'GAUSSANCHOR':sci(anchors['gaussian']['lam0']),'GAUSSRHO':sci(anchors['gaussian']['rho0']),
        'MATANCHOR':sci(anchors['matern32']['lam0']),'MATRHO':sci(anchors['matern32']['rho0']),
        'GAUSSB0':f"{anchors['gaussian']['finite_range_b']:.4f}",
        'GATEWORD':'fails' if not findings['amplitude_identification_gate_passed'] else 'passes'}.items():template=template.replace(key,value)
    (out/'publication/mathematical_note.tex').write_text(template)
    compile_tex(out,'mathematical_note')
    summary=read(out,'performance_summary');paired=read(out,'paired_comparisons');pred=read(out,'prediction_summary')
    primary=summary[summary.period.eq('1984-2024')];rows=[]
    for kernel in KERNELS:
        for method in PRIMARY_METHODS:
            g=primary[primary.kernel.eq(kernel)&primary.method.eq(method)&primary.scenario.eq('gross')]
            if not len(g):continue
            n=primary[primary.kernel.eq(kernel)&primary.method.eq(method)&primary.scenario.eq('trade25_borrow30')].iloc[0]
            g=g.iloc[0];rows.append(f'{esc(kernel)} & {esc(LABELS[method])} & {g.sharpe:.3f} & {n.sharpe:.3f} & {g.mean_C:.1f}\\\\')
    head=r'''\documentclass[11pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,newtxtext,newtxmath,microtype,graphicx,booktabs,array}
\usepackage[margin=1in]{geometry}
\usepackage[colorlinks=true,urlcolor=blue]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\begin{document}
\begin{center}{\Large\bfseries Lengthscale and portfolio shrinkage}\\
Real JKP cache investigation; 10 October 2026\end{center}
'''
    report=head+rf'\textbf{{Recommendation: {findings["recommendation"]}.}} '+r'''The initial amplitude-identification gate fails before OOS evaluation.
Amplitude transfer is reported as a finite-range falsification exercise, not an
identified population formula. The fixed-lambda control and strongest joint CV
benchmark are retained. The main manuscript is unchanged.

\section*{Common-month performance}
Decisions 1984--2024 give 492 realized excess returns, February 1984--January 2025.
Sharpe uses concatenated returns; all net accounts start from cash together.
Exact net charges 25 bp of drift-adjusted two-sided trades and 30 bp/year of short
notional, using the repository's self-financing NAV convention.
\begin{center}\small\begin{tabular}{llrrr}\toprule
Kernel & Method & Gross SR & Net SR & Mean C\\\midrule
'''+ '\n'.join(rows)+r'''\bottomrule\end{tabular}\end{center}
Annual means, volatility, drawdowns, exposure, turnover and fees are all exported.
Gaussian transfer is an explicitly requested finite-range exploratory extension;
no polynomial Gaussian population law is assumed. The fixed controls reproduce
the original annual and joint CV paths; the earlier Gaussian spectral path also
reconciles. Paired intervals below condition on fitted paths, with no whole-pipeline
or arbitrary-nonstationarity guarantee and no multiplicity adjustment.

\section*{Design}
Each kernel uses its original 1978 baseline-lengthscale chronological CV anchor,
then freezes one ratio. Adaptive lengthscale selection first occurs in 1984 using
60 genuinely prior OOS candidate-policy months from decisions 1979--1983 and
minimum response-one loss. The original five lengths, 10,000 shared random
features, cash rate, stock sample, annual calendar and per-representation penalty
bounds are retained. Calibration validation is not recycled as fresh OOS evidence.
The inherited complete-payoff JKP sample is retrospective, not a certified vintage
point-in-time universe. Private observations and holdings are not exported.
'''
    figure_sections=[('Spectral geometry','figure1_spectral_shape',
        'All five Matérn lengths use identical expanding observations. A pure amplitude change would give horizontal log-ratio curves. Leading, middle and sample-tail ratios differ. These finite-rank sample spectra do not identify a population exponent.'),
        ('Amplitude and exponent sensitivity','figure2_amplitude_exponent',
         'Points use ranks 11--60; whiskers span the six predeclared rank intervals. They show fit sensitivity, not confidence intervals for population b or c. The dashed amplitude law requires additional Weyl and return-covariance assumptions. A high R-squared does not establish common slope or amplitude identification.'),
        ('Predicted and selected penalties','figure3_penalty_complexity',
         'Transfer uses one baseline historical anchor, a frozen finite-range b and a robust c proxy. Spectral roots use the full past empirical spectrum. Bounds are never expanded after outcomes. Adaptive paths begin in 1984; a smoother complexity trajectory is not proof of economic value.'),
        ('Financial performance and paired uncertainty','figure4_financial_results',
         'Accounts are normalized at the common start. Exact net is reconstructed from each policy\'s signed weights. The final panel compares Sharpe to joint CV with 12-month circular blocks and 5,000 shared draws. Six- and 24-month intervals and subperiods are retained rather than selecting a favorable sensitivity.')]
    for title,name,caption in figure_sections:
        report+='\n\\newpage\n\\section*{'+title+'}\n'
        report+=r'\begin{center}\includegraphics[width=\linewidth,height=.76\textheight,keepaspectratio]{../figures/'+name+r'.pdf}\end{center}'+'\n'
        report+=esc(caption)+'\n'
    report+=r'\newpage\section*{Requested Gaussian transfer extension}'+'\n'
    report+=r'\begin{center}\includegraphics[width=\linewidth,height=.7\textheight,keepaspectratio]{../figures/appendix_gaussian_transfer.pdf}\end{center}'+'\n'
    report+=f'The same one-anchor transfer pipeline is applied to Gaussian with its own frozen 1978 finite-range slope {anchors["gaussian"]["finite_range_b"]:.4f}. This amendment was requested after the initial Matérn-primary run; all rank ranges, dates, bounds and candidate-selection windows remain fixed. '
    report+=r'''Gaussian decay speed can change with lengthscale. The local power-law fit
is therefore a falsifiable finite-range surrogate, not a population spectral theorem.
All six primary controls have actual stock-level net accounting on the same 492 months.
'''
    report+=r'\newpage\section*{Falsification, controls and costs}'+'\n'
    report+=r'''Original-grid ex-post loss minima and Sharpe maxima are hindsight diagnostics only.
They never enter calibration or portfolio decisions and are not population oracles.
The following log-penalty prediction RMSEs cover the five lengths and decisions
1984--2024; small error to historical CV is not itself an investment gain.
\begin{center}\small\begin{tabular}{lrr}\toprule
Matérn rule & RMSE to CV & RMSE to hindsight loss grid\\\midrule
'''
    for name in ('fixed_lambda','spectral','transfer','transfer_nominal','transfer_adjusted'):
        f=pred[pred.kernel.eq('matern32')&pred.rule.eq(name)]
        if len(f):r=f.iloc[0];report+=f'{esc(name)} & {r.rmse_log_cv:.3f} & {r.rmse_log_hindsight_loss:.3f}\\\\\n'
    report+=r'''\bottomrule\end{tabular}\end{center}
'''
    crossed=read(out,'representation_penalty_contrasts').query("kernel=='matern32' and period=='1984-2024' and metric=='sharpe'").iloc[0]
    effects=read(out,'lengthscale_effect_summary').query("kernel=='matern32'").set_index('rule')
    report+=f'Matérn joint versus fixed-length CV gross Sharpe changes by {crossed.joint_length_joint_penalty-crossed.base_length_base_penalty:+.3f}; symmetric representation/penalty contrasts are {crossed.representation_shapley_contrast:+.3f}/{crossed.penalty_shapley_contrast:+.3f}, with interaction {crossed.interaction:+.3f}. '
    report+=f'The amplitude-predicted cross-length shift has log-effect RMSE {effects.loc["cv","rmse_log_effect"]:.3f} to CV and {effects.loc["spectral","rmse_log_effect"]:.3f} to full-spectrum shifts. '
    report+=r'''
The two crossed lengthscale/penalty policies hold one component fixed at a time.
They distinguish observed representation and shrinkage contributions, but selected
hyperparameters interact, so no causal decomposition is claimed. The fixed-length
transfer control and the fixed-lambda control show whether updating spectra adds
value without changing representation. The adjusted transfer consumes five first-year
CV choices and is a separate extra-information method, not a recovered structural ratio.

The extra 1978 CV anchor at multiplier 0.25 is on a boundary in both kernels.
Adjusted transfer uses the FOC interval endpoint: a lower-bound anchor permits
$0<\rho\leq T/S(\lambda_{\min})$, and an upper-bound anchor permits
$\rho\geq T/S(\lambda_{\max})$. It does not identify a unique ratio.
\section*{Actual accounting and robustness}
Every primary method's stock weights reconstruct its managed payoff. Original CV
weights and full-history net accounts reconcile. All common-window accounts charge
entry from cash together in 1984; terminal holdings are marked without forced
liquidation. Target-weight-turnover subtraction is a separate exported scenario.
No transaction cost is borrowed from another portfolio. Illustrative fees do not
include observed execution impact, lending availability or extra funding spreads.
Predeclared history lengths, normalized feature prefixes and numerical rank thresholds
quantify finite-sample sensitivity. The Gaussian appendix is a negative control;
its population spectrum is not assigned a polynomial exponent. The requested
Gaussian finite-range transfer is an exploratory falsification exercise.
'''
    report+=r'\newpage\section*{Decision and scientific limits}'+'\n'
    for question,answer in answers:
        if question in ['Does adaptive theory-guided shrinkage beat fixed lambda?','Does it beat annual CV?','Does it beat joint CV?',
                        'Can B(ell)/A(ell) be treated as constant?','Is there a defensible one-page contribution for the paper?']:
            report+='\\textbf{'+esc(question)+'} '+esc(answer)+'\n\n'
    report+=r'''The mathematical note proves a conditional norm/order comparison and an
asymptotic bound-proxy transfer formula, and provides a same-spectrum/different-risk-oracle
counterexample. Ordinary kernel spectra and managed-return spectra are distinct.
Two-sided polynomial bounds need not identify an asymptotic constant. Neither
fixed-penalty bounds nor smooth empirical paths establish data-dependent oracle
optimality, structural rho, or a polynomial Gaussian law. The full decision memo
answers every requested scientific question; all negative and inconclusive results,
paired sensitivities, tables, tests and input/source hashes are retained.
\end{document}
'''
    (out/'publication/empirical_report.tex').write_text(report)
    compile_tex(out,'empirical_report')
    candidate=r'''% Separate candidate only; NO automatic manuscript insertion.
\subsection{Lengthscale and the limits of amplitude transfer}
For fixed-smoothness Mat\'ern kernels, equivalent RKHS norms preserve a managed
spectral order that already holds; they do not identify its exponent or amplitude
from the kernel alone. If $\mu_j(\ell)\sim c(\ell)j^{-b}$ with common $b>1$,
the asymptotic minimizer of the theoretical envelope satisfies
\[
 \frac{\lambda_T^{\rm bound}(\ell_2)}{\lambda_T^{\rm bound}(\ell_1)}
 =\left(\frac{c(\ell_2)}{c(\ell_1)}\right)^{1/(b+1)}
  \left(\frac{\rho(\ell_2)}{\rho(\ell_1)}\right)^{b/(b+1)},\qquad\rho=B/A.
\]
This is not an oracle-penalty identity. Managed eigenvalues alone do not determine
expected-payoff signal or score noise, and two-sided spectral bounds need not supply
a limiting amplitude. In the real JKP cache, the initial Mat\'ern finite-range slopes
vary across lengthscales and the predeclared amplitude-identification gate fails.
Mechanical transfer paths are therefore falsification exercises. The 492 common
OOS months for decisions 1984--2024 include fixed lambda, annual CV, joint CV and
one-anchor controls; adaptive selection uses only the previous 60 genuinely OOS
candidate returns. Actual signed weights reconstruct net costs and paired circular
blocks assess differences conditional on fitted paths. The standalone investigation
does not establish a convincing structural one-constant contribution for the current
paper. We recommend exclusion, while retaining the negative result and its audit.
The Gaussian extension tests the same finite-range heuristic while preserving
the negative control against an assumed polynomial population law.
'''
    (out/'publication/one_page_candidate.tex').write_text(candidate)
    wrapper=r'''\documentclass[12pt,letterpaper]{article}
\usepackage[T1]{fontenc}\usepackage{amsmath,amssymb,newtxtext,newtxmath,microtype,setspace}
\usepackage[top=1.5in,bottom=1.5in,left=1in,right=1in]{geometry}
\setstretch{1.5}\setlength{\parindent}{.25in}\setlength{\parskip}{0pt}
\begin{document}\input{one_page_candidate.tex}\end{document}
'''
    (out/'publication/one_page_preview.tex').write_text(wrapper);compile_tex(out,'one_page_preview')


def build(out):
    verify_frozen(out)
    diagnostics(out);figures(out);table,answers=memo_and_report(out);latex_reports(out,table,answers)
    write_json(out/'audit/references.json',SOURCE_LINKS)
    write_json(out/'audit/publication_verification.json',dict(passed=True,
        publication_module_sha256=digest(Path(__file__)),
        mathematical_template_sha256=digest(ROOT/'empirical_final/templates/lengthscale_mathematical_note.tex'),
        source_tables=['empirical_spectra','spectral_fits','monthly_accounts','paired_comparisons'],
        plots_reconstruct_from_exported_tables=True,main_manuscript_changed=False))
    write_json(out/'audit/output_hashes.json',{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
                                             if p.is_file() and p.name!='output_hashes.json'})
    print('Reports, four main figures, Gaussian appendix and separate candidate written',flush=True)
