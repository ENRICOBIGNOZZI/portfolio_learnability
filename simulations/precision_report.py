"""Report every rank and fit window of the paired precision pilot."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from .asymptotic_dense import write_json
from .asymptotic_report import draw, X_LABEL
from .report import COLORS, matrix

METRICS=[('SR_T',-2/3),('C_T',1/3),('SR_C',-2),('C_lambda',-.5)]
TITLES={'SR_T':r'Sharpe gap vs. $T_{\mathrm{train}}$',
        'C_T':r'Complexity vs. $T_{\mathrm{train}}$',
        'SR_C':'Sharpe gap vs. complexity',
        'C_lambda':r'Complexity vs. $\lambda$ (fixed spectrum)'}


def select_rank(config, certificates, spectra):
    """Use only certified approximation error, never slopes or fit quality."""
    checks=[]
    constant=config['precision_frozen_C0']
    low=constant*config['spectrum_fit_T_max']**(-2/3)
    high=constant*config['spectrum_fit_T_min']**(-2/3)
    for rank in sorted(config['precision_ranks']):
        c=certificates[(certificates['rank']==rank)&(certificates.lambda_multiplier==1)]
        s=spectra[(spectra['rank']==rank)&(spectra.lambda_reg>=low*(1-1e-10))&
                  (spectra.lambda_reg<=high*(1+1e-10))]
        a=float(c.relative_C_error_bound.max());b=float(s.relative_C_error_bound.max())
        checks.append(dict(rank=rank,maximum_balanced_relative_C_error_bound=a,
            maximum_fit_window_relative_C_error_bound=b,
            passes_certified_tolerance=bool(max(a,b)<=config['precision_relative_C_error_tolerance'])))
    eligible=[c['rank'] for c in checks if c['passes_certified_tolerance']]
    chosen=min(eligible) if eligible else None
    return dict(selected_rank=chosen,checks=checks,
        relative_C_error_tolerance=config['precision_relative_C_error_tolerance'],
        selection_uses_slopes_or_R_squared=False,
        benchmark_curve_rank=chosen if chosen is not None else max(config['precision_ranks']),
        note='If no rank certifies the tolerance, the highest studied rank only anchors benchmark curves; it is not declared numerically certified.')


def balanced(ax,frame,y,label,color,horizontal_T_train=None):
    """Keep each Sharpe observation paired with complexity at the same T."""
    cs=matrix(frame,'T_train','complexity')
    values=matrix(frame,'T_train',y)
    x=cs.mean(axis=0).to_numpy();avg=values.mean(axis=0).to_numpy()
    sd=values.std(axis=0,ddof=1).to_numpy()
    p95=values.quantile(.95,axis=0,interpolation='linear').to_numpy()
    ax.plot(x,avg,'o-',color=color,label=label,markersize=3,linewidth=1.3)
    ax.fill_between(x,avg-sd,avg+sd,color=color,alpha=.16)
    ax.plot(x,p95,color=color,linestyle=(0,(4,2)),linewidth=1,gid='trial_percentile_95')
    anchors=np.ones(len(x),dtype=bool) if horizontal_T_train is None else cs.columns.isin(horizontal_T_train)
    ax.errorbar(x[anchors],avg[anchors],xerr=cs.std(axis=0,ddof=1).to_numpy()[anchors],fmt='none',
                ecolor=color,linewidth=.6,alpha=.65,capsize=1.5)
    ax.set_xscale('log')
    if y=='SR_gap':
        if np.all(avg-sd>0):
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog',linthresh=max(np.max(np.abs(avg))/100,1e-8))


def build(output):
    out=Path(output)
    config=json.loads((out/'config.json').read_text())
    ranks=config['precision_ranks']
    frames=[];certificates=[];spectra=[];fits={};rows=[]
    for rank in ranks:
        # The report requires every prespecified rank, not an early winner.
        json.loads((out/f'precision_nystrom{rank}_completed.json').read_text())
        frames.append(pd.read_csv(out/f'precision_nystrom{rank}_trials.csv'))
        certificates.append(pd.read_csv(out/f'precision_nystrom{rank}_certificates.csv'))
        spectra.append(pd.read_csv(out/f'precision_nystrom{rank}_spectrum.csv'))
        fits[rank]=json.loads((out/f'precision_nystrom{rank}_fits.json').read_text())
        for key,fit in fits[rank].items():
            metric,cutoff=key.rsplit('_Tmin',1)
            rows.append(dict(rank=rank,metric=metric,T_train_min=int(cutoff),
                T_train_max=max(config['T_train']),expected_exponent=fit['expected_exponent'],
                exponent=fit['exponent'],exponent_ci_low=fit['exponent_ci95'][0],
                exponent_ci_high=fit['exponent_ci95'][1],fixed_power_R_squared=fit['r_squared'],
                fit_points=fit['fit_points']))
    frame=pd.concat(frames,ignore_index=True);cert=pd.concat(certificates,ignore_index=True)
    spectrum=pd.concat(spectra,ignore_index=True);rates=pd.DataFrame(rows)
    selection=select_rank(config,cert,spectrum)
    write_json(out/'precision_selection.json',selection)
    rates.to_csv(out/'precision_window_sensitivity.csv',index=False)
    summary=frame.groupby(['rank','lambda_multiplier','T_train']).agg(
        SR_mean=('SR_hat','mean'),SR_std=('SR_hat','std'),SR_p95=('SR_hat',lambda v:v.quantile(.95)),
        gap_mean=('SR_gap','mean'),gap_std=('SR_gap','std'),gap_p95=('SR_gap',lambda v:v.quantile(.95)),
        C_mean=('complexity','mean'),C_std=('complexity','std'),C_p95=('complexity',lambda v:v.quantile(.95)),
        replications=('trial','count')).reset_index()
    summary.to_csv(out/'precision_summary.csv',index=False)
    chosen=frame[frame.lambda_multiplier==1]
    primary=config['asymptotic_T_min'];anchor=selection['benchmark_curve_rank']
    figures=out/'figures';figures.mkdir(exist_ok=True)
    stems=[]
    ticks=config['T_train']
    with PdfPages(out/'precision_plots.pdf') as pages:
        def save(fig,stem,footer):
            fig.text(.5,.012,footer,ha='center',fontsize=7,color='#444444',linespacing=1.5)
            for j,ax in enumerate(fig.axes):
                title=ax.get_title();ax.set_title('')
                ax.set_title(f'({chr(97+j)}) {title}',loc='left',pad=6)
                ax.set_axisbelow(True);ax.grid(axis='y',color='#dddddd',linewidth=.4,alpha=.6)
            fig.tight_layout(rect=(0,.105,1,.98),pad=.7,w_pad=1.7,h_pad=1.6)
            pages.savefig(fig);fig.savefig(figures/(stem+'.png'),dpi=300)
            stems.append(stem);plt.close(fig)

        def train_axis(ax):
            ax.set_xscale('log',base=2);ax.set_xlabel(X_LABEL)
            ax.set_xticks(ticks,[str(t) for t in ticks],rotation=35)
            ax.xaxis.set_minor_formatter(NullFormatter())

        def benchmark(ax,metric,level=False):
            fit=fits[anchor][f'{metric}_Tmin{primary}']
            prediction=np.asarray(fit['prediction'])
            ax.plot(fit['x'],3-prediction if level else prediction,'k--',linewidth=1.2,
                    label=rf'Benchmark $p={fit["expected_exponent"]:.3f}$')

        footer=(f'{config["trials"]} paired pilot replications; N={config["fixed_N"]}, '
                rf'$\rho={config["fixed_rho"]}$. Solid: mean; shading/bars: $\pm$1 SD; colored dashes: empirical 95th percentile.'
                '\nComplexity and Sharpe are paired at the same training T; the Sharpe percentile is placed at mean complexity.')
        fig,axes=plt.subplots(1,2,figsize=(7.6,3.9))
        for rank,color in zip(ranks,COLORS):
            f=chosen[chosen['rank']==rank];label=f'Nyström r={rank}'
            draw(axes[0],f,'T_train','SR_hat',label,color)
            balanced(axes[1],f,'SR_hat',label,color)
        benchmark(axes[0],'SR_T',True);benchmark(axes[1],'SR_C',True)
        train_axis(axes[0]);axes[0].set(ylabel=r'Annualized $SR(\widehat W)$',title='Sharpe recovery')
        axes[1].set(xlabel=r'Effective complexity $\widehat C(\lambda_T)$',ylabel=r'Annualized $SR(\widehat W)$',title='Balanced path')
        for ax in axes:
            ax.axhline(3,color='#777777',linestyle=':',linewidth=.8);ax.legend(fontsize=6.5)
        save(fig,'01_precision_sharpe',footer)

        fig,axes=plt.subplots(2,2,figsize=(7.6,6.4))
        for ax,(metric,_) in zip(axes.flat,METRICS):
            for rank,color in zip(ranks,COLORS):
                f=chosen[chosen['rank']==rank];label=f'Nyström r={rank}'
                if metric=='SR_C':
                    balanced(ax,f,'SR_gap',label,color)
                elif metric=='C_lambda':
                    draw(ax,spectrum[spectrum['rank']==rank],'lambda_reg','complexity',label,color,True)
                else:
                    draw(ax,f,'T_train','SR_gap' if metric=='SR_T' else 'complexity',label,color,True)
            benchmark(ax,metric)
            if metric in ['SR_T','C_T']:
                train_axis(ax)
            else:
                ax.set_xscale('log');ax.set_xlabel(r'Effective complexity $\widehat C(\lambda_T)$' if metric=='SR_C' else r'Regularization $\lambda$')
            ax.set_ylabel(r'$SR^\star-SR(\widehat W)$' if metric.startswith('SR') else r'Effective complexity $\widehat C$')
            ax.set_title(TITLES[metric]);ax.legend(fontsize=6.5)
        status='certified rank' if selection['selected_rank'] is not None else 'highest studied rank; tolerance not certified'
        save(fig,'02_precision_rates',footer.split('\n')[0]+'\n'+
            f'Benchmark amplitude: {status} {anchor}; main fit starts at T_train={primary}; spectrum T_train={config["spectrum_T_train"]}.')

        fig,axes=plt.subplots(2,2,figsize=(7.6,6.4))
        for ax,(metric,target) in zip(axes.flat,METRICS):
            for rank,color in zip(ranks,COLORS):
                f=rates[(rates['rank']==rank)&(rates.metric==metric)].sort_values('T_train_min')
                ax.plot(f.T_train_min,f.exponent,'o-',color=color,markersize=3,label=f'Nyström r={rank}')
            ax.axhline(target,color='black',linestyle='--',linewidth=1,label='Theoretical exponent')
            ax.axvline(primary,color='#777777',linestyle=':',linewidth=.8)
            ax.set(xscale='log',xlabel=r'Lower training cutoff $T_{\mathrm{train,min}}$',ylabel='Fitted exponent',title=TITLES[metric])
            cuts=config['sensitivity_T_min'];ax.set_xticks(cuts,[str(t) for t in cuts],rotation=35)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.legend(fontsize=6.5)
        save(fig,'03_precision_windows',
            'Every prespecified cutoff is shown; no window or rank is selected by its slope or fit quality.\n'
            'For complexity vs. lambda, the cutoff defines a lambda interval on the same fixed training spectrum.')
    write_json(out/'precision_plot_audit.json',dict(replications=config['trials'],ranks=ranks,
        full_training_grid=ticks,main_fit_T_train_min=primary,all_sensitivity_cutoffs=config['sensitivity_T_min'],
        standard_error_plotted=False,SD_ddof=1,percentile_plotted=95,percentile_interpolation='linear',
        paired_complexity_axis='mean complexity at the same T; horizontal bars are sample SD',
        x_axis='T_train only; separate pilot validation and evaluation observations excluded',
        selection=selection,figure_stems=stems,pdf_pages=3))
    main=rates[rates.T_train_min==primary]
    checks=pd.DataFrame(selection['checks'])
    (out/'precision_report.html').write_text('''<!doctype html><meta charset="utf-8">
<title>Pilot appaiato: maggiore precisione numerica</title>
<style>body{font:16px Georgia,serif;max-width:1100px;margin:35px auto;padding:0 20px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd}img{width:100%;margin:20px 0}</style>
<h1>Pilot appaiato a maggiore precisione numerica</h1>
<p>'''+f'{config["trials"]} repliche per rango, separate dalla conferma a 100 per cella. Stessa economia,'+'''
kernel non periodico, semi appaiati e C0 congelato. Tutte le ascisse temporali
indicano soltanto il training. STD campionaria tra repliche e 95esimo percentile
empirico: con otto repliche la stima del percentile e ancora preliminare.</p>
<p>I benchmark sono asintotici. Nessun dato e tagliato per far coincidere la pendenza.
Tutti i tagli piccoli e grandi sono pubblicati; il fit principale resta da '''+str(primary)+'''.
L'intervallo temporale arriva a '''+str(max(config['T_train']))+''', mentre la prima tabella preliminare
terminava a 12800: le due finestre non sono intercambiabili.</p>
<p><a href="precision_plots.pdf">Figure PDF vettoriali</a>;
<a href="precision_summary.csv">media, STD e percentile</a>;
<a href="precision_window_sensitivity.csv">tutte le pendenze e finestre</a>.</p>
<h2>Fit principali</h2>'''+main.to_html(index=False,float_format=lambda x:f'{x:.6f}')+
        '<h2>Certificato della complessita empirica</h2><p>'+html.escape(selection['note'])+'</p>'+
        checks.to_html(index=False,float_format=lambda x:f'{x:.6f}')+
        '<p>Il bound della traccia e conservativo: il mancato superamento della soglia non prova un errore effettivo superiore alla soglia.</p>'+
        ''.join(f'<img src="figures/{stem}.png">' for stem in stems)+
        '<h2>Tutti i tagli</h2>'+rates.to_html(index=False,float_format=lambda x:f'{x:.6f}'))
    return selection


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    build(p.parse_args().output)
