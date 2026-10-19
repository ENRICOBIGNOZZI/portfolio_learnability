"""Full-grid SD and percentile plots with prespecified asymptotic tails."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .report import COLORS, matrix, rate_fit

X_LABEL=r'Training observations $T_{\mathrm{train}}$'
DOUBLE=(7.6,3.8)
SINGLE=(5.2,3.8)
MARKERS=dict(zip(COLORS,['o','s','^']))


def draw(ax,frame,x,y,label,color,logy=False):
    table=matrix(frame,x,y)
    xx=table.columns.to_numpy(float)
    avg=table.mean(axis=0).to_numpy()
    sd=table.std(axis=0,ddof=1).to_numpy()
    p95=table.quantile(.95,axis=0,interpolation='linear').to_numpy()
    ax.plot(xx,avg,linestyle='-',marker=MARKERS.get(color,'D'),color=color,label=label,linewidth=1.4,markersize=2.3)
    ax.fill_between(xx,avg-sd,avg+sd,color=color,alpha=.16)
    ax.plot(xx,p95,color=color,linestyle=(0,(4,2)),linewidth=1,alpha=.85,
            gid='trial_percentile_95')
    if logy:
        ax._sd_crosses_zero=getattr(ax,'_sd_crosses_zero',False) or not np.all(avg-sd>0)
        if not ax._sd_crosses_zero:
            ax.set_yscale('log')
        else:
            # Preserve the complete SD interval, including crossings of zero.
            threshold=max(float(np.min(np.abs(avg[avg!=0])))/4,1e-7)
            ax.set_yscale('symlog',linthresh=threshold)
    return xx,avg,sd


def tail_fits(config,frame,spectrum):
    nfix,rhofix=config['fixed_N'],config['fixed_rho']
    chosen=frame[frame.lambda_multiplier==1]
    results={}
    windows=[]
    for (N,rho),group in chosen.groupby(['N','rho']):
        gaps=matrix(group,'T_train','SR_gap')
        cs=matrix(group,'T_train','complexity')
        all_t=gaps.columns.to_numpy(float)
        for cutoff in config['sensitivity_T_min']:
            keep=all_t>=cutoff
            if keep.sum()<3:
                continue
            for metric,x,values,power,xtrials in [
                ('SR_T',all_t[keep],gaps.to_numpy()[:,keep],-2/3,None),
                ('C_T',all_t[keep],cs.to_numpy()[:,keep],1/3,None),
                ('SR_C',cs.to_numpy()[:,keep].mean(axis=0),gaps.to_numpy()[:,keep],-2,cs.to_numpy()[:,keep])]:
                fit=rate_fit(x,values,power,config['bootstrap_resamples'],xtrials)
                fit.update(T_train_full_grid=all_t.tolist(),T_train_fit_grid=all_t[keep].tolist(),
                           all_points_used=bool(keep.all()),all_points_in_declared_window_used=True,
                           T_train_cutoff=cutoff)
                key=f'N{int(N)}_rho{rho:g}_{metric}_Tmin{cutoff}'
                results[key]=fit
                windows.append(dict(N=N,rho=rho,metric=metric,T_train_min=cutoff,
                    T_train_max=float(all_t[keep][-1]),points=int(keep.sum()),
                    expected_exponent=power,exponent=fit['exponent'],
                    exponent_ci_low=fit['exponent_ci95'][0],exponent_ci_high=fit['exponent_ci95'][1],
                    r_squared=fit['r_squared'],floor=fit['floor'],floor_r_squared=fit['floor_r_squared']))
    ref=chosen[(chosen.N==nfix)&(chosen.rho==rhofix)]
    constant=ref.C_lambda.iloc[0]
    lower=constant*config['spectrum_fit_T_max']**(-2/3)
    upper=constant*config['spectrum_fit_T_min']**(-2/3)
    subset=spectrum[(spectrum.N==nfix)&(spectrum.rho==rhofix)&
                    (spectrum.lambda_reg>=lower*(1-1e-10))&(spectrum.lambda_reg<=upper*(1+1e-10))]
    tab=matrix(subset,'lambda_reg','complexity')
    results['C_lambda_reference']=rate_fit(tab.columns.to_numpy(float),tab.to_numpy(),-.5,config['bootstrap_resamples'])
    return results,pd.DataFrame(windows),(lower,upper)


def build(out):
    out=Path(out)
    config=json.loads((out/'config.json').read_text())
    frame=pd.read_csv(out/'confirmation_trials.csv')
    spectrum=pd.read_csv(out/'complexity_lambda_trials.csv')
    fidelity=pd.read_csv(out/'fidelity_trials.csv')
    chosen=frame[frame.lambda_multiplier==1]
    fits,windows,lambda_window=tail_fits(config,frame,spectrum)
    (out/'rate_fits.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
    windows.to_csv(out/'window_sensitivity.csv',index=False)
    summary=frame.groupby(['N','rho','representation','rank','lambda_multiplier','T_train']).agg(
        SR_mean=('SR_hat','mean'),SR_std=('SR_hat','std'),gap_mean=('SR_gap','mean'),gap_std=('SR_gap','std'),
        C_mean=('complexity','mean'),C_std=('complexity','std'),regret_mean=('regret','mean'),
        SR_raw_mean=('SR_raw','mean'),SR_p95=('SR_hat',lambda v:v.quantile(.95)),
        gap_p95=('SR_gap',lambda v:v.quantile(.95)),C_p95=('complexity',lambda v:v.quantile(.95)),
        replications=('trial','count')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    nfix,rhofix=config['fixed_N'],config['fixed_rho']
    cutoff=config['asymptotic_T_min']
    reference=chosen[(chosen.N==nfix)&(chosen.rho==rhofix)]
    main_keys={metric:f'N{nfix}_rho{rhofix:g}_{metric}_Tmin{cutoff}' for metric in ['SR_T','SR_C','C_T']}
    figures=out/'figures';figures.mkdir(exist_ok=True)
    pages=PdfPages(out/'requested_plots.pdf')
    diagnostics=PdfPages(figures/'06_asymptotic_diagnostics.pdf')
    stems=[]

    def save(fig,stem,diagnostic=False,replications=None):
        count=config['trials'] if replications is None else replications
        footer=(f'{count} independent training replications per cell.' if stem=='06_window_sensitivity' else
                f'Solid: mean; shading/bars: $\\pm$1 SD; colored dashes: 95th percentile.\n{count} replications; benchmark fits: N={nfix}, $\\rho={rhofix}$, $T_{{\\mathrm{{train}}}}\\geq {cutoff}$.')
        if stem=='07_numerical_fidelity':
            footer=f'Solid: mean; shading: $\\pm$1 SD; colored dashes: 95th percentile.\n{count} paired trials; N={nfix}, $\\rho={rhofix}$; same kernel, economy and primitive shocks.'
        elif stem=='08_scaled_gap_and_evaluation':
            footer=f'Solid: mean; shading: $\\pm$1 SD; colored dashes: 95th percentile.\n{count} independent training replications; N={nfix}, $\\rho={rhofix}$.'
        fig.text(.5,.012,footer,ha='center',fontsize=7,color='#444444',linespacing=1.5)
        for index,ax in enumerate(fig.axes):
            ax.set_axisbelow(True)
            ax.grid(axis='y',which='major',color='#d8d8d8',linewidth=.4,alpha=.6)
            if len(fig.axes)>1:
                title=ax.get_title();ax.set_title('')
                ax.set_title(f'({chr(97+index)}) {title}',loc='left',pad=6)
        fig.tight_layout(rect=(0,.11,1,.995),pad=.7,w_pad=1.7)
        if diagnostic:
            diagnostics.savefig(fig)
        else:
            fig.savefig(figures/(stem+'.pdf'))
        fig.savefig(figures/(stem+'.png'),dpi=300)
        pages.savefig(fig);stems.append(stem);plt.close(fig)

    def power(ax,fit,label):
        x=np.asarray(fit['x'])
        ax.plot(x,fit['prediction'],'k--',linewidth=1.4,label=label)
        exponent=fit['exponent']
        estimate='n/a' if exponent is None else f'{exponent:.3f}'
        ax.text(.03,.04,rf"$\widehat p={estimate}$; $R^2={fit['r_squared']:.3f}$",
                transform=ax.transAxes,fontsize=7,color='#222222',
                bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1.5))

    def train_axis(ax,shade=True):
        ax.set_xscale('log',base=2)
        times=sorted(chosen.T_train.unique())
        anchors=[100*2**j for j in range(9)]
        ticks=[t for t in anchors if min(times)<=t<=max(times)]
        if len(ticks)<2:
            ticks=times[::max(1,len(times)//8)]
        ax.set_xticks(ticks,[str(int(t)) for t in ticks],rotation=35)
        ax.set_xlabel(X_LABEL)
        if shade:
            ax.axvspan(cutoff,max(times),color='#888888',alpha=.06)

    for grouping,values,stem in [('N',config['N_values'],'01_sharpe_T_by_N'),
                                ('rho',config['rho_values'],'02_sharpe_T_by_rho')]:
        fig,axes=plt.subplots(1,2,figsize=DOUBLE)
        for value,color in zip(values,COLORS):
            f=chosen[(chosen.N==value)&(chosen.rho==rhofix)] if grouping=='N' else chosen[(chosen.N==nfix)&(chosen.rho==value)]
            label=f'N = {value}' if grouping=='N' else rf'$\rho={value:g}$'
            draw(axes[0],f,'T_train','SR_hat',label,color)
            draw(axes[1],f,'T_train','SR_gap',label,color,True)
        axes[0].axhline(3,color='black',linestyle=':',label=r'$SR^\star=3$')
        axes[0].set(ylabel=r'Annualized $SR(\widehat W)$',title='Sharpe ratio')
        axes[1].set(ylabel=r'$SR^\star-SR(\widehat W)$',title='Sharpe gap')
        power(axes[1],fits[main_keys['SR_T']],r'$aT_{\mathrm{train}}^{-2/3}$')
        for ax in axes:
            train_axis(ax);ax.legend(fontsize=7,loc='best')
        condition=rf'$\rho={rhofix}$' if grouping=='N' else f'N={nfix}'
        fig.suptitle('Sharpe ratio and training sample size; '+condition,fontsize=10)
        save(fig,stem)

    fig,axes=plt.subplots(1,2,figsize=DOUBLE)
    for N,color in zip(config['N_values'],COLORS):
        f=chosen[(chosen.N==N)&(chosen.rho==rhofix)]
        a=f.groupby('T_train').agg(C=('complexity','mean'),Csd=('complexity','std'),
                                   SR=('SR_hat','mean'),SRsd=('SR_hat','std'),gap=('SR_gap','mean'),gapsd=('SR_gap','std'),
                                   SRp95=('SR_hat',lambda v:v.quantile(.95)),gapp95=('SR_gap',lambda v:v.quantile(.95)))
        anchors=a.index.isin([100*2**j for j in range(9)])
        for ax,column,sd,p95 in [(axes[0],'SR','SRsd','SRp95'),(axes[1],'gap','gapsd','gapp95')]:
            ax.plot(a.C,a[column],linestyle='-',marker=MARKERS[color],markersize=2.3,color=color,label=f'N = {N}',linewidth=1.4)
            ax.fill_between(a.C,a[column]-a[sd],a[column]+a[sd],color=color,alpha=.16)
            ax.errorbar(a.C[anchors],a[column][anchors],xerr=a.Csd[anchors],fmt='none',
                        color=color,capsize=1.5,elinewidth=.7,alpha=.7)
            ax.plot(a.C,a[p95],color=color,linestyle=(0,(4,2)),linewidth=1,gid='trial_percentile_95')
        if (a.gap-a.gapsd<=0).any():
            axes[1]._sd_crosses_zero=True
    axes[0].axhline(3,color='black',linestyle=':')
    axes[0].set(xlabel=r'Mean complexity $\widehat C(\lambda_T)$',ylabel=r'Annualized $SR(\widehat W)$',title='Sharpe ratio')
    axes[1].set(xscale='log',yscale='log',xlabel=r'Mean complexity $\widehat C(\lambda_T)$',ylabel=r'$SR^\star-SR(\widehat W)$',title='Sharpe gap')
    if getattr(axes[1],'_sd_crosses_zero',False):
        axes[1].set_yscale('symlog',linthresh=max(float(reference.SR_gap.abs().median())/4,1e-7))
    power(axes[1],fits[main_keys['SR_C']],r'$aC^{-2}$')
    for ax in axes:ax.legend(fontsize=7)
    fig.suptitle(rf'Sharpe ratio and complexity; balanced path, $\rho={rhofix}$',fontsize=10)
    save(fig,'03_sharpe_complexity_by_N')

    fig,ax=plt.subplots(figsize=SINGLE)
    for N,color in zip(config['N_values'],COLORS):
        draw(ax,chosen[(chosen.N==N)&(chosen.rho==rhofix)],'T_train','complexity',f'N = {N}',color,True)
    power(ax,fits[main_keys['C_T']],r'$aT_{\mathrm{train}}^{1/3}$')
    train_axis(ax);ax.set_ylabel(r'Effective complexity $\widehat C(\lambda_T)$');ax.legend(fontsize=7)
    fig.suptitle(rf'Complexity and training sample size; $\rho={rhofix}$',fontsize=10)
    save(fig,'04_complexity_T')

    fig,ax=plt.subplots(figsize=SINGLE)
    for N,color in zip(config['N_values'],COLORS):
        draw(ax,spectrum[(spectrum.N==N)&(spectrum.rho==rhofix)],'lambda_reg','complexity',f'N = {N}',color,True)
    ax.set_xscale('log');ax.set_xlabel(r'Regularization $\lambda$');ax.set_ylabel(r'$\widehat C_{T_{train}}(\lambda)$')
    ax.axvspan(*lambda_window,color='#999999',alpha=.09)
    power(ax,fits['C_lambda_reference'],r'$a\lambda^{-1/2}$')
    ax.legend(fontsize=7)
    fig.suptitle(rf"Complexity and regularization; $T_{{train}}={config['spectrum_T_train']}$, $\rho={rhofix}$",fontsize=10)
    save(fig,'05_complexity_lambda')

    # Every prescribed cutoff, including windows that fit the theory poorly.
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.8))
    w=windows[(windows.N==nfix)&(windows.rho==rhofix)]
    for ax,(metric,target) in zip(axes,[('SR_T',-2/3),('C_T',1/3),('SR_C',-2)]):
        f=w[w.metric==metric].sort_values('T_train_min')
        ax.plot(f.T_train_min,f.exponent,'o-',color=COLORS[1],label='Exponent of mean curve')
        ax.axhline(target,color='black',linestyle='--',label='Theoretical exponent')
        ax.axvline(cutoff,color='#666666',linestyle=':',label='Main cutoff chosen in advance')
        title={'SR_T':r'Sharpe gap vs. $T_{\mathrm{train}}$',
               'C_T':r'Complexity vs. $T_{\mathrm{train}}$',
               'SR_C':'Sharpe gap vs. complexity'}[metric]
        ax.set(xscale='log',xlabel=r'Lower training cutoff $T_{train,min}$',ylabel='Fitted exponent',title=title)
        ax.set_xticks(f.T_train_min,[str(int(t)) for t in f.T_train_min],rotation=35)
        ax.legend(fontsize=7)
    fig.suptitle(rf'Fit-window sensitivity; N={nfix}, $\rho={rhofix}$',fontsize=10)
    save(fig,'06_window_sensitivity',True)

    fig,axes=plt.subplots(1,2,figsize=DOUBLE)
    ff=fidelity[fidelity.lambda_multiplier==1]
    settings=list(ff.groupby(['representation','rank']))
    for (kind,rank),f in settings:
        color='#7c7c7c' if kind=='rff' else dict(zip(config['candidate_ranks'],COLORS))[rank]
        label=rf'RFF ($M={rank}$)' if kind=='rff' else rf'Nyström ($r={rank}$)'
        draw(axes[0],f,'T_train','SR_gap',label,color,True)
        draw(axes[1],f,'T_train','complexity',label,color,True)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xlabel(X_LABEL);ax.legend(fontsize=8)
        ax.set_xticks(config['fidelity_T_train'],[str(t) for t in config['fidelity_T_train']],rotation=35)
    axes[0].set(ylabel=r'$SR^\star-SR(\widehat W)$',title='Sharpe gap')
    axes[1].set(ylabel='Effective complexity',title='Complexity')
    fig.suptitle('Kernel approximation: independent paired pilot',fontsize=10)
    save(fig,'07_numerical_fidelity',True,config['fidelity_trials'])

    fig,axes=plt.subplots(1,2,figsize=DOUBLE)
    for multiplier,color in zip(config['lambda_multipliers'],COLORS):
        f=frame[(frame.N==nfix)&(frame.rho==rhofix)&(frame.lambda_multiplier==multiplier)].copy()
        f['scaled_gap']=f.SR_gap*f.T_train**(2/3)
        draw(axes[0],f,'T_train','scaled_gap',rf'$\lambda_T={multiplier:g}C_0T_{{train}}^{{-2/3}}$',color)
    draw(axes[1],reference,'T_train','SR_hat','Cross-fitted moments',COLORS[1])
    draw(axes[1],reference,'T_train','SR_raw','Unadjusted moments','#777777')
    axes[0].set_ylabel(r'$T_{train}^{2/3}[SR^\star-SR(\widehat W)]$')
    axes[0].set_title('Rate-normalized gap')
    axes[1].set_ylabel('Annualized Sharpe');axes[1].set_title('Evaluation sensitivity')
    axes[1].axhline(3,color='black',linestyle=':')
    for ax in axes:
        train_axis(ax);ax.legend(fontsize=7.5)
    fig.suptitle('Regularization and numerical evaluation',fontsize=10)
    save(fig,'08_scaled_gap_and_evaluation',True)
    diagnostics.close();pages.close()

    main=[dict(graph=label,**{k:v for k,v in fits[key].items() if k in ['expected_exponent','exponent','r_squared','exponent_ci95','floor','floor_r_squared']})
          for label,key in [('SR vs T, different N',main_keys['SR_T']),('SR vs T, different rho',main_keys['SR_T']),('SR vs complexity',main_keys['SR_C']),('C vs T',main_keys['C_T']),('C vs lambda','C_lambda_reference')]]
    pd.DataFrame(main).to_csv(out/'main_rate_fits.csv',index=False)
    meta=dict(x_axis='T_train only; pilot/validation/evaluation observations excluded',
              dispersion='mean +/- 1 sample standard deviation across independent training replications, ddof=1',
              standard_error_plotted=False,main_fit_T_train_min=cutoff,full_grid_shown=True,
              percentile_plotted=95,percentile_interpolation='linear',percentile_unit='independent training replications',
              temporal_points_per_comparison=len(config['T_train']),temporal_points_reference=len(config['T_train_reference']),
              publication_style=dict(pdf='vector, embedded TrueType serif fonts',png_dpi=300,
                  colors=COLORS,mean_style='solid',percentile_style='colored dashed',benchmark_style='black dashed',
                  series_markers=MARKERS,
                  complexity_horizontal_SD='at original dyadic training anchors; all vertical SD bands and mean points shown'),
              main_reference_N=nfix,main_reference_rho=rhofix,figure_stems=stems,
              max_control_variate_SR_change=float(np.max(np.abs(frame.SR_hat-frame.SR_raw))),
              all_finite=bool(np.isfinite(frame.select_dtypes('number')).all().all()))
    (out/'plot_audit.json').write_text(json.dumps(meta,indent=2)+'\n')
    table=pd.DataFrame(main).to_html(index=False,float_format=lambda x:f'{x:.5f}')
    windows_html=windows.to_html(index=False,float_format=lambda x:f'{x:.5f}')
    selection=json.loads((out/'rank_selection.json').read_text())
    image_html=''.join(f'<figure><img src="figures/{s}.png"><figcaption>{html.escape(s)}</figcaption></figure>' for s in stems)
    text=f'''<!doctype html><html lang="it"><meta charset="utf-8"><title>Simulazioni asintotiche e deviazione standard</title>
<style>body{{font:16px/1.5 system-ui;max-width:1250px;margin:35px auto;padding:0 24px;color:#18313b}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #ddd}}figure{{margin:35px 0}}</style>
<h1>Training piu lungo, precisione del kernel e fit asintotici</h1>
<p>Economia originale invariata: D=s=3, b=2, Sharpe ottimo annualizzato 3, stessa calibrazione e distribuzione degli shock.
{config['trials']} repliche indipendenti per ciascuna delle cinque celle. Tutte le ascisse temporali indicano esclusivamente T_train.
Le bande e le barre sono media +/- una deviazione standard campionaria (ddof=1). I tratteggi colorati mostrano il 95esimo percentile empirico tra repliche, calcolato separatamente per ogni punto e metrica con interpolazione lineare: non sono SE o intervalli di confidenza della media.
Nel grafico Sharpe-complexity il percentile dello Sharpe e posto alla complexity media dello stesso T; non e un quantile congiunto delle due coordinate.</p>
<p>I PDF sono vettoriali, con font serif incorporati e palette Okabe-Ito. Per mantenere leggibile la griglia fitta nel grafico Sharpe-complexity, la STD verticale e mostrata come banda su tutti i punti e la STD orizzontale come barra alle ancore temporali originarie; i valori di tutte le STD restano in summary.csv.</p>
<p>Griglia completa di {len(config['T_train'])} valori effettivi fino a {max(config['T_train'])} date di training; riferimento N={nfix}, rho={rhofix} con {len(config['T_train_reference'])} valori fino a {max(config['T_train_reference'])}. Ogni punto e stimato nuovamente, senza interpolare simulazioni.
Il pilot usa separatamente {config['pilot_train']} date di training e {config['pilot_validation']} di validazione: non vengono aggiunte all'asse X della conferma.
Il fit principale usa T_train &gt;= {cutoff}, fissato prima di osservare la nuova simulazione. Tutti gli altri tagli prefissati sono pubblicati.</p>
<p><a href="requested_plots.pdf">PDF completo</a>; <a href="confirmation_trials.csv">repliche integrali</a>; <a href="window_sensitivity.csv">sensibilita ai tagli iniziali</a>.</p>
<h2>Fit principali, senza floor</h2>{table}
<p>Benchmark: gap(T_train) = O_P(T_train^(-2/3)), C(T_train) ~ T_train^(1/3), C(lambda) ~ lambda^(-1/2).
Il gap(C) ~ C^(-2) si riferisce al percorso bilanciato. R2 descrive l'adattamento finito; non dimostra un minimax o un'equivalenza asintotica per una singola economia.
Gli intervalli bootstrap degli esponenti sono conservati nei CSV, mentre le bande delle performance mostrano SD tra repliche.
I fit con floor restano diagnostici nei JSON; non vengono usati per dichiarare recupero del rate verso il vero Sharpe 3.</p>
<h2>Approssimazione dello stesso kernel</h2>
<p>Nyström non modifica il kernel Matérn-3/2: phi(z)=K(z,C)L^(-T), con LL'=K(C,C)+10^(-12)I.
I centri Sobol sono fissi, annidati e indipendenti dai dati. Il confronto numerico indipendente sceglie il rango {selection['selected_rank']} rispetto al riferimento {selection['reference_rank']}.
La scelta usa esclusivamente tolleranze dichiarate sugli errori di complexity e Sharpe, senza usare pendenze o R2.
Tutti gli esiti, inclusi RFF1000 e ranghi che falliscono, sono in <a href="fidelity_trials.csv">fidelity_trials.csv</a> e rank_selection.json.
Come riferimento metodologico: <a href="https://arxiv.org/abs/1507.04717">Rudi, Camoriano e Rosasco (2015), Less is More: Nystrom Computational Regularization</a>.
Quel lavoro riguarda regressione kernel; non costituisce da solo una verifica delle ipotesi finanziarie o di mixing di questa simulazione.</p>
<h2>Valutazione e controllo del rumore numerico</h2>
<p>Si integrano analiticamente gli shock e si usano {config['evaluation_states']} stati indipendenti dal training, divisi in due meta indipendenti.
E noto p*=3/7. Per ogni meta f, m_f e s_f stimano primo e secondo momento del payoff, p_f quello dell'oracolo.
La correzione e m_f-c_(-f)(p_f-p*), s_f-c_(-f)^2(p_f-p*), con c_(-f)=m_(-f)/p_(-f) scelto sull'altra meta.
L'indipendenza delle meta preserva l'assenza di bias dei momenti corretti; il rapporto di Sharpe resta una stima numerica non lineare.
Il training non usa l'oracolo. SR_raw e i momenti corretti sono salvati per ciascuna policy: nessun gap e tagliato a zero.
La figura di sensibilita confronta gli stessi portafogli con e senza correzione.</p>
<h2>Finestre di fit: tutti i risultati</h2>{windows_html}
<p>Aumentare le repliche riduce l'incertezza della media ma non un bias di rappresentazione. Per questo si mantengono 100 repliche e si aumentano training e accuratezza del kernel.
La rappresentazione resta finita: C e limitata dal rango, e i risultati non certificano il limite a kernel infinito. Per rho fissato |rho|&lt;1 restano gli stessi esponenti teorici; le costanti dei bound possono cambiare.</p>
{image_html}</html>'''
    (out/'report.html').write_text(text)
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',nargs='?',type=Path,default=Path(__file__).parent/'outputs/asymptotic_v2')
    print(json.dumps(build(parser.parse_args().output),indent=2))
