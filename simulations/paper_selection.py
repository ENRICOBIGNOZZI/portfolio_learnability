"""Post hoc paper illustrations from complete, unchanged simulation datasets.

Choose one common window per study, minimizing the worst relative exponent
error subject to all four fixed-power R2 >= .99. This is explicitly an
exploratory presentation choice, never a replacement for the primary analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from .report import COLORS, matrix
from .precision_report import METRICS, TITLES
from .asymptotic_report import X_LABEL

ROOT=Path(__file__).resolve().parent
TARGETS=dict(METRICS)
MIN_R2=.99


def rank_candidates(metrics):
    """Score complete four-rate specifications, without combining windows."""
    rows=[]
    keys=['study','scenario','rank','N','rho','T_train_min']
    for key,g in metrics.groupby(keys,dropna=False):
        if set(g.metric)!=set(TARGETS) or len(g)!=4:
            raise ValueError('Every candidate must contain all four rates exactly once')
        targets=g.metric.map(TARGETS).to_numpy()
        np.testing.assert_allclose(g.expected_exponent,targets,atol=1e-12)
        errors=np.abs(g.exponent.to_numpy()-targets)/np.abs(targets)
        row=dict(zip(keys,key))
        row.update(T_train_max=float(g.T_train_max.max()),
                   common_window=bool(g.T_train_max.nunique()==1),
                   C_lambda_window_T_train_max=float(g.loc[g.metric=='C_lambda','T_train_max'].iloc[0]))
        row.update(max_relative_exponent_error=float(errors.max()),
                   mean_relative_exponent_error=float(errors.mean()),
                   minimum_fixed_power_R_squared=float(g.r_squared.min()))
        for _,v in g.iterrows():
            row[v.metric+'_exponent']=v.exponent
            row[v.metric+'_R_squared']=v.r_squared
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ['max_relative_exponent_error','mean_relative_exponent_error','T_train_min','scenario'])


def choose_window(candidates):
    eligible=candidates[candidates.common_window&(candidates.minimum_fixed_power_R_squared>=MIN_R2)&
                        np.isfinite(candidates.max_relative_exponent_error)]
    if eligible.empty:raise ValueError('No complete specification meets the declared R2 threshold')
    return eligible.sort_values(['max_relative_exponent_error','mean_relative_exponent_error',
                                'T_train_min','scenario']).iloc[0].to_dict()


def candidates(output_root):
    tables=[]
    precision=output_root/'asymptotic_precision_v4'
    d=pd.read_csv(precision/'precision_window_sensitivity.csv').rename(
        columns={'fixed_power_R_squared':'r_squared'})
    d['study']='precision8';d['scenario']='rank'+d['rank'].astype(str)
    d['N']=600;d['rho']=.5;tables.append(d)
    rank=json.loads((output_root/'asymptotic_dense_v3/completed.json').read_text())['rank']
    for f in sorted((output_root/'asymptotic_dense_v3/cells').glob('*/window_sensitivity.csv')):
        d=pd.read_csv(f)
        d['study']='confirmation100';d['scenario']=f.parent.name
        d['rank']=rank;tables.append(d)
    ranked=rank_candidates(pd.concat(tables,ignore_index=True))
    selected_rank=json.loads((precision/'precision_selection.json').read_text())['selected_rank']
    if selected_rank is None:raise ValueError('The high-precision study has no certified rank')
    # Rank is fixed by the existing numerical certificate, before window scoring.
    selected=choose_window(ranked[(ranked.study=='precision8')&(ranked['rank']==selected_rank)])
    confirmation=choose_window(ranked[ranked.study=='confirmation100'])
    return ranked,{'high_precision':selected,'confirmation':confirmation}


def load_choice(base,choice):
    if choice['study']=='precision8':
        p=base/'asymptotic_precision_v4';prefix=f'precision_nystrom{int(choice["rank"])}_'
        paths=[p/(prefix+s) for s in ['trials.csv','spectrum.csv','fits.json']]
        frame=pd.read_csv(paths[0]);spectrum=pd.read_csv(paths[1]);allfits=json.loads(paths[2].read_text())
        fits={m:allfits[f'{m}_Tmin{int(choice["T_train_min"])}'] for m in TARGETS}
    else:
        p=base/'asymptotic_dense_v3/cells'/choice['scenario']
        paths=[p/'confirmation_trials.csv',p/'complexity_lambda_trials.csv',p/'rate_fits.json']
        frame=pd.read_csv(paths[0]);spectrum=pd.read_csv(paths[1]);allfits=json.loads(paths[2].read_text())
        fits={m:allfits[(f'{choice["scenario"]}_' if m!='C_lambda' else '')+
                       f'{m}_Tmin{int(choice["T_train_min"])}'] for m in TARGETS}
    balanced=frame[(frame.lambda_multiplier==1)&(frame.T_train>=choice['T_train_min'])]
    return balanced,spectrum,fits,paths


def panel_stats(frame,spectrum,metric,fit):
    if metric=='C_lambda':
        x0,x1=min(fit['x']),max(fit['x'])
        part=spectrum[(spectrum.lambda_reg>=x0*(1-1e-10))&
                      (spectrum.lambda_reg<=x1*(1+1e-10))]
        values=matrix(part,'lambda_reg','complexity');x=values.columns.to_numpy(float);xsd=x*0
    else:
        values=matrix(frame,'T_train','SR_gap' if metric.startswith('SR') else 'complexity')
        cs=matrix(frame,'T_train','complexity')
        x=cs.mean(axis=0).to_numpy() if metric=='SR_C' else values.columns.to_numpy(float)
        xsd=cs.std(axis=0,ddof=1).to_numpy() if metric=='SR_C' else x*0
    mean=values.mean(axis=0).to_numpy();sd=values.std(axis=0,ddof=1).to_numpy()
    p95=values.quantile(.95,axis=0,interpolation='linear').to_numpy()
    np.testing.assert_allclose(x,fit['x'],rtol=1e-11)
    np.testing.assert_allclose(mean,fit['observed'],rtol=1e-11,atol=1e-13)
    lx=np.log(x);ly=np.log(mean);dx=lx-lx.mean()
    slope=float(dx@(ly-ly.mean())/(dx@dx))
    np.testing.assert_allclose(slope,fit['exponent'],atol=1e-11)
    basis=x**TARGETS[metric];amp=float(basis@mean/(basis@basis));prediction=amp*basis
    np.testing.assert_allclose(prediction,fit['prediction'],rtol=1e-11,atol=1e-13)
    r2=1-np.sum((mean-prediction)**2)/np.sum((mean-mean.mean())**2)
    np.testing.assert_allclose(r2,fit['r_squared'],atol=1e-11)
    if np.any(mean-sd<=0):raise ValueError('Unmodified SD band does not fit a positive log axis')
    return pd.DataFrame(dict(x=x,x_std=xsd,mean=mean,std=sd,p95=p95,
                             theoretical_prediction=prediction,replications=len(values)))


def plot_choice(out,name,choice,panels,fits):
    blue,orange=COLORS[:2]
    fig,axes=plt.subplots(2,2,figsize=(7.6,6.3))
    for j,(ax,(metric,target)) in enumerate(zip(axes.flat,METRICS)):
        d=panels[metric];x=d.x.to_numpy();mean=d['mean'].to_numpy();sd=d['std'].to_numpy()
        ax.fill_between(x,mean-sd,mean+sd,color=blue,alpha=.16,linewidth=0)
        ax.plot(x,mean,'o-',color=blue,markersize=3,linewidth=1.25)
        ax.plot(x,d.p95,color=orange,linestyle=(0,(4,2)),linewidth=1)
        # Dense benchmark line uses only the fixed power and fitted amplitude.
        xx=np.geomspace(x.min(),x.max(),160)
        ax.plot(xx,fits[metric]['amplitude']*xx**target,'k--',linewidth=1.05)
        if metric=='SR_C':
            anchors=np.asarray(fits[metric]['T_train_fit_grid'])
            keep=np.array([t in [100*2**k for k in range(10)] for t in anchors])
            ax.errorbar(x[keep],mean[keep],xerr=d.x_std.to_numpy()[keep],fmt='none',
                        ecolor=blue,linewidth=.6,capsize=1.5)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_axisbelow(True);ax.grid(axis='y',color='#dddddd',linewidth=.4,alpha=.6)
        ax.set_title(f'({chr(97+j)}) {TITLES[metric]}',loc='left',pad=8)
        if metric in ['SR_T','C_T']:
            anchors=[t for t in [100*2**k for k in range(10)] if x.min()<=t<=x.max()]
            ax.set_xticks(anchors,[str(t) for t in anchors],rotation=30)
            ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlabel(X_LABEL)
        else:
            ax.set_xlabel(r'Effective complexity $\widehat C(\lambda_T)$' if metric=='SR_C' else r'Regularization $\lambda$')
        ax.set_ylabel(r'$SR^\star-SR(\widehat W)$' if metric.startswith('SR') else r'Effective complexity $\widehat C$')
        top=metric=='C_T'
        annotation=(rf'$\widehat p={fits[metric]["exponent"]:+.4f}$; $p_0={target:+.4f}$'
                    '\n'+rf'$R^2_{{p_0}}={fits[metric]["r_squared"]:.4f}$')
        ax.text(.04,.94 if top else .05,annotation,transform=ax.transAxes,
                va='top' if top else 'bottom',fontsize=7,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2))
    trials=int(panels['SR_T'].replications.iloc[0])
    title='High-precision illustration' if name=='high_precision' else '100-replication comparison'
    fig.suptitle(f'{title}: N={int(choice["N"])}, '+rf'$\rho={choice["rho"]:g}$',y=.985,fontsize=10)
    handles=[Line2D([],[],color=blue,marker='o',markersize=3,label='Mean'),
             Patch(facecolor=blue,alpha=.16,label=r'$\pm$1 SD'),
             Line2D([],[],color=orange,linestyle=(0,(4,2)),label='95th percentile'),
             Line2D([],[],color='black',linestyle='--',label='Theoretical power')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,fontsize=7)
    footer=(f'{trials} replications; balanced path; training window '
            f'{int(choice["T_train_min"])}-{int(choice["T_train_max"])}. '
            f'Spectrum for C(lambda): T_train={int(choice["spectrum_T_train"])}.'
            '\nIllustrative window selected post hoc using all four rates; full-window sensitivity is retained separately.')
    fig.text(.5,.018,footer,ha='center',fontsize=6.7,color='#444444',linespacing=1.5)
    fig.tight_layout(rect=(.01,.095,.99,.96),pad=.65,w_pad=1.8,h_pad=1.8)
    fig.savefig(out/f'{name}_rates.pdf');fig.savefig(out/f'{name}_rates.png',dpi=300)
    plt.close(fig)


def build(output=None):
    base=ROOT/'outputs';out=Path(output) if output else base/'paper_selection_v1';out.mkdir(parents=True,exist_ok=True)
    ranked,choices=candidates(base);ranked.to_csv(out/'all_candidates.csv',index=False)
    tables=[];sources=set();choice_panels={};choice_fits={}
    for name,choice in choices.items():
        frame,spectrum,fits,paths=load_choice(base,choice)
        sources.update(paths);trials=frame.trial.nunique();choice['replications']=int(trials)
        scales=frame.lambda_reg*frame.T_train**(2/3)
        choice['C0']=float(scales.median())
        np.testing.assert_allclose(scales,choice['C0'],rtol=1e-11)
        choice['spectrum_T_train']=int(spectrum.T_train.unique().item())
        panels={m:panel_stats(frame,spectrum,m,fits[m]) for m in TARGETS}
        choice_panels[name]=panels;choice_fits[name]=fits
        for m,d in panels.items():
            d.to_csv(out/f'{name}_{m}_curve.csv',index=False)
            tables.append(dict(figure=name,metric=m,theory=TARGETS[m],exponent=fits[m]['exponent'],
                               fixed_power_R_squared=fits[m]['r_squared'],fit_points=fits[m]['fit_points'],
                               replications=trials,T_train_min=choice['T_train_min'],T_train_max=choice['T_train_max']))
    sources.update((base/'asymptotic_dense_v3/cells').glob('*/window_sensitivity.csv'))
    sources.update([base/'asymptotic_precision_v4/precision_window_sensitivity.csv',
                    base/'asymptotic_precision_v4/precision_selection.json'])
    source_hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    decision=dict(post_hoc=True,confirmatory=False,primary_T_train_min_unchanged=1600,
        rule='One common cutoff and training endpoint for all four rates; require all fixed-power R2 >= .99, then minimize maximum relative exponent error. Ties: mean relative error, cutoff, scenario.',
        candidate_universe='Completed dense confirmation (five cells, seven cuts) and paired precision study (two ranks, seven cuts). Legacy pilots excluded. Candidates with mismatched temporal endpoints across metrics are ineligible. Precision rank is fixed by the existing independent complexity certificate.',
        candidate_count=len(ranked),choices=choices,inputs_sha256=source_hashes,
        uncertainty='Sample SD, ddof=1, and empirical percentile95 across training replications; no SE. Selection is post hoc, without selection-adjusted inference.')
    (out/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    pd.DataFrame(tables).to_csv(out/'rates.csv',index=False)
    for name,choice in choices.items():plot_choice(out,name,choice,choice_panels[name],choice_fits[name])
    captions=[]
    for name,choice in choices.items():
        label='Higher-precision illustration' if name=='high_precision' else 'Comparison with 100 Monte Carlo replications'
        mantissa,power=f"{choice['C0']:.6e}".split('e')
        c0_latex=rf'$C_0={mantissa}\times10^{{{int(power)}}}$'
        captions.append(r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../simulations/outputs/paper_selection_v1/'''+name+r'''_rates.pdf}
\caption{'''+label+f'''. $N={int(choice['N'])}$, $\\rho={choice['rho']:g}$, Nystr\\"om rank {int(choice['rank'])},
{choice['replications']} replications, and $T_{{\\mathrm{{train}}}}\\in[{int(choice['T_train_min'])},{int(choice['T_train_max'])}]$.
'''+r'''The regularization follows the balanced path $\lambda_T=C_0T_{\mathrm{train}}^{-2/3}$.
'''+f'The scale is {c0_latex}.\n'+r'''Means, one sample standard deviation across replications, and empirical 95th percentiles of the displayed ordinates are shown.
The black curves fit one amplitude with the theoretical exponent fixed and no intercept;
$\widehat p$ is the unrestricted log--log slope of the mean curve.
In panel (c), horizontal bars are one SD of complexity; percentile ordinates are placed at mean complexity.
For $C(\lambda)$, the training spectrum is held fixed at $T_{\mathrm{train}}='''+str(int(choice['spectrum_T_train']))+r'''$.
This illustrative specification was selected after observing the simulations: one common cutoff
minimizes the largest relative discrepancy of the four exponents, subject to all fixed-power
$R^2\geq0.99$. The rank of the higher-precision study was fixed separately by its complexity-error
certificate. This is a descriptive comparison; the primary cutoff of 1600 and all sensitivity
results remain available in the full simulation reports.}
\label{fig:simulation-'''+name.replace('_','-')+r'''}
\end{figure}
''')
    (out/'paper_figures.tex').write_text('\n'.join(captions))
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Quantity & Theory & Higher precision (8) & Comparison (100) \\',r'\midrule']
    labels={'SR_T':r'Sharpe gap vs. $T_{\mathrm{train}}$','C_T':r'$C$ vs. $T_{\mathrm{train}}$',
            'SR_C':r'Sharpe gap vs. $C$','C_lambda':r'$C$ vs. $\lambda$'}
    for m,t in METRICS:
        lines.append(f'{labels[m]} & {t:+.4f} & {choice_fits["high_precision"][m]["exponent"]:+.4f} & {choice_fits["confirmation"][m]["exponent"]:+.4f} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'paper_table.tex').write_text('\n'.join(lines)+'\n')
    assert all(hashlib.sha256((ROOT.parent/p).read_bytes()).hexdigest()==sha for p,sha in source_hashes.items())
    validation=dict(passed=True,raw_curve_means_slopes_and_fixed_power_predictions_independently_checked=True,
        source_inputs_unchanged=True,SD_ddof=1,percentile=95,SE_plotted=False,
        common_window_per_figure=True,all_candidates_retained=len(ranked),post_hoc_disclosed=True,
        visual_review_pending=True,source_inputs_sha256=source_hashes)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(choices,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();build(a.output)
