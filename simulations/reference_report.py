"""Review a fully exported reference cell while the other cells continue."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from .asymptotic import scientific_hash
from .asymptotic_analysis import analysis_design
from .asymptotic_dense import validate_cell,write_json
from .asymptotic_report import draw,tail_fits,X_LABEL
from .precision_report import balanced,METRICS,TITLES
from .report import COLORS


def summary_stats(frame):
    return frame.groupby(['lambda_multiplier','T_train']).agg(
        SR_mean=('SR_hat','mean'),SR_std=('SR_hat','std'),SR_p95=('SR_hat',lambda v:v.quantile(.95)),
        gap_mean=('SR_gap','mean'),gap_std=('SR_gap','std'),gap_p95=('SR_gap',lambda v:v.quantile(.95)),
        C_mean=('complexity','mean'),C_std=('complexity','std'),C_p95=('complexity',lambda v:v.quantile(.95)),
        replications=('trial','count')).reset_index()


def build_tables(output,N=None,rho=None):
    """Analyze an exported cell without changing the primary reference design."""
    out=Path(output);config=json.loads((out/'config.json').read_text())
    N=config['fixed_N'] if N is None else N
    rho=config['fixed_rho'] if rho is None else rho
    if [N,rho] not in config['cells']:
        raise ValueError('The requested cell is outside the frozen design')
    cell=out/'cells'/f'N{N}_rho{rho:g}'
    marker=json.loads((cell/'completed.json').read_text())
    assert marker['source_hash']==scientific_hash(config)
    for name,sha in marker['files_sha256'].items():
        assert hashlib.sha256((cell/name).read_bytes()).hexdigest()==sha
    frame=pd.read_csv(cell/'confirmation_trials.csv')
    spectrum=pd.read_csv(cell/'complexity_lambda_trials.csv')
    validate_cell(config,N,rho,frame,spectrum)
    analysis_config,analysis=analysis_design(config,out)
    # This selects the cell-local lambda curve for descriptive fitting only.
    # The frozen simulation config and the primary N=600 reference stay intact.
    local=dict(analysis_config,fixed_N=N,fixed_rho=rho)
    fits,windows,_=tail_fits(local,frame,spectrum)
    write_json(cell/'rate_fits.json',fits)
    windows.to_csv(cell/'window_sensitivity.csv',index=False)
    summary_stats(frame).to_csv(cell/'summary.csv',index=False)
    write_json(cell/'analysis_status.json',dict(status='Complete cell; other cells outside this report',
        analyzed_cell=[N,rho],original_primary_reference=[config['fixed_N'],config['fixed_rho']],
        completed_reference_replications=config['trials'],
        required_experiment_replications=config['trials']*len(config['cells']),
        covers_full_experiment=False,source_hash_matches=True,analysis_design=analysis,
        source_cell_manifest_sha256=hashlib.sha256((cell/'completed.json').read_bytes()).hexdigest(),
        x_axis='T_train only',full_grid_retained=True,SD_ddof=1,standard_error_plotted=False,
        percentile_reported=95,percentile_plotted=None,percentile_interpolation='linear',tables_only=True))
    print(windows[['metric','T_train_min','points','exponent','r_squared']].to_string(index=False))


def build(output):
    out=Path(output)
    config=json.loads((out/'config.json').read_text())
    N,rho=config['fixed_N'],config['fixed_rho']
    cell=out/'cells'/f'N{N}_rho{rho:g}'
    marker=json.loads((cell/'completed.json').read_text())
    assert marker['source_hash']==scientific_hash(config)
    for name,sha in marker['files_sha256'].items():
        assert hashlib.sha256((cell/name).read_bytes()).hexdigest()==sha
    frame=pd.read_csv(cell/'confirmation_trials.csv')
    spectrum=pd.read_csv(cell/'complexity_lambda_trials.csv')
    validate_cell(config,N,rho,frame,spectrum)
    np.testing.assert_allclose(frame.SR_gap,3-frame.SR_hat,atol=1e-13)
    np.testing.assert_allclose(frame.SR_hat,np.sqrt(12)*frame.payoff_mean/
        np.sqrt(frame.payoff_second-frame.payoff_mean**2),rtol=1e-11)
    np.testing.assert_allclose(frame.lambda_reg,frame.C_lambda*frame.T_train**(-2/3)*frame.lambda_multiplier,rtol=1e-11)
    config,analysis=analysis_design(config,out)
    fits,windows,lambda_window=tail_fits(config,frame,spectrum)
    write_json(cell/'rate_fits.json',fits)
    windows.to_csv(cell/'window_sensitivity.csv',index=False)
    summary=summary_stats(frame)
    summary.to_csv(cell/'summary.csv',index=False)
    chosen=frame[frame.lambda_multiplier==1]
    cutoff=config['asymptotic_T_min'];color=COLORS[0]
    anchors=[100*2**j for j in range(9)]
    stems=[]
    with PdfPages(cell/'reference_plots.pdf') as pdf:
        def save(fig,stem,footer):
            for j,ax in enumerate(fig.axes):
                title=ax.get_title();ax.set_title('')
                ax.set_title(f'({chr(97+j)}) {title}',loc='left',pad=6)
                ax.set_axisbelow(True);ax.grid(axis='y',color='#dddddd',linewidth=.4,alpha=.6)
            fig.text(.5,.014,footer,ha='center',fontsize=7,color='#444444',linespacing=1.5)
            fig.tight_layout(rect=(0,.085,1,.975),pad=.7,w_pad=1.6,h_pad=1.7)
            pdf.savefig(fig);fig.savefig(cell/(stem+'.png'),dpi=300)
            plt.close(fig);stems.append(stem)

        def time_axis(ax):
            ax.set_xscale('log',base=2);ax.set_xlabel(X_LABEL)
            ax.set_xticks(anchors,[str(t) for t in anchors],rotation=35)
            ax.xaxis.set_minor_formatter(NullFormatter())

        def power(ax,metric,level=False):
            key='C_lambda_reference' if metric=='C_lambda' else f'N{N}_rho{rho:g}_{metric}_Tmin{cutoff}'
            fit=fits[key];pred=np.asarray(fit['prediction'])
            ax.plot(fit['x'],3-pred if level else pred,'k--',linewidth=1.1,
                label=rf'Benchmark $p={fit["expected_exponent"]:.3f}$')
            if not level:
                slope='n/a' if fit['exponent'] is None else f'{fit["exponent"]:.3f}'
                ax.text(.03,.04,rf'$\widehat p={slope}$; $R^2={fit["r_squared"]:.3f}$',
                    transform=ax.transAxes,fontsize=7,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1.5))

        fig,axes=plt.subplots(3,2,figsize=(7.6,9.4))
        draw(axes[0,0],chosen,'T_train','SR_hat',f'N={N}',color)
        draw(axes[0,1],chosen,'T_train','SR_gap',f'N={N}',color,True)
        for ax in axes[0]:time_axis(ax)
        power(axes[0,0],'SR_T',True);power(axes[0,1],'SR_T')
        axes[0,0].axhline(3,color='#777777',linestyle=':',linewidth=.8)
        axes[0,0].set(ylabel=r'Annualized $SR(\widehat W)$',title='Sharpe recovery')
        axes[0,1].set(ylabel=r'$SR^\star-SR(\widehat W)$',title=TITLES['SR_T'])
        draw(axes[1,0],chosen,'T_train','complexity',f'N={N}',color,True)
        time_axis(axes[1,0]);power(axes[1,0],'C_T')
        axes[1,0].set(ylabel=r'Effective complexity $\widehat C(\lambda_T)$',title=TITLES['C_T'])
        draw(axes[1,1],spectrum,'lambda_reg','complexity',f'N={N}',color,True)
        axes[1,1].set(xscale='log',xlabel=r'Regularization $\lambda$',
            ylabel=r'Effective complexity $\widehat C$',title=TITLES['C_lambda'])
        axes[1,1].axvspan(*lambda_window,color='#999999',alpha=.09)
        power(axes[1,1],'C_lambda')
        for ax,y,level in [(axes[2,0],'SR_hat',True),(axes[2,1],'SR_gap',False)]:
            balanced(ax,chosen,y,f'N={N}',color,horizontal_T_train=anchors)
            power(ax,'SR_C',level)
            ax.set(xlabel=r'Effective complexity $\widehat C(\lambda_T)$',
                ylabel=r'Annualized $SR(\widehat W)$' if level else r'$SR^\star-SR(\widehat W)$',
                title='Sharpe on the balanced path' if level else TITLES['SR_C'])
        for ax in axes.flat:ax.legend(fontsize=6.5)
        fig.suptitle(rf'Complete reference cell: N={N}, $\rho={rho}$; Nyström rank {marker["rank"]}',fontsize=10)
        save(fig,'reference_curves',
            f'{config["trials"]} independent training replications; all {len(config["T_train_reference"])} actual training sizes shown. '
            r'Solid: mean; shading/bars: $\pm$1 sample SD; colored dashes: trial 95th percentile.'+'\n'+
            f'Main benchmark fits: T_train >= {cutoff}; fixed spectrum T_train={config["spectrum_T_train"]}. Reference cell only; comparisons across N and rho are reported separately.'+'\n'+
            'Sharpe percentiles use mean complexity at the same T; horizontal complexity SD bars at original training anchors.')

        fig,axes=plt.subplots(2,2,figsize=(7.6,6.4))
        for ax,(metric,target) in zip(axes.flat,METRICS):
            w=windows[windows.metric==metric].sort_values('T_train_min')
            ax.plot(w.T_train_min,w.exponent,'o-',color=color,markersize=3,label='Exponent of mean curve')
            ax.axhline(target,color='black',linestyle='--',linewidth=1,label='Theoretical exponent')
            ax.axvline(cutoff,color='#777777',linestyle=':',linewidth=.8,label='Prespecified primary cutoff')
            ax.set(xscale='log',xlabel=r'Lower training cutoff $T_{\mathrm{train,min}}$',
                   ylabel='Fitted exponent',title=TITLES[metric])
            ax.set_xticks(w.T_train_min,[str(int(t)) for t in w.T_train_min],rotation=35)
            ax.xaxis.set_minor_formatter(NullFormatter());ax.legend(fontsize=6.5)
        save(fig,'reference_windows',
            f'Complete reference cell: {config["trials"]} replications, N={N}, rho={rho}. Every declared cutoff is retained.'+'\n'+
            'The complexity-lambda cutoff defines a penalty interval on the same fixed training spectrum; no fit window is selected by its slope.')
    write_json(cell/'analysis_status.json',dict(status='Complete reference cell; other four cells outside this report',
        completed_reference_replications=config['trials'],required_experiment_replications=config['trials']*len(config['cells']),
        covers_full_experiment=False,source_hash_matches=True,analysis_design=analysis,
        source_cell_manifest_sha256=hashlib.sha256((cell/'completed.json').read_bytes()).hexdigest(),
        x_axis='T_train only',full_grid_shown=True,SD_ddof=1,standard_error_plotted=False,
        percentile_plotted=95,percentile_interpolation='linear',pdf_pages=2,figure_stems=stems))
    print(windows[['metric','T_train_min','points','exponent','r_squared']].to_string(index=False))


def validate(output,N=None,rho=None):
    """Recompute summary statistics and fit exponents from exported CSVs."""
    out=Path(output);config=json.loads((out/'config.json').read_text())
    N=config['fixed_N'] if N is None else N
    rho=config['fixed_rho'] if rho is None else rho
    if [N,rho] not in config['cells']:
        raise ValueError('The requested cell is outside the frozen design')
    cell=out/'cells'/f'N{N}_rho{rho:g}'
    marker=json.loads((cell/'completed.json').read_text())
    assert marker['source_hash']==scientific_hash(config)
    for name,sha in marker['files_sha256'].items():
        assert hashlib.sha256((cell/name).read_bytes()).hexdigest()==sha
    frame=pd.read_csv(cell/'confirmation_trials.csv');spectrum=pd.read_csv(cell/'complexity_lambda_trials.csv')
    validate_cell(config,N,rho,frame,spectrum)
    np.testing.assert_allclose(frame.SR_gap,3-frame.SR_hat,atol=1e-13)
    np.testing.assert_allclose(frame.SR_hat,np.sqrt(12)*frame.payoff_mean/
        np.sqrt(frame.payoff_second-frame.payoff_mean**2),rtol=1e-11)
    np.testing.assert_allclose(frame.lambda_reg,frame.C_lambda*frame.T_train**(-2/3)*frame.lambda_multiplier,rtol=1e-11)
    np.testing.assert_allclose(frame.Q_hat,1-2*frame.payoff_mean+frame.payoff_second,atol=1e-12)
    np.testing.assert_allclose(frame.regret,frame.Q_hat-4/7,atol=1e-12)
    summary=pd.read_csv(cell/'summary.csv')
    assert not summary.duplicated(['lambda_multiplier','T_train']).any()
    for (mult,t),g in frame.groupby(['lambda_multiplier','T_train']):
        row=summary[(summary.lambda_multiplier==mult)&(summary.T_train==t)]
        assert len(row)==1;row=row.iloc[0];assert row.replications==config['trials']
        for name,column in [('SR','SR_hat'),('gap','SR_gap'),('C','complexity')]:
            values=g[column].to_numpy()
            np.testing.assert_allclose(row[name+'_mean'],np.mean(values),rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(row[name+'_std'],np.std(values,ddof=1),rtol=1e-10,atol=1e-13)
            np.testing.assert_allclose(row[name+'_p95'],np.quantile(values,.95,method='linear'),rtol=1e-11,atol=1e-13)
    config,analysis=analysis_design(config,out)
    fits=json.loads((cell/'rate_fits.json').read_text());windows=pd.read_csv(cell/'window_sensitivity.csv')
    chosen=frame[frame.lambda_multiplier==1]
    assert len(windows)==4*len(config['sensitivity_T_min'])
    assert not windows.duplicated(['metric','T_train_min']).any()
    for cutoff in config['sensitivity_T_min']:
        means=chosen[chosen.T_train>=cutoff].groupby('T_train')[['complexity','SR_gap']].mean().sort_index()
        T=means.index.to_numpy(float);constant=chosen.C_lambda.iloc[0]
        low=constant*config['spectrum_fit_T_max']**(-2/3);high=constant*cutoff**(-2/3)
        s=spectrum[(spectrum.lambda_reg>=low*(1-1e-10))&
                   (spectrum.lambda_reg<=high*(1+1e-10))].groupby('lambda_reg').complexity.mean().sort_index()
        for metric,x,y,power in [('SR_T',T,means.SR_gap.to_numpy(),-2/3),
            ('C_T',T,means.complexity.to_numpy(),1/3),('SR_C',means.complexity.to_numpy(),means.SR_gap.to_numpy(),-2),
            ('C_lambda',s.index.to_numpy(float),s.to_numpy(),-.5)]:
            key=f'C_lambda_Tmin{cutoff}' if metric=='C_lambda' else f'N{N}_rho{rho:g}_{metric}_Tmin{cutoff}'
            fit=fits[key];row=windows[(windows.metric==metric)&(windows.T_train_min==cutoff)]
            assert len(row)==1;row=row.iloc[0]
            np.testing.assert_allclose(fit['x'],x,rtol=1e-11)
            np.testing.assert_allclose(fit['observed'],y,rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(row.exponent,np.polyfit(np.log(x),np.log(y),1)[0],atol=1e-10)
            np.testing.assert_allclose(fit['exponent'],row.exponent,atol=1e-10)
            basis=x**power;scale=max(0,float(basis@y/(basis@basis)))
            r2=1-np.sum((y-scale*basis)**2)/np.sum((y-y.mean())**2)
            np.testing.assert_allclose(row.r_squared,r2,atol=1e-10)
            np.testing.assert_allclose(fit['prediction'],scale*basis,rtol=1e-11,atol=1e-13)
    status=json.loads((cell/'analysis_status.json').read_text())
    assert status['analysis_design']==analysis and status['covers_full_experiment'] is False
    assert status['standard_error_plotted'] is False and status['SD_ddof']==1
    assert status.get('percentile_reported',status.get('percentile_plotted'))==95
    files=[p for p in cell.iterdir() if p.is_file() and p.name not in ['reference_validation.json','visual_review.json']]
    primary=windows[windows.T_train_min==config['asymptotic_T_min']]
    result=dict(passed=True,completed_reference_replications=config['trials'],observations=len(frame),
        temporal_points=frame.T_train.nunique(),all_28_window_fits_verified=True,
        SD_ddof=1,standard_error_plotted=False,percentile_plotted=status.get('percentile_plotted'),
        percentile_reported=95,source_hash_matches=True,
        covers_full_experiment=False,minimum_primary_fixed_power_R_squared=float(primary.r_squared.min()),
        note='Bookkeeping and fixed-power R2 do not imply exact agreement of finite-window free exponents with asymptotic powers.',
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    write_json(cell/'reference_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--validate',action='store_true');p.add_argument('--tables-only',action='store_true')
    p.add_argument('--N',type=int);p.add_argument('--rho',type=float);args=p.parse_args()
    if args.validate:
        validate(args.output,args.N,args.rho)
    elif args.tables_only:
        build_tables(args.output,args.N,args.rho)
    elif args.N is not None or args.rho is not None:
        p.error('--N and --rho apply to --tables-only or --validate')
    else:
        build(args.output)
