"""Paper figure on a denser simulated grid, with the prior window held fixed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_selection import ROOT,TARGETS,METRICS,panel_stats,plot_choice
from .precision_enclosures import complexity_slope_enclosures
from .precision_validate import validate,validate_complexity_slope_enclosures


def verify_anchors(frame,previous):
    """Every old balanced policy remains paired with its original trial/T."""
    old=previous[previous.lambda_multiplier==1]
    current=frame[frame.lambda_multiplier==1]
    merged=old.merge(current,on=['trial','T_train'],suffixes=('_old','_dense'),validate='one_to_one')
    if len(merged)!=len(old):raise ValueError('An original training anchor is missing')
    errors={}
    for column in ['complexity','SR_hat','payoff_mean','payoff_second']:
        x=merged[column+'_old'].to_numpy();y=merged[column+'_dense'].to_numpy()
        np.testing.assert_allclose(x,y,rtol=1e-8,atol=1e-9)
        errors[column]=float(np.max(np.abs(x-y)))
    return dict(passed=True,paired_anchor_policies=len(merged),maximum_absolute_changes=errors)


def build(source=None,tables_only=False):
    out=Path(source) if source else ROOT/'outputs/paper_density_v2'
    plan=json.loads((out/'plan.json').read_text());config=json.loads((out/'config.json').read_text())
    previous=ROOT/'outputs/asymptotic_precision_v4'
    frozen=ROOT/'outputs/paper_selection_v1/selection.json'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==plan['prior_selection_sha256']
    choice=dict(json.loads(frozen.read_text())['choices']['high_precision'])
    cutoff=int(plan['frozen_paper_cutoff']);rank=int(choice['rank'])
    assert cutoff==int(choice['T_train_min'])==800 and rank==8192
    assert config['precision_frozen_C0']==choice['C0']
    assert config['fidelity_seed']==plan['original_seed']
    assert config['trials']==choice['replications']==8
    backend_audit=None
    if (out/'training_backend.json').exists():
        from .paper_density_compute import validate_backend
        backend_audit=validate_backend(out)
    rank_audit=validate(out,rank=rank)
    prefix=f'precision_nystrom{rank}_'
    frame=pd.read_csv(out/(prefix+'trials.csv'))
    spectrum=pd.read_csv(out/(prefix+'spectrum.csv'))
    cert=pd.read_csv(out/(prefix+'certificates.csv'))
    old=pd.read_csv(previous/(prefix+'trials.csv'))
    anchors=verify_anchors(frame,old)
    selected=frame[(frame.lambda_multiplier==1)&(frame.T_train>=cutoff)]
    actual=sorted(selected.T_train.unique().tolist())
    assert actual==plan['paper_training_grid'] and len(actual)==plan['paper_grid_size']
    allfits=json.loads((out/(prefix+'fits.json')).read_text())
    fits={m:allfits[f'{m}_Tmin{cutoff}'] for m in TARGETS}
    panels={m:panel_stats(selected,spectrum,m,fits[m]) for m in TARGETS}
    choice.update(T_train_max=max(actual),T_train_min=cutoff,
                  spectrum_T_train=int(spectrum.T_train.unique().item()),
                  paper_training_points=len(actual),window_reselected_after_densification=False)
    # Update descriptive numbers for the denser grid without selecting a new window.
    errors=[]
    for m in TARGETS:
        choice[m+'_exponent']=fits[m]['exponent']
        choice[m+'_R_squared']=fits[m]['r_squared']
        errors.append(abs(fits[m]['exponent']-TARGETS[m])/abs(TARGETS[m]))
    choice.update(max_relative_exponent_error=float(max(errors)),
                  mean_relative_exponent_error=float(np.mean(errors)),
                  minimum_fixed_power_R_squared=float(min(f['r_squared'] for f in fits.values())))
    rates=[]
    for m,d in panels.items():
        d.to_csv(out/f'paper_{m}_curve.csv',index=False)
        rates.append(dict(metric=m,theoretical_exponent=TARGETS[m],exponent=fits[m]['exponent'],
                          fixed_power_R_squared=fits[m]['r_squared'],fit_points=fits[m]['fit_points']))
    pd.DataFrame(rates).to_csv(out/'paper_rates.csv',index=False)
    labels={'SR_T':r'Sharpe gap vs. $T_{\mathrm{train}}$','C_T':r'$C$ vs. $T_{\mathrm{train}}$',
            'SR_C':r'Sharpe gap vs. $C$','C_lambda':r'$C$ vs. $\lambda$'}
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',
           r'Quantity & Theory & Simulation & Fixed-power $R^2$ \\',r'\midrule']
    for m,target in METRICS:
        lines.append(f'{labels[m]} & {target:+.4f} & {fits[m]["exponent"]:+.4f} & {fits[m]["r_squared"]:.4f} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'paper_table.tex').write_text('\n'.join(lines)+'\n')
    analysis_config=dict(config,precision_ranks=[rank])
    enclosures=complexity_slope_enclosures(analysis_config,frame,cert,spectrum)
    validate_complexity_slope_enclosures(config,frame,cert,spectrum,enclosures)
    enclosures.to_csv(out/'complexity_slope_enclosures.csv',index=False)
    result=dict(passed=True,numeric_rank_validation=rank_audit['passed'],backend_validation=backend_audit,
        dense_training_grid_verified=True,actual_paper_training_points=len(actual),
        replications=8,same_eight_training_streams=True,anchors=anchors,
        frozen_paper_window=[cutoff,max(actual)],window_reselected_after_densification=False,
        original_primary_cutoff_retained=config['asymptotic_T_min'],
        standard_error_plotted=False,SD_ddof=1,percentile=95,
        passes_complexity_tolerance=rank_audit['passes_certified_tolerance'],
        descriptive_selection_from_prior_pilot=True,visual_review_pending=True,
        choice=choice)
    if not tables_only:
        plot_choice(out,'high_precision',choice,panels,fits)
        caption=(r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{../simulations/outputs/paper_density_v1/high_precision_rates.pdf}
\caption{Higher-precision simulation on a denser training grid.
$N=600$, $\rho=0.5$, Nystr\"om rank 8192, and eight Monte Carlo replications.
The balanced path uses $\lambda_T=5.623413\times10^{-7}T_{\mathrm{train}}^{-2/3}$.
The figure contains '''+str(len(actual))+r''' actual training sizes from 800 to 25600, with the same
underlying eight trial streams as the earlier pilot. Markers are computed observations.
Means, one sample standard deviation across replications, and empirical 95th percentiles
of the displayed ordinates are shown. In panel (c), horizontal bars show one SD of complexity,
and percentile ordinates are placed at mean complexity. Black curves fit one amplitude with
the theoretical exponent fixed and no intercept; $\widehat p$ is the unrestricted log--log
slope of the mean curve. The spectrum for $C(\lambda)$ is held fixed at $T_{\mathrm{train}}=25600$.
The initial training cutoff of 800 was selected descriptively using the completed pilot
and held fixed before computing the denser grid. Every training point in this new window
is included, without selecting another cutoff from the densified results.
The primary analysis from 1600 and the full sensitivity results remain available separately.}
\label{fig:simulation-dense-precision}
\end{figure}
''')
        (out/'paper_figure.tex').write_text(caption.replace('outputs/paper_density_v1/',f'outputs/{out.name}/'))
    source_files=[out/'plan.json',out/'config.json',frozen,previous/(prefix+'trials.csv'),
                  *[out/(prefix+s) for s in ['trials.csv','spectrum.csv','certificates.csv','fits.json','completed.json']]]
    if backend_audit is not None:
        source_files.extend(out/name for name in ['training_backend.json','prior_config.json','prior_eigenvalues.csv','prior_trace_anchors.csv','prior_provenance.json'])
    result['inputs_sha256']={f'{f.parent.name}/{f.name}':hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    (out/'paper_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(pd.DataFrame(rates).to_string(index=False))
    print(json.dumps(anchors,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--tables-only',action='store_true')
    a=p.parse_args();build(a.source,a.tables_only)
