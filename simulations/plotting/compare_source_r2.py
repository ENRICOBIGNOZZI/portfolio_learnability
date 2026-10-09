"""Compare the proposed r=2 rate with existing r=1-policy Monte Carlo data.

This is a benchmark comparison, not a refit, a new source-condition proof,
or a claim that the observed fixed-rank slopes equal an asymptotic exponent.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from simulations.plotting.journal_rich6d import STYLE, BLUE, GRAY, COLORS, decorate, save_figure

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'simulations/outputs/rich6d_sr3_annual_monthly'
OUT=RUN/'comparison_r2'
STEM='sharpe_rate_comparison_r1_r2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    provenance=json.loads((RUN/'figures_min150/provenance.json').read_text())
    for name in ('methods.csv','path_methods.csv','population.json'):
        assert digest(RUN/name)==provenance['input_sha256'][name]
    population=json.loads((RUN/'population.json').read_text())
    s=population['parameters']['nu']+population['parameters']['D']/2
    dimension=population['parameters']['D']
    b=2*s/dimension
    rates={r:2*s*r/(2*s*r+dimension) for r in (1,2)}
    penalties={r:b/(b*r+1) for r in (1,2)}
    assert s==4.5 and dimension==6 and rates=={1:.6,2:.75}
    f=pd.read_csv(RUN/'methods.csv')
    f=f[(f.method=='theory_1')&(f['T']>=150)].copy()
    assert len(f)==7 and f['T'].min()==180
    raw=pd.read_csv(RUN/'path_methods.csv')
    raw=raw[(raw.method=='theory_1')&(raw['T']>=150)]
    check=raw.groupby('T').population_sharpe.agg(['mean','std','count'])
    assert (check['count']==100).all()
    np.testing.assert_allclose(f.population_sharpe_mean,check['mean'],rtol=1e-12)
    np.testing.assert_allclose(f.population_sharpe_mcse,check['std']/10,rtol=1e-12)
    scale=np.sqrt(population['periods_per_year'])
    T=f['T'].to_numpy()
    f['annual_sharpe']=f.population_sharpe_mean*scale
    f['annual_gap']=f.sharpe_gap_mean*scale
    for r,alpha in rates.items():
        f[f'annual_gap_guide_r{r}']=f.annual_gap.iloc[0]*(T/T[0])**(-alpha)
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(8.8,3.9))
        fig.subplots_adjust(left=.075,right=.985,bottom=.22,top=.89,wspace=.33)
        ax=axes[0]
        ax.plot(T,f.annual_sharpe,'-o',color=BLUE,markersize=3.5,label='Existing policy: Monte Carlo mean')
        ax.fill_between(T,f.population_sharpe_ci_low*scale,f.population_sharpe_ci_high*scale,color=BLUE,alpha=.12,linewidth=0)
        ax.axhline(3,color=GRAY,linestyle='--',linewidth=1.2,label=r'Analytic annual $SR^\star=3$')
        ax.set(ylabel='Annualized population Sharpe ratio',ylim=(0,3.3))
        ax.legend(loc='lower right',frameon=False,fontsize=8)
        decorate(ax,'(a) Sharpe recovery: unchanged estimates')
        ax=axes[1]
        ax.set(xscale='log',yscale='log',ylabel='Annualized population Sharpe gap')
        ax.plot(T,f.annual_gap,'-o',color=BLUE,markersize=3.5,label='Measured gap: existing policy')
        ax.fill_between(T,f.sharpe_gap_ci_low*scale,f.sharpe_gap_ci_high*scale,color=BLUE,alpha=.12,linewidth=0)
        ax.plot(T,f.annual_gap_guide_r1,'--',color=GRAY,linewidth=1.25,label=r'$r=1$: $T^{-0.60}$ guide')
        ax.plot(T,f.annual_gap_guide_r2,linestyle=(0,(5,2,1,2)),color=COLORS[2],linewidth=1.4,label=r'$r=2$: $T^{-0.75}$ proposed guide')
        ax.set_yticks([.1,.2,.4],['0.10','0.20','0.40'])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.legend(loc='lower left',frameon=False,fontsize=7.7)
        decorate(ax,'(b) Comparing the two rate benchmarks')
        for ax in axes:
            ax.set_xticks([180,360,720,1440],['180','360','720','1440'])
            ax.set_xlim(150,1500)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel(r'Sample size, $T$')
        fig.text(.53,.025,r'Both guides anchored at $T=180$. Existing $\lambda_T=aT^{-0.6}$ estimates; no policy refit.',ha='center',color=GRAY,fontsize=8)
        save_figure(fig,OUT,STEM)
    f.to_csv(OUT/'comparison_curve_data.csv',index=False)
    comparison={'s_sobolev':s,'matern_nu':population['parameters']['nu'],'d':dimension,'b':b,
                'rate_exponent':rates,'hypothetical_lambda_exponent_if_bias_is_lambda_to_r':penalties,
                'guide_anchor_T':int(T[0]),'existing_policy':'theory_1: lambda=a*T^-0.6',
                'policies_refitted':False,'source_condition_r2_proved':False,
                'descriptive_slope':json.loads((RUN/'figures_min150/rate_window.json').read_text()),
                'scientific_run_hash':population['run_hash'],
                'input_sha256':{name:digest(RUN/name) for name in ('methods.csv','path_methods.csv','population.json')},
                'plotting_source_sha256':digest(__file__)}
    notes='''Proposed exponent: -2*s*r/(2*s*r+d). Here s is the Sobolev/RKHS smoothness, not the Matern parameter nu: nu=1.5, d=6, s=nu+d/2=4.5. Thus b=2s/d=1.5; r=1 gives -0.6 and r=2 gives -0.75.

This figure overlays two benchmark lines on the EXISTING 100-replication theory_1 Sharpe curves, with annual optimum 3, monthly convention sqrt(12), and T>=150 (available points start at 180). Both lines are normalized to the same measured gap at T=180; neither exponent is fitted. The Monte Carlo means, pointwise intervals and trained policies are unchanged. The historical T=60 anchor is deliberately replaced by a common T=180 anchor for this comparison only.

If the population bias were O(lambda^r) and complexity O(lambda^(-1/b)), balancing lambda^r with lambda^(-1/b)/T would give lambda proportional to T^(-b/(br+1)), risk proportional to T^(-br/(br+1)), and complexity proportional to T^(1/(br+1)). For r=2 these exponents are respectively -0.375, -0.75, +0.25; relative complexity has exponent -0.75. The exponent -0.75 is NOT the proposed lambda exponent.

The present manuscript proof establishes source order r=1 (paper/theory/proofs_learning.tex lines 373-375 and 444-445). Its current Rich6D admissibility argument establishes RKHS membership; this does not by itself prove the stronger operator source condition for r=2. This comparison does not extend the theorem, certify r=2 for Rich6D, determine T_0, or refit an r=2 estimator. The observed finite-window bootstrap slope is descriptive and is not a test of an asymptotic big-O bound.

Provenance: methods.csv and path_methods.csv in the parent run, with independent reconstruction of means/MCSE from all 100 saved path evaluations. Formula conventions: paper/theory/representation_tables.tex (Matern smoothness) and simulations/docs/dgp_methodology.md (s=9/2). Complete displayed values are in comparison_curve_data.csv. The manuscript and previous figures are unchanged.
'''
    (OUT/'notes.txt').write_text(notes,encoding='utf-8')
    comparison['output_sha256']={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'}
    (OUT/'provenance.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':
    main()
