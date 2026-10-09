"""Display the saved annual-SR=3 experiment for T>=150; no new DGP draws.

The first available T is 180. This presentation cutoff is not the theoretical
T_0(delta). Theory guides retain their original T=60 normalization. A separate
finite-rank empirical stability diagnostic covers ALL ten original horizons.
"""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from scipy.linalg import eigvalsh

from simulations.plotting import journal_rich6d_final as figures
from simulations.diagnostics.confirmation_pilot import seed

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'simulations/outputs/rich6d_sr3_annual_monthly'
OUT = RUN / 'figures_min150'
DISPLAY = (180, 360, 720, 1440)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ticks(ax, *, logarithmic=False, compact=False):
    if logarithmic:
        ax.set_xscale('log')
    ax.set_xticks(DISPLAY, [str(t) for t in DISPLAY])
    ax.set_xlim(150, 1500)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'Sample size, $T$')


def tradeoff(d):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.95))
    fig.subplots_adjust(left=.075, right=.985, top=.87, bottom=.24, wspace=.22)
    positions = [( .53, .63), (.57, .73), (.61, .83), (.64, .94)]
    for T, color, label_position in zip(DISPLAY, figures.COLORS, positions):
        f = d['final_paths'][d['final_paths']['T'] == T]
        row = d['final_theory'][d['final_theory']['T'] == T].iloc[0]
        x, y = row.resolved_population_complexity_800, row.population_sharpe_mean*np.sqrt(12)
        for i, ax in enumerate(axes):
            start = len(ax.lines)
            figures.resolution_curve(ax, f.resolved_population_complexity_800,
                                     f.population_sharpe_mean*np.sqrt(12), f.rank_audit_passed, color)
            if i == 1:
                for line in ax.lines[start:]:
                    line.set_alpha(.27)
            ax.plot(x, y, 'D', color=color, markersize=6, markeredgecolor='white', markeredgewidth=.6, zorder=5)
        axes[1].annotate(f'$T={T}$, $\\mathcal{{C}}_{{800}}={x:.2f}$', xy=(x,y),
                         xytext=label_position, textcoords='axes fraction', color=color,
                         fontsize=8, va='center', arrowprops={'arrowstyle':'-', 'color':color, 'lw':.65})
    for ax, title in zip(axes, ('(a) Full diagnostic penalty paths', '(b) The theory-scaled policy')):
        ax.set(xscale='log', xlabel=figures.COMPLEXITY_LABEL, ylabel='Annualized population Sharpe ratio', ylim=(0,3.3))
        figures.decorate(ax,title)
    axes[0].legend(handles=[Line2D([0],[0],color=c,label=f'$T={T}$') for T,c in zip(DISPLAY,figures.COLORS)],
                   loc='upper left',ncol=2,frameon=False,handlelength=1.6)
    figures.resolution_legend(fig)
    fig.text(.52,.01,'Sample sizes above 150; all 96 diagnostic penalties retained at each displayed T.',ha='center',fontsize=8,color=figures.GRAY)
    figures.save_figure(fig,OUT,figures.STEMS[3])


def stability(verification):
    with np.load(RUN/'population.npz') as z:
        S=z['second']
    mu,U=np.linalg.eigh(S)
    assert mu.min()>0
    rows=[]
    for r in range(100):
        path=RUN/'replications'/f'{r:04d}.npz'
        assert sha(path)==verification['raw_checkpoint_sha256'][str(path.relative_to(RUN))]
        with np.load(path) as z:
            assert int(z['managed_rank'])==512
            Y=z['managed_training']@U
            Ts=z['T'];penalties=z['penalties'][:,-1]
        for T,lam in zip(Ts,penalties):
            inv=1/np.sqrt(mu+lam)
            gram=Y[:T].T@Y[:T]/T
            delta=-gram*inv[:,None]*inv[None,:]
            delta[np.diag_indices_from(delta)]+=mu/(mu+lam)
            ev=eigvalsh(delta,check_finite=False,overwrite_a=True)
            norm=max(abs(ev[0]),abs(ev[-1]))
            rows.append({'replication':r,'T':int(T),'lambda':lam,'stability_operator_norm_rank512':norm,'at_most_one_half':bool(norm<=.5)})
        if (r+1)%20==0:
            print(f'Stability diagnostic: {r+1}/100',flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'stability_by_replication.csv',index=False)
    summary=raw.groupby('T',as_index=False).agg(replications=('replication','size'),passed=('at_most_one_half','sum'),pass_fraction=('at_most_one_half','mean'),median_norm=('stability_operator_norm_rank512','median'),maximum_norm=('stability_operator_norm_rank512','max'))
    summary.to_csv(OUT/'stability_summary.csv',index=False)
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    prior=json.loads((RUN/'figures/provenance.json').read_text())
    for name,expected in prior['output_sha256'].items():
        assert sha(RUN/'figures'/name)==expected
    assert prior['figure_source_sha256']==sha(figures.__file__)
    verification=json.loads((RUN/'verification.json').read_text())
    spectrum=pd.read_csv(ROOT/'simulations/outputs/confirmation_v2/audit/spectrum/eigenvalues.csv')
    spectrum=spectrum[(spectrum.operator=='rich6d')&(spectrum.groups==4096)&(spectrum.seed_index==0)].sort_values('index')
    population=json.loads((RUN/'population.json').read_text())
    figures.DISPLAY_T=DISPLAY
    figures.time_ticks=ticks
    d=figures.prepare({'curves':pd.read_csv(RUN/'curves.csv'),'theory':pd.read_csv(RUN/'methods.csv'),
                       'rank':pd.read_csv(RUN/'rank_resolution.csv'),'spectrum':spectrum,
                       'sr_star':population['reference_sharpe'],'sharpe_display_scale':np.sqrt(12),
                       'sharpe_axis_label':'Annualized population Sharpe ratio','sharpe_gap_axis_label':'Annualized population Sharpe gap'})
    d['final_theory']=d['final_theory'][d['final_theory']['T']>=150].copy()
    assert d['final_theory']['T'].min()==180
    assert d['final_paths']['T'].min()==180
    with plt.rc_context(figures.STYLE):
        for fn in (figures.figure_zero,figures.figure_one,figures.figure_two):
            fn(d,OUT)
        tradeoff(d)
        figures.figure_four(d,OUT)
    for name,frame in [('theory_sequences',d['final_theory']),('full_penalty_paths',d['final_paths']),('spectrum_filters_cumulative',d['final_spectrum'])]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    paths=pd.read_csv(RUN/'path_methods.csv')
    paths=paths[paths['T']>=150].pivot(index='replication',columns='T',values='population_sharpe')
    gaps=population['reference_sharpe']-paths.to_numpy()
    x=np.log(paths.columns.to_numpy());x-=x.mean()
    slope=np.log(gaps.mean(axis=0))@x/(x@x)
    rng=np.random.default_rng(seed('bootstrap'))
    boot=np.array([np.log(gaps[rng.integers(100,size=100)].mean(axis=0))@x/(x@x) for _ in range(1000)])
    rate={'window':'T>=150, available T=180..1440','slope':float(slope),'bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_draws':1000}
    (OUT/'rate_window.json').write_text(json.dumps(rate,indent=2)+'\n')
    summary=stability(verification)
    note='''PRESENTATION WINDOW: T >= 150. First available simulated horizon: 180. No T=150 observation is invented or interpolated. Time curves show 180,240,360,540,720,1080,1440; spectrum filters and penalty paths show 180,360,720,1440. Figure 4 remains at T=1440. All data come from the same 100 annual-SR*=3 Rich6D replications. The theoretical guide curves retain the original T=60 normalization, although that point is outside the display window.

THEORETICAL THRESHOLD: The latest supplied PDF, The_Law_of_Portfolio_Learnability_3.pdf, Appendix A.8, printed page 68, equation (A.5), gives T^(gamma/(gamma+1))*lambda >= C_delta*[log(C_delta*T/lambda)]^2 as a sufficient stability condition for sufficiently large T. Page 69 states the existence of T_0(delta). Source: paper/theory/proofs_learning.tex lines 586-626; unbounded_concentration.tex lines 29-62. With b=1.5 and lambda=a*T^-0.6 this becomes a*T^((gamma-1.5)/(2.5*(gamma+1))) >= C_delta*[log((C_delta/a)*T^1.6)]^2. The paper does not supply a numeric C_delta or a calibrated T_0. The Rich6D geometric mixing argument does not numerically specify C_beta for a chosen gamma. Thus 150 is a display choice, not a derived theoretical threshold. The condition is sufficient, not a necessary date at which asymptotics suddenly begin.

EMPIRICAL STABILITY DIAGNOSTIC: For all 100 saved paths and ALL ten original horizons, stability_by_replication.csv computes the operator norm of (S_512+lambda*I)^(-1/2)*(S_512-S_hat_512,T)*(S_512+lambda*I)^(-1/2), using the fixed 8192-group production population quadrature and exact theory lambda. The criterion is norm<=1/2. This evaluates only the finite-rank stability event on the observed cohort. The pass fractions do not certify its true probability, infinite-rank stability, score bounds, the complete theorem, or a population asymptotic threshold. They are distinct from the earlier 512-versus-1024 regret-based numerical audit.

PROVENANCE: Input curves.csv, methods.csv, path_methods.csv, rank_resolution.csv, population.npz/json and replications/*.npz are in the parent run folder. The fixed 800-eigenvalue spectrum is the same audited 4096-group Rich6D spectrum as before. Input checkpoint SHA256 hashes are verified. All 96 penalties and numerical resolution flags are preserved. No manuscript or simulation data is modified.
'''
    (OUT/'notes_and_provenance.txt').write_text(note,encoding='utf-8')
    manifest={'presentation_min_T':150,'first_observed_T':180,'display_T':DISPLAY,'rate':rate,
              'scientific_run_hash':population['run_hash'],'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(__file__),str(Path(figures.__file__).relative_to(ROOT)):sha(figures.__file__)},
              'input_sha256':{name:sha(RUN/name) for name in ('curves.csv','methods.csv','path_methods.csv','rank_resolution.csv','population.npz','population.json','verification.json')},
              'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'}}
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(rate),flush=True)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
