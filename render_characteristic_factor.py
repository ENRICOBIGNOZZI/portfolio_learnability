"""Publication plots and interpretation for the audited characteristic-factor simulation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress
from empirical_closeout import style,save
from data_pipeline import write_json,digest


def heatmap(ax,surface,oracle,validation,absolute=False):
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    ts=sorted(surface['T'].unique())
    vmax=max(np.log10(g.mean_regret/g.mean_regret.min()).max() for _,g in surface.groupby('T'))
    norm=Normalize(0,vmax)
    xmin=np.inf;xmax=0
    for k,t in enumerate(ts):
        s=surface[surface['T']==t].sort_values('population_C')
        x=s.population_C.to_numpy()/(1 if absolute else t)
        # Geometric Voronoi edges, bounded by the actual sampled endpoints (no extrapolation).
        edges=np.r_[x[0],np.sqrt(x[:-1]*x[1:]),x[-1]]
        z=np.log10(s.mean_regret.to_numpy()/s.mean_regret.min())
        mesh=ax.pcolormesh(edges,[k-.5,k+.5],z[None,:],cmap='cividis',norm=norm,rasterized=True)
        xmin=min(xmin,x[0]);xmax=max(xmax,x[-1])
    ox=oracle.C_oracle.to_numpy()/(1 if absolute else oracle['T'].to_numpy())
    vx=validation.median_population_C_over_T.to_numpy()*(validation['T'].to_numpy() if absolute else 1)
    ax.plot(ox,np.arange(len(ts)),'o-',color='black',lw=1.3,ms=4,label='MC oracle',zorder=3)
    ax.scatter(vx,np.arange(len(ts)),s=29,facecolor='white',edgecolor='black',lw=.8,label='Median validation',zorder=4)
    ax.set(xscale='log',xlim=(xmin,xmax),ylim=(len(ts)-.5,-.5),yticks=np.arange(len(ts)),yticklabels=ts,
        ylabel='Historical months T',xlabel='Population effective complexity C' if absolute else 'Population relative complexity C/T')
    ax.legend(loc='upper left',fontsize=9,framealpha=.95)
    return mesh


def rate_panel(ax,oracle,rate,metric,title,ylabel,theory):
    t=oracle['T'].to_numpy();y=oracle[metric].to_numpy();ix=np.arange(len(t)//2,len(t))
    row=rate[(rate.subset=='upper_half')&(rate.metric==metric)].iloc[0]
    fit=linregress(np.log(t[ix]),np.log(y[ix]));xx=np.geomspace(t[ix[0]],t[-1],100)
    ax.axvspan(t[ix[0]],t[-1],color='#eff3f5',zorder=0)
    lo=oracle[metric+'_ci_low'].to_numpy();hi=oracle[metric+'_ci_high'].to_numpy()
    ax.vlines(t,lo,hi,color='#245b7d',lw=1.1)
    ax.loglog(t,y,'o',color='#245b7d',ms=5,label='MC oracle')
    ax.plot(xx,np.exp(fit.intercept)*xx**fit.slope,color='#245b7d',lw=1.6,label='Fitted: upper half')
    anchor=np.exp(np.mean(np.log(y[ix])));tx=np.exp(np.mean(np.log(t[ix])))
    ax.plot(xx,anchor*(xx/tx)**theory,'--',color='#b17b2e',lw=1.6,label='General r = 1 benchmark')
    ax.set(title=title,xlabel='Historical months T',ylabel=ylabel)
    ax.grid(which='major',color='#dce2e8',lw=.6);ax.legend(fontsize=8.8,loc='upper left' if metric=='C_oracle' else 'upper right',framealpha=.9)
    txt=f'Slope {row.slope:+.3f}  |  OLS SE {row.ols_se:.3f}  |  R² {row.r_squared:.3f}\nBenchmark {theory:+.3f}  |  Bootstrap SE {row.bootstrap_se:.3f}'
    if metric=='C_oracle':
        txt=f'Slope {row.slope:+.3f}  |  OLS SE {row.ols_se:.3f}\nR² {row.r_squared:.3f}  |  Benchmark {theory:+.3f}\nBootstrap SE {row.bootstrap_se:.3f}'
    ax.text(.43 if metric=='C_oracle' else .025,.035,txt,transform=ax.transAxes,fontsize=9.3,ha='left',va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=.9))


def main(output='outputs',headline_only=False):
    out=Path(output);sim=out/'simulations';figs=out/'figures';notes=out/'notes'
    figs.mkdir(parents=True,exist_ok=True);notes.mkdir(parents=True,exist_ok=True)
    audits=json.loads((sim/'characteristic_factor_dgp_audit.json').read_text())
    assert all(a['passed'] and a['long_path']['passed'] for a in audits)
    surface=pd.read_parquet(sim/'characteristic_factor_full_surface.parquet')
    oracle=pd.read_csv(sim/'characteristic_factor_oracle_by_T.csv')
    validation=pd.read_csv(sim/'characteristic_factor_validation_by_T.csv')
    rate=pd.read_csv(sim/'characteristic_factor_rate_slopes.csv')
    def headline(d):return d[(d.economy=='NL')&(d.b==1.5)&(d.J==2000)].sort_values('T') if 'T' in d else d[(d.economy=='NL')&(d.b==1.5)&(d.J==2000)]
    s,o,v,r=map(headline,(surface,oracle,validation,rate))
    plt=style();fig,axes=plt.subplots(2,2,figsize=(13,11.2));fig.subplots_adjust(left=.08,right=.96,bottom=.135,top=.84,wspace=.35,hspace=.40)
    fig.text(.08,.956,'Learnability in a persistent characteristic-factor economy',fontsize=20,weight='bold')
    fig.text(.08,.917,'Nonlinear policy | b = 1.5 | 5,000 stocks | 2,000 directions | 500 independent paths',fontsize=12.4,color='#4e5661')
    mesh=heatmap(axes[0,0],s,o,v);axes[0,0].set_title('(a) Finite-sample learnability map',pad=12)
    cb=fig.colorbar(mesh,ax=axes[0,0],pad=.018,fraction=.045);cb.set_label(r'$\log_{10}(\mathrm{regret}/\mathrm{row\ minimum})$',fontsize=10)
    rate_panel(axes[0,1],o,r,'lambda_oracle','(b) Oracle ridge',r'$\lambda_T^{\mathrm{oracle}}$',-.6)
    rate_panel(axes[1,0],o,r,'C_oracle','(c) Oracle effective complexity',r'$C(\lambda_T^{\mathrm{oracle}})$',.4)
    rate_panel(axes[1,1],o,r,'regret_oracle','(d) Exact population regret','Mean population regret',-.6)
    fig.text(.08,.035,'Risk is evaluated analytically, not on a simulated test set. Bars: 95% bootstrap intervals across independent paths.\n'
        'Slope fits use the predeclared upper half, T = 360-1,440 (shaded); all sample sizes remain visible.\n'
        'Dashed slopes describe the general RKHS envelope. The fixed smooth target need not attain its worst-case rate.\n'
        'The full C/T range is shown on a log axis and can exceed one. Median validation markers do not describe mean validation risk.',fontsize=10,color='#4e5661',linespacing=1.45)
    save(fig,figs,'simulation_learnability_law_b150',pdf=True);plt.close(fig)
    if headline_only:
        return
    fig,ax=plt.subplots(figsize=(10,7));fig.subplots_adjust(left=.1,right=.86,bottom=.18,top=.80)
    fig.text(.1,.935,'Learnability map in absolute complexity',fontsize=20,weight='bold')
    fig.text(.1,.879,'Same population regret surface and selections | Nonlinear economy, b = 1.5',fontsize=12,color='#4e5661')
    mesh=heatmap(ax,s,o,v,absolute=True);cb=fig.colorbar(mesh,ax=ax,pad=.025)
    cb.set_label(r'$\log_{10}(\mathrm{regret}/\mathrm{row\ minimum})$')
    fig.text(.1,.045,'Full sampled population complexity, including the near-unregularized finite-rank limit J = 2,000.\n'
        'White markers select lambda using chronological validation; black markers use the mean population-risk oracle.',fontsize=10,color='#4e5661',linespacing=1.45)
    save(fig,figs,'simulation_heatmap_absolute_complexity_b150');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,5.4));fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.76,wspace=.33)
    fig.text(.09,.94,'Economic spectrum and the observed rate of learning',fontsize=20,weight='bold')
    fig.text(.09,.883,'Nonlinear fixed policy | J = 2,000 | 500 paths at b = 1.5; 200 at other spectra',fontsize=12,color='#4e5661')
    for b,color in zip([1.25,1.5,2.],['#245b7d','#34918a','#b17b2e']):
        d=oracle[(oracle.economy=='NL')&(oracle.J==2000)&(oracle.b==b)]
        for ax,metric,ylabel in zip(axes,['lambda_oracle','C_oracle','regret_oracle'],['Oracle ridge','Population complexity','Population regret']):
            ax.loglog(d['T'],d[metric],'o-',lw=1.5,ms=4,label=f'b = {b:g}',color=color)
            ax.set(xlabel='Historical months T',ylabel=ylabel);ax.grid(which='major',color='#dce2e8',lw=.6)
    axes[0].legend(frameon=False)
    fig.text(.09,.035,'All curves use exact population risk. Full-grid, upper-half and largest-four slope estimates, with uncertainty, are saved in the tables.\n'
        'The general r = 1 bounds are not pointwise oracle identities for this fixed smooth target.',fontsize=10,color='#4e5661',linespacing=1.5)
    save(fig,figs,'simulation_spectrum_comparison_b125_b150_b200');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,5.4));fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.76,wspace=.33)
    fig.text(.09,.94,'Low-complexity policy: a separate sanity check',fontsize=20,weight='bold')
    fig.text(.09,.883,'Only the first characteristic direction carries the target | b = 1.5 | J = 2,000 | 200 paths',fontsize=12,color='#4e5661')
    for economy,label,color in [('L','Single characteristic tilt','#245b7d'),('NL','Nonlinear target','#b17b2e')]:
        d=oracle[(oracle.economy==economy)&(oracle.b==1.5)&(oracle.J==2000)]
        for ax,metric,ylabel in zip(axes,['lambda_oracle','C_oracle','regret_oracle'],['Oracle ridge','Population complexity','Population regret']):
            ax.loglog(d['T'],d[metric],'o-',lw=1.5,ms=4,label=label,color=color)
            ax.set(xlabel='Historical months T',ylabel=ylabel);ax.grid(which='major',color='#dce2e8',lw=.6)
    axes[0].legend(frameon=False,fontsize=9)
    fig.text(.09,.035,'The estimator still fits all characteristic directions. The finite-support target is a sanity check, not evidence for a nonlinear worst-case law.\n'
        'The first cosine is a low-frequency monotone characteristic tilt; it is not literally an affine policy in the rank.',fontsize=10,color='#4e5661',linespacing=1.5)
    save(fig,figs,'simulation_linear_sanity_check');plt.close(fig)
    # Prespecified finite-rank stability: compare upper-half slopes to J=2,000, 0.08 tolerance.
    rob=rate[(rate.economy=='NL')&(rate.b==1.5)&(rate.subset=='upper_half')]
    wide=rob.pivot(index='metric',columns='J',values='slope');differences=wide.subtract(wide[2000],axis=0).abs()
    risk_last=oracle[(oracle.economy=='NL')&(oracle.b==1.5)&(oracle['T']==1440)].set_index('J').regret_oracle
    stability=dict(max_slope_difference=float(differences.to_numpy().max()),slope_tolerance=.08,
        max_largest_T_relative_regret_difference=float(abs(risk_last/risk_last.loc[2000]-1).max()),regret_tolerance=.10)
    stability['passed']=stability['max_slope_difference']<.08 and stability['max_largest_T_relative_regret_difference']<.10
    write_json(sim/'characteristic_factor_finite_J_check.json',stability)
    make_note(out,audits,oracle,validation,rate,stability)
    write_json(out/'manifest_simulation_figures.json',dict(source_sha256=digest(__file__),dgp_audits_passed=True,
        finite_J_check=stability,inputs={p.name:digest(p) for p in sim.iterdir() if p.suffix in ('.csv','.parquet','.json')},
        outputs={p.name:digest(p) for p in figs.glob('simulation_*') if p.suffix in ('.png','.pdf')}))


def make_note(out,audits,oracle,validation,rate,stability):
    run=json.loads((out/'simulations/characteristic_factor_run.json').read_text())
    rate=rate[(rate.economy=='NL')&(rate.J==2000)&(rate.subset=='upper_half')]
    rows='\n'.join(f'| {r.b:g} | {r.metric} | {r.slope:+.4f} | {r.ols_se:.4f} | {r.bootstrap_se:.4f} | {r.r_squared:.4f} | {r.theory_r1:+.4f} |' for r in rate.itertuples())
    auditrows='\n'.join(f"| {a['economy']} | {a['b']} | {a['J']} | {a['c_theta']:.6g} | {a['min_eigenvalue_V_F']:.6g} | {max(x['orthogonality_error'] for x in a['months']):.2e} | {max(x['payoff_error'] for x in a['months']):.2e} | {a['long_path']['estimated_ar']:.5f} |" for a in audits)
    headline_validation=validation[(validation.economy=='NL')&(validation.b==1.5)&(validation.J==2000)].sort_values('T')
    validation_start=headline_validation.iloc[0];validation_end=headline_validation.iloc[-1]
    text=f'''# Persistent characteristic-factor economy

## Economic construction and exact identities

N=5,000 stocks have stationary latent Gaussian AR(1) characteristics, rho_Z=0.90.
At formation t, U=(rank-0.5)/N, using characteristics only. The actual stock audit
constructs three months for each economy/J configuration and five random policies per
month. Phi_ij=sqrt(2)cos(pi*j*U_i), j=1,...,J<N, satisfies Phi'Phi/N=I.
R_(t+1)=Phi_t F_(t+1)+epsilon_(t+1). Independent stock shocks with sigma_e=0.15
are projected as epsilon=sigma_e*(v-Phi Phi'v/N); the identity normalization makes this
exactly the requested orthogonal-complement projector. Stock residual noise is nonzero.
For weights W=Phi theta/N, W'R=theta'F up to the audited numerical precision.
The admissible policies are static functions of the current characteristic; theta*
is their unconditional population optimum, not an optimum over arbitrary history-dependent
strategies. Positions are unconstrained long-short exposures, not normalized unlevered
index weights. For each finite J the coefficient norm defines the finite-dimensional
RKHS with kernel sum_j phi_j(u)phi_j(v). This is a controlled finite-rank approximation,
not a proof of a uniformly bounded infinite-J point-evaluation kernel.
After checking this identity, every Monte Carlo path uses these same managed factors
without rebuilding the observationally irrelevant stock matrices. It is the same DGP.

mu_j=0.0004*j^(-b), b in {{1.25,1.5,2}}. The NL target is c_theta/j (a=1).
The L sanity target has only coordinate one. In each configuration c_theta normalizes
theta*'S theta*=0.16<1; mean factors are fbar=S theta*. Stationary covariance is
V_F=S-fbar fbar', NOT S. A diagonal-minus-rank-one square root samples this covariance
without a dense Cholesky. Factor AR persistence is 0.30, with stationary initialization
and innovation covariance (1-rho_F^2)V_F. Thus E[FF']=S exactly, theta* is optimal,
and Q(theta*)=0.84. One 200,000-month stationary path audits each configuration:
all coordinate means/second-moment diagonals, a whitened 16x16 second-moment block,
and aggregate centered AR coefficient. Recorded empirical/target means and moments
are included in the JSON, together with every parameter and residual volatility.
Predeclared long-path tolerances: maximum standardized mean error <6.5, maximum
relative diagonal error <0.035, 16x16 block error <0.025, AR error <0.005.

| Economy | b | J | c_theta | min eig(V_F) | max orthogonality error | max payoff error | estimated AR |
|---|---:|---:|---:|---:|---:|---:|---:|
{auditrows}

## Estimation, tuning and uncertainty

T={{60,90,120,180,240,360,540,720,1080,1440}}. Headline NL b=1.5 J=2000 has
500 independent paths; other b values, L, and J=1000/4000 have 200 each.
Within one replica the T paths are nested prefixes; replicas are independent.
Seeds use SeedSequence([20261003,100*b,J,economy_code,replica]); no seed is selected
for attractive results. The population risk is the exact quadratic
(theta_hat-theta*)'S(theta_hat-theta*), not a noisy holdout loss.
The estimator uses uncentered sample second moments and mean response-one payoffs.
Population ridge-target regret and empirical complexity diagnostics are saved too.

Each spectrum uses 360 positive logarithmic penalties from mu_J*1e-5 to mu_1*1e3.
The script automatically extends/reruns the grid if any MC oracle is on a boundary.
Simulation lambda never equals zero: it includes near-unregularized positive ridge.
Population C=sum mu/(mu+lambda); unlike sample rank, population C/T can exceed one.
No large-C portion is dropped. Both heatmaps use log x axes over the full sampled range.
Their geometric bin edges lie inside the sampled endpoints; no curve smoothing or
out-of-range extrapolation is used. Color is log10(mean exact regret / row minimum).

Validation trains on the first T-V, V=min(60,floor(T/3)), selects minimum response-one
loss on the last V, then refits all T. Exact ties choose largest lambda. Neither
population oracle nor regret enters validation. The validation table reports selected
medians, mean population regret, MCSE, ratio to oracle and boundary frequency.
Bootstrap resamples whole independent replicas, paired across T and lambda, 500 times.
Intervals reselect the oracle within each bootstrap. They describe Monte Carlo error,
not sampling uncertainty of a single investment history. Grid discreteness remains.
Slopes report OLS SE and bootstrap SE/95% intervals. The predeclared headline subset
is the upper half, T=360,...,1440; full-grid and largest-four estimates are also saved.
OLS regression errors alone do not account for paired Monte Carlo dependence.

In the headline economy, mean validation-selected regret is
{validation_start.validation_oracle_ratio:.3f} times oracle regret at T=60 and
{validation_end.validation_oracle_ratio:.3f} times oracle regret at T=1440.
At T=1440 it is {validation_end.mean_validation_regret:.6f}, versus oracle
{validation_end.oracle_regret:.6f}. Median complexity markers near the oracle curve
therefore must not be interpreted as near-oracle mean performance: they conceal
loss dispersion and costly selection errors. The validation window is capped at
60 observations; it does not grow with T beyond 180. This experiment does not
establish rate-optimality of that finite-validation tuning rule.

## Honest comparison with the r=1 envelope

| b | Quantity | fitted slope | OLS SE | MC bootstrap SE | R squared | r=1 reference |
|---:|---|---:|---:|---:|---:|---:|
{rows}

RKHS membership supplies a bias upper bound, not a matching bias lower bound for
every fixed target. The supplied manuscript explicitly makes this distinction in
proofs_learning.tex, in the proof of effective complexity (around lines 950-953).
This simulation therefore checks the controlled operator/estimator and compares
rates with the envelope; it must not be described as establishing equality with a
worst-case minimax rate for this particular target.

Here the population ridge bias is exactly sum_j mu_j theta_j^2 [lambda/(mu_j+lambda)]^2.
For a=1, infinite power-law spectra and b>1 give bias of order lambda^(1+1/b),
which is smaller than the general O(lambda) bound. Balancing this pointwise bias
with order lambda^(-1/b)/T suggests lambda of order T^(-b/(b+2)), C of order
T^(1/(b+2)), and regret of order T^(-(b+1)/(b+2)). These are explanatory asymptotic
calculations, not fitted constraints or replacements for the requested r=1 comparison.
Finite T, AR dependence, empirical covariance estimation and finite J can alter slopes.
The smooth one-direction L target is reported separately and is not used to claim
the nonlinear law. No r parameter was added to the baseline.

## Finite-rank check and reproducibility

J=1000,2000,4000 are compared using the same prespecified T subsets, with independent
seeds across J. Stability tolerances declared before results: maximum upper-half slope
difference from J=2000 <0.08; largest-T mean-regret difference <10%.
Observed check: {json.dumps(stability)}.
If this check fails, the finite-rank acceptance condition remains unmet; a failure
cannot be concealed by switching the slope interval. The JSON and CSV retain it.

Compute command: `VECLIB_MAXIMUM_THREADS=1 python3 -u simulation_characteristic_factor.py --workers 2`.
The initial run used 2 BLAS threads/one process, then resumed from deterministic
checkpoints using four, subsequently two, one-thread workers to limit memory pressure;
results do not depend on scheduling.
Render: `VECLIB_MAXIMUM_THREADS=1 python3 render_characteristic_factor.py`.
Recorded completed-checkpoint compute time: {run['checkpoint_compute_seconds']:.1f} seconds.
The run manifest records code hashes, counts, seeds and timing. Licensed empirical
inputs are unrelated to the artificial economy and are never required for this run.
'''
    (out/'notes'/'simulation_characteristic_factor_methodology.md').write_text(text)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--headline-only',action='store_true')
    args=parser.parse_args();main(args.output,args.headline_only)
