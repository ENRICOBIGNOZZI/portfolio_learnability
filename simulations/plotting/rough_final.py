"""Four publication figures and honest finite-spectrum numerical reporting."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from simulations.provenance import json_write, file_hash

COLORS = ['#21618C','#138D75','#D68910','#A93226']
DISPLAY = [60,240,720,1440]
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
         'axes.labelsize': 10, 'legend.fontsize': 8, 'axes.spines.top': False,
         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .16,
         'pdf.fonttype': 42, 'savefig.facecolor': 'white'}


def slope(x,y):
    return float(np.polyfit(np.log(x),np.log(y),1)[0])


def save(fig,folder,name):
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),dpi=320,bbox_inches='tight')
    plt.close(fig)


def render(c,output,tables):
    curves, methods, rank, rates, verification = tables
    folder = output/'figures'; folder.mkdir(exist_ok=True)
    data = folder/'source_data'; data.mkdir(exist_ok=True)
    assert len(methods) == 10 and len(curves) == 960
    assert (methods.replications == 100).all() and (curves.replications == 100).all()
    assert set(methods['method']) == {'theory_1'}
    assert np.array_equal(methods['T'], [60,90,120,180,240,360,540,720,1080,1440])
    np.testing.assert_allclose(methods.lambda_mean, c['spec']['a']*methods['T'].to_numpy(dtype=float)**(-.6), rtol=1e-14)
    assert np.all(methods.sharpe_gap_mean > 0)
    mu = c['spectrum']
    assert len(mu) == 1024 and np.all(mu > 0) and np.all(np.diff(mu) <= 0)
    trace = lambda lam: np.sum(mu[:,None]/(mu[:,None]+np.atleast_1d(lam)),axis=0)
    theory = methods.copy()
    theory['population_complexity_1024'] = trace(theory.lambda_mean.to_numpy())
    theory['relative_population_complexity_1024'] = theory.population_complexity_1024/theory['T']
    theory['rank_audit_passed'] = rank[rank.lambda_index==96].sort_values('T').certified.to_numpy()
    theory['quadrature_audit_passed'] = pd.read_csv(output/'quadrature_resolution.csv').sort_values('T').certified.to_numpy()
    paths = curves.merge(rank[rank.lambda_index<96][['T','lambda','certified']],on=['T','lambda'],validate='one_to_one')
    paths['population_complexity_1024'] = trace(paths['lambda'].to_numpy())
    spectral = pd.DataFrame({'rank':np.arange(1,len(mu)+1),'eigenvalue':mu})
    for T in DISPLAY:
        lam = theory.loc[theory['T']==T,'lambda_mean'].iloc[0]
        spectral[f'filter_T{T}'] = mu/(mu+lam)
        spectral[f'cumulative_T{T}'] = np.cumsum(mu/(mu+lam))
    spectral.to_csv(data/'figure_0_spectrum.csv',index=False)
    theory.to_csv(data/'figures_1_2_theory.csv',index=False)
    paths[paths['T'].isin(DISPLAY)].to_csv(data/'figure_3_paths.csv',index=False)
    theory[theory['T'].isin(DISPLAY)].to_csv(data/'figure_3_exact_theory.csv',index=False)
    gap_rate = rates[(rates.quantity=='sharpe_gap')&(rates.window=='full')].iloc[0]
    slopes = {'sharpe_gap': float(gap_rate.OLS_slope),
              'gap_bootstrap_95': [float(gap_rate.bootstrap_low),float(gap_rate.bootstrap_high)],
              'population_complexity_1024': slope(theory['T'],theory.population_complexity_1024),
              'population_complexity_512': slope(theory['T'],theory.population_complexity_mean),
              'empirical_complexity_512': slope(theory['T'],theory.complexity_mean),
              'relative_population_complexity_1024': slope(theory['T'],theory.relative_population_complexity_1024),
              'penalty': slope(theory['T'],theory.lambda_mean)}
    json_write(output/'observed_slopes.json',slopes)
    pd.DataFrame([{'quantity':k,'slope':v} for k,v in slopes.items() if not isinstance(v,list)]).to_csv(output/'complexity_slopes.csv',index=False)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1,3,figsize=(13.6,4.2),layout='constrained')
        fig.suptitle('Figure 0. The economic spectrum',fontsize=15)
        axes[0].loglog(spectral['rank'],mu,color=COLORS[0],lw=1.8)
        axes[0].set(title='(a) Population eigenvalues',xlabel='Eigenvalue rank',ylabel=r'$\mu_j$')
        for T,color in zip(DISPLAY,COLORS):
            axes[1].semilogx(spectral['rank'],spectral[f'filter_T{T}'],color=color,label=f'T = {T}')
            axes[2].semilogx(spectral['rank'],spectral[f'cumulative_T{T}'],color=color,label=f'T = {T}')
        axes[1].set(title='(b) Shrinkage filters',xlabel='Eigenvalue rank',ylabel=r'$\mu_j/(\mu_j+\lambda_T)$')
        axes[2].set(title='(c) Cumulative effective complexity',xlabel='Eigenvalue rank',ylabel='Cumulative filter sum')
        axes[1].legend(frameon=False)
        axes[2].text(.04,.96,'Fixed rank-1024 population approximation\nInfinite-spectrum tail is not extrapolated',
                     transform=axes[2].transAxes,va='top',fontsize=8,color='#555555')
        save(fig,folder,'figure_0_economic_spectrum')

        T = theory['T'].to_numpy(); sr = theory.population_sharpe_mean.to_numpy(); gap = theory.sharpe_gap_mean.to_numpy()
        fig,axes = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        fig.suptitle('Figure 1. Theorem 1: Sharpe learnability',fontsize=15)
        axes[0].plot(T,sr,'o-',color=COLORS[0],label='Learned policy: theory_1',markersize=4)
        axes[0].fill_between(T,theory.population_sharpe_ci_low,theory.population_sharpe_ci_high,color=COLORS[0],alpha=.16,label='95% Monte Carlo interval')
        axes[0].axhline(c['p'].sr_star,color='#555555',ls='--',label=r'Analytical $SR^\star$')
        axes[0].set(title='(a) Population Sharpe ratio',xlabel='Training observations, T',ylabel='Sharpe ratio per period')
        unresolved_theory = ~(theory.rank_audit_passed & theory.quadrature_audit_passed).to_numpy()
        if np.any(unresolved_theory):
            axes[0].plot(T[unresolved_theory],sr[unresolved_theory],'x',color=COLORS[3],ms=8,label='Numerical check unresolved')
        axes[0].legend(frameon=False,loc='lower right')
        axes[1].loglog(T,gap,'o-',color=COLORS[0],label='Observed mean gap',markersize=4)
        axes[1].fill_between(T,theory.sharpe_gap_ci_low,theory.sharpe_gap_ci_high,color=COLORS[0],alpha=.16)
        axes[1].loglog(T,gap[0]*(T/T[0])**(-.6),ls='--',color='#555555',label=r'$T^{-0.6}$ reference')
        axes[1].set(title='(b) Distance from the optimum',xlabel='Training observations, T',ylabel=r'$SR^\star-SR(\widehat W)$')
        if np.any(unresolved_theory):
            axes[1].plot(T[unresolved_theory],gap[unresolved_theory],'x',color=COLORS[3],ms=8,label='Numerical check unresolved')
        axes[1].legend(frameon=False)
        axes[1].text(.04,.05,f'Full-grid slope: {slopes["sharpe_gap"]:.3f}\nPath-bootstrap 95% CI: [{gap_rate.bootstrap_low:.3f}, {gap_rate.bootstrap_high:.3f}]',transform=axes[1].transAxes,fontsize=9)
        save(fig,folder,'figure_1_sharpe_learnability')

        C = theory.population_complexity_1024.to_numpy()
        fig,axes = plt.subplots(1,3,figsize=(13.6,4.2),layout='constrained')
        fig.suptitle('Figure 2. Theorem 2: Effective complexity over time',fontsize=15)
        axes[0].loglog(T,theory.lambda_mean,'o-',color=COLORS[0],markersize=4,label=r'$aT^{-0.6}$')
        axes[0].set(title='(a) Prescribed regularization',xlabel='Training observations, T',ylabel=r'$\lambda_T$')
        axes[0].legend(frameon=False)
        for ax,relative in zip(axes[1:],[False,True]):
            divisor = T if relative else np.ones_like(T)
            ax.loglog(T,C/divisor,'o-',color=COLORS[0],markersize=4,label='Population: rank 1024')
            ax.loglog(T,theory.population_complexity_mean/divisor,ls='-.',color=COLORS[1],label='Population: rank 512')
            ax.loglog(T,theory.complexity_mean/divisor,ls=':',color=COLORS[3],label='Empirical: rank 512 (mean)')
            exponent = -.6 if relative else .4
            ax.loglog(T,C[0]/divisor[0]*(T/T[0])**exponent,ls='--',color='#555555',label=rf'$T^{{{exponent}}}$ reference')
            ax.set(xlabel='Training observations, T',ylabel=r'$\mathcal{C}_{1024}(\lambda_T)/T$' if relative else r'$\mathcal{C}_{1024}(\lambda_T)$')
        axes[1].set_title('(b) Effective complexity')
        axes[2].set_title('(c) Relative effective complexity')
        axes[1].legend(frameon=False,fontsize=7.4)
        axes[2].text(.04,.06,f'Population slopes: {slopes["population_complexity_1024"]:.3f} (absolute)\n{slopes["relative_population_complexity_1024"]:.3f} (relative)',transform=axes[2].transAxes,fontsize=8)
        save(fig,folder,'figure_2_effective_complexity')

        fig,ax = plt.subplots(figsize=(10,5.5),layout='constrained')
        fig.suptitle('Figure 3. Sharpe versus effective complexity',fontsize=15)
        for T,color in zip(DISPLAY,COLORS):
            f = paths[paths['T']==T].sort_values('population_complexity_1024')
            x,y,ok = f.population_complexity_1024.to_numpy(),f.population_sharpe_mean.to_numpy(),f.certified.to_numpy()
            passed_segments = ok[:-1] & ok[1:]
            boundaries = np.r_[0, np.flatnonzero(passed_segments[1:] != passed_segments[:-1])+1, len(passed_segments)]
            for start,stop in zip(boundaries[:-1],boundaries[1:]):
                ax.plot(x[start:stop+1],y[start:stop+1],color=color,lw=1.9,
                        ls='-' if passed_segments[start] else (0,(4,2.5)))
            row = theory[theory['T']==T].iloc[0]
            ax.plot(row.population_complexity_1024,row.population_sharpe_mean,'D',color=color,ms=7,mec='white',zorder=5)
        ax.axhline(c['p'].sr_star,color='#888888',lw=1,ls=':')
        ax.set(xscale='log',xlabel=r'Population effective complexity, $\mathcal{C}_{1024}(\lambda)$',ylabel='Population Sharpe ratio per period')
        handles = [Line2D([0],[0],color=co,label=f'T = {t}') for t,co in zip(DISPLAY,COLORS)]
        handles += [Line2D([0],[0],color='#555555',marker='D',ls='None',label='Exact theory penalty'),
                    Line2D([0],[0],color='#555555',label='Rank audit passed'),
                    Line2D([0],[0],color='#555555',ls='--',label='Rank sensitivity unresolved')]
        ax.legend(handles=handles,frameon=False,loc='upper left',ncol=2)
        save(fig,folder,'figure_3_sharpe_complexity')
    diagnostic_rows = []
    for groups in (8192,32768):
        for index in range(4):
            with np.load(output/f'moments/quadrature_{groups}_{index}.npz') as z:
                values = np.linalg.eigvalsh(z['second'])
            for row in theory.itertuples():
                diagnostic_rows.append({'groups': groups, 'seed_index': index, 'T': row.T,
                    'lambda': row.lambda_mean, 'population_complexity_512': float(np.sum(values/(values+row.lambda_mean)))})
    pd.DataFrame(diagnostic_rows).to_csv(output/'complexity_quadrature_sensitivity.csv',index=False)
    pd.DataFrame({'T': theory['T'], 'lambda': theory.lambda_mean,
        'population_complexity_512_production': theory.population_complexity_mean,
        'population_complexity_512_rank_audit': np.sum(np.linalg.eigvalsh(c['rank_low_second'])[:,None]/(np.linalg.eigvalsh(c['rank_low_second'])[:,None]+theory.lambda_mean.to_numpy()),axis=0),
        'population_complexity_1024_rank_audit': theory.population_complexity_1024
        }).to_csv(output/'complexity_rank_sensitivity.csv',index=False)
    from simulations.diagnostics import rough_operators
    rough_operators.validate(c,output)
    write_report(c,output,theory,paths,slopes,gap_rate)
    json_write(folder/'provenance.json', {'run_hash': c['run_hash'],
        'renderer_sha256': file_hash(__file__), 'operator_validator_sha256': file_hash(rough_operators.__file__),
        'matplotlib_version': matplotlib.__version__, 'backend': matplotlib.get_backend(), 'population_spectrum_sha256': file_hash(output/'spectrum.csv'),
        'artifact_sha256': {str(f.relative_to(folder)): file_hash(f) for f in folder.rglob('*')
            if f.is_file() and f.name != 'provenance.json'}})


def write_report(c,output,theory,paths,slopes,rate):
    failures = theory.loc[~theory.rank_audit_passed,'T'].tolist()
    qfail = theory.loc[~theory.quadrature_audit_passed,'T'].tolist()
    unresolved = int((~paths.certified).sum())
    captions = [
        'Figure 0. The economic spectrum. Panel (a) reports the 1,024 eigenvalues of the managed-payoff second-moment operator compressed to a nested Matérn-3/2 Nyström subspace, computed using 32,768 independent balanced groups. Panels (b) and (c) apply the exact theory-scaled penalties at T=60,240,720,1440 to this same fixed spectrum, showing shrinkage weights and their cumulative sums. This finite-rank population approximation neither supplies the infinite spectrum nor extrapolates its omitted tail. Eigenvalues are computed for the new rough loading map only.',
        f'Figure 1. Sharpe learnability. Panel (a) displays the mean population Sharpe ratio of the rank-512 learned policy across 100 independent economic replications and the analytical optimum {c["p"].sr_star:.9f}. Panel (b) reports the Sharpe gap and a T^(-0.6) theoretical reference anchored at T=60. Shading gives pointwise 95% Monte Carlo intervals conditional on the fixed numerical evaluator. All ten sample sizes and all replications are retained. The full-grid descriptive log-log slope is {slopes["sharpe_gap"]:.6f}, with 1,000-whole-path-bootstrap interval [{rate.bootstrap_low:.6f}, {rate.bootstrap_high:.6f}]. Sharpe ratios retain the original per-period units.',
        f'Figure 2. Effective complexity over time. Panel (a) shows lambda_T={c["spec"]["a"]:.12g} T^(-0.6), fixed by an independent population pilot. Panels (b) and (c) report C_1024(lambda_T) and C_1024(lambda_T)/T using the same population spectrum as Figure 0. The rank-512 population trace and mean empirical rank-512 complexity are labeled separately. The T^(0.4) and T^(-0.6) references are anchored at T=60, without estimating their exponents. The finite-spectrum population complexity slope is {slopes["population_complexity_1024"]:.6f}; the relative-complexity slope is {slopes["relative_population_complexity_1024"]:.6f}.',
        'Figure 3. Sharpe versus effective complexity. The full 96-penalty diagnostic paths at T=60,240,720,1440 plot 100-replication mean population Sharpe against the fixed population spectral coordinate C_1024(lambda). Diamonds locate policies estimated at the exact theory penalties, without snapping to the grid. Solid segments pass the predeclared 512-versus-1024 regret-based rank diagnostic on 50 matched paths; dashed segments touch a point failing it and remain visible. The audit is a regret-based numerical sensitivity check, not a uniform Sharpe-error guarantee. These paths are diagnostic and do not select a competing strategy.'
    ]
    (output/'figures/captions.txt').write_text('\n\n'.join(captions)+'\n')
    tr = json_load(output/'fourier_truncation.json')
    first,last = theory.iloc[0],theory.iloc[-1]
    window_rates = pd.read_csv(output/'rates.csv').set_index('window')
    compatible = rate.bootstrap_low <= -.6 <= rate.bootstrap_high
    interpretation = ('The full-grid estimate is compatible with the -.6 reference within its whole-path bootstrap interval; the point estimate is not an exact exponent equality.' if compatible else 'The full-grid bootstrap interval excludes the -.6 reference over these accessible sample sizes; this is a finite-sample finding.')
    upper = window_rates.loc['upper_half']
    largest = window_rates.loc['largest_four']
    interpretation += f' The prespecified upper-half window (T=360 through 1440) gives slope {upper.OLS_slope:.6f}, with interval [{upper.bootstrap_low:.6f}, {upper.bootstrap_high:.6f}]; the largest-four window gives {largest.OLS_slope:.6f}, with interval [{largest.bootstrap_low:.6f}, {largest.bootstrap_high:.6f}].'
    if upper.OLS_slope < -.6:
        interpretation += ' The faster decay in the upper-half window is compatible with smoother effective behavior at larger accessible sample sizes. It cautions against interpreting the full-grid agreement as identification of critical roughness. All windows are reported; the full grid remains the principal result.'
    pilot_seed = json_load(output/'calibration.json')['pilot_seed']
    report = f'''Rough Rich6D: critical numerical report

Design and scope
N=600, D=6, three Gaussian economic factors, stationary AR(1) characteristics (rho=.95), balanced triplets, c_beta=.04^2, sigma_eps=.08, mu_F=(.10,.06,.03), Matérn-3/2 with lengthscale 1. The new loading map is rich6d_rough, eta=.35. Original simulations and manuscript are not inputs or outputs. No empirical section is edited. The ideal nonlinear target is sum_(m=2)^infinity cos(pi*m*u)/(m^5 log(m+1)), normalized by Parseval to the sine RMS 1/sqrt(2). The normalization is {tr['normalization']:.15g} and is independent of portfolio performance.

Numerical Fourier implementation
M=128 was frozen before production after the 128/512/2048 common-innovation comparison (240 dates, 8,192 independent evaluation groups, fixed penalties 1e-7,1e-6,1e-5). Detailed absolute errors and tolerances are in fourier_truncation.csv/json. Vectorized Clenshaw evaluation does not use interpolation. The selected uniform psi tail bound is {tr['records'][0]['uniform_psi_tail_bound']:.6g}. Finite M is smooth. Agreement of finite-M values, payoffs, Sharpe and regret is not mathematical evidence of the exact source condition r=1 or of critical Sobolev regularity of the composed optimal policy. No proof or theoretical assumption is modified.

Calibration and independent evaluation
The existing theory_1 scale convention is retained: a is the largest eigenvalue of a fresh rank-512 population second-moment pilot with 8,192 groups, seed {pilot_seed}. a={c['spec']['a']:.15g}; lambda_T=a*T^(-.6). The exponent is prescribed from b=1.5 and r=1, never fitted. The frozen pilot, production evaluation, rank audit, quadrature audit, truncation check and economic replications use separate seed families. The population production evaluator uses 8,192 groups and integrates Gaussian returns analytically; it does not estimate Sharpe from a noisy future return sample. The pilot is a population-calibrated benchmark and is not a historically implementable selector. There is no annual Sharpe target or annualization.

Results across the complete grid
All 100 independent replications are used at each of the ten prescribed horizons. Histories are nested within replication; whole-path bootstrapping preserves this dependence. Mean population Sharpe rises from {first.population_sharpe_mean:.8f} at T=60 to {last.population_sharpe_mean:.8f} at T=1440, against analytical SR*={c['p'].sr_star:.8f}. The full-grid Sharpe-gap slope is {slopes['sharpe_gap']:.6f} (95% bootstrap interval [{rate.bootstrap_low:.6f}, {rate.bootstrap_high:.6f}]), compared with the reference -.6. Additional prespecified-window slopes are in rates.csv; no window is selected to improve agreement. C_1024 grows from {first.population_complexity_1024:.6f} to {last.population_complexity_1024:.6f}, with slope {slopes['population_complexity_1024']:.6f}, compared with +.4; its ratio to T has slope {slopes['relative_population_complexity_1024']:.6f}, compared with -.6. Population rank-512 and empirical rank-512 slopes are {slopes['population_complexity_512']:.6f} and {slopes['empirical_complexity_512']:.6f}.

Interpretation
{interpretation} The m=2 harmonic accounts for {100*(tr["normalization"]/(2**5*np.log(3)))**2:.4f}% of the normalized Fourier variance by Parseval. Low-frequency terms therefore dominate target values: rapid convergence of values does not make a roughness-driven rate observable. Neither a descriptive slope nor disagreement with an asymptotic reference proves or disproves an asymptotic theorem. No coefficient, amplitude, sample size, kernel or penalty exponent was adjusted after inspecting the rates. The finite spectrum also need not exhibit the asymptotic +.4 complexity exponent. No fitted spectral exponent replaces b=1.5 and no minimax-optimality claim is made.

Numerical resolution
The estimator stays rank 512 throughout the principal 100-path experiment. Fifty predeclared paths are also estimated in the nested rank-1024 basis, with identical returns. Both policies are evaluated on the same fresh 32,768-group operator. The predeclared criterion requires the rank-512 projection floor and |paired mean regret difference|+1.96*paired MCSE each to be no more than 5% of rank-512 regret. Theory horizons failing this criterion: {failures}. Across all ten full 96-penalty paths, {unresolved} of {len(paths)} cells fail and remain in the source tables and appropriate plotted curves. Quadrature diagnostics compare 8,192 and 32,768 groups using four independent seeds on the same 50 fixed theory policies; seeds use nested groups for paired differences. Theory horizons failing the 5% criterion: {qfail}. Checks are not uniform certificates for the infinite kernel. The finite projection floors are {c['floors']}.

All population complexity coordinates in Figures 0,2,3 use one fixed rank-1024, 32,768-group spectrum. This is a compressed finite-rank operator, not a list of the exact leading 1,024 infinite-population eigenvalues. The rank-512 trace and empirical sample complexity are explicitly separate. The numerical diagnostics audit policy regret near the exact chosen penalties; they cannot bound an uncomputed infinite-spectrum tail. Monte Carlo bands exclude numerical approximation uncertainty. verification.json independently reconstructs saved theory normal equations and population Sharpe, and checks the signed regret decomposition in every path and penalty cell.

Reproduction and artifacts
Run VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough --workers 2. protocol.json is frozen before main replications and binds science source hashes, configuration and truncation checks. calibration.json records the pilot. Replication checkpoints preserve rank-512 managed training histories, exact theory coefficients, every diagnostic metric, matched rank-comparison results, seeds and stock-return hashes. Rank-1024 training histories are reproducible from seeds. No failed economic replication is silently removed. manifest.json hashes output artifacts and records software versions. Four vector PDFs, 320-dpi PNGs, English captions and figure-level source CSVs are in figures/.
'''
    (output/'critical_report.txt').write_text(report)
    (output/'README.txt').write_text('Rough Rich6D, original per-period Sharpe units.\nRead critical_report.txt and figures/captions.txt for numerical scope and limitations.\nReproduce: VECLIB_MAXIMUM_THREADS=1 python3 -m simulations.run_rough --workers 2\n')


def json_load(path):
    import json
    return json.loads(Path(path).read_text())
