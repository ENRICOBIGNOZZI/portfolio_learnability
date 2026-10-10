"""Critical report assembled from complete production tables, with explicit limits."""
import json
import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT,ENVIRONMENTS,parameters
from simulations.extended.elasticity_comparison import run as compare_N_elasticities


def render():
    verification=json.loads((OUTPUT/'verification.json').read_text())
    if not verification['passed']:
        raise ValueError('Complete numerical verification before writing the final report.')
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    methods=pd.read_csv(OUTPUT/'theory_distributions.csv')
    slopes=pd.read_csv(OUTPUT/'slopes_all_windows.csv')
    common_slopes=pd.read_csv(OUTPUT/'slopes_robustness_common_grid.csv')
    pop=pd.read_csv(OUTPUT/'population_summary.csv')
    resolution=pd.read_csv(OUTPUT/'production_resolution.csv')
    quad=pd.read_csv(OUTPUT/'quadrature_audit.csv')
    paired=pd.read_csv(OUTPUT/'paired_robustness.csv')
    comparison=pd.read_csv(OUTPUT/'comparison_previous_100.csv')
    old_slopes=pd.read_csv(OUTPUT/'comparison_slopes_previous_100.csv')
    lines=[]
    add=lines.append
    def estimate(name,metric,T):
        return methods[(methods.environment==name)&(methods.metric==metric)&(methods['T']==T)].iloc[0]
    add('EXTENDED ROUGH RICH6D STUDY: CRITICAL REPORT')
    add(f"Run identity: {protocol['run_hash']}. Fixed Nyström rank {protocol['rank']}; {protocol['population_groups']} independent evaluation groups. There are 300 independent replications in each of five unique economies (1,500 economy-replications). Histories are nested across T and innovations are paired across economies. The baseline and rho variants have analytical annualized optimal Sharpe 3. All annualization is sqrt(12) times monthly marginal Sharpe, not compounded annual-return Sharpe.")
    add('1. NUMERICAL RESOLUTION')
    final_resolution=[]
    for _,row in resolution.iterrows():
        mean_regret=float(estimate(row.environment,'regret',int(row['T']))['mean'])
        floor_fraction=float(row.projection_floor/mean_regret)
        paired_fraction=float(row.paired_upper_bound/mean_regret)
        final_resolution.append(dict(environment=row.environment,T=int(row['T']),
            rank=int(row['rank']),projection_floor=float(row.projection_floor),
            production_mean_regret=mean_regret,floor_fraction=floor_fraction,
            paired_fraction=paired_fraction,floor_pass=floor_fraction<=.05,
            paired_pass=paired_fraction<=.05,
            forward_reference_available=bool(row.forward_reference_available),
            certified=bool(floor_fraction<=.05 and paired_fraction<=.05 and row.forward_reference_available)))
    final_resolution=pd.DataFrame(final_resolution)
    final_resolution.to_csv(OUTPUT/'production_resolution_final.csv',index=False)
    for name in ENVIRONMENTS:
        r=final_resolution[final_resolution.environment==name]
        failed=r[~(r.floor_pass&r.paired_pass)]
        horizons=', '.join(map(str,failed['T'].astype(int))) or 'none'
        add(f"{name}: failed checked 5% conditions at T = {horizons}; largest floor/production-regret fraction {r.floor_fraction.max():.2%}. The acceptance threshold remains 5%.")
    if not final_resolution.certified.all():
        add('High-T Nyström errors are NOT fully resolved. In addition to failed floor cells, the largest investigated rank has no independent larger-rank reference; its backward rank comparison is an agreement diagnostic, not an upper-rank certificate. The analytical economic optimum is retained everywhere. Increasing Monte Carlo replication count does not eliminate this approximation error.')
    else:
        add('All investigated production cells pass the declared numerical checks against the available higher-rank reference. This is a numerical certificate over the studied cells, not proof of an infinite-dimensional asymptotic law.')
    add('The optional T=7290 was investigated in the resource/rank pilots. '+
        ('It was omitted from production because the numerical feasibility requirement was not met; the decision was not based on a fitted rate.' if protocol['omitted_optional_T7290'] else 'It was included following the frozen numerical and resource decision.'))
    add(f"The largest quadrature discrepancy bound/regret fraction across the recorded pilot cells is {quad.fraction.max():.2%}; consult the cell-specific quadrature table. Full-penalty quadrature diagnostics on the first pilot are exploratory and do not have replication-based uncertainty. The selected theory rule is checked across all 12 independent pilot policies.")
    add('2. COMPLEXITY GEOMETRY')
    for name in ('baseline','N300','N1200'):
        r=pop[pop.environment==name].sort_values('T')
        add(f"{name}: exact local T elasticity runs from {r.local_T_elasticity.iloc[0]:.6f} at T={int(r['T'].iloc[0])} to {r.local_T_elasticity.iloc[-1]:.6f} at T={int(r['T'].iloc[-1])}. It is computed from the same finite population spectrum and independently checked by finite differences.")
    elasticities=compare_N_elasticities()
    for T in (min(protocol['robustness_T']),max(protocol['robustness_T'])):
        cases=elasticities[elasticities['T']==T]
        r=cases[cases.primary].iloc[0]
        add(f"At the common T={T}, local population-complexity elasticities for N=300,600,1200 are {r.N300:.6f}, {r.baseline:.6f}, {r.N1200:.6f}; the cross-N spread is {r.cross_N_range:.6f}. Across the {len(cases)} matched rank/basis/quadrature cases, the N300-minus-N600 difference ranges from {cases.N300_minus_baseline.min():.6f} to {cases.N300_minus_baseline.max():.6f}, and N1200-minus-N600 ranges from {cases.N1200_minus_baseline.min():.6f} to {cases.N1200_minus_baseline.max():.6f}.")
    add('These finite-window differences across N are economically distinct population geometries under the fixed factor economy and independently calibrated constants. The tabulated numerical-case ranges are sensitivity diagnostics, not confidence intervals or a certificate against the infinite-dimensional operator. They do not establish different asymptotic spectral exponents. The full common-grid comparison is in population_elasticity_N_comparison.csv; each source spectrum and hash is recorded in population_elasticity_N_audit.csv.')
    add('The baseline local elasticity passes near 0.4 and then exceeds it over the extended candidate horizons. This does not establish convergence to 0.4. A full-window log-log slope, a local population elasticity and the empirical-complexity distribution are different objects. The saturation diagnostic extends only the finite matrix, not the simulated economic histories: eventually C approaches P and its local elasticity approaches zero. Actual-study C/P is small, so the pronounced policy projection floor must not be confused with global trace saturation.')
    add('Unsmoothed clusters persist across basis and quadrature seeds and multiple ranks. Similar clusters occur in the kernel operator before factor weighting. They are consistent with permutation/reflection symmetries of a radial Matérn kernel on the uniform six-dimensional cube; the factor economy changes the leading directions. This is an interpretation of the operator comparisons, not a proof identifying every eigenspace. Component eigenvalues were never added as if the component operators shared eigenvectors. The j^(-1.5) curve is an order reference, not a fitted law.')
    add('3. ALL PREDECLARED SLOPES')
    add('The baseline extended-grid slopes are reported first. All five economies are then compared on their identical frozen robustness grid; for nonbaseline economies that common grid is also their full production grid.')
    for _,r in slopes[slopes.environment=='baseline'].iterrows():
        if pd.isna(r.get('slope',np.nan)):
            continue
        uncertainty='' if pd.isna(r.get('MCSE_delta',np.nan)) else f"; delta MCSE {r.MCSE_delta:.6f}, 95% Monte Carlo interval [{r.CI_low:.6f}, {r.CI_high:.6f}], jackknife MCSE {r.MCSE_jackknife:.6f}"
        add(f"{r.environment}, {r.quantity}, {r.window}, T <= {int(r.T_max)}: {r.slope:.6f}{uncertainty}.")
    add('Matched-grid robustness slopes use the same frozen horizons through T=3240 for every economy, including the baseline. They are reported separately from the baseline full-grid analysis through T=4860; differences across economies are not inferred from unequal fitting ranges.')
    for _,r in common_slopes.iterrows():
        uncertainty='' if pd.isna(r.get('MCSE_delta',np.nan)) else f"; delta MCSE {r.MCSE_delta:.6f}, 95% Monte Carlo interval [{r.CI_low:.6f}, {r.CI_high:.6f}], jackknife MCSE {r.MCSE_jackknife:.6f}"
        add(f"Common grid, {r.environment}, {r.quantity}, {r.window}, T <= {int(r.T_max)}: {r.slope:.6f}{uncertainty}.")
    add('These are descriptive finite-window exponents. Monte Carlo intervals quantify uncertainty from 300 independent paths, preserving full within-path cross-T covariance. They do not include population quadrature or finite-rank error. An O_P(T^(-0.6)) upper-rate prediction does not require a finite-grid regression slope to equal -0.6; a faster observed slope can be compatible with that bound. Failed high-T numerical checks preclude a decisive infinite-dimensional asymptotic interpretation, irrespective of proximity to -0.6.')
    add('4. CROSS-SECTIONAL BREADTH AND TEMPORAL DEPENDENCE')
    first=min(protocol['robustness_T']);last=max(protocol['robustness_T'])
    for name in ENVIRONMENTS:
        start=estimate(name,'annual_SR',first);end=estimate(name,'annual_SR',last)
        gap=estimate(name,'annual_gap',last)
        add(f"{name}: analytical optimum {parameters(name).sr_star*np.sqrt(12):.6f}; learned mean annual Sharpe {start['mean']:.6f} at T={first} and {end['mean']:.6f} at T={last}. At the common last horizon, the own-optimum gap is {gap['mean']:.6f}, MCSE {gap.mcse:.6f}; the learned-Sharpe central 95% replication interval is [{end.p025:.6f}, {end.p975:.6f}].")
    for name in ('N300','N1200','rho000','rho075'):
        r=paired[(paired.environment==name)&(paired.metric=='annual_gap')&(paired['T']==last)].iloc[0]
        add(f"At T={last}, {name} minus baseline own-optimum gap: {r['mean']:.6f}, paired MCSE {r.mcse:.6f} (Monte Carlo mean interval [{r['mean']-1.96*r.mcse:.6f}, {r['mean']+1.96*r.mcse:.6f}]). A negative difference means better recovery relative to that economy's own opportunity set.")
    add('N changes residual diversification and the analytical opportunity set, with factor means and covariance held fixed. Absolute Sharpe and own-optimum recovery are therefore reported separately. The rho experiments share the same stationary operator, population complexity, analytical optimum and regularization constant; only serial dependence changes. No ad hoc effective-sample-size substitution is made. End-horizon differences and slope tables must not be generalized into a universal monotonic learning-rate ordering.')
    add('The N-dependent projection floors must also be considered when interpreting cross-sectional comparisons: unequal finite-rank approximation error can contribute to the apparent difference in recovery. The rho comparison has a common population projection floor, but that fact alone does not certify every fitted-policy difference against a larger numerical rank.')
    add('5. MORE REPLICATIONS, MORE HISTORY, AND NUMERICAL APPROXIMATION')
    np.testing.assert_allclose(comparison.new300_selected_rank-comparison.old100_rank512_old_quadrature,
        comparison.quadrature_contribution+comparison.rank_contribution+comparison.additional_replications_contribution,atol=1e-12)
    for T in (60,1440):
        r=comparison[comparison['T']==T].iloc[0]
        add(f"At T={T}, the previous 100-path mean annual Sharpe {r.old100_rank512_old_quadrature:.6f} becomes {r.new300_selected_rank:.6f}. The additive changes are evaluation quadrature {r.quadrature_contribution:+.6f}, numerical rank on the same first 100 return histories {r.rank_contribution:+.6f}, and the additional 200 paths {r.additional_replications_contribution:+.6f}. All first-100 baseline stock-return hashes agree with the reference run.")
    for _,r in old_slopes[(old_slopes.window=='full')].iterrows():
        add(f"Comparison slope, {r.comparison}: {r.slope:.6f}.")
    complexity_comparison=pd.read_csv(OUTPUT/'complexity_comparison_previous.csv')
    reference=complexity_comparison[(complexity_comparison.source=='previous_rank1024')&
        (complexity_comparison.grid=='original')&(complexity_comparison.window=='full')].iloc[0]
    add(f"The previous rank-1024 population-complexity slope on the original grid is {reference.slope:.6f}. Population complexity does not change merely because the Monte Carlo replication count increases.")
    current=complexity_comparison[(complexity_comparison.source=='new_population')&
        (complexity_comparison['rank']==protocol['rank'])&(complexity_comparison.window=='full')]
    chosen_grid='required_extended' if protocol['omitted_optional_T7290'] else 'candidate_7290'
    for _,r in current[current.grid.isin(['original',chosen_grid])].iterrows():
        add(f"New selected-rank population-complexity slope, {r.grid}: {r.slope:.6f}. The same-grid change reflects numerical resolution; the subsequent grid change reflects the longer range of prescribed penalties.")
    add('The original-grid versus extended-grid comparison at the same 300-path numerical specification isolates the fitting-window/history extension. It is not interchangeable with the effect of more replications or a changed finite-rank approximation.')
    add('6. UNCERTAINTY AND UNSUPPORTED CLAIMS')
    add('All main and robustness bands are central 95% replication-percentile bands, not confidence intervals for the mean. Means, medians, standard deviations, both percentiles and mean MCSE are retained for every T and candidate penalty. Regularization bias, estimation norm and the cross term sum to analytical-optimum regret for the exact theory choice. These quantities remain distinct from the projection floor, quadrature error and serial-dependence effects.')
    add('The 300-path distributions provide Monte Carlo evidence for the finite, stated experiment. They do not cure numerical under-resolution. Unsupported claims include a proof of the r=1 source condition (the numerical Fourier truncation is finite and smooth), a minimax theorem, an exact power law at every eigenvalue, exact asymptotic exponents inferred from finite windows, an everywhere-optimal multiplicative regularization constant, a universal N or rho ordering, and an independent higher-rank certificate for the largest tested rank. Diagnostic penalty paths are ex post; the reported rule always remains a*T^(-0.6).')
    add('No manuscript or existing empirical results were modified. All ten figures require separate visual review in addition to the numerical verifier.')
    (OUTPUT/'critical_report.txt').write_text('\n\n'.join(lines)+'\n')

if __name__=='__main__':
    render()
