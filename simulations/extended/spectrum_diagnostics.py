"""Unsmoothed spectrum geometry, local elasticity and finite-rank saturation."""
import json
import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT,BASIS_SEED,BASE_A,T_CANDIDATE,seed
from simulations.extended.population import spectral_path
from simulations.extended.statistics import deterministic_slopes
from simulations.provenance import json_write,utc_now


def run():
    primary=OUTPUT/'pilot'/f'spectra_baseline_P4096_B{BASIS_SEED}_Q32768_S{seed("population")}.npz'
    with np.load(primary) as z:
        spectra={key:z[key] for key in z.files}
    # These horizons diagnose a finite matrix only; they are not economic
    # replications and must never be added to the Monte Carlo learning curves.
    diagnostic_T=np.geomspace(60,1e24,241)
    C,E,local=spectral_path(spectra['managed'],BASE_A*diagnostic_T**(-.6))
    pd.DataFrame(dict(diagnostic_T=diagnostic_T,complexity=C,lambda_elasticity=E,
        local_T_elasticity=local,rank_fraction=C/4096,
        scope='Finite-matrix saturation diagnostic, not simulated economic histories')).to_csv(
            OUTPUT/'finite_rank_saturation_diagnostic.csv',index=False)
    comparisons=[]
    for path in sorted((OUTPUT/'pilot').glob('spectra_baseline_*.npz')):
        with np.load(path) as z:
            for operator in ('managed','kernel','factor','idiosyncratic'):
                v=z[operator]
                for j in (1,2,3,7,8,22,23,28,48,78):
                    comparisons.append(dict(source=path.name,operator=operator,rank_boundary=j,
                        preceding_eigenvalue=float(v[j-1]),following_eigenvalue=float(v[j]),
                        adjacent_ratio=float(v[j-1]/v[j])))
    table=pd.DataFrame(comparisons)
    table.to_csv(OUTPUT/'spectral_cluster_seed_comparison.csv',index=False)
    rows=[]
    for operator in ('managed','kernel','factor','idiosyncratic'):
        sub=table[table.operator==operator]
        for j,g in sub.groupby('rank_boundary'):
            rows.append(dict(operator=operator,rank_boundary=int(j),cases=len(g),
                adjacent_ratio_min=float(g.adjacent_ratio.min()),
                adjacent_ratio_max=float(g.adjacent_ratio.max())))
    pd.DataFrame(rows).to_csv(OUTPUT/'spectral_cluster_range_summary.csv',index=False)
    production_T=np.asarray(T_CANDIDATE,dtype=float)
    prodC,prodE,prodlocal=spectral_path(spectra['managed'],BASE_A*production_T**(-.6))
    report=dict(created_utc=utc_now(),operator='Baseline managed payoff',rank=4096,
        local_elasticity_at_candidate_T=dict(zip(map(str,T_CANDIDATE),map(float,prodlocal))),
        candidate_grid_complexity_slopes=deterministic_slopes(production_T,prodC),
        candidate_grid_complexity_fraction_rank=(prodC/4096).tolist(),
        complexity_fraction_rank_at_largest_candidate=float(prodC[-1]/4096),
        half_rank_diagnostic_T=float(diagnostic_T[np.flatnonzero(C>=2048)[0]]),
        saturation_end_complexity=float(C[-1]),saturation_end_local_elasticity=float(local[-1]),
        spectrum_interpretation='Large adjacent-eigenvalue gaps persist in independent basis and quadrature cases. Kernel clusters also occur before factor weighting, so they cannot all be artifacts of the factor economy. The radial Matérn kernel and uniform six-dimensional cube are invariant under coordinate permutations/reflections; the observed near-multiplicities are consistent with that symmetry. Three leading factor-related directions change the managed spectrum. This is a numerical interpretation, not a proof identifying every eigenspace.',
        component_warning='Ordered component eigenvalues are not added; managed, kernel, factor and idiosyncratic operators were diagonalized separately (noise is exactly a scalar multiple of the kernel operator).',
        scope='Candidate-horizon diagnostics precede production. Actual final-grid slopes must be recalculated on the frozen production grid. No segment regression is used to infer b.')
    json_write(OUTPUT/'spectrum_geometry_and_saturation.json',report)

if __name__=='__main__':
    run()
