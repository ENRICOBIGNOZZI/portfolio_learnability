"""Read-only diagnostics of the frozen 300-path study; never select a strategy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT, PENALTIES, ENVIRONMENTS, parameters
from simulations.extended.statistics import gap_slopes
from simulations.provenance import file_hash, json_write, utc_now

DEST = OUTPUT.parent / 'rich6d_sr3_numerical_followup'


def oracle_diagnostic(sr, regret, theory_index=-1):
    """Common ex-post grid optimum, plus disjoint-index selection/evaluation.

    Cross-fitting only measures grid-maximization optimism. It uses population
    outcomes and is NOT an implementable tuning procedure.
    """
    n, k = sr.shape
    candidates = k - 1
    j = int(np.argmax(sr[:, :candidates].mean(axis=0)))
    j_loss = int(np.argmin(regret[:, :candidates].mean(axis=0)))
    difference = sr[:, j] - sr[:, theory_index]
    crossfit = np.empty(n)
    selected = []
    for parity in (0, 1):
        train = np.arange(n) % 2 == parity
        pick = int(np.argmax(sr[train, :candidates].mean(axis=0)))
        crossfit[~train] = sr[~train, pick]
        selected.append(pick)
    return dict(grid_SR_index=j, grid_regret_index=j_loss,
                theory_SR=float(sr[:, theory_index].mean()),
                grid_max_mean_SR=float(sr[:, j].mean()),
                SR_improvement=float(difference.mean()),
                fixed_selected_index_paired_MCSE=float(difference.std(ddof=1)/np.sqrt(n)),
                theory_regret=float(regret[:, theory_index].mean()),
                SR_grid_choice_regret=float(regret[:, j].mean()),
                grid_min_mean_regret=float(regret[:, j_loss].mean()),
                crossfit_population_oracle_SR=float(crossfit.mean()),
                crossfit_even_selected_index=selected[0],
                crossfit_odd_selected_index=selected[1],
                grid_max_at_boundary=j in (0, candidates-1),
                note='Ex-post population diagnostic; fixed-index MCSE ignores selection uncertainty; cross-fit is also nonimplementable.'), crossfit


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    rows, decompositions, slopes, sources = [], [], [], {}
    for name in ENVIRONMENTS:
        source = OUTPUT/'distributions'/f'{name}.npz'
        sources[str(source.relative_to(OUTPUT))] = file_hash(source)
        with np.load(source) as z, np.load(OUTPUT/'population'/f'{name}_theory.npz') as pop:
            T, sr, regret = z['T'], z['annual_SR'], z['regret']
            floor = float(pop['projection_floor'])
            q = parameters(name).q_star
            subspace_SR = np.sqrt(12*(q-floor)/(1-q+floor))
            crossfit = []
            for t, horizon in enumerate(T):
                row, heldout = oracle_diagnostic(sr[:, t], regret[:, t])
                crossfit.append(heldout)
                lam = protocol['a'][name]*float(horizon)**(-.6)
                j = row['grid_SR_index']
                rows.append(dict(environment=name, T=int(horizon), replications=300,
                    theory_lambda=lam, grid_SR_lambda=PENALTIES[j],
                    grid_regret_lambda=PENALTIES[row['grid_regret_index']],
                    theory_over_grid_SR_lambda=lam/PENALTIES[j], **row))
                bias = float(z['theory_population_bias'][:, t].mean())
                mean_regret = float(regret[:, t, -1].mean())
                economic_SR = np.sqrt(12)*parameters(name).sr_star
                gap = economic_SR-sr[:, t, -1].mean()
                decompositions.append(dict(environment=name, T=int(horizon),
                    economic_optimal_SR=economic_SR, quadrature_subspace_optimal_SR=subspace_SR,
                    estimated_mean_SR=sr[:, t, -1].mean(), SR_gap=gap,
                    subspace_SR_gap=economic_SR-subspace_SR,
                    subspace_fraction_of_SR_gap=(economic_SR-subspace_SR)/gap,
                    projection_regret_floor=floor, mean_regret=mean_regret,
                    floor_fraction_of_regret=floor/mean_regret, population_bias=bias,
                    population_bias_excluding_floor=bias-floor,
                    bias_fraction_of_regret=bias/mean_regret,
                    estimation_norm=z['theory_estimation_norm'][:, t].mean(),
                    signed_cross_term=z['theory_cross_term'][:, t].mean()))
            crossfit = np.asarray(crossfit).T
            for label, gaps in [('theory', np.sqrt(12)*parameters(name).sr_star-sr[:, :, -1]),
                                ('crossfit_population_oracle', np.sqrt(12)*parameters(name).sr_star-crossfit)]:
                for r in gap_slopes(T, gaps):
                    if label == 'crossfit_population_oracle':
                        # Shared selection folds make these outcomes dependent
                        # across paths. Do not mislabel an iid delta/jackknife
                        # interval as unconditional oracle uncertainty.
                        for key in ('MCSE_delta', 'CI_low', 'CI_high', 'MCSE_jackknife'):
                            r.pop(key, None)
                        r['uncertainty'] = 'Descriptive only; shared cross-fit selection induces cross-path dependence.'
                    slopes.append(dict(environment=name, diagnostic=label, **r))
    pd.DataFrame(rows).to_csv(DEST/'oracle_diagnostic.csv', index=False)
    pd.DataFrame(decompositions).to_csv(DEST/'gap_decomposition.csv', index=False)
    pd.DataFrame(slopes).to_csv(DEST/'diagnostic_slopes.csv', index=False)
    json_write(DEST/'saved_results_audit.json', dict(completed_utc=utc_now(),
        source_run_hash=protocol['run_hash'], source_files=sources, rows=len(rows),
        replications_added=0, original_outputs_modified=False,
        caution='Bias plus estimation norm plus signed cross term is the exact regret decomposition. Bias/regret is not an additive variance share.'))


if __name__ == '__main__':
    run()
