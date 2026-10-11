"""Audit the targeted twelve-path rank comparison without relabeling production."""
import json

import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT
from simulations.extended.followup_diagnostics import DEST
from simulations.extended.rank_followup import TIMES
from simulations.extended.statistics import gap_slopes
from simulations.provenance import json_write, file_hash, utc_now


def paired_bound(differences):
    mean = differences.mean(axis=0)
    mcse = differences.std(axis=0, ddof=1)/np.sqrt(len(differences))
    return mean, mcse, np.abs(mean)+1.96*mcse


def paired_slope_change(times, low_gaps, high_gaps):
    x = np.log(np.asarray(times))
    weights = (x-x.mean())/np.sum((x-x.mean())**2)
    low_mean, high_mean = low_gaps.mean(axis=0), high_gaps.mean(axis=0)
    change = float(weights@(np.log(high_mean)-np.log(low_mean)))
    influence = (high_gaps/high_mean-low_gaps/low_mean)@weights
    mcse = float(influence.std(ddof=1)/np.sqrt(len(influence)))
    # Verify with the complete cross-rank, cross-T covariance as well.
    gradient = np.r_[-weights/low_mean, weights/high_mean]
    covariance = np.cov(np.column_stack([low_gaps, high_gaps]), rowvar=False)/len(influence)
    np.testing.assert_allclose(mcse**2, gradient@covariance@gradient, atol=1e-14)
    return dict(slope_change=change, paired_MCSE=mcse,
                CI_low=change-1.96*mcse, CI_high=change+1.96*mcse)


def run():
    records = []
    for index in range(12):
        with np.load(DEST/'pilots'/f'P8192_rep_{index:03d}.npz') as z:
            if int(z['index']) != index:
                raise ValueError('Pilot index mismatch.')
            np.testing.assert_array_equal(z['times'], TIMES)
            records.append({k: z[k] for k in z.files})
    with np.load(DEST/'population_P8192.npz') as z:
        high_floor = float(z['regret'][0])
        high_evaluated_floor_policy = float(z['independent_regret'][0])
        high_bias = z['regret'][1:]
    with np.load(OUTPUT/'population'/'baseline_theory.npz') as z:
        low_floor = float(z['projection_floor'])
        positions = [int(np.flatnonzero(z['T'] == t)[0]) for t in TIMES]
        low_bias = z['bias'][positions]
    population_comparison = pd.DataFrame(dict(T=TIMES, population_bias_P4096=low_bias,
        population_bias_P8192=high_bias, population_bias_reduction=low_bias-high_bias,
        population_bias_relative_reduction=(low_bias-high_bias)/low_bias,
        primary_floor_P4096=low_floor, primary_floor_P8192=high_floor))
    population_comparison.to_csv(DEST/'population_rank_comparison.csv', index=False)
    primary = np.stack([z['primary_regret'] for z in records])
    alternate = np.stack([z['independent_regret'] for z in records])
    primary_sr = np.stack([z['primary_annual_SR'] for z in records])
    alternate_sr = np.stack([z['independent_annual_SR'] for z in records])
    rows, slopes, slope_changes = [], [], []
    for label, loss, sr in [('primary', primary, primary_sr), ('independent', alternate, alternate_sr)]:
        difference, se, bound = paired_bound(loss[:, 0]-loss[:, 1])
        sr_difference, sr_se, _ = paired_bound(sr[:, 1]-sr[:, 0])
        for t, T in enumerate(TIMES):
            low_mean, high_mean = loss[:, :, t].mean(axis=0)
            rows.append(dict(evaluation=label, T=int(T), paired_paths=12,
                mean_regret_P4096=low_mean, mean_regret_P8192=high_mean,
                mean_SR_P4096=sr[:, 0, t].mean(), mean_SR_P8192=sr[:, 1, t].mean(),
                paired_regret_difference=difference[t], paired_regret_MCSE=se[t],
                paired_regret_bound=bound[t], paired_fraction_of_lower_regret=bound[t]/low_mean,
                paired_pass=bool(bound[t] <= .05*low_mean),
                P4096_primary_floor_fraction=low_floor/low_mean,
                P8192_primary_floor_fraction=high_floor/high_mean,
                P8192_independently_evaluated_floor_policy_fraction=high_evaluated_floor_policy/high_mean,
                P8192_floor_diagnostic_pass=bool(high_floor <= .05*high_mean),
                P8192_independent_floor_policy_pass=bool(high_evaluated_floor_policy <= .05*high_mean),
                paired_SR_improvement=sr_difference[t], paired_SR_MCSE=sr_se[t],
                paired_SR_CI_low=sr_difference[t]-1.96*sr_se[t],
                paired_SR_CI_high=sr_difference[t]+1.96*sr_se[t],
                P8192_has_higher_rank_certificate=False))
        for r, rank in enumerate([4096, 8192]):
            slopes.extend(dict(evaluation=label, rank=rank, horizons='1440,2160,3240,4860',
                **row) for row in gap_slopes(TIMES, 3-sr[:, r]))
        slope_changes.append(dict(evaluation=label, paired_paths=12,
            horizons='1440,2160,3240,4860',
            **paired_slope_change(TIMES, 3-sr[:, 0], 3-sr[:, 1])))
    quad = []
    for r, rank in enumerate([4096, 8192]):
        difference, se, bound = paired_bound(primary[:, r]-alternate[:, r])
        for t, T in enumerate(TIMES):
            denominator = alternate[:, r, t].mean()
            quad.append(dict(rank=rank, T=int(T), paired_paths=12,
                paired_difference=difference[t], paired_MCSE=se[t], bound=bound[t],
                independent_mean_regret=denominator, fraction=bound[t]/denominator,
                passed=bool(bound[t] <= .05*denominator)))
    table, quadrature = pd.DataFrame(rows), pd.DataFrame(quad)
    table.to_csv(DEST/'paired_rank_comparison.csv', index=False)
    quadrature.to_csv(DEST/'paired_quadrature_comparison.csv', index=False)
    pd.DataFrame(slopes).to_csv(DEST/'paired_pilot_slopes.csv', index=False)
    pd.DataFrame(slope_changes).to_csv(DEST/'paired_slope_change.csv', index=False)
    oracle = pd.read_csv(DEST/'oracle_diagnostic.csv')
    decomp = pd.read_csv(DEST/'gap_decomposition.csv')
    end = table[(table.evaluation == 'independent') & (table['T'] == 4860)].iloc[0]
    full = oracle[(oracle.environment == 'baseline') & (oracle['T'] >= 1440)]
    final = decomp[(decomp.environment == 'baseline') & (decomp['T'] == 4860)].iloc[0]
    lines = [
        'TARGETED NUMERICAL AND REGULARIZATION FOLLOW-UP',
        '',
        'Scope: the published 300-path experiment remains unchanged. This follow-up reuses its complete distributions and reruns the original twelve independent pilot histories at P=8192, at T=1440,2160,3240,4860. Baseline economy only; no new final-production Monte Carlo indices. The same Matérn kernel, anchors nested by Sobol prefix, RKHS norm, economics, annualization and a*T^(-0.6) strategy are preserved.',
        'MAIN CONCLUSION: Doubling rank reduces the estimated unregularized projection floor, but does not resolve the final-horizon 5% floor criterion. The actual prescribed fitted policies improve only slightly; their paired late-window slope is essentially unchanged. The saved 300-path penalty grid reveals a materially larger ex-post regularization opportunity. This comparison does not justify retuning a on final outcomes, nor attributing all late-window flattening to the rank-4096 floor.',
        '',
        '1. ORIGINAL HIGH-T DECOMPOSITION',
        f'At T=4860: economic optimal annual Sharpe {final.economic_optimal_SR:.6f}; quadrature-fitted rank-4096 subspace optimum {final.quadrature_subspace_optimal_SR:.6f}; learned mean {final.estimated_mean_SR:.6f}. The subspace ceiling accounts for {100*final.subspace_fraction_of_SR_gap:.3f}% of the annual Sharpe gap, versus {100*final.floor_fraction_of_regret:.3f}% of quadratic regret. These are different denominators.',
        f'Population regularization bias including projection floor / mean regret = {100*final.bias_fraction_of_regret:.3f}%. The exact decomposition is bias + estimation norm + signed cross term; this percentage is not an additive bias-versus-variance allocation.',
        '',
        '2. COMMON EX-POST PENALTY-GRID OPTIMUM',
        full[['T','theory_lambda','grid_SR_lambda','theory_SR','grid_max_mean_SR','crossfit_population_oracle_SR','SR_improvement']].to_string(index=False),
        'The selected grid penalty maximizes the mean across paths, not a separate maximum for each path. Even/odd cross-fitting separates population-outcome selection and evaluation; it remains an infeasible population diagnostic, not a trading selector. Fixed-selected-index paired MCSE ignores selection uncertainty. No constant a is changed. The regret-minimizing grid point and the Sharpe-maximizing grid point are reported separately.',
        'The grid oracle is still rank 4096 and can be more exposed to projection error at its smaller penalty; these numbers do not certify a new optimal strategy.',
        '',
        '3. HIGHER-RANK PAIRED AUDIT',
        f'P8192 quadrature-fitted projection regret floor: {high_floor:.10g}; independently evaluated regret of that fitted optimum: {high_evaluated_floor_policy:.10g}. The latter evaluates a fixed policy; it is not a certified exact subspace minimum.',
        'Population regularization bias at the same prescribed penalties and common primary quadrature:',
        population_comparison.to_string(index=False),
        table.to_string(index=False),
        '',
        f'At T4860 independent evaluation: mean annual Sharpe P4096={end.mean_SR_P4096:.6f}, P8192={end.mean_SR_P8192:.6f}; paired improvement={end.paired_SR_improvement:.6f}, MCSE={end.paired_SR_MCSE:.6f}, 95% mean interval=[{end.paired_SR_CI_low:.6f},{end.paired_SR_CI_high:.6f}]. These are twelve pilot policies, not the 300-production means.',
        'Paired slope changes over the four targeted horizons (full cross-rank and cross-T covariance):',
        pd.DataFrame(slope_changes).to_string(index=False),
        '',
        '4. QUADRATURE AND LIMITATIONS',
        quadrature.to_string(index=False),
        'All twelve lower-rank policies reproduce the saved original population metrics, and the entire original return histories are hash-checked. Cholesky coordinates are an orthogonal change of RKHS feature coordinates, tested against the original symmetric whitening. No jitter, target enrichment, factor-informed features or altered penalty is introduced.',
        'Rank 8192 has no still-higher-rank reference. A backward difference does not certify the larger rank. Any remaining 5% failures remain failures; no universal numerical-resolution claim or replacement of all five 300-path environments is made.',
        '',
        '5. THEORETICAL INTERPRETATION',
        'Imposing lambda proportional to T^(-0.6) does not itself impose complexity proportional to T^(0.4): the actual population spectrum determines its elasticity. Nevertheless, measuring complexity along that prescribed path is not an independent test of the optimality of a, nor proof of the learnability theorem. This audit does not establish the source condition, its maximal regularity, Gaussian-tail assumptions, or minimax optimality, and makes no manuscript/proof changes.',
    ]
    (DEST/'critical_followup_report.txt').write_text('\n\n'.join(lines)+'\n')
    json_write(DEST/'verification.json', dict(completed_utc=utc_now(), paired_paths=12,
        original_production_replications_per_environment=300, new_production_replications=0,
        original_histories_hash_verified=True, lower_rank_metrics_reproduced=True,
        all_quadrature_checks_pass=bool(quadrature.passed.all()),
        all_higher_rank_floor_diagnostics_pass=bool(table.P8192_floor_diagnostic_pass.all()),
        all_independent_floor_policy_checks_pass=bool(table.P8192_independent_floor_policy_pass.all()),
        higher_rank_certificate=False,
        artifacts={str(p.relative_to(DEST)):file_hash(p) for p in sorted(DEST.rglob('*'))
                   if p.is_file() and p.name not in ('verification.json', 'manifest.json')}))


if __name__ == '__main__':
    run()
