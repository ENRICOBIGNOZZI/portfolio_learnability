"""V2 simulation manuscript, with cell-level lineage for every reported number."""
import json
from pathlib import Path

import pandas as pd

from simulations.plotting.tables import emit
from simulations.provenance import file_hash, json_write, require


def generate(output):
    output = Path(output)
    directory = output/'paper'; directory.mkdir(parents=True, exist_ok=True)
    audit = json.loads((output/'audit/confirmation_reconstruction.json').read_text())
    require(audit['passed'] and not audit['partial'] and not audit['raw_only'], 'Full independent reconstruction is required')
    protocol = json.loads((Path(__file__).parent/'config/confirmation_v2.json').read_text())
    frames, lineage, macros = {}, [], []
    def cell(file, filters, column):
        if file not in frames:
            frames[file] = pd.read_csv(output/file)
        frame = frames[file]
        selected = frame
        for key, value in filters.items():
            selected = selected[selected[key] == value]
        require(len(selected) == 1, f'Publication cell is ambiguous: {file} {filters}')
        value = selected.iloc[0][column]
        return value.item() if hasattr(value, 'item') else value
    def number(file, filters, column, fmt='.5f'):
        value = cell(file, filters, column)
        name = 'simNumber'+''.join(chr(65+int(d)) for d in str(len(macros)))
        rendered = format(value, fmt)
        macros.append('\\newcommand{\\'+name+'}{'+rendered+'}')
        lineage.append({'kind': 'prose', 'macro': name, 'file': file, 'filters': filters,
                        'column': column, 'format': fmt, 'rendered': rendered})
        return '\\'+name+'{}'
    def table(name, specifications, caption):
        rows = []
        for index, spec in enumerate(specifications):
            row = {}
            for column, value in spec.items():
                if isinstance(value, tuple):
                    source, filters, field = value
                    row[column] = cell(source, filters, field)
                    lineage.append({'kind': 'table', 'table': f'tables/{name}.csv', 'row': index,
                                    'target_column': column, 'file': source, 'filters': filters, 'column': field})
                else:
                    row[column] = value
            rows.append(row)
        emit(pd.DataFrame(rows), output/'tables', name, caption, 'tab:'+name)
    def source(name, T, method, column):
        return (f'data/{name}/methods.csv', {'T': T, 'method': method}, column)
    friendly = {'baseline_original': 'Baseline', 'rich6d': 'Rich6D'}
    method_labels = {'holdout25': 'Holdout', 'rolling3': 'Rolling', 'theory_1': 'Theory a=1',
                     'ensemble_plugin_loss_oracle': 'Plugin oracle', 'crossfit_loss_oracle': 'Crossfit oracle',
                     'zero': 'Zero', 'linear_validation': 'Affine', 'equal_weight': 'Equal weight'}
    rows = []
    for name in friendly:
        for method, label in method_labels.items():
            rows.append({'Map': friendly[name], 'Method': label,
                         'R': source(name, 1440, method, 'replications'),
                         'Loss': source(name, 1440, method, 'population_loss_mean'),
                         'SR': source(name, 1440, method, 'population_sharpe_mean'),
                         'Regret': source(name, 1440, method, 'regret_mean'),
                         'OOS loss': source(name, 1440, method, 'oos_loss_mean'),
                         'Future loss': source(name, 1440, method, 'forward_loss_mean')})
    table('confirmation_main', rows, 'Prospective confirmation at T=1440. Population risk uses independent quadrature; independent and contiguous-future test losses remain distinct. Sharpe is per period.')
    rows = []
    for name in friendly:
        for method in ('holdout25', 'rolling3'):
            file, filters = f'data/{name}/selection_diagnostics.csv', {'T': 1440, 'selector': method}
            rows.append({'Map': friendly[name], 'Selector': method_labels[method],
                         'Mean lambda': (file, filters, 'lambda_mean'), 'Median': (file, filters, 'lambda_median'),
                         'Geometric mean': (file, filters, 'lambda_geometric_mean'),
                         'Lower freq.': (file, filters, 'lower_frequency'), 'Upper freq.': (file, filters, 'upper_frequency')})
    table('confirmation_selection', rows, 'Separate penalty summaries and endpoint selection frequencies at T=1440. Frequencies are fractions; the zero policy is not eligible for validation.')
    rows = []
    for name in friendly:
        for method in ('theory_0.25', 'theory_1', 'theory_4'):
            rows.append({'Map': friendly[name], 'a': method.split('_')[1],
                         'R': source(name, 7290, method, 'replications'),
                         'lambda': source(name, 7290, method, 'lambda_mean'),
                         'Regret': source(name, 7290, method, 'regret_mean'),
                         'Scaled regret': source(name, 7290, method, 'scaled_regret_mean'),
                         'C / T': source(name, 7290, method, 'complexity_over_T_mean')})
    table('confirmation_theory', rows, 'Exact theory penalties at T=7290 in the extended cohort. The scaled risk is descriptive; no equality with an asymptotic envelope is required. These extra horizons are not certified by the practical-grid rank audit.')
    rows = []
    for name in friendly:
        for method in ('holdout25', 'rolling3', 'theory_1'):
            rankfile = f'audit/{name}_rank_policy_resolution.csv'
            quadfile = f'audit/{name}_quadrature_resolution.csv'
            filters = {'T': 1440, 'method': method}
            rows.append({'Map': friendly[name], 'Method': method_labels[method],
                         'Floor / risk': (rankfile, filters, 'floor_fraction'),
                         'Rank bound / risk': (rankfile, filters, 'difference_confidence_fraction'),
                         'Rank pass': (rankfile, filters, 'certified'),
                         'Quadrature pass': (quadfile, filters, 'certified')})
    table('confirmation_resolution', rows, 'Numerical resolution at T=1440. The rank bound is absolute paired mean difference plus 1.96 MCSE. Both ratios must be at most 0.05, and the independent quadrature check must also pass. False flags retain unresolved results.')
    unresolved_rows = []
    for name in friendly:
        rankfile = f'audit/{name}_rank_policy_resolution.csv'
        quadfile = f'audit/{name}_quadrature_resolution.csv'
        rank_frame, quad_frame = pd.read_csv(output/rankfile), pd.read_csv(output/quadfile)
        combined = rank_frame.merge(quad_frame, on=['T', 'method'], suffixes=('_rank', '_quad'), validate='one_to_one')
        require(len(combined) == len(rank_frame), 'Missing paired numerical audit cells')
        for _, row in combined[~(combined.certified_rank&combined.certified_quad)].iterrows():
            filters = {'T': int(row['T']), 'method': row.method}
            unresolved_rows.append({'Map': friendly[name], 'Method': (rankfile, filters, 'method'),
                                    'T': (rankfile, filters, 'T'),
                                    'Floor / risk': (rankfile, filters, 'floor_fraction'),
                                    'Rank bound / risk': (rankfile, filters, 'difference_confidence_fraction'),
                                    'Rank pass': (rankfile, filters, 'certified'),
                                    'Quadrature pass': (quadfile, filters, 'certified')})
    unresolved_tables = []
    for first in range(0, len(unresolved_rows), 18):
        name = f'confirmation_unresolved_{first//18+1}'
        unresolved_tables.append(name)
        table(name, unresolved_rows[first:first+18], 'Practical-grid policy cells failing at least one predeclared numerical check. Every failing cell is retained here and in the machine-readable results. The extra horizons have no corresponding rank audit and remain outside the certified region regardless of quadrature status.')
    maximum_tables = []
    for name in friendly:
        file = f'audit/{name}_rank_resolution.csv'
        frame = pd.read_csv(output/file)
        rows = []
        for T, group in frame[frame.lambda_index < 96].groupby('T'):
            ordered = group.sort_values('lambda_index')
            peak = ordered.loc[ordered.regret512.idxmax()]
            filters = {'T': int(T), 'lambda_index': int(peak.lambda_index)}
            rows.append({'T': (file, filters, 'T'), 'lambda': (file, filters, 'lambda'),
                         'Max regret512': (file, filters, 'regret512'),
                         'Same lambda regret1024': (file, filters, 'regret1024'),
                         'C512': (file, filters, 'complexity512'), 'C1024': (file, filters, 'complexity1024'),
                         'Payoff error': (file, filters, 'payoff_squared_difference_512_1024'),
                         'Rank pass': (file, filters, 'certified')})
        table_name = 'confirmation_curve_maximum_'+name
        maximum_tables.append(table_name)
        table(table_name, rows, friendly[name]+': sensitivity at the largest mean population loss on the fixed 96-penalty grid, in the 50-path numerical audit. Maximizing regret is equivalent because the analytic optimum is constant. The rank-1024 comparison uses the same penalty, without reselection. C512 and C1024 show proximity to the sample-rank boundary; payoff error is the squared difference on common quadrature. This is an ex-post diagnostic, not a tuning rule.')
    gradient_rows = []
    for coordinate in range(1, 7):
        file, filters = 'audit/rich6d_gradient_moments.csv', {'loading_map': 'rich6d', 'coordinate': coordinate}
        gradient_rows.append({'Coordinate': coordinate,
                              'Analytic mean square': (file, filters, 'analytic_squared_derivative_expectation'),
                              'Uniform-cube MC mean': (file, filters, 'uniform_cube_MC_mean'),
                              'MCSE': (file, filters, 'MCSE')})
    table('confirmation_gradient', gradient_rows, 'Squared coordinate derivatives of the rich-map optimal policy. The analytic expectation covers the full uniform cube; Monte Carlo uses 131,072 independent points.')
    rows = []
    for name in friendly:
        for cohort in ('practical', 'extended'):
            for window in ('full', 'upper_half', 'largest_four'):
                file = f'data/{name}/rates.csv'
                filters = {'cohort': cohort, 'method': 'theory_1', 'quantity': 'regret', 'window': window}
                rows.append({'Map': friendly[name], 'Cohort': cohort, 'Window': window.replace('_', ' '),
                             'R': (file, filters, 'replications'),
                             'Slope': (file, filters, 'OLS_slope'), 'OLS SE': (file, filters, 'OLS_SE'),
                             'Boot. low': (file, filters, 'bootstrap_low'), 'Boot. high': (file, filters, 'bootstrap_high')})
    table('confirmation_rates', rows, 'Descriptive log-regret slopes for the exact theory path with a=1, in every fixed window and both declared cohorts. The extended cohort uses only its 50 complete paths at all training lengths. Bootstrap intervals resample whole paths; the theoretical exponent is an upper-envelope benchmark, not a required equality. All other path and metric fits are retained in the machine-readable rate tables.')
    rows = []
    for name in friendly:
        for T in (1440, 7290):
            file = f'data/{name}/oracle_optimism_bootstrap.csv'
            filters = {'T': T, 'cohort': 'practical' if T == 1440 else 'extended'}
            rows.append({'Map': friendly[name], 'T': T,
                         'R': (file, filters, 'replications'),
                         'Crossfit minus plugin': (file, filters, 'crossfit_minus_plugin_regret'),
                         'Boot. low': (file, filters, 'bootstrap_low'),
                         'Boot. high': (file, filters, 'bootstrap_high')})
    table('confirmation_optimism', rows, 'Monte Carlo loss-oracle selection optimism, measured as cross-fitted minus plugin population regret. Pointwise percentile intervals resample whole paths and recalibrate both oracle rules in every draw; fold labels, anchors and population quadrature remain fixed. These intervals include calibration variability, unlike the separately retained descriptive paired-MCSE intervals.')
    text = r'''\input{\simoutput/paper/simulation_numbers.tex}
\section{Simulation: confirmation in two balanced economies}
\label{sec:balanced_simulation}
We preserve the original stock-level baseline and add a smooth loading map that
depends on all six characteristics. The historical experiment contains 3,500
environment/rank evaluations on 3,100 distinct economic replication seeds;
the rank comparisons reuse baseline innovations. Its outputs and theoretical
source bytes are archived unchanged. The following confirmation uses new seeds
and a protocol frozen before performance was examined.

\subsection{Economy and analytic reference}
There are $N=600$ stocks in triplets, six characteristics and three return
factors. Group states start from their stationary Gaussian law and follow
independent AR(1) dynamics with persistence $0.95$. With
$U_{g,d,t}=\Phi(S_{g,d,t})$, stock roles $r=0,1,2$ have
$Z_{g,r,d,t}=2((U_{g,d,t}+r/3)\bmod1)-1$. These population ranks have continuous
support; persistence is in characteristics, while factor shocks are iid.
Define
\[
\theta_\eta(z)=\pi(z_1+1)+\eta\sum_{d=2}^6\sin\{\pi(z_d-z_1)\},\qquad
\beta_\eta(z)=\sqrt{c_\beta}
\begin{pmatrix}\sqrt{2/3}\cos\theta_\eta(z)\\
\sqrt{2/3}\sin\theta_\eta(z)\\1/\sqrt3\end{pmatrix}.
\]
The original map uses $\eta=0$ and the richer map uses $\eta=0.35$.
The common triplet phase rotates the first two loadings by $2\pi/3$ and
preserves $\|\beta\|^2=c_\beta$ and $B_t'B_t/N=(c_\beta/3)I_3$.
Returns are $R_{t+1}=B_tF_{t+1}+\varepsilon_{t+1}$, with
$F_{t+1}\sim N(\mu_F,I-\mu_F\mu_F')$ and independent
$\varepsilon_{t+1}\sim N(0,\sigma_\varepsilon^2I_N)$.
We retain $c_\beta=0.0016$, $\sigma_\varepsilon=0.08$ and
$\mu_F=(0.10,0.06,0.03)'$. For the paper's weights $W(Z_t)/N$, both maps have
\[
W^\star(z)=\frac{\beta_\eta(z)'\mu_F}{c_\beta/3+\sigma_\varepsilon^2/N},\quad
q^\star=\frac{c_\beta/3}{c_\beta/3+\sigma_\varepsilon^2/N}\|\mu_F\|^2,
\quad Q^\star=1-q^\star,\quad \SR^\star=\sqrt{q^\star/(1-q^\star)}.
\]
The original conditional-pricing and Gaussian-moment audits apply to both
maps. An independent stock-dimensional solve checks E4, and analytic derivatives
are compared with finite differences in all six coordinates. Smooth extensions
with compactly supported cutoffs establish the required Sobolev membership;
a finite Nystr\"om norm is not used as a proof of infinite-RKHS membership.
The operator sandwich underlying E5 is retained. Three economic factors do
not imply rank three after integrating over characteristic states.

\subsection{Frozen design, selectors and evaluation}
The local-resource protocol uses 100 new paths per map on the practical grid
\[
T\in\{60,90,120,180,240,360,540,720,1080,1440\}.
\]
The first 50 are coherently
extended to $T\in\{2160,3240,4860,7290\}$. Smoothness sensitivities
$\nu\in\{0.5,2.5\}$ use 50 paths per map on the practical grid. These counts
are a prospective reduction from the proposed maximum 500/200/200 targets,
recorded after a cost and projection pilot and before confirmation performance.
Common random numbers pair maps, kernels and ranks; independent paths within
an environment, rather than all environment evaluations, are the sampling units.
The 400 environment evaluations cover 200 distinct map/innovation-index pairs:
the two maps share 100 innovation seed indices, and alternative kernels reuse
the corresponding economic paths. These are not 400 independent trajectories.

The response-one ridge estimator uses the actual Mat\'ern kernel, unit diagonal,
length scale one and rank 512 with independent nested Sobol anchors. Its
uncentered second moment and penalty normalization are unchanged. Holdout25
uses the last quarter for validation and preserves its first-minimum tie rule.
Rolling3 uses chronological train/validation endpoints at fractions
$(0.4,0.6)$, $(0.6,0.8)$ and $(0.8,1)$, rounded down to integers. Validation
loss is weighted by the number of observations; the selected policy is refitted
on all $T$ observations. Ties within $10^{-12}$ choose the larger penalty.
Neither selector receives population moments, the optimum, oracle objects or
test observations. A conservative one-standard-error variant is omitted
because no dependent-data uncertainty rule was prospectively justified.
The zero policy is an explicit benchmark with loss one, complexity zero and
Sharpe convention zero; it is never silently added to a selector.

The independent stationary test and contiguous future $T+1,\ldots,T+720$
each contain 720 observations. The latter uses the same long economic path,
with no future data entering fitting. Sharpe is per period. Whole-path
resampling preserves overlap across training lengths. Pointwise Monte Carlo
intervals are conditional on the anchors and population quadrature, whose
uncertainty is assessed separately.

For $b=1+2\nu/6$, the separate theory experiment evaluates
$\lambda_T(a)=a s_{\rm ref}T^{-b/(b+1)}$, $a\in\{0.25,1,4\}$, at the actual
penalties, without grid snapping. The positive scale $s_{\rm ref}$ is fixed
once per map/kernel from an independent population pilot. It uses population
calibration information and is not an implementable tuning rule.
The ensemble plugin loss oracle minimizes mean population regret on the fixed
96-point grid. Its Sharpe is evaluated at that loss-optimal penalty; it is not
a Sharpe oracle. A five-fold oracle calibrates on four whole-path folds and
evaluates on the fifth, exposing Monte Carlo selection optimism.

\subsection{Evidence and numerical limits}
'''
    for name in friendly:
        file = f'data/{name}/methods.csv'
        hold = number(file, {'T': 1440, 'method': 'holdout25'}, 'regret_mean', '.6f')
        rolling = number(file, {'T': 1440, 'method': 'rolling3'}, 'regret_mean', '.6f')
        plugin = number(file, {'T': 1440, 'method': 'ensemble_plugin_loss_oracle'}, 'regret_mean', '.6f')
        crossfit = number(file, {'T': 1440, 'method': 'crossfit_loss_oracle'}, 'regret_mean', '.6f')
        text += (f'For {friendly[name]}, mean population regret at $T=1440$ is {hold} for holdout, '
                 f'{rolling} for rolling validation, {plugin} for the plugin loss oracle and {crossfit} '
                 'for the cross-fitted loss oracle. These are observed comparisons, not a promise of a particular selector improvement.\n\n')
    text += r'''
Mean, median and geometric-mean penalties are distinct because endpoint choices
can dominate the mean. The tables and distribution figures report both endpoints
separately. Negative sample OOS excess losses are retained on a signed linear
scale. The exact population decomposition is
\[
Q(\widehat W_\lambda)-Q^\star
=\{Q(W_\lambda)-Q^\star\}
+\|\widehat W_\lambda-W_\lambda\|_\Sigma^2
+2\langle\widehat W_\lambda-W_\lambda,\Sigma W_\lambda-\mu\rangle.
\]
The cross term is signed. Population bias includes the measured projection
floor. Sample complexity is at most $\min(T,r_{\rm num})$; population complexity
divided by $T$ need not be at most one.

Small exact temporal-Gram solves use $N\in\{30,60\}$ and $T\in\{60,120\}$,
including the indispensable $T\lambda$ dual penalty. The practical-grid
resolution audit compares ranks 256,512,1024 at fixed penalties on 50 paired
paths, using common quadrature. Certification requires projection floor below
5 percent of relevant mean regret and absolute paired rank difference plus
1.96 MCSE below 5 percent. Fixed selected/theory policies are also evaluated
with 8,192 and 32,768 groups at four independent quadrature seeds.
Only regions passing the applicable rank and quadrature checks are described as
resolved relative to the audited rank-1024 reference and quadrature. This finite
comparison does not prove equality with the infinite kernel. False flags, near-interpolating
regions and extra horizons without a corresponding rank check remain explicitly
unresolved. No points or economic parameters are removed to secure a pass.

Spectrum calculations use 1,024,2,048,4,096 groups at three seeds, with
independent dense-operator checks and partial eigenpair residuals. The windows
30--200,60--400,120--800 are fixed; unresolved indices are reported. A slope
near $-b$ does not prove E5, and a distant finite-window fit does not disprove
its analytic argument. The historical tolerance band is only descriptive.

The source condition is a theorem assumption. Neither smooth map is claimed
to be least favourable, so neither must attain the minimax envelope with
equality. The displayed theory paths, ex-post oracle fits and validation fits
answer different questions. Their slopes and whole-path bootstrap intervals
are reported without calibrating the economy or choosing windows after seeing
the results. Cross-sectional size stays fixed within every rate experiment.

\input{\simoutput/tables/confirmation_main.tex}
\input{\simoutput/tables/confirmation_selection.tex}
\input{\simoutput/tables/confirmation_theory.tex}
\input{\simoutput/tables/confirmation_resolution.tex}
'''
    main_figures = [(name+'_four_panel', friendly[name]+': full practical-grid OOS curves and training-only selection in the rank-512 representation. The entire curve is not certified as a full-kernel calculation. Bands are pointwise Monte Carlo intervals. Heatmaps interpolate only inside computed complexity support.') for name in friendly]
    main_figures += [(name+'_risk_complexity', friendly[name]+': raw population loss, regret and normalized regret against sample complexity and complexity per observation, at numerical rank 512. High-complexity regions failing the rank check remain diagnostic.') for name in friendly]
    main_figures += [(name+'_theory_paths', friendly[name]+': distinct theory, oracle and validation paths. Larger training lengths use the 50-path extended cohort and retain finite-resolution qualifications. Panels labeled symlog preserve confidence intervals crossing zero, with linear threshold $10^{-5}$. A negative penalty interval bound comes from the Gaussian Monte Carlo interval; selected penalties remain positive.') for name in friendly]
    main_figures += [('rich6d_minus_baseline', 'Paired rich-map minus original-map differences on common economic innovations; pointwise intervals use paired path differences as the sampling units.')]
    def figures(items):
        return ''.join('\n\\begin{figure}[p]\\centering\n\\includegraphics[width=\\textwidth]{\\simoutput/figures/'+name+'.pdf}\n\\caption{'+caption+'}\n\\end{figure}\n' for name, caption in items)
    text += figures(main_figures)
    appendix = r'''\clearpage
\section*{Simulation confirmation appendix}
All aggregates are reconstructed independently from saved path surfaces. The
five stored training-only and theory policies also retain coefficients and
actual test payoffs for population and OOS evaluation checks. Public
synthetic archives contain checksums for each checkpoint; no licensed empirical
observations are included. Historical checks reconstruct the original 3,500
evaluations and bootstrap intervals. That archive lacks stock returns and
coefficients, so its aggregation check is not misrepresented as a fitting replay.

The historical empirical-rank robustness has finite support at fixed N and
does not meet the baseline infinite-spectrum E5 interpretation. Its original
one-coordinate loading map is preserved. Bounded heteroskedasticity does not
impose baseline conditional pricing, and its historical regret reference is the
finite-subspace optimum, not the unrestricted analytic baseline optimum.
The licensed empirical exercise elsewhere in the paper is unchanged and
separate from these artificial economies.
'''
    appendix_figures = []
    appendix += '\n\\input{\\simoutput/tables/confirmation_gradient.tex}\n'
    appendix += '\n\\input{\\simoutput/tables/confirmation_rates.tex}\n'
    appendix += '\n\\input{\\simoutput/tables/confirmation_optimism.tex}\n'
    for name in unresolved_tables:
        appendix += '\n\\input{\\simoutput/tables/'+name+'.tex}\n'
    for name in maximum_tables:
        appendix += '\n\\input{\\simoutput/tables/'+name+'.tex}\n'
    for name in friendly:
        appendix_figures.extend([(name+'_decomposition', friendly[name]+': regularization bias including projection floor, nonnegative estimation norm and signed cross term.'),
                                 (name+'_selection_distributions', friendly[name]+': separate penalty location summaries, endpoint frequencies, and median/interquartile ranges of complexity and regret.'),
                                 (name+'_oos_types', friendly[name]+': independent stationary and contiguous future OOS. Negative sample excess losses remain visible.'),
                                 (name+'_rolling_minus_holdout', friendly[name]+': paired rolling-minus-holdout differences; no selector was redesigned after observing confirmation outcomes.'),
                                 (name+'_numerical_resolution', friendly[name]+': fixed-penalty rank comparisons and independent quadrature sensitivity. Unresolved cells are retained.')])
    appendix_figures.extend([('spectrum_resolution', 'Three quadrature resolutions and three independent seeds. Shading spans the seed range, not a confidence interval. Fits use fixed windows and do not certify an asymptotic exponent.'),
                             ('kernel_sensitivity', 'Prospective smoothness sensitivities on the practical grid; 50 paths per alternative map/kernel environment.')])
    appendix += figures(appendix_figures)
    for name, caption in [('appendix_N_robustness', 'Historical cross-sectional-size sensitivity, archived unchanged and independently checked.'),
                          ('appendix_persistence', 'Historical characteristic-persistence sensitivity, archived unchanged and independently checked.')]:
        appendix += '\n\\begin{figure}[p]\\centering\n\\includegraphics[width=\\textwidth]{\\simoutput/../paper/figures/'+name+'.pdf}\n\\caption{'+caption+'}\n\\end{figure}\n'
    (directory/'simulation_section.tex').write_text(text)
    (directory/'simulation_appendix.tex').write_text(appendix)
    (directory/'simulation_numbers.tex').write_text('\n'.join(macros)+'\n')
    (directory/'simulation_preview.tex').write_text(r'''\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,amssymb,graphicx,booktabs,longtable,array}
\usepackage[margin=1in]{geometry}
\usepackage[hidelinks]{hyperref}
\newcommand{\simoutput}{..}
\newcommand{\SR}{\operatorname{SR}}
\begin{document}
\input{simulation_section.tex}
\input{simulation_appendix.tex}
\end{document}
''')
    generated = set(directory.glob('*.tex'))
    for entry in lineage:
        if entry['kind'] == 'table':
            generated.add(output/entry['table'])
            generated.add((output/entry['table']).with_suffix('.tex'))
    json_write(directory/'number_lineage.json', {'schema': 'publication-cell-lineage/2.0',
               'protocol_hash': protocol['protocol_hash'], 'cells': lineage,
               'source_hashes': {file: file_hash(output/file) for file in frames},
               'generated_hashes': {str(p.relative_to(output)): file_hash(p) for p in sorted(generated)}})
    return directory
