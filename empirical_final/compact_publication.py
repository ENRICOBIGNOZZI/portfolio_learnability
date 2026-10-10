"""Compact PDFs and full manuscript, preserving the latest supplied theory bytes."""
import argparse
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import jinja2
from data_pipeline import digest
from empirical_final.compact_common import ROOT, OUT, PROTOCOL, setup, save, audit, source

ARCHIVE=Path('/Users/enrico/Downloads/Portfolio_Paper (2).zip')


def read(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')


def escaped(value):
    return str(value).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def simple_table(headers,rows,caption,label,spec=None,note=''):
    spec=spec or 'l'+'r'*(len(headers)-1)
    text=r'\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}'+'\n'
    text+=f'\\caption{{{caption}}}\\label{{{label}}}\n'
    text+=r'\begin{tabular}{@{}'+spec+r'@{}}\toprule'+'\n'
    text+=' & '.join(headers)+r' \\ \midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    text+='\n'+r'\bottomrule\end{tabular}'+'\n'
    if note:text+=r'\par\vspace{6pt}\begin{minipage}{\textwidth}\footnotesize '+note+r'\end{minipage}'+'\n'
    return text+r'\end{table}'+'\n'


def performance_table(perf):
    rows=[]
    labels={'linear':'Linear','gaussian':'Gaussian','matern32':r"Mat\'ern-$3/2$",'neural':'Neural + ridge'}
    for p,label in labels.items():
        g=perf[(perf.policy==p)&(perf.scenario=='gross')].iloc[0]
        n=perf[(perf.policy==p)&(perf.scenario=='trade25_borrow30')].iloc[0]
        rows.append([label,f'{100*g.annual_excess_return:.2f}',f'{100*g.annual_volatility:.2f}',f'{g.sharpe:.3f}',
                     f'{n.sharpe:.3f}',f'{g.mean_gross:.2f}',f'{n.annual_turnover:.2f}',
                     f'{100*(n.annual_trading_fees+n.annual_borrow_fees):.2f}'])
    return simple_table(['Policy',r'Mean \%',r'Vol. \%',r'$\SR_G$',r'$\SR_N$','Gross','Trades',r'Fees \%'],rows,
        'Common-protocol investment performance and illustrative costs','tab:kernel_performance',
        note='All policies: 564 months, February 1978--January 2025; one frozen ex-ante scale. Mean and volatility are annualized gross excess returns. '
        r'$\SR_G$ and $\SR_N$ denote gross and net Sharpe; net uses 25 bps two-sided commissions and 30 bps/year short fees. '
        'Gross is average account gross/NAV. Trades are annual drift-adjusted traded notional/NAV on the net path; fees are annual trading plus borrowing charges/NAV. '
        'These fees are hypothetical; their return drag also includes execution timing. Neural uses fixed primary seed 0.')


def spectral_table(g):
    rows=[]
    for r,label in zip(g.itertuples(),[r'0--10\%',r'10--50\%',r'50--90\%',r'90--100\%']):
        rows.append([label,f'{100*r.mean_spectral_share:.4f}',f'{r.component_gross:.3f}',f'{r.account_gross:.3f}',
            f'{100*r.annual_mean:.3f}',f'{r.cumulative_gross_sharpe:.3f}',f'{r.cumulative_net_sharpe:.3f}'])
    return simple_table(['Ranks',r'Mass \%','Own gross','Account gross',r'Mean \%',r'$\SR_G$',r'$\SR_N$'],rows,
        'Spectral groups and incremental investment value','tab:spectral_value',
        note='Gaussian fixed-bandwidth selected policy, unchanged penalty at each annual refit. Mass averages annual training second-moment shares. '
        'Own gross and mean refer to individual group contributions; account gross and Sharpes refer to all groups through that boundary. '
        r'Stock weights are summed before account gross, trading and short fees. The first 10\% refers to positive spectral ranks, not stocks or mass. '
        r'Gross $\SR$ increments at the successive boundaries are '+', '.join(f'{d:+.3f}' for d in np.diff(g.cumulative_gross_sharpe))+
        r'; net increments are '+', '.join(f'{d:+.3f}' for d in np.diff(g.cumulative_net_sharpe))+'.')


def macro_values(perf,spectral):
    values={};lineage=[]
    def put(name,file,column,filters,scale=1,decimals=3):
        frame=read(file)
        for k,v in filters.items():frame=frame[frame[k].eq(v)]
        if len(frame)!=1:raise ValueError(f'Claim does not identify one row: {name}')
        raw=float(frame.iloc[0][column])*scale
        values[name]=f'{raw:.{decimals}f}'
        lineage.append(dict(macro=name,input=f'tables/{file}.csv',column=column,filters=filters,scale=scale,raw_value=raw,
                            rendered_value=values[name]))
    missing=json.loads(source(ROOT/'outputs/tables/formation_timing_audit.json').read_text())
    for name,key in [('MissingCount','unresolved_payoffs'),('FormationCount','formation_stock_months')]:
        values[name]=f'{missing[key]:,}';lineage.append(dict(macro=name,input='outputs/tables/formation_timing_audit.json',key=key,raw_value=missing[key]))
    values['MissingShare']=f'{100*missing["unresolved_payoffs"]/missing["formation_stock_months"]:.3f}'
    lineage.append(dict(macro='MissingShare',input='outputs/tables/formation_timing_audit.json',operation='100*unresolved_payoffs/formation_stock_months'))
    for name,sc in [('NeuralGross','gross'),('NeuralNet','trade25_borrow30')]:put(name,'table1_performance','sharpe',dict(policy='neural',scenario=sc))
    for prefix,contrast in [('Kernel','Gaussian minus Linear'),('Group','Add intermediate group')]:
        for part,sc in [('Gross','gross'),('Net','trade25_borrow30')]:
            for tail,col in [('Delta','delta_sharpe'),('Low','low'),('High','high')]:
                put(prefix+part+tail,'paired_contrasts',col,dict(scenario=sc,contrast=contrast,block_length=12))
    for name,col,group,scale,d in [('LeadingMass','mean_spectral_share',1,100,3),('LeadingComplexity','mean_active_C',1,1,1),
        ('LeadingGross','component_gross',1,1,3),('LeadingMean','annual_mean',1,100,2),
        ('IntermediateMean','annual_mean',2,100,2),('IntermediateGross','component_gross',2,1,3),
        ('IntermediateCovariance','annual_covariance_with_previous',2,1,6),('IntermediateVariance','incremental_annual_variance',2,1,6),
        ('LeadingSR','cumulative_gross_sharpe',1,1,3),('TwoGroupSR','cumulative_gross_sharpe',2,1,3),
        ('LeadingNetSR','cumulative_net_sharpe',1,1,3),('TwoGroupNetSR','cumulative_net_sharpe',2,1,3)]:
        put(name,'spectral_value',col,dict(group=group),scale,d)
    values['TotalComplexity']=f'{spectral.mean_active_C.sum():.1f}'
    values['IncrementalGross']=f'{spectral.account_gross.iloc[1]-spectral.account_gross.iloc[0]:.3f}'
    lineage.extend([dict(macro='TotalComplexity',input='tables/spectral_value.csv',operation='sum(mean_active_C)'),
                    dict(macro='IncrementalGross',input='tables/spectral_value.csv',operation='account_gross[group2]-account_gross[group1]')])
    curve=read('single_window_complexity');peak=curve.loc[curve.oos_sharpe.idxmax()]
    for name,col,d in [('SinglePeakC','C',1),('SinglePeakSR','oos_sharpe',3)]:
        values[name]=f'{peak[col]:.{d}f}';lineage.append(dict(macro=name,input='tables/single_window_complexity.csv',operation='argmax(oos_sharpe)',column=col))
    panels=read('nine_panel_summary')
    values['InteriorCount']=str(int((~panels.boundary_maximum).sum()))
    values['MinimumPeakC']=f'{panels.maximum_C.min():.1f}';values['MaximumPeakC']=f'{panels.maximum_C.max():.1f}'
    for name,op in [('InteriorCount','sum(~boundary_maximum)'),('MinimumPeakC','min(maximum_C)'),('MaximumPeakC','max(maximum_C)')]:
        lineage.append(dict(macro=name,input='tables/nine_panel_summary.csv',operation=op))
    for label,k in [('Gaussian','gaussian'),('Matern','matern32')]:
        for part,p in [('Fixed',k),('Tuned',k+'_tuned')]:
            for suffix,sc in [('', 'gross'),('Net','trade25_borrow30')]:put(label+part+suffix,'bandwidth_performance','sharpe',dict(policy=p,scenario=sc))
    return values,lineage


APPENDIX = r"""
\section{Empirical Appendix}
\label{app:empirical_compact}
\subsection{Sample, chronology, provenance, and reconciliation}
The canonical revision is pinned by \texttt{audit/protocol.json}, code and input
SHA-256 hashes in \texttt{audit/run\_manifest.json}, and claim-level mappings in
\texttt{audit/source\_map.json}. This execution freeze is not retrospective
preregistration. Licensed stock-level inputs, fitted extractors, coefficients,
and holdings stay private. Public aggregates retain formation/realization dates,
candidate indices and annual selection diagnostics. The archive's nonempirical
source bytes are preserved in the delivered full manuscript.

The supplied archive's Gaussian table reports Sharpe 3.350 and average gross
1.76. The canonical path gives 3.300 and 2.326. The archive lacks monthly paths,
weights, data identifiers, feature-bank hashes, and an execution manifest, so
its discrepancy cannot be uniquely attributed to a numerical step. Both describe
an initial 1963--1972 sample, five-year validation, 10,000 features, and fixed
median-distance bandwidth, but those verbal settings do not prove identical
estimation. Constant positive rescaling cannot explain a Sharpe change. The
reconciliation CSV records known, unknown, and numerically verified differences;
archived figures and table values are legacy artifacts, never inputs to the
main results. Canonical holdings reconstruct all three kernel returns and the
Gaussian spectral components, resolving the operational mismatch without
pretending to recover an undocumented archived run.

Formation-only preprocessing excludes no name because of its next payoff. It
recovers zero of 7,489 unavailable leads by documented adjacent-month source
matches, across 638 formation months. Separate CRSP/CIZ source crosschecks test
supplied finite leads rather than provide all missing-payoff observations.
The missing source requirement is an identified next-calendar total/delisting
payoff for every eligible unresolved name and month, with compatible cash-rate
units. Neither deleting positions after outcomes nor assigning zero is an
admissible reconstruction. The retained conditional sample therefore remains
distinct from an implementable formation universe; the maximum observed rank
change relative to formation-only ranks is approximately 0.005875. Aggregate
coverage is public; it does not bound portfolio-performance bias.

All future-payoff perturbation checks distinguish values from availability.
Changing subsequent values cannot change earlier fitted parameters or
validation choices. Formation-only ranks are invariant to availability changes.
The conditional retained universe itself does depend on availability, and we
do not assert otherwise. January formation decisions can use payoffs realized
at the January close, never February's outcome.

\input{tables/coverage}
\clearpage
\subsection{Estimation and exact financial accounting}
The maximum-Sharpe criterion uses monthly managed payoffs
$X_t=N_t^{-1}\sum_i R^e_{i,t+1}\phi(Z_{i,t})$. With $G$ containing their rows,
the fitted quadratic criterion is the monthly mean of $(1-Ga)^2$ plus
$\lambda\|a\|_2^2$. The solution is
$a=(G'G/T+\lambda I)^{-1}G'\mathbf1/T$.
The same validation response-one loss is applied at fitted scale, before
exposure calibration. Each initial-training complexity grid is reused through
all 47 decisions. Boundary choices are reported rather than expanded in
response to OOS maxima. A minimum-norm zero-penalty diagnostic uses a
$10^{-12}$ relative positive-rank tolerance and remains ineligible for selection.

The common scale is 0.03740000227285338: initial Linear validation median
gross exposure 48.12833932115886 is mapped to 1.8. All 60 calibration payoffs
are known by January 1978 close. Linear uses a penalized constant feature;
Gaussian and Mat\'ern have their original Fourier maps without an added free
intercept. No full-sample volatility, feature whitening, or posttest gross
normalization enters a policy.

Let $w$ be target stock weights relative to post-execution NAV and $d$ the
drifted positions relative to pretrade NAV. A proportional two-sided commission
$\tau$ gives $c=\tau\sum_i|(1-c)w_i-d_i|$, solved before payoff accounting.
The cash residual finances stock targets and fees exactly. With cash rate
$r_f$, annual short fee $b$, and $s=\sum_i(-w_i)_+$, end-period growth is
$(1-c)[1+r_f+w'R^e-bs/12]$. The next drift is
$d_i'=w_i(1+r_f+R_i^e)/(1+r_f+w'R^e-bs/12)$.
Costs begin with entry from cash; terminal positions are marked rather than
forced to liquidate. Exited names incur liquidation trades but receive no
invented next payoff. The same cash rate finances both positive cash and
leverage; additional funding spreads and market impact are absent.

Upstream JKP payoffs include its documented CRSP return/delisting treatment.
These formulas infer value drift from total payoffs and stable identifiers;
there is no independent share-count, split, corporate-action or execution-tape
reconstruction. Friction results are illustrative self-financing sensitivities
on a conditional sample. Target turnover $\sum_i|w_{i,t}-w_{i,t-1}|$ is separate
from drift-adjusted actual traded notional within this accounting model. A
component's own gross or fee cannot be added across groups to price a netted
account. Each cumulative spectral account is recomputed from summed stock weights.

\input{tables/costs}
\clearpage
\subsection{Complexity paths, uncertainty, and historical diagnostics}
\input{tables/panel_dates}
For each block and candidate, the public monthly path is concatenated across
annual refits. Sharpe uses sample monthly volatility with $T-1$ denominator and
annualizes by $\sqrt{12}$. Mean effective complexity weights annual fits by
their actual OOS month counts. Individual curve coordinates and maxima, the
selected-policy summaries, and all month identifiers accompany the square
PDF/PNG. A changing selected annual penalty cannot be mapped to one point on
a fixed-penalty path. Common axes retain the entire observed supports; no
hump, interior maximum, or significance is imposed.

The circular-block bootstrap uses 5,000 shared month-index draws and seed
20261010. Intervals are pointwise, conditional on fitted policy paths. Models,
representations, validation choices, and holdings are not re-estimated in a
draw. Approximate stability and dependence decay motivate block resampling;
arbitrary nonstationarity invalidates a blanket bootstrap justification.
Six-, 12-, and 24-month results are displayed separately. Gaussian minus
Linear and adding the intermediate group are priority economic contrasts;
all highlighted intervals remain descriptive and unadjusted for multiplicity.
No claim of a significant grid maximum is made.

\input{tables/inference}
\input{tables/subperiods}
\clearpage
\subsection{Fixed versus chronologically tuned bandwidth}
The five multipliers share original frequency, radial and phase draws; each
multiplier has its own 120-candidate initial-training penalty scale. Prior
validation jointly minimizes raw $Q$ over multiplier and penalty. The full
pretest history refits the selected pair. Fixed controls must reproduce the
canonical selected payoffs before tuned results are accepted. Tuned costs
use newly reconstructed stock holdings and the identical friction scenarios.
The primary spectral and complexity figures retain fixed bandwidth.
Boundary selections are diagnostics under the original past-only grid rule;
test peaks do not broaden that grid.

\input{tables/bandwidth}
\input{tables/bandwidth_boundaries}
\input{tables/bandwidth_uncertainty}
\clearpage
\subsection{Neural portfolio policy with ridge readout}
The extractor has two 64-unit tanh layers, a scalar linear readout, and an
explicit penalized constant coordinate. Raw hidden features are retained;
there is no whitening, per-year feature normalization, or arbitrary spectral
matching to a Fourier kernel. Torch 2.9.1 runs a float32 extractor on one CPU
thread, with float64 exact ridge algebra and financial accounting. The primary
seed is 0; seeds 0--4 are stability checks, never selected on OOS Sharpe.

The hidden layers minimize the direct monthly maximum-Sharpe criterion using
Adam, learning rate 0.003, and mean squared hidden-parameter penalty $10^{-5}$.
Each update samples four complete historical months uniformly with replacement,
aggregates every stock's contribution within each month, and then differentiates
the squared monthly payoff criterion. At a fixed head this has the intended
full-history gradient in expectation. It is not a random-stock minibatch loss.
The head is refreshed by an exact full-history ridge solve every 16 steps.
Its extraction penalty is fixed within the stage at 0.01 times the initial
random-representation mean eigenvalue. This alternating method differs from
an unmodified end-to-end network.

Inner training restarts deterministically from its seed. Budgets 32 and 96
steps are evaluated with five exact-head penalty ratios
$\{10^{-6},10^{-4},10^{-2},1,100\}$ times the frozen extractor's training
mean eigenvalue. Validation selects the budget and ratio using prior monthly
$Q$. Full pretest-history refitting restarts from the same seed for the chosen
budget, freezes the extractor, recomputes its training-only penalty scale,
and solves the final head at the selected ratio. Validation is not re-used
for selection after entering that refit. The scale rule itself is the selected
specification; the final absolute penalty is recorded annually.

\begin{align*}
 W_y^{NN}(z)&=a_y'\phi_{\widehat\theta_y}(z),&
 K_y^{NN}(z,z')&=\phi_{\widehat\theta_y}(z)'\phi_{\widehat\theta_y}(z'),\\
 \widehat\Sigma_y^{NN}&=\frac1{T_y}\sum_{t\in H_y}X_t^{NN}(X_t^{NN})',&
 \widehat a_y&=(\widehat\Sigma_y^{NN}+\lambda_yI)^{-1}\widehat\mu_y^{NN},\\
 \widehat C_y^{head}(\lambda)&=
 \Tr[\widehat\Sigma_y^{NN}(\widehat\Sigma_y^{NN}+\lambda I)^{-1}].
\end{align*}
The head has the stated Euclidean penalty including the intercept. Independent
SVD, direct solve, and eigenspace summation reconstruct the actual policy.
Conditional head complexity is bounded by managed-matrix rank. Raw and
trace-normalized diagnostics are published together; eigenvalue normalization
does not change the policy penalty. This learned-feature kernel is distinct
from the tangent kernel of Jacot, Gabriel and Hongler (2018),
\url{https://arxiv.org/abs/1806.07572}. Hidden outcomes are reused when learning
the representation, so conditioning on the extractor does not establish the
paper's fixed-kernel theorem. Deployment freezing alone proves no rate claim.

\input{tables/neural_checks}
\input{tables/neural_seeds}
\clearpage
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/appendix_neural_spectrum.pdf}
\caption{January 2024 seed-0 frozen neural readout spectrum. Raw and
mass-normalized eigenvalues are distinct diagnostics; neither measures the
complexity of estimating the hidden parameters.}\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{figures/appendix_neural_optimization.pdf}
\caption{Full-history penalized objective during selected-budget extractor
refits. Periodic exact head refreshes and complete-month stochastic updates
can produce nonmonotonic trajectories. All annual trajectories, parameter
changes, and residuals are public; finite budgets do not certify global
nonconvex convergence.}\end{figure}
\input{neural_diagnostics_text}
\clearpage
\subsection{Spectral attribution, grouping, seed stability, and factors}
For training eigenpairs $(\mu_j,v_j)$, the actual ridge contribution is
$v_j(v_j'\widehat\mu)/(\mu_j+\lambda)$. We group positive ranks at 10/50/90\%,
round upward, and include near ties with relative gap $10^{-8}$. Stock scores
are divided by the contemporaneous $N_t$ and multiplied by the common scale.
Group weights sum to the selected policy and returns reconstruct it month by
month. No group is separately tuned. Alternative boundaries 5/25/75\% and
20/50/80\% keep the same policy and annual penalty. Feature seeds 0 and 1
form a predeclared, deliberately small computational sensitivity, separate
from the neural five-seed comparison. Each draw has its own initial-training
penalty scale and annual prior-validation selection. No seed is selected for
OOS performance.

Long means weight each input rank by positive positions and divide by long
notional; short means use absolute negative positions divided by short
notional. Their difference is a normalized profile. Signed exposure sums
the rank times signed weights, preserving notional. Families are explicitly
listed in \texttt{audit/economic\_dictionary.json}, including financial-liability
growth and net debt issuance for debt financing, and share growth and the
opposite sign of net equity payout for equity issuance. Overlapping families
and interaction cells are alternative views of the same positions, never
additive economic components. Empty cells have zero payoff but no observed
density.

\input{tables/grouping}
\input{tables/feature_seeds}
The CSV economic profiles expose sensitivity rather than fixing factor names
to particular spectral ranks. Rotations within near-tied eigenspaces prevent
stable identification of individual eigenvectors through time. A component
can raise mean and diversify covariance yet raise total variance or net
turnover. All cumulative accounts are priced after stock-level netting.

Factor regressions use FF5 and momentum on identical realized months, with
an intercept and Bartlett/Newey--West lag-12 errors. Full-sample coefficients
and $R^2$ are descriptive. Past-only slopes estimated before each test year
produce a separate hedge series; no full-sample regression is represented as
deployed trading. Low $R^2$ identifies neither a new risk premium nor causal
anomaly returns. Detailed coefficients and financing--profitability cells
are included as source CSVs and compact appendix summaries.

\input{tables/factors}
\input{tables/financing_cells}
\clearpage
\subsection{History-length diagnostic and reproduction}
The retained earlier history-length experiment compares 60, 120, 240 and 360
months with a 20-month validation tail, at their complete common decision
years. These are secondary diagnostics with their own disclosed history
windows, not an alternative main backtest or a memory-selection strategy.
All use the identical fixed Gaussian feature map and conditional stock
snapshot. Results are reused only after hashing the inputs and verifying
the expanding-path identity. Public monthly paths support independent
recomputation.
\input{tables/history}

Run \texttt{python3 -m empirical\_final.compact\_reproduce} with the private
licensed snapshot to execute numerical regeneration, five neural seeds,
costs, bandwidth and grouping sensitivity, tests, plots and documents.
\texttt{--public-only} rebuilds the five main figures and PDFs from committed
aggregate tables and the included manuscript sources. The complete LaTeX ZIP
contains all images and tables, with relative paths. Neither command downloads
a different factor vintage or publishes confidential holdings.
"""


def appendix_tables(folder):
    write=lambda name,txt:(folder/f'{name}.tex').write_text(txt)
    missing=json.loads(source(ROOT/'outputs/tables/formation_timing_audit.json').read_text())
    write('coverage',simple_table(['Quantity','Value'],[
        ['Formation stock-months',f'{missing["formation_stock_months"]:,}'],['Unidentified payoffs',f'{missing["unresolved_payoffs"]:,}'],
        ['Months with unidentified outcomes',str(missing['months_with_unresolved_payoffs'])],
        ['Exact next-calendar recoveries',str(missing['recovered_exact_next_calendar_current_excess'])]],
        'Formation coverage audit','tab:coverage','lr'))
    p=read('table1_performance');rows=[]
    for r in p.itertuples():
        rows.append([escaped(r.policy),escaped(r.scenario),f'{r.sharpe:.3f}',f'{100*r.annual_excess_return:.2f}',
            f'{100*r.annual_volatility:.2f}',f'{r.annual_turnover:.2f}',f'{r.annual_target_turnover:.2f}'])
    write('costs',simple_table(['Policy','Scenario','SR',r'Mean \%',r'Vol. \%','Trades','Target'],rows,'Cost scenarios and turnover conventions','tab:costs',
        note='Trades use drift and fee-dependent NAV; Target is annualized target-weight change. Full monthly fees and borrowing are in the public accounting tables.'))
    panels=read('nine_panel_summary')
    write('panel_dates',simple_table(['Decisions','Months','First payoff','Last payoff','Peak $C$'],[
        [r.period,str(r.months),r.first_return,r.last_return,f'{r.maximum_C:.1f}'] for r in panels.itertuples()],
        'Exact nine-panel realization coverage','tab:panel_dates','lrr rr'.replace(' ','')))
    c=read('paired_contrasts');rows=[]
    for r in c.itertuples():rows.append([('G--L' if r.contrast.startswith('Gaussian') else 'Add group 2'),
        ('Gross' if r.scenario=='gross' else 'Net'),str(r.block_length),f'{r.delta_sharpe:+.3f}',f'[{r.low:+.3f}, {r.high:+.3f}]'])
    write('inference',simple_table(['Contrast','Payoffs','Block','Effect',r'95\% interval'],rows,'Paired conditional Sharpe differences','tab:inference','lllrr',
        note='Descriptive, no pipeline refits or multiplicity correction. Dependence/stability assumptions remain necessary.'))
    sub=read('subperiod_contrasts');pivot=sub.pivot(index='period',columns='contrast',values='delta_sharpe')
    write('subperiods',simple_table(['Decision years','Gaussian--Linear','Add group 2'],[
        [i,f'{r["gaussian minus linear"]:+.3f}',f'{r["nested_2 minus nested_1"]:+.3f}'] for i,r in pivot.iterrows()],
        'Net economic contrasts by historical block','tab:subperiods'))
    bw=read('bandwidth_performance');rows=[]
    for r in bw[bw.scenario.isin(['gross','trade25_borrow30'])].itertuples():
        rows.append([escaped(r.policy),('Gross' if r.scenario=='gross' else 'Net'),f'{r.sharpe:.3f}',f'{r.mean_gross:.2f}',f'{r.annual_turnover:.2f}'])
    write('bandwidth',simple_table(['Policy','Payoffs','SR','Gross','Trades'],rows,'Fixed and joint-tuned bandwidth with common accounting','tab:bandwidth','llrrr'))
    s=read('bandwidth_selections');rows=[]
    for (k,m),f in s.groupby(['kernel','mode']):rows.append([escaped(k),m,str(int(f.lambda_boundary.sum())),str(int(f.bandwidth_boundary.sum())),str(int(f.bandwidth_multiplier.ne(1).sum()))])
    write('bandwidth_boundaries',simple_table(['Kernel','Rule','Ridge boundary','Bandwidth boundary','Nonmedian'],rows,
        'Validation boundary diagnostics, out of 47 decisions','tab:bandwidth_boundaries','llrrr'))
    b=read('bandwidth_uncertainty');write('bandwidth_uncertainty',simple_table(['Kernel','Payoffs','Block','Tuned--fixed',r'95\% interval'],[
        [escaped(r.kernel),r.scenario,str(r.block_length),f'{r.delta_sharpe:+.3f}',f'[{r.low:+.3f}, {r.high:+.3f}]'] for r in b.itertuples()],
        'Conditional paired bandwidth sensitivity','tab:bandwidth_uncertainty','lllrr'))
    n=read('neural_annual_seed_0');write('neural_checks',simple_table(['Diagnostic','Value'],[
        ['Annual refits',str(len(n))],['Maximum head residual',f'{n.head_residual.max():.2e}'],
        ['Mean conditional head complexity',f'{n.C.mean():.2f}'],['Ridge boundary selections',str(int(n.lambda_boundary.sum()))],
        ['96-step budget selections',str(int(n.budget.eq(96).sum()))],
        ['Refits improving penalized training criterion',str(int((n.refit_final_objective<n.refit_initial_objective).sum()))]],
        'Seed-0 neural implementation checks','tab:neural_checks','lr'))
    n=read('neural_seed_stability');rows=[]
    for seed,f in n.groupby('seed'):
        g=f[f.scenario=='gross'].iloc[0];net=f[f.scenario=='trade25_borrow30'].iloc[0]
        rows.append([str(seed),f'{g.sharpe:.3f}',f'{net.sharpe:.3f}',f'{g.mean_gross:.2f}',f'{net.annual_turnover:.2f}'])
    write('neural_seeds',simple_table(['Seed','Gross SR','Net SR','Gross','Trades'],rows,'Neural seed stability; no seed selection','tab:neural_seeds',
        note='Individual paths and Sharpes are reported. Their average is not the Sharpe of an ensemble.'))
    robust=read('robustness_performance');rows=[]
    for boundary,label in [(0,'10/50/90'),(1,'5/25/75'),(2,'20/50/80')]:
        f=robust[(robust.seed==0)&(robust.boundary==boundary)]
        for sc in ['gross','trade25_borrow30']:
            fsc=f[f.scenario.eq(sc)].set_index('group')
            rows.append([label,('Gross' if sc=='gross' else 'Net'),f'{fsc.loc[1,"sharpe"]:.3f}',f'{fsc.loc[2,"sharpe"]:.3f}',
                         f'{fsc.loc[2,"sharpe"]-fsc.loc[1,"sharpe"]:+.3f}'])
    write('grouping',simple_table([r'Boundaries \%','Payoffs','First SR','Two-group SR','Increment'],rows,'Alternative spectral boundaries','tab:grouping','llrrr'))
    rows=[]
    for seed in [0,1]:
        f=robust[(robust.seed==seed)&(robust.boundary==0)&(robust.scenario=='trade25_borrow30')].set_index('group')
        rows.append([str(seed),f'{f.loc[1,"sharpe"]:.3f}',f'{f.loc[2,"sharpe"]:.3f}',f'{f.loc[4,"sharpe"]:.3f}',
                     f'{f.loc[2,"sharpe"]-f.loc[1,"sharpe"]:+.3f}'])
    write('feature_seeds',simple_table(['Feature seed','First net SR','Two-group net SR','Full net SR','Increment'],rows,'Small Gaussian feature-draw sensitivity','tab:feature_seeds'))
    f=read('group_factor_regressions');rows=[]
    for j,group in f.groupby('group'):
        alpha=group[group.factor.eq('alpha_monthly')].iloc[0]
        rows.append([str(j),f'{1200*alpha.coefficient:.3f}',f'{1200*alpha.se_hac12:.3f}',f'{alpha.r_squared:.3f}'])
    write('factors',simple_table(['Group',r'Annual alpha \%',r'HAC SE \%','$R^2$'],rows,'Descriptive six-benchmark regressions','tab:factors'))
    # Detailed interaction sources are copied with their provenance, never aggregated across partitions.
    cells=read('financing_cell_return_attribution')
    rows=[]
    for (group,partition),f in cells[cells.period.eq('all')].groupby(['group','partition']):
        # Sum mutually exclusive cells within this partition only.
        col='annual_mean' if 'annual_mean' in f else 'annual_payoff'
        if col in f:rows.append([str(group),escaped(partition),f'{100*f[col].sum():.3f}'])
    write('financing_cells',simple_table(['Group','Partition',r'Annual mean \%'],rows,
        'Joint financing partitions: full cells in accompanying CSV','tab:financing_cells','lp{3.6in}r',
        note='Each partition separately reconstructs the same group. Different overlapping partitions must never be summed.'))
    h=read('e2_summary');rows=[]
    for r in h.itertuples():
        if getattr(r,'period','all')=='all':
            rows.append([str(r.T),f'{r.sharpe:.3f}',f'{getattr(r,"mean_C",float("nan")):.1f}',str(r.months)])
    write('history',simple_table(['History months','SR','Mean $C$','OOS months'],rows,'Earlier common-period history diagnostic','tab:history'))


def compile_pdf(folder,name):
    r=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=folder,capture_output=True,text=True)
    (OUT/'audit'/f'latex_{name}.txt').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('LaTeX build failed: '+name)
    return folder/(name+'.pdf')


def build(draft=False):
    setup()
    # Final delivery never substitutes unavailable results.
    required=['table1_performance','bandwidth_performance','robustness_performance','bandwidth_uncertainty']
    required += [f'neural_performance_seed_{i}' for i in PROTOCOL['neural']['seeds']]
    if not draft:
        for name in required:
            if not (OUT/'tables'/f'{name}.csv').exists():raise FileNotFoundError('Incomplete study: '+name)
    pub=OUT/'publication';pub.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        # Public-only reproduction uses the byte-preserved delivered source.
        archive_source=pub/'full_source'
        if not (archive_source/'source_archive_manifest.json').exists():raise FileNotFoundError('Latest manuscript source unavailable')
        meta=json.loads((archive_source/'source_archive_manifest.json').read_text())
        prefix=(archive_source/'original_main.tex').read_text().split(r'\begin{document}')[0]
    else:
        z=zipfile.ZipFile(source(ARCHIVE));prefix=z.read('Portfolio/main.tex').decode().split(r'\begin{document}')[0]
        archive_source=pub/'full_source';archive_source.mkdir(exist_ok=True)
        meta=dict(archive=str(ARCHIVE),archive_sha256=digest(ARCHIVE),files={})
        for name in z.namelist():
            if name.endswith('/') or not name.startswith('Portfolio/'):continue
            target=archive_source/name.removeprefix('Portfolio/')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
            meta['files'][str(target.relative_to(archive_source))]=digest(target)
        shutil.copyfile(archive_source/'main.tex',archive_source/'original_main.tex')
        (archive_source/'source_archive_manifest.json').write_text(json.dumps(meta,indent=2))
    for d in ['figures','tables']:(pub/d).mkdir(exist_ok=True)
    for p in (OUT/'figures').glob('*.pdf'):shutil.copyfile(p,pub/'figures'/p.name)
    perf=read('table1_performance');spectral=read('spectral_value')
    values,lineage=macro_values(perf,spectral)
    macros='\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in values.items())+'\n'
    (pub/'empirical_numbers.tex').write_text(macros)
    text=(ROOT/'empirical_final/templates/compact_section_vi.tex').read_text()
    (pub/'empirics.tex').write_text(text)
    (pub/'tables/performance.tex').write_text(performance_table(perf))
    (pub/'tables/spectral_value.tex').write_text(spectral_table(spectral))
    appendix_tables(pub/'tables')
    (pub/'empirical_appendix.tex').write_text(APPENDIX)
    annual=read('neural_annual_seed_0')
    failed=int((annual.refit_final_objective>=annual.refit_initial_objective).sum())
    nntext=f'The selected seed-0 refit lowers the full-history penalized objective in {47-failed} of 47 decisions. '
    nntext+='These finite-budget stochastic diagnostics establish actual representation learning and reveal optimization limitations; they do not certify a global optimum. '
    nntext+='The numerical pilot checks the complete-month gradient by finite differences and the exact readout by an independent SVD solve. '
    (pub/'neural_diagnostics_text.tex').write_text(nntext+'\n')
    for name,body in [('empirical_main',r'\setcounter{section}{5}\input{empirics}'),
                      ('empirical_appendix_standalone',r'\appendix\renewcommand{\thesection}{\Alph{section}}\input{empirical_appendix}'),
                      ('empirical_combined',r'\setcounter{section}{5}\input{empirics}\clearpage\appendix\renewcommand{\thesection}{\Alph{section}}\input{empirical_appendix}')]:
        (pub/f'{name}.tex').write_text(prefix+'\n'+r'\input{empirical_numbers}\begin{document}'+'\n'+body+'\n'+r'\end{document}'+'\n')
        compile_pdf(pub,name)
    # Full source: original nonempirical files preserved byte for byte.
    for name in ['empirics.tex','empirical_appendix.tex','empirical_numbers.tex','neural_diagnostics_text.tex']:
        shutil.copyfile(pub/name,archive_source/name)
    for d in ['figures','tables']:
        for p in (pub/d).glob('*'):shutil.copyfile(p,archive_source/d/p.name)
    original=(archive_source/'original_main.tex').read_text()
    full=original.replace(r'\begin{document}',r'\input{empirical_numbers}\begin{document}')
    full=full.replace(r'\end{document}',r'\clearpage\input{empirical_appendix}\end{document}')
    (archive_source/'main.tex').write_text(full)
    compile_pdf(archive_source,'main')
    shutil.copyfile(archive_source/'main.pdf',pub/'full_manuscript.pdf')
    # Remove stale empirical graphics that no final source references.
    references='\n'.join(p.read_text() for p in archive_source.glob('*.tex') if p.name!='original_main.tex')
    for p in (archive_source/'figures').glob('*'):
        if p.stem not in references:p.unlink()
    with zipfile.ZipFile(pub/'Portfolio_Empirics_Compact_LaTeX.zip','w',zipfile.ZIP_DEFLATED) as zipped:
        for p in archive_source.rglob('*'):
            if p.is_file() and (p.suffix in ['.tex','.bib','.pdf','.png','.json']):
                # Exclude full build PDF, retain final figures and sources.
                if p.name not in ['main.pdf','original_main.tex']:zipped.write(p,Path('Portfolio')/p.relative_to(archive_source))
    audit('publication',dict(manuscript_archive=meta,figures=5,tables=2,
        original_theory_preserved=True,source_template_sha256=digest(ROOT/'empirical_final/templates/compact_section_vi.tex'),
        PDFs={p.name:digest(p) for p in pub.glob('*.pdf')},zip_sha256=digest(pub/'Portfolio_Empirics_Compact_LaTeX.zip')))
    audit('numerical_claims',lineage)
    print(pub,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--draft',action='store_true')
    build(parser.parse_args().draft)
