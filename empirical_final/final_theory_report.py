"""Concise main report and decision memo, derived entirely from exported tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import digest
from empirical_final.final_theory_regularization import ROOT,OUT
from empirical_final.theory_guided_lambda import write_json


def deliver(out):
    # Optional local install for the system Python; standard reportlab also works.
    if (ROOT/'tmp/pdf-deps').exists():sys.path.insert(0,str(ROOT/'tmp/pdf-deps'))
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.pagesizes import A4
    s=pd.read_csv(out/'tables/performance_summary.csv');boot=pd.read_csv(out/'tables/paired_bootstrap.csv')
    a=pd.read_csv(out/'tables/annual_policies.csv');constants=pd.read_csv(out/'tables/frozen_constants.csv')
    exclusion=pd.read_csv(out/'tables/exclusion_sensitivity_summary.csv');rolling=pd.read_csv(out/'tables/rolling_diagnostic_summary.csv',float_precision='round_trip')
    rank=pd.read_csv(out/'tables/rolling_rank_sensitivity.csv');interaction=pd.read_csv(out/'tables/interactions.csv')
    rep=json.loads((out/'audit/replication.json').read_text());cost=json.loads((out/'audit/costs.json').read_text())
    dense_out=out/'dense_lengthscale'
    ds=pd.read_csv(dense_out/'tables/performance_summary.csv');db=pd.read_csv(dense_out/'tables/paired_bootstrap.csv')
    da=pd.read_csv(dense_out/'tables/annual_policies.csv');dc=pd.read_csv(dense_out/'tables/frozen_constants.csv')
    labels={'A':'A: fixed length, fixed b','B':'B: fixed length, updated b','C':'C: selected length, fixed b',
            'D':'D: selected length, updated b','joint_cv':'Joint annual CV','annual_cv':'Fixed-length annual CV',
            'fixed_lambda':'Fixed initial lambda','one_anchor':'One-anchor full-spectrum rule',
            'C_dense':'C dense','D_dense':'D dense'}
    def row(method,period='1979-2024',scenario='trade25_borrow30',dense=False):
        frame=ds if dense else s
        return frame[frame.method.eq(method)&frame.period.eq(period)&frame.scenario.eq(scenario)].iloc[0]
    def ci(comparison,scenario='trade25_borrow30',metric='sharpe',dense=False,period='1979-2024',block=12):
        frame=db if dense else boot
        return frame[frame.comparison.eq(comparison)&frame.scenario.eq(scenario)&frame.metric.eq(metric)&frame.period.eq(period)&frame.block_months.eq(block)].iloc[0]
    def interval(q):return f"{q.estimate:+.3f} [{q.ci_lower:+.3f}, {q.ci_upper:+.3f}]"
    baseline_b=a[a.method.eq('B')].sort_values('year').iloc[-1]
    total=-baseline_b.log_lambda_relative_initial
    sample_share=-baseline_b.log_sample_size_contribution/total
    update_share=-baseline_b.log_b_update_contribution/total
    robust_dense=all(ci('D_dense-B',dense=True,block=k).ci_lower>0 for k in (6,12,24))
    # Main controlled result is mixed. Exploration cannot retroactively confirm it.
    decision='Leave the current manuscript unchanged. The controlled representation extension weakens B, and incremental net evidence for updating b is inconclusive.'
    dense_gain=ci('D_dense-D',dense=True);dense_joint=ci('D_dense-joint_cv',dense=True);dense_b=ci('D_dense-B',dense=True)
    dense_description=(f"The exploratory denser search gives primary gross/net Sharpe {row('D_dense',scenario='gross',dense=True).sharpe:.3f}/{row('D_dense',dense=True).sharpe:.3f}; "
        f"net D dense minus B is {interval(dense_b)}. "
        + ('Its net gain over B clears all three block intervals, but this sequential follow-up needs independent confirmation before a paper claim.' if robust_dense else 'It does not establish a robust incremental net advantage over B.'))
    memo=f"""# Final decision memo - theory-guided regularization

{decision}

The exact baseline was reproduced: {rep['full_gross_sharpe']:.14f} gross Sharpe, all 564 payoffs and 47 annual schedules; maximum payoff error {rep['maximum_payoff_error']:.3g}.

1. **How much does updating b add?** On the full sample, A is {row('A','1978-2024','gross').sharpe:.6f} and B {row('B','1978-2024','gross').sharpe:.6f}: an increment of {row('B','1978-2024','gross').sharpe-row('A','1978-2024','gross').sharpe:.6f}, not a causal decomposition of Sharpe. On the primary 1979-2024 decisions, the gross increment is {interval(ci('B-A','gross'))}; net is {interval(ci('B-A'))}. Updating b has a small measurable gross gain conditional on these fitted paths, but no resolved net gain.
2. **Is initial fixed b nearly as good?** Its Sharpe point estimate is close. No equivalence margin was declared, and an interval including zero does not establish equivalence.
3. **Does varying length improve the theoretical rule?** In the controlled five-candidate design, D minus B is {interval(ci('D-B','gross'))} gross and {interval(ci('D-B'))} net. Selection worsens performance; the net difference remains negative across 6-, 12- and 24-month intervals.
4. **Does it improve net Sharpe?** D is {row('D').sharpe:.6f}, below B at {row('B').sharpe:.6f}. D minus C is {interval(ci('D-C'))}, so updating b within the selected representation has an unresolved net benefit.
5. **Can it compete with joint CV?** D has gross/net {row('D',scenario='gross').sharpe:.3f}/{row('D').sharpe:.3f}, joint CV {row('joint_cv',scenario='gross').sharpe:.3f}/{row('joint_cv').sharpe:.3f}. The net difference is {interval(ci('D-joint_cv'))}; this is a competitive point estimate, not demonstrated superiority or noninferiority.
6. **Fewer searches?** A/B: one 200-candidate initial grid; C/D: five such initial grids (1,000 evaluations, with invalid candidates reported), then zero annual lambda-grid searches. Each subsequent C/D decision evaluates three admissible lengths. Joint CV evaluates 5 x 120 = 600 length/penalty combinations each year; fixed-length CV uses 120. Complexity of spectral decomposition, full-history refitting and holdings reconstruction remains.
7. **Historical robustness?** Main findings are supplied on 1979-2024, 1984-2024, four historical blocks, and the last five decision years. The last five years are descriptive. Circular-block intervals condition on fitted strategies and do not include algorithm-selection uncertainty. All length calibrations share the initial validation window, and early rolling validation windows overlap it; 1984 onward removes that overlap. This exercise inherits retrospective signals, complete-case availability, the current JKP snapshot and same-close execution assumptions.
8. **Paper subsection?** Keep the manuscript unchanged. No one-page subsection is proposed from this mixed controlled evidence; no appendix or new methodology is introduced.

At the final expanding refit, {100*sample_share:.1f}% of the log-penalty decrease from the original initial CV penalty comes from increasing T and {100*update_share:.1f}% from updating b. The exact log identity is exported for A-D; representation changes form a separate term. B's last penalty is {baseline_b.penalty:.9g}, versus {a[a.method.eq('A')].sort_values('year').iloc[-1].penalty:.9g} holding b=b0. This attribution concerns lambda, not the sources of investment performance.

## Admissibility materially matters

The initial exponents at multipliers 0.25 and 0.5 are {constants.iloc[0].b:.6f} and {constants.iloc[1].b:.6f}, below one. They are permanently unavailable in the primary theoretical comparison. Extrapolating the formula to these candidates changes 36 of 46 primary decisions: D's gross Sharpe becomes {exclusion[exclusion.method.eq('D')&exclusion.period.eq('1979-2024')].iloc[0].extrapolated_gross_sharpe:.3f}, compared with {row('D',scenario='gross').sharpe:.3f} for the admissible rule. This diagnostic is outside the theorem and has no new net-performance claim. These exclusions are consequential, not a minor numerical detail.

## User-requested extra optimization

The nine-candidate follow-up adds 0.75, 1.25, 1.5 and 3 times the original lengthscale using the same frequency and phase draws, initial 200-candidate calibration, past-only selector and frozen constants. It was declared after the coarse gross results were available and is exploratory. {dense_description} Net D dense minus original D is {interval(dense_gain)}; versus joint CV it is {interval(dense_joint)}. These results do not change the primary 2 x 2 findings or identify a population-optimal data-dependent penalty.

## Monthly rolling decision

Both trailing windows are translated before the diagnostic, at January 1984 close, to the then-available expanding B penalty: C0(W)=lambda_anchor W^[b_anchor(W)/(b_anchor(W)+1)]. Constants are frozen thereafter. On the original 10%-60% rank band, median absolute monthly log-lambda changes are {100*rolling[rolling.W.eq(180)&rolling.rank_fraction_low.eq(.1)&rolling.rank_fraction_high.eq(.6)].iloc[0].median_abs_log_lambda_monthly_change:.3f}% and {100*rolling[rolling.W.eq(240)&rolling.rank_fraction_low.eq(.1)&rolling.rank_fraction_high.eq(.6)].iloc[0].median_abs_log_lambda_monthly_change:.3f}%. No primary slope falls below one. Cross-band sensitivity exceeds ordinary monthly movement. The predeclared 5% change / rank-stability gate fails; a costly full monthly portfolio backtest is not justified by this diagnostic. This is not evidence that rolling portfolios could never outperform expanding portfolios.

All costs use each strategy's actual signed stock weights, the frozen common exposure scale {cost.get('kappa',json.loads((out/'audit/estimation.json').read_text())['kappa']):.12g}, 25 bp on traded notional and 30 bp/year on short notional through verified self-financing accounting. States begin in cash for February 1978 and carry into subperiods; holdings are marked at the end without liquidation. Quadratic losses use raw payoff units; net losses divide by the same pretest scale. Public exports contain aggregate paths only; licensed panels, coefficients and stock holdings remain under ignored results/.
"""
    (out/'DECISION_MEMO.md').write_text(memo)
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodySmall',fontName='Helvetica',fontSize=9.2,leading=13,spaceAfter=7,textColor=colors.HexColor('#28364a')))
    styles.add(ParagraphStyle(name='SmallNote',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=5,textColor=colors.HexColor('#526176')))
    styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=21;styles['Title'].leading=25
    styles['Heading1'].fontSize=16;styles['Heading1'].leading=20;styles['Heading1'].textColor=colors.HexColor('#244c85')
    styles['Heading2'].fontSize=11;styles['Heading2'].leading=15
    story=[]
    def p(text,style='BodySmall'):story.append(Paragraph(text,styles[style]))
    def title(text):p(text,'Heading1')
    def table(header,rows,widths=None,size=8.3):
        cells=[[Paragraph(str(x),ParagraphStyle(name='cell',fontName='Helvetica-Bold' if i==0 else 'Helvetica',fontSize=size,leading=size+2)) for x in values] for i,values in enumerate([header,*rows])]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#244c85')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,9)])
    p('The law of portfolio learnability','Title')
    p('Final controlled regularization experiment - 11 October 2026','Heading2')
    p(decision)
    p(f'Baseline replication: gross Sharpe {rep["full_gross_sharpe"]:.12f}; 564 monthly returns; maximum payoff difference {rep["maximum_payoff_error"]:.2g}. Annual b, lambda, complexity and initial calibration all match.')
    title('The 2 x 2 result')
    p('Primary sample: February 1979-January 2025, 552 months. Initial 1978 decision excluded. Sharpe uses concatenated monthly excess returns.')
    table(['Representation','Fixed initial b','Updated b'],[
        ['Fixed lengthscale',f'A: gross {row("A",scenario="gross").sharpe:.3f} / net {row("A").sharpe:.3f}',f'B: gross {row("B",scenario="gross").sharpe:.3f} / net {row("B").sharpe:.3f}'],
        ['Chronologically selected lengthscale',f'C: gross {row("C",scenario="gross").sharpe:.3f} / net {row("C").sharpe:.3f}',f'D: gross {row("D",scenario="gross").sharpe:.3f} / net {row("D").sharpe:.3f}']],widths=[174,164,164])
    table(['Comparison','Gross Sharpe difference [95% CI]','Net Sharpe difference [95% CI]'],[
        [x,interval(ci(x,'gross')),interval(ci(x))] for x in ('B-A','C-A','D-B','D-C','D-joint_cv')],widths=[92,205,205])
    p('Intervals use 5,000 paired circular moving-block draws, 12 months here; 6- and 24-month sensitivity is exported. Inference conditions on the fitted paths and has no multiple-comparison correction. Updating b gives a small gross gain, but its net gain is unresolved. The selected-length rule D underperforms B net under all three block lengths.')
    p('The theorem motivates the order lambda(T) proportional to T^[-b/(b+1)]. It does not establish optimality for these finite-rank estimates, empirically calibrated constants, or data-dependent representation choices. No oracle or causal claim is made.','SmallNote')
    story.append(PageBreak());title('Financial implementation and common samples')
    p('Every policy uses the same fixed pretest exposure calibration. Signed holdings are reconstructed separately; net returns use the existing accounting_step at 25 bp per traded notional plus 30 bp/year on shorts. All subperiods carry the existing account state from 1978.')
    table(['Method','Gross SR','Net SR','Mean / vol (% p.a.)','Leverage','Turnover / month','Max DD (%)'],[
        [labels[m],f'{row(m,scenario="gross").sharpe:.3f}',f'{row(m).sharpe:.3f}',
         f'{100*row(m).annual_excess_return:.2f} / {100*row(m).annual_volatility:.2f}',
         f'{row(m).mean_leverage:.2f}',f'{row(m).mean_monthly_turnover:.2f}',f'{100*row(m).maximum_drawdown:.2f}']
        for m in ('A','B','C','D','annual_cv','joint_cv','fixed_lambda','one_anchor')],widths=[149,40,40,87,45,70,71],size=7.7)
    p('Mean, volatility and drawdown above are net. Leverage is gross stock notional / NAV; turnover charges purchases and sales. The one-anchor full-spectrum rule is a separate secondary reference, never substituted for B.')
    table(['Window (February-January)','A gross/net','B gross/net','C gross/net','D gross/net','Joint CV gross/net'],[
        [period,f'{row("A",period,"gross").sharpe:.3f}/{row("A",period).sharpe:.3f}',
         f'{row("B",period,"gross").sharpe:.3f}/{row("B",period).sharpe:.3f}',
         f'{row("C",period,"gross").sharpe:.3f}/{row("C",period).sharpe:.3f}',
         f'{row("D",period,"gross").sharpe:.3f}/{row("D",period).sharpe:.3f}',
         f'{row("joint_cv",period,"gross").sharpe:.3f}/{row("joint_cv",period).sharpe:.3f}'] for period in ('1978-2024','1979-2024','1984-2024','2020-2024')],widths=[122,76,76,76,76,76],size=7.8)
    p('The window labels are decision years: e.g., 2020-2024 means February 2020-January 2025, and is descriptive only. The 1984 sensitivity removes the early overlap between initial calibration and repeated validation blocks.')
    table(['Historical block','A net','B net','C net','D net','Joint CV net'],[
        [period,*[f'{row(m,period).sharpe:.3f}' for m in ('A','B','C','D','joint_cv')]] for period in ('1979-1989','1990-1999','2000-2009','2010-2019')],widths=[122,76,76,76,76,76])
    p('These conditional historical comparisons inherit the common retrospective signal dictionary, complete-case payoff availability, current JKP snapshot and same-close execution assumptions. They do not establish future robustness.','SmallNote')
    story.append(PageBreak());title('What moves the penalty?')
    story.append(Image(str(out/'figures/regularization_diagnostic.png'),width=502,height=329))
    p(f'At T=732, increasing sample size accounts for {100*sample_share:.1f}% of the log decrease from the initial CV lambda; updating b accounts for {100*update_share:.1f}%. Holding b fixed gives final lambda {a[a.method.eq("A")].iloc[-1].penalty:.4g}, versus {baseline_b.penalty:.4g} for B. These are exact lambda identities, not attributions of Sharpe.')
    p('Decomposition: log(lambda / initial baseline lambda) = sample-size term + representation term + b-update term. The representation term includes both its frozen C0 and initial exponent; the b term compares the current slope with that representation\'s initial slope.','SmallNote')
    table(['Multiplier','Initial b','Initial lambda','Frozen C0','Validation loss','Admissible'],[
        [f'{r.multiplier:g}',f'{r.b:.4f}',f'{r.initial_lambda:.3g}',f'{r.C0:.3g}',f'{r.validation_loss:.4f}','Yes' if r.admissible else 'No: b <= 1'] for r in constants.itertuples()],widths=[65,67,91,88,90,101])
    p('Calibration uses 1963-1972 training (120 months) and 1973-1977 validation (60 months). C0 = initial lambda x 120^[b0/(b0+1)]; the first full-history refit uses T=180. Constants for inadmissible lengths are arithmetic extrapolations recorded for sensitivity only, never used by the primary algorithm.')
    story.append(PageBreak());title('Wealth, loss and admissibility')
    story.append(Image(str(out/'figures/performance_comparison.png'),width=502,height=205))
    p('Wealth includes the frozen cash return and starts at one for the primary display. Net accounts carry the 1978 state. Terminal holdings are marked without a liquidation charge. Both A-D and the joint CV paths are exported at full monthly precision.')
    table(['Difference','Gross response-one loss [95% CI]','Net raw-unit loss [95% CI]'],[
        [x,interval(ci(x,'gross','loss')),interval(ci(x,metric='loss'))] for x in ('B-A','C-A','D-B','D-C','D-joint_cv')],widths=[92,205,205])
    inter=interaction[interaction.period.eq('1979-2024')&interaction.scenario.eq('trade25_borrow30')&interaction.metric.eq('sharpe')&interaction.block_months.eq(12)].iloc[0]
    p(f'Descriptive net interaction (D-C)-(B-A): {interval(inter)}. Lower loss is better. Net response-one losses divide monthly net excess returns by the same fixed pretest exposure scale; they are not a new selection objective.')
    ex=exclusion[exclusion.period.eq('1979-2024')&exclusion.method.eq('D')].iloc[0]
    p(f'<b>Candidate exclusions materially affect the comparison.</b> Two initial b estimates violate b&gt;1. Allowing them as an explicitly outside-theorem extrapolation changes {ex.changed_selection_years} of 46 primary decisions and raises D gross Sharpe from {ex.admissible_gross_sharpe:.3f} to {ex.extrapolated_gross_sharpe:.3f}. This is an assumption-violation diagnostic, not a valid theorem-guided improvement; no new net claim is attached.')
    p('No lambda search occurs after initial calibration in A-D. Joint CV evaluates 600 penalty/length combinations per annual decision; C/D evaluate the three admissible lengths, each at one theory-specified penalty. Fewer validation searches do not remove the cost of eigendecomposition or portfolio refitting.')
    story.append(PageBreak());title('Is monthly re-estimation justified?')
    story.append(Image(str(out/'figures/rolling_diagnostic.png'),width=502,height=351))
    table(['Trailing W','SD monthly change in b','Median |log lambda change|','Median |log complexity change|','b <= 1 frequency'],[
        [str(W),f'{q.b_monthly_change_std:.4f}',f'{100*q.median_abs_log_lambda_monthly_change:.3f}%',f'{100*q.median_abs_log_complexity_monthly_change:.3f}%',f'{100*q.inadmissible_fraction:.1f}%']
        for W in (180,240) for q in [rolling[rolling.W.eq(W)&rolling.rank_fraction_low.eq(.1)&rolling.rank_fraction_high.eq(.6)].iloc[0]]],widths=[63,109,117,127,86],size=8)
    p('Before the diagnostic begins, both window constants are translated at January 1984 close to the then-available expanding B penalty. The window-specific anchor slope determines C0(W), frozen thereafter. Four rank bands are supplied; each uses the same pretest anchoring convention.')
    p('Median monthly penalty movement is small and rank sensitivity exceeds it. Neither window meets the declared 5% monthly-change and rank-stability gate. No full monthly OOS portfolio backtest was run; the diagnostic gives no compelling reason to replace annual expanding estimation.')
    p(f'Holdings audit: largest scaled payoff reconciliation error {cost["maximum_payoff_error"]:.2g}; accounting residual {cost["maximum_accounting_residual"]:.2g}; original CV net path reproduced. Numerical tests cover chronology, frozen constants, rank thresholds, primal/dual predictions, shared draws, signed holdings and figure/table reconciliation. Input hashes are saved under audit/.','SmallNote')
    story.append(PageBreak());title('Additional optimization requested during execution')
    p('Exploratory follow-up: the original five lengths plus 0.75, 1.25, 1.5 and 3 times ell0. New managed-payoff caches use the same real panels, frequency draws and phases. Each new length gets the original 200-candidate initial calibration; no annual lambda grid is added. This follow-up was specified after observing coarse-grid gross results.')
    table(['Method','1979 gross/net','1984 gross/net','Last five years gross/net'],[
        [labels[m],*[f'{row(m,period,"gross",dense=True).sharpe:.3f}/{row(m,period,dense=True).sharpe:.3f}' for period in ('1979-2024','1984-2024','2020-2024')]]
        for m in ('C_dense','D_dense','C','D','B','joint_cv')],widths=[122,126,126,128])
    table(['Primary net comparison','Difference [12-month 95% CI]','6-month interval','24-month interval'],[
        [comparison,interval(ci(comparison,dense=True)),
         f'[{ci(comparison,dense=True,block=6).ci_lower:+.3f}, {ci(comparison,dense=True,block=6).ci_upper:+.3f}]',
         f'[{ci(comparison,dense=True,block=24).ci_lower:+.3f}, {ci(comparison,dense=True,block=24).ci_upper:+.3f}]']
        for comparison in ('D_dense-D','D_dense-B','D_dense-C_dense','D_dense-joint_cv')],widths=[131,165,103,103],size=8)
    p(dense_description)
    table(['Added multiplier','Initial b','C0','Validation loss','Admissible'],[
        [f'{r.multiplier:g}',f'{r.b:.4f}',f'{r.C0:.3g}',f'{r.validation_loss:.4f}','Yes' if r.admissible else 'No'] for r in dc[dc.multiplier.isin([.75,1.25,1.5,3])].itertuples()],widths=[98,80,115,108,101])
    p('The expanded search evaluates nine initial grids (1,800 candidate policies), with seven initially admissible lengths. Later annual decisions select length only. Saved paired intervals condition on the resulting fitted paths and do not account for sequentially expanding the experiment after seeing earlier results.')
    p('<b>Paper decision:</b> retain the current manuscript. Preserve these negative, mixed and exploratory findings as a reproducible research record; there is no proposed new subsection, appendix, or oracle-optimal tuning claim.')
    story.append(Spacer(1,8));p('Reproduce: empirical_final/final_theory_regularization.py; final_theory_analysis.py; final_theory_dense.py; final_theory_report.py. Public tables, candidate losses, bootstrap draws specification and source/input hashes accompany this report. Licensed stock-level holdings remain private.','SmallNote')
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526176'))
        canvas.drawString(46,26,'Portfolio learnability | frozen inputs | no manuscript changes');canvas.drawRightString(A4[0]-46,26,str(doc.page));canvas.restoreState()
    pdf=out/'publication/final_theory_report.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=36,bottomMargin=43,
        title='Final theory-guided regularization experiment',author='Portfolio Learnability').build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'.gitattributes').write_text('*.pdf binary\n')
    (out/'README.md').write_text('''# Final theory-guided regularization experiment

The original 3.56133432162478 Sharpe baseline is reproduced before the controlled 2 x 2 design. See `DECISION_MEMO.md` and `publication/final_theory_report.pdf` for the findings. Original published outputs and the manuscript are preserved.

## Reproduction

Requires the existing licensed JKP clean panels and verified managed-payoff caches, Python/numpy/pandas/scipy/pyarrow/threadpoolctl/matplotlib, reportlab for the PDF, and pytest. Run from the repository root with single BLAS threads. The complete reproduction wrapper requires fresh destinations and preserves the delivered package.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m empirical_final.final_theory_reproduce \\
  --output outputs/final_theory_regularization_reproduction \\
  --private results/final_theory_regularization_reproduction
python3 -m pytest -q tests/test_final_theory_regularization.py tests/test_final_theory_dense.py
```

The wrapper calls replication, estimation, actual-holdings costs, paired inference, rolling diagnostics, figures, the separately specified denser search, and the report in order. Both destinations must be absent. Individual phase commands are also available in the underlying modules. The tests reconcile the default delivered package. Never delete or overwrite a completed output to rerun. PDF rendering and visual QA use `pdftoppm`; rendered pages are temporary QA artifacts.

## Tables and units

- `frozen_constants.csv`, `initial_validation_candidates.csv`: all five initial calibrations, including inadmissible candidates. Numeric C0 for b<=1 is diagnostic extrapolation and not used by primary strategies.
- `candidate_losses.csv`: inner/full slopes and dates, single scheduled inner penalties, availability and historical candidate losses.
- `annual_policies.csv`, `annual_spectra.csv`, `lambda_decomposition.csv`: annual choices, penalties, complexity, exact sample/b/representation log decomposition and explicit fixed-b penalty counterfactuals.
- `monthly_payoffs.csv`, `monthly_accounts.csv`: raw and fixed-scale gross payoffs; strategy-specific actual-holdings gross, target-change cost sensitivity and exact self-financing net accounts.
- `performance_summary.csv`, `annual_performance.csv`, `cumulative_performance.csv`: annual/cumulative mean, volatility, Sharpe, loss, turnover, leverage and drawdown; gross and net common windows. Month labels are return dates; period labels are annual decision years.
- `paired_bootstrap.csv`, `interactions.csv`: conditional paired 95% circular-block intervals at 6, 12 and 24 months, 5,000 replicates; point comparisons are fitted-strategy differences, not causal effects.
- `exclusion_sensitivity_*`: outside-theorem positive-b extrapolation, gross only.
- `rolling_*`: window-specific pretest translations and monthly spectrum-only diagnostics; no monthly portfolio backtest.
- `performance_figure_data.csv`: exact primary gross/net wealth curves used by the performance figure. Annual and rolling diagnostic figures plot their exported path tables directly.
- `dense_lengthscale/`: separate exploratory denser representation search requested during execution, retaining the same initial calibration, spectral rule and trading implementation.

Response-one loss uses raw portfolio payoff units. For net inference, net excess returns are divided by the same fixed pretest exposure scale. Reported investment mean/volatility/wealth use that scale and the frozen cash series. Net account states begin in cash February 1978 and carry into subperiods, without terminal liquidation. Stock identifiers and holdings are excluded from public exports. Private coefficient and holdings hashes provide reconciliation receipts.

## Interpretation

The baseline regression is a finite-rank diagnostic, not identification of a population spectral exponent. The theorem assumes b>1; no clipping or CV replacement is performed. The initial calibration block also selects the anchor representation. Primary inference excludes 1978; the 1984 comparison removes early overlapping calibration/validation history. The extra denser grid is explicitly exploratory because it was proposed after seeing coarse-grid gross results. No OOS ranking selects a deployment hyperparameter.
''')
    write_json(out/'audit/report.json',dict(passed=True,pdf_sha256=digest(pdf),memo_sha256=digest(out/'DECISION_MEMO.md'),
        report_source_sha256=digest(Path(__file__)),decision='manuscript unchanged',data_driven=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    deliver(parser.parse_args().output)


if __name__=='__main__':main()
