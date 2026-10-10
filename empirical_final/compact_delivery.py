"""Audit coverage, map public numerical claims, and prepare the critical report."""
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import digest
from portfolio import sharpe
from empirical_final.three_experiments import block_counts,boot_metrics,interval
from empirical_final.compact_common import ROOT,OUT,PROTOCOL,save,audit,setup


def bandwidth_intervals():
    fixed=pd.read_csv(OUT/'tables/monthly_accounting.csv')
    others=[pd.read_csv(OUT/'tables'/f'bandwidth_{k}_accounting.csv') for k in ['gaussian','matern32']]
    m=pd.concat([fixed,*others]);rows=[]
    for kernel in ['gaussian','matern32']:
        for scenario in ['gross','trade25_borrow30']:
            p=m[m.scenario.eq(scenario)&m.policy.isin([kernel,kernel+'_tuned'])].pivot(
                index='return_date',columns='policy',values='excess_return').sort_index()
            for length in PROTOCOL['bootstrap']['blocks']:
                boot=boot_metrics(p[[kernel+'_tuned',kernel]].to_numpy(),block_counts(len(p),length,20261010,5000))['sharpe']
                ci=interval(boot[:,0]-boot[:,1])
                rows.append(dict(kernel=kernel,scenario=('Gross' if scenario=='gross' else 'Net'),block_length=length,
                    delta_sharpe=float(sharpe(p[kernel+'_tuned'])-sharpe(p[kernel])),low=ci[0],high=ci[1]))
    save('bandwidth_uncertainty',rows)


def reconciliation():
    # Preserve uncertainty about the archived source; do not invent a causal reconciliation.
    rows=[
        ('Gaussian Sharpe','3.350 (rounded archival table)','3.299902664375','Different; constant positive scale cannot explain'),
        ('Average Gaussian gross','1.76 (rounded archival table)','2.326054439524','Different; archival scaling rule not recoverable'),
        ('Sample membership','JKP, excludes nano, 130 characteristics; no input hash','Frozen clean manifest ca6e97aa..., 7,489 payoff exclusions','Exact old membership unverified'),
        ('Formation/realization dates','1978-2024, 564 months stated; shift not disclosed','Jan-Dec formation -> Feb-next Jan payoffs','Archive chronology insufficient to prove equivalence'),
        ('Preprocessing','Cross-sectional ranks, 30% missing cutoff; coverage window unstated','Initial-only characteristic coverage; payoff filter before ranks and N_t','Old formation-eligibility ordering not documented'),
        ('Feature bank','10,000 RFF stated; hashes absent','Seed-0 frequencies/phases SHA in source manifests','Exact old feature draw unverified'),
        ('Bandwidth','Initial-sample median stated; value/sample hash absent','Initial sample checksum and exact bank ell','Verbal rule alone cannot establish same bandwidth'),
        ('Penalty grid/validation','Fixed grid, five-year validation stated; numerical grid absent','120 initial-spectrum candidates; minimum fitted-scale validation Q','Old candidate spacing and selections unverified'),
        ('Refitting','Annual refit stated; head and selection paths absent','Train+validation through January close','Old refit implementation not reproducible'),
        ('Exposure scaling','Not documented by archive; different means/gross','Common scale 0.03740000227285338, initial Linear validation','Scale might explain much of levels, cannot explain SR difference'),
        ('Units','Percent annual means/volatility table; excess wealth caption','Decimal monthly excess payoffs; total wealth includes rf','Both appear to report excess SR, old series unavailable'),
        ('Cost accounting','Archive figure only; no stock holdings or fee path','Drift-adjusted two-sided trades, netted account weights, annual short fee/12','Old cost path unverified'),
        ('Canonical cross-check','No numerical output paths in archive','Kernel baseline -> recomputed managed path -> stock holdings -> spectral sum','Maximum reconstruction errors recorded in canonical audits')]
    save('snapshot_reconciliation',[dict(dimension=d,archived_snapshot=a,canonical_snapshot=c,assessment=s) for d,a,c,s in rows])
    # Coverage inputs copied as aggregate source records, stock inputs stay private.
    import shutil
    for name in ['formation_timing_by_month.csv','formation_timing_audit.json']:
        src=ROOT/'outputs/tables'/name
        target=OUT/('tables' if src.suffix=='.csv' else 'audit')/name
        shutil.copyfile(src,target)
    materiality=ROOT/'outputs/final_empirical_20261008/tables/missing_payoff_materiality.csv'
    if materiality.exists():shutil.copyfile(materiality,OUT/'tables'/materiality.name)
    for name in ['financing_cell_return_attribution']:
        p=ROOT/'outputs/spectral_economic_content_20261009/tables'/f'{name}.csv'
        shutil.copyfile(p,OUT/'tables'/p.name)


def source_map():
    claims=json.loads((OUT/'audit/numerical_claims.json').read_text())
    figures={
        'fig01_kernel_wealth':['figure01_wealth','monthly_accounting'],
        'fig02_managed_spectrum':['figure02_spectrum','managed_spectra'],
        'fig03_gaussian_complexity':['single_window_complexity','e1_monthly_paths','kernel_annual_paths'],
        'fig04_gaussian_nine_panels':['nine_panel_curves','nine_panel_summary','nine_panel_month_identifiers','e1_monthly_paths','kernel_annual_paths'],
        'fig05_economic_holdings':['economic_exposure_summary','monthly_economic_exposures']}
    table_map={'performance':['table1_performance','neural_monthly_seed_0','monthly_accounting'],
               'spectral_value':['spectral_value','e3_monthly','e3_annual_groups','monthly_accounting','monthly_economic_exposures']}
    audit('source_map',dict(main_text_claims=claims,figures={k:[f'tables/{n}.csv' for n in v] for k,v in figures.items()},
        main_tables={k:[f'tables/{n}.csv' for n in v] for k,v in table_map.items()},
        point_identifiers='Figure 1: policy/scenario/return_date. Figure 2: policy/decision_year/rank. Figure 3: candidate. Figure 4: period/candidate, selected paths separate. Figure 5: group/family.',
        computations={'wealth':'cumprod(1+total_return)', 'Sharpe':'sqrt(12)*mean(monthly excess)/sample sd',
                      'complexity':'sum(mu/(mu+lambda))', 'economic_profiles':'mean stock-holding-weighted formation scores; signed notional separately'},
        producer_modules=['compact_study.py','compact_figures.py','neural_portfolio.py','compact_robustness.py','compact_publication.py']))
    code={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'empirical_final').glob('compact*')) if p.is_file()}
    code.update({str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'empirical_final/neural_portfolio.py',ROOT/'empirical_final/templates/compact_section_vi.tex',ROOT/'tests/test_compact_empirical.py']})
    inputs={}
    for p in (OUT/'audit').glob('*.json'):
        obj=json.loads(p.read_text())
        if isinstance(obj,dict):inputs.update(obj.get('inputs',{}))
    artifacts={str(p.relative_to(OUT)):digest(p) for p in (OUT/'tables').glob('*.csv')}
    artifacts.update({str(p.relative_to(OUT)):digest(p) for p in (OUT/'figures').glob('*') if p.suffix in ['.png','.pdf']})
    audit('run_manifest',dict(schema='compact-empirics/1.0',starting_commit=PROTOCOL['starting_commit'],
        protocol_sha256=digest(ROOT/'empirical_final/compact_protocol.json'),python=platform.python_version(),
        code_sha256=code,input_sha256=inputs,artifact_sha256=artifacts,
        formation_dates=['1963-01-31','2024-12-31'],OOS_realization_dates=['1978-02-28','2025-01-31'],
        main_bandwidth='fixed initial-sample median',kernels=['linear','gaussian','matern32'],feature_count=10000,seed=0,
        scale=0.03740000227285338,selection=PROTOCOL['selection'],neural=PROTOCOL['neural'],
        freeze_scope='Execution protocol for a revision, not retrospective preregistration',
        privacy='Stock observations, identifiers, holdings and extractor checkpoints remain ignored under data/ and results/'))


def report():
    perf=pd.read_csv(OUT/'tables/table1_performance.csv');bw=pd.read_csv(OUT/'tables/bandwidth_performance.csv')
    spectral=pd.read_csv(OUT/'tables/spectral_value.csv');panels=pd.read_csv(OUT/'tables/nine_panel_summary.csv')
    neural=pd.read_csv(OUT/'tables/neural_annual_seed_0.csv')
    sr=lambda frame,policy,sc:frame[(frame.policy==policy)&(frame.scenario==sc)].sharpe.iloc[0]
    text=f'''# Critical empirical revision report

Starting commit: `{PROTOCOL['starting_commit']}`. Current branch: main.

One fixed-bandwidth conditional research-sample protocol now supplies all five main figures, both tables, and the financial attribution. The 3x3 figure is generated as actual PDF/PNG, with 564 distinct realized months, eight 60-month blocks and one 84-month block. {int((~panels.boundary_maximum).sum())}/9 maxima are interior, descriptively; paired pointwise bands do not validate hindsight-selected optima.

Gaussian/Linear gross Sharpe is {sr(perf,'gaussian','gross'):.3f}/{sr(perf,'linear','gross'):.3f}; net is {sr(perf,'gaussian','trade25_borrow30'):.3f}/{sr(perf,'linear','trade25_borrow30'):.3f}. The 12-month paired gross interval includes zero. All intervals remain conditional on fitted paths and unadjusted for multiple contrasts. Six-/24-month and subperiod evidence is reported rather than choosing a favorable block length.

Fixed/tuned Gaussian gross Sharpe is {sr(bw,'gaussian','gross'):.3f}/{sr(bw,'gaussian_tuned','gross'):.3f}; net is {sr(bw,'gaussian','trade25_borrow30'):.3f}/{sr(bw,'gaussian_tuned','trade25_borrow30'):.3f}. Matérn gross is {sr(bw,'matern32','gross'):.3f}/{sr(bw,'matern32_tuned','gross'):.3f}; net is {sr(bw,'matern32','trade25_borrow30'):.3f}/{sr(bw,'matern32_tuned','trade25_borrow30'):.3f}. Fixed controls are reproduced; tuning remains secondary. Boundary choices, paired uncertainty and actual tuned account costs are public.

The genuinely learned seed-0 neural portfolio with exact ridge readout has gross/net Sharpe {sr(perf,'neural','gross'):.3f}/{sr(perf,'neural','trade25_borrow30'):.3f}. Hidden parameters change in every refit. {int((neural.refit_final_objective<neural.refit_initial_objective).sum())}/47 selected refits improve the penalized training objective; full trajectories reveal finite-budget limitations. This is an alternating complete-month direct portfolio learner, not a stock-return predictor or an unmodified end-to-end network. Its feature kernel and conditional head spectrum do not establish the fixed-kernel theorem for hidden-layer learning. Five individual seeds are reported without seed selection or invalid averaging of Sharpes.

Adding the intermediate spectral group changes gross Sharpe {spectral.cumulative_gross_sharpe.iloc[0]:.3f} -> {spectral.cumulative_gross_sharpe.iloc[1]:.3f}, and net {spectral.cumulative_net_sharpe.iloc[0]:.3f} -> {spectral.cumulative_net_sharpe.iloc[1]:.3f}. Its component gross {spectral.component_gross.iloc[1]:.3f} increases netted account gross by only {spectral.account_gross.iloc[1]-spectral.account_gross.iloc[0]:.3f}. Negative covariance coexists with positive incremental annual variance {spectral.incremental_annual_variance.iloc[1]:.6f}. Turnover and short fees are recomputed after stock netting; low incremental gross does not imply negligible costs. Weak-tail net increments and alternate groupings are retained even when adverse.

The archived 3.350/1.76 Gaussian snapshot has no supplied monthly path, weights, or execution manifest. Its historical causal discrepancy remains unverified; constant positive scaling cannot explain Sharpe. `tables/snapshot_reconciliation.csv` traces each available setting and explicitly marks the missing provenance. The operational discrepancy is resolved by rebuilding every main result from the reproducible 3.300/2.326 snapshot. Old artifacts remain legacy outputs, never mixed into this run.

Formation-only auditing leaves 7,489 unresolved payoffs out of 2,504,402 eligible stock-months (0.299%). No documented adjacent-calendar source observations recover these outcomes. No zero returns or post-outcome position deletions were introduced. The results are conditional on complete-payoff availability; an implementable universe still requires source-identified next-month total/delisting outcomes. Fees are illustrative assumptions and corporate-action handling relies on upstream JKP return construction, not an execution tape.

The newest locally supplied `Portfolio_Paper (2).zip` is the manuscript base. `Portfolio_Empirics_Full_Methodology_Rewrite.zip` was not present. Nonempirical archive bytes are preserved; its proofs differ from the repository's theory source, which is recorded as an external source-consistency issue rather than silently edited. There are no simulations or theorem changes in this revision.

Run `python3 -m empirical_final.compact_reproduce` to reproduce with the private frozen data. Run `python3 -m empirical_final.compact_reproduce --public-only` for figure/PDF reconstruction from public aggregates. All numerical claims and figure points have `audit/source_map.json` mappings and `audit/run_manifest.json` hashes. Licensed observations and holdings stay private. See `audit/page_quality.json` and `audit/final_verification.json` for rendered page counts, visual inspection and numerical test results.
'''
    (OUT/'critical_report.md').write_text(text)


if __name__=='__main__':
    setup();reconciliation();bandwidth_intervals();source_map();report()
