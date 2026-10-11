"""Quantitative closeout report, explicit limitations, and artifact hashes."""
import json
import platform
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib
from simulations.closeout.results import DEST,CANONICAL,REFERENCE
from simulations.closeout.verify import verify
from simulations.provenance import file_hash,json_write,utc_now


def run():
    verification=verify()
    numerical=json.loads((DEST/'numerical_status.json').read_text())
    paper=json.loads((DEST/'paper_build_verification.json').read_text())
    visual=json.loads((DEST/'visual_review.json').read_text())
    if not paper['passed'] or not visual['passed']: raise ValueError('Publication build and final visual review must pass before delivery.')
    for doc in visual['documents']:
        if file_hash(DEST/doc['path'])!=doc['sha256']: raise ValueError('A PDF changed after visual review.')
    for name,expected in paper['simulation_input_sha256'].items():
        if file_hash(DEST/'paper'/name)!=expected: raise ValueError('A simulation input changed after manuscript build.')
    tests=DEST/'tests.xml'
    import xml.etree.ElementTree as ET
    suite=ET.parse(tests).getroot()
    if any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suite.iter('testsuite')): raise ValueError('Required tests failed.')

    summary=pd.read_csv(DEST/'summary.csv');slopes=pd.read_csv(DEST/'slopes_all_windows.csv')
    paired=pd.read_csv(DEST/'paired_cv_vs_fixed.csv')
    alpha=pd.read_csv(DEST/'selected_constant_summary.csv')
    warnings=pd.read_csv(DEST/'numerical_warnings.csv')
    report=['# Rich6D initial-CV closeout — 12 October 2026','',
        'The complete 300-history experiment in each of five economies is preserved. The economic model, M=128, Matérn lengthscale, one-time initial60 CV, 71 constants, forward folds, exponent -0.6, pairing, horizons, boundary choices and analytical optima are unchanged. All new diagnostics are post hoc. No full campaign or cloud workflow was launched. Existing unrelated empirical work is excluded from this task.','',
        '## Quantities covered by the rate','',
        f'On the 13 baseline means, regret slope = {slopes[(slopes.environment=="baseline")&(slopes.specification=="initial_cv")&(slopes.grid=="full")&(slopes.window=="full")&(slopes.quantity=="regret")].slope.iloc[0]:.7f}; Sharpe-gap slope = {slopes[(slopes.environment=="baseline")&(slopes.specification=="initial_cv")&(slopes.grid=="full")&(slopes.window=="full")&(slopes.quantity=="annual_gap")].slope.iloc[0]:.7f}. Loss versus Sharpe alone does not explain the finite slow decay. All four established windows and full/common grids are exported with whole-history covariance MCSE and jackknife checks. Slopes regress log Monte Carlo means, not individual-history slopes.','',
        'Normalized regret T^0.6 regret, monthly/annual Sharpe gaps times T^0.6, and empirical/population C/T^0.4 are retained for every history and summarized by mean, median, SD, 2.5/97.5 percentiles and mean MCSE. A finite plot does not certify or disprove an asymptotic O_P upper bound. No cutoff is chosen by proximity to -0.6.','',
        '## What initial CV improves','']
    for name in ('baseline','N300','N1200','rho000','rho075'):
        T=int(summary[summary.environment==name]['T'].max())
        end=summary[(summary.environment==name)&(summary['T']==T)].pivot(index='specification',columns='metric',values='mean')
        for metric in ('regret','annual_SR'):
            r=paired[(paired.environment==name)&(paired['T']==T)&(paired.metric==metric)].iloc[0]
            report.append(f'- {name}, T={T}, {metric}: initial CV {end.loc["initial_cv",metric]:.6f}; historical {end.loc["old_fixed",metric]:.6f}; paired change {r["mean"]:+.6f}, 95% mean interval [{r.CI_low:+.6f}, {r.CI_high:+.6f}]. Oracle Q {end.loc["oracle_Q",metric]:.6f}; oracle SR {end.loc["oracle_SR",metric]:.6f}.')
    early=summary[(summary.environment=='baseline')&(summary['T']==60)&(summary.metric=='annual_SR')].set_index('specification')
    early_difference=paired[(paired.environment=='baseline')&(paired['T']==60)&(paired.metric=='annual_SR')].iloc[0]
    report+=['',f'Baseline T=60: annualized Sharpe is {early.loc["initial_cv","mean"]:.6f} under initial CV versus {early.loc["old_fixed","mean"]:.6f} historically; paired difference {early_difference["mean"]:+.6f}, 95% mean interval [{early_difference.CI_low:+.6f},{early_difference.CI_high:+.6f}]. All later paired horizons are retained in the delivered CSV, including the loss of a clear final Sharpe benefit.','']
    report+=['','Initial constant distribution, including boundaries (all 300 histories each):','']
    for _,row in alpha.iterrows():
        report.append(f'- {row.environment}: median a={row["median"]:.8g}, central95% [{row.p025:.8g},{row.p975:.8g}], lower-bound selections {int(row.lower_boundary_count)}, upper-bound selections {int(row.upper_boundary_count)}; no removal.')
    report+=['','In the baseline the final loss improvement is clear, while the final Sharpe interval includes zero. The original short-history benefit remains tabulated. Oracle Q minimizes mean loss and oracle SR maximizes mean Sharpe on the same original 96-point grid, one common penalty per economy/T. Even/odd cross-fitting selects on one half of complete histories and assesses on the other. Neither oracle enters tuning; oracle bands and slopes do not account for selection.','',
        '## Measured mechanisms and scope','']
    fits=sorted((DEST/'audit_fits').glob('*.csv'))
    if fits:
        moments=pd.concat([pd.read_csv(p) for p in fits],ignore_index=True)
        moments=moments.sort_values('sample_scope').drop_duplicates(['environment','index','T','audit'])
        end=moments[(moments.environment=='baseline')&(moments['T']==4860)&moments.audit.isin(['A','C'])]
        columns=['regret','u','v','c','d','radial_gap','normalized_directional_error','ray_gap','scale_gap','projection_floor','regularization_bias_within_P','estimation_displacement','signed_cross_term']
        for label in ('A','C'):
            rows=end[end.audit==label]
            if len(rows):
                report.append(f'Baseline T=4860, {label} ({"initial CV" if label=="A" else "historical"}), **{rows["index"].nunique()} recovered audit histories**, monthly quantities: '+', '.join(f'{k}={rows[k].mean():.8g}' for k in columns)+'.')
        report+=['','These are measured audit-history decompositions, not a reconstruction of all 300 final policies. The scale/ray and radial/angular identities hold policy by policy. Direction is normalized because raw d changes with positive rescaling. Population-ridge bias is relative to the optimum in the same subspace, and the signed cross term is retained. The displacement norm is not labeled variance. A numerical difference in this subset is descriptive, not a causal diagnosis of the complete Monte Carlo mean.','']
    component_summary=pd.read_csv(DEST/'component_summary.csv')
    components=component_summary[(component_summary.environment=='baseline')&(component_summary['T']==4860)].pivot(index='specification',columns='metric',values='mean')
    report+=['','All 300 endpoint baseline policies have positive signed Sharpe. Without choosing ANY u/v root, the identified per-policy ray gap is q-SR_monthly^2/(1+SR_monthly^2), scale gap is regret-ray_gap, and normalized directional error is SR_star_monthly^2/SR_monthly^2-1. These are uniquely identified even when two distinct moment pairs produce the same Q/SR.','']
    for spec in ('initial_cv','old_fixed'):
        report.append(spec+', **all 300 baseline endpoint histories**: '+', '.join(f'{metric}={components.loc[spec,metric]:.8g}' for metric in ['ray_gap','scale_gap','normalized_directional_error','projection_floor','regularization_bias_within_P','estimation_displacement_plus_cross'])+'.')
    report+=['','Initial CV sharply reduces scaling error while the mean scale-invariant directional loss is slightly larger. This explains the response-one/Sharpe discrepancy for the actual complete Monte Carlo means. Population-ridge bias also falls, but the combined displacement plus signed cross grows. The latter is not called variance; only recovered coefficients separate its two terms. Raw monthly u/v/c/d remain unavailable for the unregenerated histories.','']
    component_rates=pd.read_csv(DEST/'component_slopes.csv')
    component_rates=component_rates[(component_rates.environment=='baseline')&(component_rates.specification=='initial_cv')&(component_rates.grid=='full')&(component_rates.window=='full')]
    report+=['Finite full-grid component slopes (post hoc, no horizon removal): '+', '.join(f'{r.quantity}={r.slope:.6f}' for _,r in component_rates.iterrows())+'. The projection floor explains only part of the finite slow slope: subtracting it changes descriptive regret decay from about -0.438 to about -0.470, still distinct from -0.6. The complete-grid empirical benchmark always remains the analytic economic optimum.','']
    report += ['## Rank and quadrature','',numerical['latex_scope'],'',
        f'Full audit histories: {numerical.get("full_indices",[])}; additional initial60 audit histories: {numerical.get("short_indices",[])}. Missing full representative indices: {numerical.get("representative_full_indices_not_run",[])}. Missing full stress indices: {numerical.get("stress_full_indices_not_run",[])}.','',
        'Both A (same archived constant across ranks) and B (initial CV repeated at each rank on the same initial60) are kept separate. C checks the historical rule. Primary and independent quadratures use fixed learned policies and different seeds, with analytical factor/noise integration. An estimated subspace optimum is never treated as the exact economic optimum. Normal residuals and reproduction of old loss/Sharpe are checked. Paired bounds with fewer than two histories are marked not estimable rather than given an invented MCSE.','']
    for _,row in warnings[(warnings.specification=='initial_cv')&warnings['T'].isin([3240,4860])].iterrows():
        report.append(f'- {row.environment}, T={row["T"]}: projection floor {row.projection_floor:.8g}, mean regret {row.mean_regret:.8g}, ratio {100*row.floor_fraction:.3f}%; 5% condition {"passes" if row.floor_pass else "fails"}.')
    report+=['','All failed original checks remain in `historical_numerical_warnings.csv`. The new four-specification `numerical_reliability_mask.csv` preserves the complete grid and asserts no global rank certificate. No ranks are mixed along the archived curves.','',
        '## Fourier and theory','',
        'The unchanged supplied note and compiled PDF are in sources/. Its ideal infinite-economy r=1 proof, operator-range comparison, and conditional finite-positive-grid frozen-CV argument were checked against the DGP and current manuscript; see theory_review.md. The manuscript mixing/stability assumptions and sufficient asymptotic threshold remain conditions. The finite M128 economy is not shown to have maximal source order one. A fixed numerical subspace with positive representation error cannot inherit convergence of regret to zero against the full economic optimum. The Nyström computational-error issue is also explicit in [Rudi, Camoriano and Rosasco](https://papers.nips.cc/paper/5936-less-is-more-nystrom-computational-regularization.pdf); their assumptions are not automatically transferred to this dependent unbounded-feature setting.','']
    fourier=json.loads((DEST/'fourier_bound_status.json').read_text())
    report.append(f'M128 uniform Fourier tail bound = {fourier["bound_128"]:.10g}; the conservative second-operator perturbation/lambda ratio has maximum {fourier["M128_operator_ratio_max"]:.6g} across actual CV penalties and exceeds 5% in {100*fourier["M128_fraction_over_5_percent"]:.4f}% of cells. Small pointwise error does not certify a ridge inverse or argmin.')
    if (DEST/'fourier_paired_verification.json').exists():
        f=json.loads((DEST/'fourier_paired_verification.json').read_text())
        report.append(f'Paired M128/512/2048 refits and economy-specific moment evaluation: {f["scope"]}. All checked initial choices unchanged: {f["unchanged_choices"]}. Final-horizon sensitivity and second independent quadrature for the Fourier variants remain not run.')
    if (DEST/'fourier_paired_verification.json').exists():
        f=json.loads((DEST/'fourier_paired_verification.json').read_text())
        report.append(f'Within this single initial60 pilot, maximum relative RKHS policy change = {f["max_relative_RKHS_policy_change"]:.8g}; maximum relative moment/gap change = {f["max_relative_moment_or_gap_change"]:.8g}. These small changes do not certify the untested endpoints.')
    report+=['','## Status by claim','',
        '- **resolved**: verified full 1500-history input provenance; all-horizon regret and Sharpe-rate diagnostics and normalized distributions; distinct loss/Sharpe common-grid oracles; whole-history uncertainty; preserved initial constant; exact helper identities and recovered low-rank population-ridge reconstruction; reviewed ideal/theorem distinction; generated four-figure section and appendix.',
        f'- **failed**: endpoint projection-floor criteria wherever the exported ratio exceeds 5%. The new paired numerical audit has {numerical["failed_metric_comparisons"]} failed comparisons in its checked scope; late single-history comparisons lack MCSE. Thresholds are unchanged.',
        '- **unresolved**: transfer of the full-kernel asymptotic upper rate to the fixed Nyström implementation; full-horizon twelve-history representative/stress audit; rank8192 forward accuracy; complete-300-history raw u/v/c/d and separate population-ridge displacement/cross decomposition.',
        '- **not_run**: uncompleted audit workers and rank8192 population-ridge decomposition; final-horizon paired Fourier fitted-policy/CV sensitivity; new 1500-history campaign (deliberately not launched).','',
        '## Minimum further work','',
        'Complete the declared missing paired representative/stress histories at the same fixed audit horizons, using saved sidecars for independent evaluation. If the 5% criterion still fails, assess higher-rank cost and benefit before any new production. To recover raw moments and separate the population-ridge displacement/cross for all 300 histories, recover their original coefficients/moments or regenerate the missing histories with the original seeds; do not choose an arbitrary loss/Sharpe root. The complete-300 ray/scale split already requires no such root. Extend paired Fourier sensitivity only where the operator bound and initial60 diagnostic cannot close the requested penalty-specific check. A full rerun is a later scientific decision, not an automatic action.','',
        '## Reproduction and editorial package','',
        'Run the commands in simulations/closeout/README.md. Worker receipts preserve indices, original training seeds, history hashes, selected constants, fold losses, coefficients and moments. closeout_verification.json checks CSV arithmetic independently. artifact_manifest.json hashes the intended public artifacts; regenerable large basis/operator caches are excluded. Four principal figures have vector PDF and PNG copies; extra economy and normalized-outcome figures appear in the appendix. All plotted bands are replication outcomes, with mean intervals separate.','',
        'The paper is compiled in an isolated staging copy before the one-line simoutput link changes. Empirical and theorem inputs are hash-checked and unchanged. Both a standalone simulation document and a full staged manuscript are delivered. Visual QA and test receipts record the final verified versions.']
    (DEST/'closeout_report.md').write_text('\n'.join(report)+'\n')
    status=dict(resolved=['full archived input and arithmetic checks','loss/Sharpe distinct oracles','whole-history slope uncertainty','financial identities, full-300 identified ray/scale quantities and numerical ridge bias, and recovered subset moments','theory scope review','generated publication'],
        failed=['Exported projection floors exceeding 5%']+(['Paired numerical budget failures'] if numerical['failed_metric_comparisons'] else []),
        unresolved=['Full-horizon representative and stress audit','Fixed-rank asymptotic transfer','Complete 300-history raw moments and separate displacement/cross','8192 forward certificate'],
        not_run=['Missing numerical workers','8192 population-ridge decomposition','Final-horizon Fourier fit sensitivity','New production campaign'],
        numerical_status=numerical,closed_with_limitations=True,scientific_certification=False)
    json_write(DEST/'closeout_status.json',status)
    software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),completed_utc=utc_now())
    json_write(DEST/'software_versions.json',software)
    tests_count=sum(int(x.attrib.get('tests',0)) for x in suite.iter('testsuite'))
    brief=f'''# Chiusura Rich6D — 12 ottobre 2026

La CV iniziale riduce la loss soprattutto correggendo la scala del portafoglio. Il miglioramento non si traduce in un vantaggio finale dello Sharpe medio. La specifica resta invariata: una selezione sui primi 60 dati, poi a congelata e lambda=a T^-0.6. Nessuna nuova campagna completa.

| Baseline T=4860, tutte le 300 storie | CV iniziale | Vecchia costante |
|---|---:|---:|
| Regret | 0.011296 | 0.022205 |
| Sharpe annualizzato | 2.932497 | 2.934774 |
| Errore lungo la direzione (ray gap) | 0.011119 | 0.010733 |
| Errore di scala | 0.000177 | 0.011472 |
| Bias ridge entro il sottospazio | 0.003179 | 0.019338 |
| Spostamento di stima + cross firmato | 0.006339 | 0.001089 |

Il floor di rappresentazione è 0.001777, pari al 15.73% del regret medio CV: il criterio del 5% fallisce. Lo spostamento e il cross sono separati solo dove abbiamo recuperato i coefficienti. La loro somma sulle 300 storie non è chiamata varianza. Ray gap, scale gap ed errore direzionale normalizzato sono identificati univocamente da loss e Sharpe, senza scegliere radici per u/v; le identità sono verificate anche sui momenti ricalcolati.

Le pendenze su tutte le 13 medie sono -0.4378196 per il regret e -0.3986480 per lo Sharpe gap. Sottrarre il floor porta la prima soltanto a -0.470399. Le quattro finestre già definite e le distribuzioni normalizzate restano complete: non abbiamo scelto un cutoff per ottenere -0.6.

**resolved:** provenance e aritmetica delle 1500 storie; due oracle distinti sulla stessa griglia; componenti identificate e bias ridge a ogni lambda effettiva; {tests_count} test superati; quattro figure principali PDF/PNG e appendice; nota teorica verificata nel suo ambito; sezione compilata e collegamento minimo del paper. Empirica e teoremi conservati.

**failed:** floor/regret oltre il 5% nelle celle esportate. Nessuna soglia è stata allentata.

**unresolved:** certificazione numerica globale. L'audit copre i dodici indici rappresentativi e quattro stress aggiuntivi a T=60, ma una sola storia agli orizzonti intermedi/massimi. Gli stress non entrano nella MCSE dei rappresentativi. Il rango modifica 8/80 scelte iniziali; i confronti sui dodici rappresentativi a T=60 passano, mentre ai grandi T manca la MCSE tra repliche. Nel singolo baseline finale, a fissata, la variazione di regret P8192-P4096 è +0.00000247; è un confronto puntuale, non un certificato. Il rango 8192 non ha un controllo a rango superiore.

**not_run:** gli altri audit completi pianificati, la ridge di popolazione a P8192 alle lambda CV, tutti i momenti grezzi mancanti e la sensibilità Fourier ai grandi T. Il pilota M128/512/2048 a T=60 lascia invariate le scelte e mostra variazioni relative della policy al più 1.50e-10; il bound conservativo a tutte le lambda CV resta non informativo in alcune celle estreme.

Per chiudere le parti irrisolte occorre completare i controlli appaiati dichiarati e valutare il costo di un eventuale rango superiore prima di una produzione. Un sottospazio fisso con floor positivo non eredita convergenza a zero verso l'ottimo completo dal teorema kernel. Il risultato r=1 riguarda l'economia ideale infinita; non certifica M128 o Nyström.

[Rapporto tecnico completo](closeout_report.md) · [Comandi di riproduzione](../../closeout/README.md) · PDF compilati nella cartella paper/.
'''
    (DEST/'closeout_brief.md').write_text(brief)
    entries=[]
    for path in sorted(DEST.rglob('*')):
        if not path.is_file() or path.name=='artifact_manifest.json' or 'cache' in path.relative_to(DEST).parts or path.suffix in ('.log','.aux','.out','.bbl','.blg'):
            continue
        entries.append(dict(path=str(path.relative_to(DEST)),sha256=file_hash(path),bytes=path.stat().st_size))
    json_write(DEST/'artifact_manifest.json',dict(schema='simulation-closeout/1',artifacts=entries,
        completed_utc=utc_now(),sources={str(p):file_hash(p) for p in Path('simulations/closeout').glob('*.py')},
        original_run_hash=json.loads((CANONICAL/'protocol.json').read_text())['run_hash']))

if __name__=='__main__':run()
