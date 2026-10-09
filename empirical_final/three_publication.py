"""Build only the replacement Section VI; never mutate the reference manuscript."""
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from data_pipeline import digest
from empirical_final.three_experiments import ROOT, OUT


def table(name, caption, label, spec, header, rows, note):
    text = '\\begin{table}[htbp]\n\\centering\\small\\setstretch{1.12}\n'
    text += f'\\caption{{{caption}}}\\label{{{label}}}\n'
    text += '\\begin{tabular}{'+spec+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(row)+' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\par\\vspace{5pt}\n'
    text += '\\begin{minipage}{\\textwidth}\\footnotesize '+note+'\\end{minipage}\n\\end{table}\n'
    (OUT/'publication'/f'table_{name}.tex').write_text(text)


def main():
    read = lambda name: pd.read_csv(OUT/'tables'/f'{name}.csv')
    s = read('e1_selected_summary')
    table('selected', 'Chronologically selected Gaussian policy, reported separately from fixed-penalty paths',
          'tab:selected', 'lrrrrr', r'Decision years & Months & Mean $\C$ & Sharpe & 95\% interval & Loss',
          [[r.period.replace('-', '--'), str(r.months), f'{r.mean_C:.1f}', f'{r.sharpe:.2f}',
            f'[{r.sr_low:.2f}, {r.sr_high:.2f}]', f'{r.raw_loss:.3f}'] for r in s.itertuples()],
          'Each decision year runs from February through the following January. The selected penalty can change across annual decisions. '
          'Complexity is averaged over those refits, and Sharpe is calculated on the concatenated OOS returns. All intervals are pointwise.')
    h = read('e2_summary').query("period == 'all'")
    table('history', 'Common-date history comparison, February 1993--January 2025', 'tab:history', 'rrrrr',
          r'$T$ & Mean $\C_T$ & Median $\widehat\lambda_T$ & OOS Sharpe & Response-one loss',
          [[str(r.T), f'{r.mean_C:.1f}', f'${r.median_lambda/1e-9:.2f}\\times10^{{-9}}$',
            f'{r.sharpe:.3f}', f'{r.loss:.3f}'] for r in h.itertuples()],
          'All specifications have 384 identical evaluation months and a common 20-month validation window. '
          'Paired full-period and subperiod contrasts, interval endpoints, annual penalties, and grid-boundary frequencies accompany the replication files.')
    g = read('e3_group_summary').query("period == 'all'")
    table('spectrum', 'Spectral group attribution over 564 out-of-sample months', 'tab:spectrum', 'lrrrr',
          r'Rank group & Mass (\%) & Mean contribution (\%) & Active $\C$ & Refit cosine',
          [[label, f'{r.mean_spectral_share*100:.3f}', f'{r.contribution_annual_mean*100:.3f}',
            f'{r.mean_active_C:.1f}', f'{r.mean_refit_cosine:.3f}']
           for label, r in zip([r'0--10\%',r'10--50\%',r'50--90\%',r'90--100\%'], g.itertuples())],
          'Mass and active complexity are averages of training quantities. Mean contributions are annualized, use the common frozen scale, '
          'and sum to the full policy mean. Cosines compare adjacent-refit group policy coefficients in the unchanged feature basis, not stock turnover.')
    n = read('e3_nested_summary').query("period == 'all'")
    table('nested', 'Nested ridge attribution and incremental out-of-sample value', 'tab:nested', 'rrrrr',
          r'Included ranks & Sharpe & Loss & Incremental loss & 95\% interval',
          [[f'{r.rank_fraction*100:.0f}\\%', f'{r.sharpe:.3f}', f'{r.loss:.4f}', f'{r.delta_loss:.4f}',
            f'[{r.delta_low:.4f}, {r.delta_high:.4f}]'] for r in n.itertuples()],
          'The first row compares the leading group with a zero-payoff policy. Later rows compare adjacent nested policies, preserving the selected ridge filter. '
          'The final row is the existing full Gaussian policy. Intervals are unadjusted paired block-bootstrap intervals.')
    target = OUT/'publication/Section_VI_Three_Empirical_Experiments.tex'
    shutil.copyfile(ROOT/'empirical_final/templates/three_section_vi.tex', target)
    reference = Path('/Users/enrico/Downloads/Portfolio_Paper (27).pdf')
    before = digest(reference) if reference.exists() else None
    result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', target.name],
                            cwd=target.parent, capture_output=True, text=True, errors='replace')
    (OUT/'audit/latex_build.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:
        raise RuntimeError('LaTeX build failed; see audit/latex_build.log')
    if before and digest(reference) != before:
        raise RuntimeError('Reference manuscript changed')
    audit = dict(reference_path=str(reference), reference_sha256=before,
                 reference_available_at_build=before is not None,
                 reference_unchanged=True if before is not None else None,
                 original_manuscript_written=False,
                 section_only=True, output_sha256=digest(target.with_suffix('.pdf')),
                 template_sha256=digest(ROOT/'empirical_final/templates/three_section_vi.tex'),
                 figure_hashes={p.name:digest(p) for p in sorted((OUT/'figures').glob('*.pdf'))})
    (OUT/'audit/publication.json').write_text(json.dumps(audit, indent=2))
    print(target.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
