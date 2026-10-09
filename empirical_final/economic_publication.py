"""Build the standalone Section VI from unchanged A/B and holdings-based C."""
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from data_pipeline import digest
from empirical_final.economic_holdings import ROOT, OUT, BASE

GROUPS=[r'0--10\%',r'10--50\%',r'50--90\%',r'90--100\%']


def tex(s):
    return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')


def table(name,caption,label,spec,header,rows,note):
    text=r'\begin{table}[p]\centering\small\setstretch{1.12}'+'\n'
    text+=f'\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n'
    text+=header+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+r' \\' for row in rows)
    text+='\n'+r'\bottomrule\end{tabular}\par\vspace{5pt}'+'\n'
    text+=r'\begin{minipage}{\textwidth}\footnotesize '+note+r'\end{minipage}\end{table}'+'\n'
    (OUT/'publication'/f'table_{name}.tex').write_text(text)


def main():
    dictionary=pd.read_csv(OUT/'tables/characteristic_dictionary.csv').set_index('characteristic')
    d=pd.read_csv(OUT/'tables/dominant_characteristics.csv')
    d=d.merge(dictionary[['name_new']],left_on='characteristic',right_index=True,validate='many_to_one')
    d['long_input_percentile']=50+100*d.long_mean
    d['short_input_percentile']=50+100*d.short_mean
    d['difference_pp']=100*d.long_minus_short
    d.to_csv(OUT/'tables/dominant_long_short_exposures_named.csv',index=False,float_format='%.10g')
    rows=[]
    for group in range(1,5):
        for side in ['positive tilt','negative tilt']:
            for r in d[(d.period=='all')&(d.group==group)&(d.tilt==side)&(d['rank']<=2)].itertuples():
                label='Piotroski F-score' if r.characteristic=='f_score' else r.name_new
                rows.append([GROUPS[group-1],tex(label),f'{r.long_input_percentile:.1f}',
                    f'{r.short_input_percentile:.1f}',f'{r.difference_pp:+.1f}'])
    table('dominant','Dominant long and short characteristic exposures in actual holdings','tab:dominant',
        r'lp{3.05in}rrr',r'Ranks & Characteristic & Long & Short & L--S',rows,
        'Two largest positive and two largest negative differences per group, ranked by average holding exposure, not realized return. '
        'Long and short columns show holding-weighted input percentile scores (neutral missing inputs equal 50); L--S is their difference in percentile points. '
        'The companion named table reports six exposures on each side for every historical period. Stocks can share multiple characteristics; this is not a disjoint strategy decomposition.')
    g=pd.read_csv(BASE/'tables/e3_group_summary.csv').query("period=='all'").set_index('group')
    b=pd.read_csv(OUT/'tables/group_holdings_summary.csv').query("period=='all'").set_index('group')
    n=pd.read_csv(OUT/'tables/stock_netting_summary.csv').query("period=='all'").set_index('group')
    rows=[]
    for j in range(1,5):
        rows.append([GROUPS[j-1],f'{100*g.loc[j,"mean_spectral_share"]:.4f}',f'{100*b.loc[j,"mean_gross"]:.2f}',
            f'{100*n.loc[j,"cumulative_gross"]:.2f}',f'{100*b.loc[j,"annual_mean"]:.3f}',f'{100*b.loc[j,"annual_volatility"]:.3f}'])
    table('financial','Spectral mass, actual investment scale, and subsequent payoff','tab:financial','lrrrrr',
        r'Ranks & Mass & Gross & Netted gross & Mean & Volatility',rows,
        'All entries are percentages. Mass is the average training second-moment share. Gross refers to the individual group; netted gross sums stock weights across all groups through the stated rank boundary before taking absolute values. '
        'Mean and volatility are annualized excess-payoff contributions of the individual group, not the cumulative portfolio. No rescaling is introduced.')
    for name in ['selected','history']:
        shutil.copyfile(BASE/'publication'/f'table_{name}.tex',OUT/'publication'/f'table_{name}.tex')
    source=ROOT/'empirical_final/templates/three_section_vi.tex'
    original=source.read_text()
    prefix=original.split(r'\subsection{Economic Content of the Managed-Payoff Spectrum}')[0]
    prefix=prefix.replace('The third examines the expected payoffs, conventional risk exposures, and incremental portfolio value of different parts of the managed-payoff spectrum.',
        'The third reconstructs actual stock holdings to identify the investment strategies, risk exposures, and incremental portfolio value represented by different parts of the managed-payoff spectrum.')
    start=prefix.index('The retained panel has an important limitation.')
    end=prefix.index('\n\nThe Gaussian map',start)
    prefix=prefix[:start]+('We accept the retrospectively constructed JKP characteristic library as the academic information set for this analysis. '
        'All results remain conditional on the existing retained, complete-payoff panel. The economic interpretation preserves that universe and its preprocessing; it does not introduce a new information set.')+prefix[end:]
    prefix=prefix.replace(r'{../figures/#1.pdf}',r'{#1.pdf}')
    prefix=prefix.replace(r'\begin{document}',r'\graphicspath{{../figures/}{../../empirical_three_experiments_20261009/figures/}}'+'\n'+r'\begin{document}')
    section_c=ROOT/'empirical_final/templates/economic_section_c.tex'
    target=OUT/'publication/Section_VI_Economic_Investment_Strategies.tex'
    target.write_text(prefix+section_c.read_text()+'\n\\end{document}\n')
    ref=Path('/Users/enrico/Downloads/Portfolio_Paper (27).pdf');before=digest(ref) if ref.exists() else None
    built=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',target.name],cwd=target.parent,capture_output=True,text=True,errors='replace')
    (OUT/'audit/latex_build.log').write_text(built.stdout+'\n'+built.stderr)
    if built.returncode:raise RuntimeError('Section VI build failed; inspect audit/latex_build.log')
    if before and digest(ref)!=before:raise RuntimeError('Reference manuscript was modified')
    a=dict(section_only=True,original_manuscript_written=False,reference_available_at_build=before is not None,
        reference_sha256=before,compatibility='Existing reference-derived 12pt newtx, 1.5 spacing, letter paper, 1-inch margins; Section VI begins on page 28. A/B reused unchanged.',
        source_prefix_sha256=digest(source),section_c_sha256=digest(section_c),output_sha256=digest(target.with_suffix('.pdf')),
        figure_hashes={p.name:digest(p) for p in (OUT/'figures').glob('*.pdf')})
    (OUT/'audit/publication.json').write_text(json.dumps(a,indent=2))
    print(target.with_suffix('.pdf'))


if __name__=='__main__':main()
