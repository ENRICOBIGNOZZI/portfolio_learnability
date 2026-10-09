"""Package aggregate interpretation and local-only stock identifiers/weights."""
import html
import json
import shutil
import pandas as pd
from data_pipeline import digest
from empirical_final.economic_holdings import ROOT, OUT, BASE, PRIVATE, FACTORS


def main():
    # Mean factor accounting uses the exact OOS dates of each existing period.
    fac=pd.read_csv(OUT/'tables/factor_attribution.csv').query("partition=='group'")
    factors=pd.read_csv(BASE/'tables/french_factors.csv',parse_dates=['return_date']).set_index('return_date')
    balances=pd.read_parquet(OUT/'tables/monthly_balances.parquet')
    rows=[]
    for period,d in fac.groupby('period'):
        lo,hi=(1978,2024) if period=='all' else map(int,period.split('-'))
        sub=balances[balances.decision_year.between(lo,hi)]
        dates=sorted(sub.return_date.unique());means=factors.loc[dates,FACTORS].mean()
        for group,dd in d.groupby('group'):
            slope=dd.set_index('factor').coefficient
            mean=12*sub[sub.group.eq(group)].total_payoff.mean()
            alpha=12*slope['alpha_monthly'];explained=12*sum(slope[f]*means[f] for f in FACTORS)
            assert abs(alpha+explained-mean)<1e-10
            for name in FACTORS:
                rows.append(dict(period=period,group=group,factor=name,annual_mean_allocation=12*slope[name]*means[name],
                                 total_group_mean=mean,annual_alpha=alpha,total_factor_mean=explained))
    pd.DataFrame(rows).to_csv(OUT/'tables/factor_mean_accounting.csv',index=False,float_format='%.10g')
    latest=pd.read_parquet(PRIVATE/'year_2024/holdings.parquet')
    latest=latest[latest.formation_date.eq(latest.formation_date.max())]
    latest.to_csv(PRIVATE/'latest_all_stock_holdings.csv',index=False,float_format='%.12g')
    tops=pd.read_csv(PRIVATE/'top_positions_all_months.csv',parse_dates=['formation_date'])
    tops=tops[tops.formation_date.eq(tops.formation_date.max())]
    content=['<!doctype html><meta charset="utf-8"><title>Actual spectral stock holdings</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:6px;text-align:right;border-bottom:1px solid #ddd}h2{margin-top:40px}</style>',
        '<h1>Actual ridge-selected stock holdings</h1><p>Formation: 31 December 2024; payoff: January 2025. Signed weights are percentages of the common policy NAV scale. Stocks are identified by verified CRSP PERMNO; company names are unavailable in this extract. Lists are sorted by position size, not returns.</p>',
        '<p><a href="latest_all_stock_holdings.csv">All stocks, latest month (CSV)</a> | <a href="top_positions_all_months.csv">Top 20 on each side, all 564 months (CSV)</a></p>',
        '<p>Full histories: year_1978 through year_2024 / holdings.parquet. Each file contains every stock and all four component weights. These stock-level files remain local.</p>']
    for group,label in enumerate(['0–10%','10–50%','50–90%','90–100%'],1):
        content.append(f'<h2>Ranks {html.escape(label)}</h2>')
        for side in ['long','short']:
            data=tops[(tops.group==group)&(tops.side==side)][['position_rank','permno','weight_pct']].copy()
            data.columns=['Position rank','CRSP PERMNO','Signed weight (% NAV)']
            content.append(f'<h3>{side.title()}</h3>'+data.to_html(index=False,float_format=lambda x:f'{x:.5f}'))
    (PRIVATE/'holdings_index.html').write_text('\n'.join(content))
    gallery=ROOT/'IMMAGINI PER CHAT/outputs/spectral_economic_content_20261009/figures'
    gallery.mkdir(parents=True,exist_ok=True)
    cards=[]
    for f in sorted((OUT/'figures').glob('EC*')):
        shutil.copy2(f,gallery/f.name)
        if f.suffix=='.png':cards.append(f'<figure><img src="figures/{f.name}" style="width:100%"><figcaption>{f.stem}</figcaption></figure>')
    (gallery.parent/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Economic investment strategies</title><main style="max-width:1100px;margin:auto;font:18px system-ui"><h1>Economic content of spectral groups</h1>'+''.join(cards)+'</main>')
    files={str(p.relative_to(ROOT)):digest(p) for directory in [OUT/'tables',OUT/'figures',OUT/'publication']
           for p in directory.iterdir() if p.suffix in ['.csv','.parquet','.png','.pdf','.tex']}
    files[str((OUT/'REPORT.txt').relative_to(ROOT))]=digest(OUT/'REPORT.txt')
    (OUT/'audit/delivery_manifest.json').write_text(json.dumps(dict(public_artifacts=files,
        private_holdings_directory=str(PRIVATE.relative_to(ROOT)),no_stock_level_weights_in_public_artifacts=True,
        code={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'empirical_final').glob('economic_*.py')}),indent=2))
    print('Packaged figures, local stock holdings viewer, and factor mean accounting')


if __name__=='__main__':main()
