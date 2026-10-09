"""Produce the predeclared 1994-1995 slice and verified gross-only legacy comparison."""
import argparse
import json
from pathlib import Path
import pandas as pd
from adaptive_memory_equities.config import PACKAGE, REPO, digest, write_json
from adaptive_memory_equities.metrics import summary, paired_comparisons
from adaptive_memory_equities.accounting import wealth


def finalize(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['status']!='REAL_RUN_COMPLETED':
        raise ValueError('Only completed real executions may be summarized.')
    monthly=pd.read_csv(directory/'report/monthly.csv')
    selected=monthly.loc[monthly.return_date.between('1994-01-31','1995-12-31')]
    wide=selected.loc[selected.method.str.startswith('joint/')].pivot(index='return_date',columns='method',values='raw_excess')
    expected=pd.date_range('1994-01-31','1995-12-31',freq='ME').strftime('%Y-%m-%d').tolist()
    if list(wide.index)!=expected:
        raise ValueError('Missing a predeclared common-slice month.')
    out=directory/'report'
    write_json(out/'reduced_common_provenance.json',dict(
        status='RETROSPECTIVE_ONLY',manifest_hash=digest(directory/'manifest.json'),
        monthly_hash=digest(out/'monthly.csv'),report_tool_hash=digest(Path(__file__)),
        metrics_code_hash=digest(PACKAGE/'metrics.py'),dates=expected,
        warning='Only 24 monthly observations and two 12-month blocks; intervals have low resolution.'))
    table=pd.DataFrame([dict(method=k,**summary(wide[k])) for k in wide])
    table.to_csv(out/'reduced_common_comparison.csv',index=False)
    pd.DataFrame(paired_comparisons(wide,'joint/full')).to_csv(out/'reduced_common_paired.csv',index=False)
    all_arch=selected.groupby('method',sort=True)
    pd.DataFrame([dict(method=k,**summary(v.raw_excess)) for k,v in all_arch]).to_csv(out/'reduced_common_by_architecture.csv',index=False)
    costs, accounting = [], []
    for k,v in selected.groupby('method',sort=True):
        for col in ['0bps','10bps','25bps','borrow_sensitivity']:
            costs.append(dict(method=k,sensitivity=col,**summary(v[col])))
        values,drawdowns,insolvent=wealth(v.raw_excess,v.risk_free)
        accounting.append(dict(method=k,gross=float(v.gross.mean()),net=float(v.net.mean()),
            mean_turnover=float(v.turnover.mean()),maximum_weight=float(v.maximum_absolute_weight.max()),
            insolvent=insolvent is not None,
            max_drawdown=float(pd.Series(drawdowns).min()),
            final_wealth=float(values[-1]) if pd.notna(values[-1]) else None))
    pd.DataFrame(costs).to_csv(out/'reduced_common_costs.csv',index=False)
    pd.DataFrame(accounting).to_csv(out/'reduced_common_accounting.csv',index=False)
    fixed_deltas, fixed_paired = [], []
    for scope in sorted({k.split('/')[0] for k in selected.method if not k.startswith('joint/')}):
        part=selected.loc[selected.method.str.startswith(scope+'/')].pivot(index='return_date',columns='method',values='raw_excess')
        fixed_paired.extend(paired_comparisons(part,scope+'/full'))
        base=summary(part[scope+'/full'])
        for k in part:
            if k.endswith('/unguarded'):
                item=summary(part[k])
                fixed_deltas.append(dict(method=k,loss_minus_full=item['response_one']-base['response_one'],
                                         sharpe_minus_full=item['sharpe']-base['sharpe']))
    pd.DataFrame(fixed_deltas).to_csv(out/'fixed_architecture_gains_and_losses.csv',index=False)
    pd.DataFrame(fixed_paired).to_csv(out/'fixed_architecture_paired.csv',index=False)
    choices=[]
    for file in sorted((directory/'events').glob('*.json')):
        event=json.loads(file.read_text())
        if event['return_date'] in expected:
            for method,allocation in event['selections'].items():
                if method.startswith('joint/'):
                    choices.extend(dict(return_date=event['return_date'],method=method,
                        expert=expert,architecture=expert.split('|')[0],allocation=value)
                        for expert,value in allocation.items())
    pd.DataFrame(choices).to_csv(out/'joint_architecture_choices.csv',index=False)
    legacy=[]
    provenance=[]
    for kernel,width in [('linear',131),('gaussian',10000),('matern32',10000)]:
        root=REPO/'results/final/public'/kernel/'seed_0'
        source=json.loads((root/'source.json').read_text())
        old=json.loads((root/f'p_{width}/manifest.json').read_text())
        path=root/f'p_{width}/monthly.csv'
        if source['clean_manifest_sha256']!=digest(REPO/'data/clean/manifest.json'):
            raise ValueError('Legacy sample manifest differs.')
        if digest(path)!=old['files']['monthly.csv']:
            raise ValueError('Legacy monthly data checksum differs.')
        if source['refit_timing']!='after first formation month close; labels known then':
            raise ValueError('Legacy refit calendar differs.')
        d=pd.read_csv(path)
        d=d.loc[d.return_date.between('1994-01-31','1995-12-31')].sort_values('return_date')
        if d.return_date.tolist()!=expected:
            raise ValueError('Legacy payoff months differ.')
        if not ((pd.to_datetime(d.formation_date)+pd.offsets.MonthEnd(1))==pd.to_datetime(d.return_date)).all():
            raise ValueError('Legacy forward labels differ.')
        k='legacy/'+kernel
        wide[k]=d.raw_excess_return.to_numpy()
        legacy.append(dict(method=k,**summary(d.raw_excess_return),reporting_scale=1.,cost_bps=0,
                           fit_status='PREEXISTING_VERIFIED_OUTPUT',interpretation='RETROSPECTIVE_ONLY'))
        provenance.append(dict(kernel=kernel,source_path=str(root/'source.json'),source_hash=digest(root/'source.json'),
            monthly_path=str(path),monthly_hash=digest(path),clean_manifest_hash=source['clean_manifest_sha256'],
            raw_scale_used=True,legacy_kappa_not_applied=True,cost_bps=0,
            legacy_net_cost_comparison='NOT_RUN',new_fits=0))
    pd.DataFrame(legacy).to_csv(out/'legacy_gross_comparison.csv',index=False)
    pd.DataFrame(paired_comparisons(wide,'joint/full')).to_csv(out/'paired_with_verified_legacy_gross.csv',index=False)
    write_json(out/'legacy_gross_provenance.json',dict(status='VERIFIED_GROSS_ONLY',records=provenance,
        report_tool_hash=digest(Path(__file__)),metrics_code_hash=digest(PACKAGE/'metrics.py'),
        reason='Identical clean snapshot, realization months, excess-payoff definition, January-close timing, scale 1 and zero costs. Net-cost comparisons are not imported.'))
    baseline=table.loc[table.method=='joint/full'].iloc[0]
    losses=[]
    for r in table.itertuples():
        if 'unguarded' in r.method:
            losses.append(dict(method=r.method,loss_minus_full=r.response_one-baseline.response_one,
                               sharpe_minus_full=r.sharpe-baseline.sharpe))
    write_json(out/'adaptive_gains_and_losses.json',losses)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotted=pd.concat([table.loc[table.method.eq('joint/full')|table.method.str.endswith('/unguarded')],
                       pd.DataFrame(legacy)],ignore_index=True)
    labels=plotted.method.str.replace('joint/','',regex=False).str.replace('/unguarded','',regex=False)
    colors=['#236b8e' if not k.startswith('legacy/') else '#74777b' for k in plotted.method]
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    axes[0].barh(labels,plotted.sharpe,color=colors)
    axes[0].set_xlabel('Annualized Sharpe, raw scale 1')
    axes[1].barh(labels,plotted.response_one,color=colors)
    axes[1].set_xlabel('Out-of-sample response-one loss')
    fig.suptitle('1994–1995 | Joint selection including affine control | RETROSPECTIVE')
    fig.supxlabel('Joint memory choices can change architecture; guarded methods equal full history (<60 past scores).\n'
                  'Legacy: verified saved seed-0 outputs; NN: average of seed-0 and seed-1 holdings.',fontsize=9)
    fig.savefig(out/'reduced_common_comparison.png',dpi=150)
    plt.close(fig)
    fixed=pd.DataFrame([dict(method=k,**summary(v.raw_excess)) for k,v in selected.groupby('method')
        if k.startswith('L2W32/') and (k.endswith('/full') or k.endswith('/unguarded'))])
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    labels=fixed.method.str.replace('L2W32/','',regex=False).str.replace('/unguarded','',regex=False)
    axes[0].barh(labels,fixed.sharpe,color='#236b8e')
    axes[0].set_xlabel('Annualized Sharpe')
    axes[1].barh(labels,fixed.response_one,color='#ac6b39')
    axes[1].set_xlabel('Response-one loss')
    fig.suptitle('Fixed NN: 2 hidden layers × 32 units | 1994–1995 | RETROSPECTIVE')
    fig.supxlabel('Same architecture, seed ensemble and optimizer budget. Unguarded memory choices; 24 monthly payoffs.',fontsize=9)
    fig.savefig(out/'fixed_L2W32_comparison.png',dpi=150)
    plt.close(fig)
    lines=['# Verified reduced common-calendar comparison','',
        '**RETROSPECTIVE_ONLY. January 1994–December 1995, 24 monthly payoffs.**','',
        'All neural comparisons use the same two seed portfolios averaged in holdings, raw scale 1 and zero trading costs.',
        'Guarded methods equal full history because the run has fewer than 60 mature prequential observations.',
        'Legacy rows use previously saved seed-0 policies whose data, calendar, return definition and scale were verified.',
        'Legacy policies were not refitted. This comparison does not reproduce the full 1994–2024 period.','',
        'Joint memory selection often chooses the affine control, while its warmup full-history baseline is fixed L2W32.',
        'Consequently joint improvements cannot be attributed to temporal memory alone. The fixed-architecture table isolates that comparison.','',
        '## Fixed NN: two hidden layers of width 32','',
        '| Method | Sharpe | Response-one |','|---|---:|---:|']
    lines += [f'| {r.method} | {r.sharpe:.4f} | {r.response_one:.6f} |' for r in fixed.itertuples()]
    lines += ['', '## Joint selection and legacy gross references','',
        '| Method | Sharpe | Response-one |','|---|---:|---:|']
    lines += [f'| {r.method} | {r.sharpe:.4f} | {r.response_one:.6f} |' for r in plotted.itertuples()]
    lines += ['', 'Memory rows above are unguarded; every guarded and soft result is in reduced_common_comparison.csv.',
        'All fixed-architecture results are in reduced_common_by_architecture.csv.',
        f'Adaptive loss is worse than its matching uniform baseline in {sum(r["loss_minus_full"]>0 for r in fixed_deltas)} of {len(fixed_deltas)} fixed-architecture/family comparisons.',
        'Fixed-architecture paired bands adjust within each scope, not jointly over all nine architecture scopes.',
        'Paired block-bootstrap intervals use common circular blocks of length 12 and 2,000 replicates.',
        'Only two blocks fit into this slice: intervals are approximate, low-resolution retrospective diagnostics.',
        'Cost sensitivities and exposure/turnover/drawdown diagnostics are saved on the same 24 dates.',
        'No net-cost legacy comparison is asserted. Existing-position turnover is inherited from the saved startup history.',
        '', '![Fixed-architecture comparison](fixed_L2W32_comparison.png)',
        '', '![Joint common-period comparison](reduced_common_comparison.png)']
    (out/'reduced_common_report.md').write_text('\n'.join(lines)+'\n')
    original_report=out/'report.md'
    if original_report.exists():
        original_report.write_text(original_report.read_text().replace(
            'Legacy kernel benchmarks: NOT_RUN (their fitted specifications/calendar/scales are not silently imported).',
            'Legacy gross outputs verified separately on the 24-month 1994–1995 slice; see reduced_common_report.md. '
            'No new legacy fits or legacy net-cost comparison.'))
    print(table.to_string(index=False))
    print(pd.DataFrame(legacy).to_string(index=False))
    return table


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    finalize(parser.parse_args().run_dir)
