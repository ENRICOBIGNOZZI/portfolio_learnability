"""Reports generated exclusively from completed immutable run ledgers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .accounting import wealth
from .config import write_json, canonical_hash
from .metrics import summary, paired_comparisons


def report(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['status'] != 'REAL_RUN_COMPLETED':
        raise ValueError('Only completed runs can be reported.')
    events = [json.loads(p.read_text()) for p in sorted((directory/'events').glob('*.json'))]
    if len(events) != manifest['months'] or events[-1]['event_hash'] != manifest['final_event_hash']:
        raise ValueError('Incomplete final ledger.')
    previous = None
    for event in events:
        check = dict(event)
        token = check.pop('event_hash')
        if canonical_hash(check) != token or event['previous_hash'] != previous:
            raise ValueError('Corrupted event ledger.')
        previous = token
    out = directory/'report'
    out.mkdir(exist_ok=True)
    rows, seed_rows, expert_rows, gates = [], [], [], []
    for e in events:
        for name,m in e['methods'].items():
            rows.append(dict(method=name,decision_date=e['decision_date'],return_date=e['return_date'],
                raw_excess=m['raw_excess'], response_one=m['response_one'],risk_free=e['risk_free'],
                gross=m['gross'],net=m['net'],cash=m['cash'],concentration=m['concentration'],
                maximum_absolute_weight=m['maximum_absolute_weight'],turnover=m['turnover'],
                turnover_kind=m['turnover_kind'],target_weight_turnover=m['target_weight_turnover'],
                **m['cost_sensitivities'],borrow_sensitivity=m['borrowing_25bps_plus_30bps_year']))
        for name,p in e['experts'].items():
            expert_rows.append(dict(expert=name,return_date=e['return_date'],raw_excess=p))
        for name,seeds in e['seed_payoffs'].items():
            for seed,p in enumerate(seeds):
                seed_rows.append(dict(expert=name,seed=seed,return_date=e['return_date'],raw_excess=p))
        for name,g in e['gates'].items():
            gates.append(dict(method=name,decision_date=e['decision_date'],active=g['active'],
                              selected=g['selected'],baseline=g['baseline'],reason=g['reason'],candidates=g['candidates']))
    frame = pd.DataFrame(rows)
    table, paths, costs = [], [], []
    for name,g in frame.groupby('method',sort=True):
        s = summary(g.raw_excess)
        v, dd, insolvent = wealth(g.raw_excess,g.risk_free)
        table.append(dict(method=name,**s,gross=g.gross.mean(),net=g.net.mean(),
                          mean_turnover=g.turnover.mean(),insolvent=insolvent is not None,
                          insolvency_date=g.return_date.iloc[insolvent] if insolvent is not None else None,
                          max_drawdown=float(np.nanmin(dd)) if np.isfinite(dd).any() else None))
        paths.extend(dict(method=name,return_date=d,wealth=float(a) if np.isfinite(a) else None,
                          drawdown=float(b) if np.isfinite(b) else None)
                     for d,a,b in zip(g.return_date,v,dd,strict=True))
        for col in ['0bps','10bps','25bps','borrow_sensitivity']:
            costs.append(dict(method=name,sensitivity=col,**summary(g[col])))
    summary_frame = pd.DataFrame(table)
    frame.to_csv(out/'monthly.csv',index=False)
    summary_frame.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(paths).to_csv(out/'wealth.csv',index=False)
    pd.DataFrame(costs).to_csv(out/'cost_sensitivities.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'seed_payoffs.csv',index=False)
    experts = pd.DataFrame(expert_rows)
    experts.to_csv(out/'expert_payoffs.csv',index=False)
    depth = [dict(expert=name,**summary(g.raw_excess)) for name,g in experts.groupby('expert')]
    pd.DataFrame(depth).to_csv(out/'depth_width_comparison.csv',index=False)
    gate_frame = pd.DataFrame(gates)
    gate_frame.to_csv(out/'gate_log.csv',index=False)
    activations = []
    for name,g in gate_frame.groupby('method'):
        activations.append(dict(method=name,activation_frequency=float(g.active.mean()),
            switches=int(g.selected.ne(g.selected.shift()).iloc[1:].sum())))
    pd.DataFrame(activations).to_csv(out/'gate_summary.csv',index=False)
    wide = frame[frame.method.str.startswith('joint/')].pivot(index='return_date',columns='method',values='raw_excess')
    paired = paired_comparisons(wide,'joint/full')
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    # Common and terminal reports require complete prespecified calendar coverage.
    coverage = {}
    for label,start,end in [('common','1994-01-31','2024-12-31'),('terminal','2020-01-31','2024-12-31')]:
        subset = wide.loc[(wide.index>=start)&(wide.index<=end)]
        expected = pd.date_range(start,end,freq='ME').strftime('%Y-%m-%d').tolist()
        complete = list(subset.index) == expected
        coverage[label] = dict(status='REAL_RUN_COMPLETED' if complete else 'NOT_RUN',months=len(subset),expected=len(expected))
        if complete:
            pd.DataFrame([dict(method=k,**summary(subset[k])) for k in subset]).to_csv(out/f'{label}_comparison.csv',index=False)
    write_json(out/'coverage.json',coverage)
    _figures(out,summary_frame,frame,directory)
    selected = summary_frame[summary_frame.method.str.startswith('joint/')]
    lines = ['# Completed run report', '', '**'+manifest['interpretation']+'**. Current licensed JKP snapshot.', '',
        f"Executed: {manifest['fits']} fits, {manifest['origins']} annual origins, {manifest['months']} monthly payoffs.",
        'Raw training scale = reporting scale = 1. No OOS volatility normalization.',
        'The sample conditions on future payoff availability. These are not point-in-time investment results.',
        'Fixed optimization budgets do not certify convergence or a global Sharpe optimum.', '',
        '| Method | Months | Annualized Sharpe | Response-one | Insolvent |', '|---|---:|---:|---:|---|']
    lines += [f'| {r.method} | {r.months} | {format(r.sharpe,".4f") if pd.notna(r.sharpe) else "NOT_IDENTIFIED"} | {r.response_one:.6f} | {r.insolvent} |'
              for r in selected.itertuples()]
    lines += ['', 'Bootstrap intervals are retrospective and conditional on these saved series. With fewer than 60',
        'mature prequential monthly scores the guard exactly uses full history; this is not evidence of stationarity.',
        'Cost sensitivities use both trade legs with no factor 1/2; borrowing at 30 bps/year is illustrative.',
        'Stock total returns for drift accounting equal source excess plus same-snapshot cash.',
        'Wealth stops at insolvency; subsequent policy payoffs remain hypothetical frictionless diagnostics.',
        'Legacy kernel benchmarks: NOT_RUN (their fitted specifications/calendar/scales are not silently imported).',
        '', 'Prespecified coverage: '+json.dumps(coverage)]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return selected[['method','months','sharpe','response_one','insolvent']].to_dict(orient='records')


def _figures(out, table, monthly, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = table[table.method.str.startswith('joint/')]
    fig,axes = plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    labels = selected.method.str.replace('joint/','',regex=False)
    axes[0].barh(labels,selected.sharpe,color='#236b8e')
    axes[0].set_xlabel('Annualized Sharpe (concatenated monthly payoffs)')
    axes[1].barh(labels,selected.response_one,color='#ac6b39')
    axes[1].set_xlabel('Out-of-sample response-one loss')
    fig.suptitle('JKP direct NN | RETROSPECTIVE COMPLETE-PAYOFF SENSITIVITY')
    fig.savefig(out/'comparison.png',dpi=150)
    plt.close(fig)
    origins = sorted((directory/'origins').glob('*.json'))
    origin = json.loads(origins[0].read_text())
    fig,ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    for w in origin['weights']:
        ax.plot(w['ages'],w['omega'],label=w['memory'])
    ax.set(xlabel='Calendar-month age of training payoff',ylabel='Temporal weight',title='Frozen memory weights at first refit')
    ax.legend(fontsize=8)
    fig.savefig(out/'memory_curves.png',dpi=150)
    plt.close(fig)
