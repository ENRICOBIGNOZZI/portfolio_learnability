"""Independently reconcile published aggregates and check immutable input lineage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/final_empirical_20261008'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def check(condition,message):
    if not condition:raise ValueError(message)

def close(actual,expected,message,tol=1e-9):
    if not np.allclose(actual,expected,rtol=tol,atol=tol,equal_nan=False):raise ValueError(message)

def verify(out=DEFAULT,revision='v2',private=True):
    out=Path(out).resolve();verified={}
    freeze=json.loads((out/'audit/protocol_freeze.json').read_text())
    def hashcheck(key,expected):
        path=ROOT/key
        check('simulations' not in path.relative_to(ROOT).parts,'Forbidden simulation input')
        if key not in verified:verified[key]=digest(path)
        check(verified[key]==expected,'Changed source/input: '+key)
    for key,value in freeze['existing_tracked_non_simulation_files'].items():hashcheck(key,value)
    for phase in ('accounting','spectral','missing'):
        report=json.loads((out/'audit'/f'{phase}_verification.json').read_text())
        check(report['passed'],phase+' did not pass')
        for key,value in report['sources'].items():hashcheck(key,value)
        for key,value in report['inputs'].items():
            path=Path(key)
            if not private and (path.parts[0] in ('data','results')):continue
            if path.is_absolute():
                check(digest(path)==value,'Changed external reproduction input')
            else:hashcheck(key,value)
    report=json.loads((out/'audit/spectral_verification.json').read_text())
    for row in report['baseline_checks']:
        for key in ('maximum_payoff_error','maximum_relative_coefficient_error','maximum_complexity_error','maximum_direct_dual_payoff_error'):
            check(row[key]<1e-8,key)
        check(row['primal_svd_dual_eigenvalue_relative_error']<1e-10,'Primal-dual equivalence')
        check(row['maximum_feature_operator_relative_residual']<1e-5,'Tail eigenpair precision')
    m=pd.read_csv(out/'tables/monthly_accounting.csv')
    p=pd.read_csv(out/'tables/performance_scenarios.csv')
    a=pd.read_csv(out/'tables/annual_complexity.csv')
    n=pd.read_csv(out/'tables/nested_spectral_monthly.csv')
    c=pd.read_csv(out/'tables/spectral_contributions.csv')
    check(len(m)==5076 and len(p)==9 and len(a)==141 and len(n)==6768 and len(c)==12,'Aggregate dimensions')
    dates=pd.date_range('1978-02-28','2025-01-31',freq='ME').strftime('%Y-%m-%d').tolist()
    for (kernel,scenario),f in m.groupby(['kernel','scenario'],sort=False):
        check(f.return_date.tolist()==dates,'Common payoff dates')
        close(f.total_return-f.rf,f.excess_return,'Cash counted exactly once')
        close(f.gross_excess_return-f.excess_return,f.trading_return_drag+f.borrowing_fee,'Net drag identity')
        check(f.accounting_residual.max()<1e-12,'Self-financing residual')
        wealth=np.cumprod(1+f.total_return.to_numpy())
        dd=wealth/np.maximum.accumulate(np.r_[1,wealth])[1:]-1
        vals={'annual_excess_return':12*f.excess_return.mean(),'annual_volatility':np.sqrt(12)*f.excess_return.std(ddof=1),
            'maximum_drawdown':dd.min(),'annualized_turnover_two_sided':12*f.turnover.mean(),
            'mean_gross_exposure':f.gross_exposure.mean(),'mean_net_exposure':f.net_exposure.mean(),
            'mean_short_notional':f.short_notional.mean(),'annual_trading_fees':12*f.trading_fee.mean(),
            'annual_borrowing_fees':12*f.borrowing_fee.mean()}
        vals['sharpe']=vals['annual_excess_return']/vals['annual_volatility']
        r=p[(p.kernel==kernel)&(p.scenario==scenario)].iloc[0]
        for key,value in vals.items():close(value,r[key],'Performance '+key)
    scale=json.loads((ROOT/'results/final/public/linear/seed_0/calibration.json').read_text())['scale'] if private else .03740000227285338
    for kernel,f in a.groupby('kernel'):
        check(f.decision_year.tolist()==list(range(1978,2025)),'Annual coverage')
        close(f['T'],np.arange(180,733,12),'Historical monthly T')
        close(f.C/f['T'],f.C_over_T,'Relative complexity')
        check(((f.C_over_T>0)&(f.C_over_T<=1)).all(),'Complexity bounds')
        check((f.last_training_payoff==f.decision_date).all(),'Last known payoff')
        check((pd.to_datetime(f.first_oos_payoff)>pd.to_datetime(f.decision_date)).all(),'OOS chronology')
        close(f.inner_training_months+f.validation_months,f['T'],'Final historical refit')
        for fraction,part in n[n.kernel==kernel].groupby('rank_fraction'):
            check(part.return_date.tolist()==dates,'Nested common dates')
            close(part.previous_payoff+part.added_payoff,part.raw_payoff,'Nested reconstruction')
            loss=np.mean((1-part.raw_payoff)**2)
            cross=-2*np.mean((1-part.previous_payoff)*part.added_payoff)
            square=np.mean(part.added_payoff**2)
            delta=loss-np.mean((1-part.previous_payoff)**2)
            r=c[(c.kernel==kernel)&(c.rank_fraction==fraction)].iloc[0]
            close([loss,cross,square,delta],[r.raw_oos_loss,r.cross_term,r.squared_added_payoff,r.incremental_loss_change],'Spectral loss decomposition')
            close(delta,cross+square,'Cross terms')
            close(np.sqrt(12)*part.raw_payoff.mean()/part.raw_payoff.std(ddof=1),r.sharpe,'Nested Sharpe')
            if fraction==1:
                baseline=m[(m.kernel==kernel)&(m.scenario=='gross')]
                close(part.raw_payoff.to_numpy()*scale,baseline.excess_return.to_numpy(),'Full spectral endpoint')
                for year,block in part.groupby('decision_year'):
                    ryear=f[f.decision_year==year].iloc[0]
                    close(np.mean((1-block.raw_payoff)**2),ryear.raw_oos_loss,'Annual OOS loss')
    missing=pd.read_csv(out/'tables/missing_payoff_materiality.csv')
    check(len(missing)==744 and missing.missing_payoffs.sum()==7489,'Missing payoff counts')
    pub=out/'publication'/revision
    render=json.loads((pub/'render_manifest.json').read_text())
    for key,value in render['inputs'].items():check(digest(out/key)==value,'Changed figure input')
    for key,value in render['generated'].items():check(digest(pub/key)==value,'Changed publication artifact')
    check(digest(ROOT/'empirical_final/render.py')==render['renderer_sha256'],'Renderer changed')
    for key,value in render['templates'].items():check(digest(ROOT/'empirical_final/templates'/key)==value,'Template changed')
    build=json.loads((pub/'build_verification.json').read_text())
    check(build['passed'],'Manuscript build')
    check(digest(pub/'manuscript_real_data.pdf')==build['pdf_sha256'],'Changed final PDF')
    for key,value in build['inputs'].items():hashcheck(key,value)
    text=(pub/'manuscript_real_data.tex').read_text()
    check('simoutput' not in text and 'simulation_' not in text,'Real-data manuscript scope')
    return {'passed':True,'private_inputs_checked':private,'verified_source_and_input_files':len(verified),
        'preserved_existing_files':len(freeze['existing_tracked_non_simulation_files']),
        'monthly_policy_scenario_rows':len(m),'annual_refits':len(a),'nested_monthly_rows':len(n),
        'publication_revision':revision,
        'aggregate_hashes':{str(path.relative_to(out)):digest(path) for path in sorted((out/'tables').glob('*.csv'))},
        'final_pdf_sha256':build['pdf_sha256'],
        'package_sources':{str(path.relative_to(ROOT)):digest(path) for path in sorted((ROOT/'empirical_final').rglob('*')) if path.is_file() and '__pycache__' not in path.parts}}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--revision',default='v2')
    parser.add_argument('--public-only',action='store_true');parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=verify(args.out,args.revision,not args.public_only)
    if args.report:
        with args.report.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
