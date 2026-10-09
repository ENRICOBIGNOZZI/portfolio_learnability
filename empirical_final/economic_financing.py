"""Holdings-led financing follow-up, without changing any learned portfolio.

The dominant formed-holdings exposures motivated this accounting lens. Its
categories are fixed before their payoff attribution is calculated. This is a
descriptive follow-up, not an ex-ante strategy-selection or causal exercise.
"""
import json
import numpy as np
import pandas as pd
from data_pipeline import digest
from empirical_final.economic_holdings import ROOT,OUT,PRIVATE,BASE,FAMILIES,exposures,terciles,family_matrix,save,BLOCKS,block_counts,interval,hac_regression,FACTORS

EXTRA='Debt financing'
EXTRA_MEMBERS={'fnl_gr1a':1,'dbnetis_at':1}
EXTRA_PAIRS=[(EXTRA,'Profitability'),(EXTRA,'Issuance')]


def main():
    design=dict(family=EXTRA,members=EXTRA_MEMBERS,pairs=EXTRA_PAIRS,
        rationale='Dominant group-level holding exposures to financial-liability growth motivated a descriptive follow-up; no portfolio, penalty or security selection changes.',
        selection_basis='Observed characteristic exposures, not ex-post profitability of financing bins',
        chronology='Same existing weights; cells use only contemporaneous characteristics',
        no_claim_of_preregistered_financing_hypothesis=True)
    p=OUT/'audit/financing_followup_design.json'
    if p.exists() and json.loads(p.read_text())!=json.loads(json.dumps(design)):raise ValueError('Follow-up design changed')
    p.write_text(json.dumps(design,indent=2))
    families,cells,profiles,netting=[],[],[],[]
    for year in range(1978,2025):
        clean=pd.read_parquet(ROOT/f'data/clean/jkp_{year}.parquet')
        held=pd.read_parquet(PRIVATE/f'year_{year}/holdings.parquet').set_index(['formation_date','permno'])
        for date,frame in clean.groupby('eom',sort=True):
            w=held.loc[[(date,int(i)) for i in frame.id],[f'group_{j}_weight' for j in range(1,5)]].to_numpy()
            n=len(frame);r=frame.r.to_numpy()
            score=sum(frame[name].to_numpy()*sign for name,sign in EXTRA_MEMBERS.items())/len(EXTRA_MEMBERS)
            l,s,lm,sm,net=exposures(score[:,None],w)
            cumulative=np.abs(np.cumsum(w,axis=1)).sum(axis=0)
            for j in range(4):
                families.append(dict(formation_date=date,return_date=date+pd.offsets.MonthEnd(1),decision_year=year,group=j+1,
                    family=EXTRA,long_mean=lm[j,0],short_mean=sm[j,0],long_minus_short=lm[j,0]-sm[j,0],net_exposure=net[j,0]))
                netting.append(dict(formation_date=date,decision_year=year,group=j+1,component_gross=l[j]+s[j],
                    cumulative_gross=cumulative[j],incremental_gross=cumulative[j]-(cumulative[j-1] if j else 0),
                    cancellation=float(np.abs(w[:,:j+1]).sum()-cumulative[j])))
            fm=family_matrix(frame);assignment={name:terciles(fm[:,k]) for k,name in enumerate(FAMILIES)}
            assignment[EXTRA]=terciles(score)
            partitions=[(EXTRA,assignment[EXTRA],3)]+[(a+' x '+b,3*assignment[a]+assignment[b],9) for a,b in EXTRA_PAIRS]
            for name,labels,count in partitions:
                reconstructed=np.zeros(4);counts=np.bincount(labels,minlength=count)
                for cell in range(count):
                    mask=labels==cell;cw=w[mask];rr=r[mask];payoff=cw.T@rr;reconstructed+=payoff
                    for j in range(4):
                        cells.append(dict(formation_date=date,return_date=date+pd.offsets.MonthEnd(1),decision_year=year,group=j+1,
                            partition=name,cell=cell,stock_count=int(counts[cell]),stock_fraction=float(mask.mean()),
                            long_notional=float(np.maximum(cw[:,j],0).sum()),short_notional=float(np.maximum(-cw[:,j],0).sum()),
                            net_notional=float(cw[:,j].sum()),payoff=payoff[j],
                            long_payoff=float(np.maximum(cw[:,j],0)@rr),short_payoff=float(np.minimum(cw[:,j],0)@rr)))
                np.testing.assert_allclose(reconstructed,w.T@r,atol=1e-12)
                if count==9:
                    design_x=np.column_stack([np.ones(9),np.arange(9)//3==1,np.arange(9)//3==2,np.arange(9)%3==1,np.arange(9)%3==2])
                    valid=counts>0
                    for j in range(4):
                        sums=np.bincount(labels,weights=w[:,j],minlength=9)
                        density=np.divide(n*sums,counts*(l[j]+s[j]),out=np.zeros(9),where=valid)
                        fit=design_x@np.linalg.lstsq(design_x[valid]*np.sqrt(counts[valid,None]),density[valid]*np.sqrt(counts[valid]),rcond=None)[0]
                        for cell in range(9):profiles.append(dict(formation_date=date,decision_year=year,group=j+1,partition=name,cell=cell,
                            weight_density=density[cell],additive_density=fit[cell],nonadditive_density=density[cell]-fit[cell]))
    f=pd.DataFrame(families);c=pd.DataFrame(cells);p=pd.DataFrame(profiles);n=pd.DataFrame(netting)
    for name,frame in [('financing_monthly_families',f),('financing_monthly_cells',c),('financing_monthly_joint_profiles',p),('monthly_stock_netting',n)]:
        frame.to_parquet(OUT/'tables'/f'{name}.parquet',index=False,compression='zstd')
    factors=pd.read_csv(BASE/'tables/french_factors.csv',parse_dates=['return_date']).set_index('return_date')
    returns=pd.read_parquet(OUT/'tables/monthly_balances.parquet').pivot(index='return_date',columns='group',values='total_payoff')
    kappa=json.loads((OUT/'audit/holdings_execution.json').read_text())['kappa']
    es,cs,js,ns,fs=[],[],[],[],[]
    max_error=0.
    old=pd.read_csv(BASE/'tables/e3_nested_summary.csv')
    for period,lo,hi in [('all',1978,2024)]+[(f'{lo}-{hi}',lo,hi) for lo,hi in BLOCKS]:
        ff=f[f.decision_year.between(lo,hi)];cc=c[c.decision_year.between(lo,hi)];pp=p[p.decision_year.between(lo,hi)];nn=n[n.decision_year.between(lo,hi)]
        dates=sorted(ff.return_date.unique());counts=block_counts(len(dates));fx=factors.loc[dates,FACTORS].to_numpy()
        for group in range(1,5):
            d=ff[ff.group==group].sort_values('return_date');ci=interval(counts@d.long_minus_short.to_numpy()/len(d))
            es.append(dict(period=period,group=group,family=EXTRA,months=len(d),long_mean=d.long_mean.mean(),short_mean=d.short_mean.mean(),
                long_minus_short=d.long_minus_short.mean(),low=ci[0],high=ci[1],fraction_positive=np.mean(d.long_minus_short>0),net_exposure=d.net_exposure.mean()))
            dd=nn[nn.group==group];ns.append(dict(period=period,group=group,component_gross=dd.component_gross.mean(),
                cumulative_gross=dd.cumulative_gross.mean(),incremental_gross=dd.incremental_gross.mean(),cancellation=dd.cancellation.mean()))
            q=returns.loc[dates,group].to_numpy();prev=returns.loc[dates].iloc[:,:group-1].sum(axis=1).to_numpy()
            for partition,dd in cc[cc.group==group].groupby('partition'):
                total_loss,total_var=0.,0.
                for cell,ddd in dd.groupby('cell'):
                    ddd=ddd.sort_values('return_date');r=ddd.payoff.to_numpy();mean_ci=interval(12*counts@r/len(r))
                    loss=-2*(1-prev/kappa)*(r/kappa)+(q/kappa)*(r/kappa);loss_ci=interval(counts@loss/len(r))
                    var=12*(2*np.cov(prev,r,ddof=1)[0,1]+np.cov(q,r,ddof=1)[0,1]);total_loss+=loss.mean();total_var+=var
                    cs.append(dict(period=period,group=group,partition=partition,cell=cell,annual_mean=12*r.mean(),mean_low=mean_ci[0],mean_high=mean_ci[1],
                        annual_long_payoff=12*ddd.long_payoff.mean(),annual_short_payoff=12*ddd.short_payoff.mean(),
                        mean_long_notional=ddd.long_notional.mean(),mean_short_notional=ddd.short_notional.mean(),
                        allocated_incremental_loss=loss.mean(),loss_low=loss_ci[0],loss_high=loss_ci[1],allocated_incremental_variance=var))
                    beta,se,r2=hac_regression(r,fx)
                    for factor,value,error in zip(['alpha_monthly']+FACTORS,beta,se):fs.append(dict(period=period,group=group,partition=partition,cell=cell,
                        factor=factor,coefficient=value,se=error,low=value-1.96*error,high=value+1.96*error,r_squared=r2))
                target=old[(old.period==period)&(old.group==group)].iloc[0]
                max_error=max(max_error,abs(total_loss-target.delta_loss),abs(total_var-target.delta_variance))
            for (partition,cell),dd in pp[pp.group==group].groupby(['partition','cell']):
                js.append(dict(period=period,group=group,partition=partition,cell=cell,weight_density=dd.weight_density.mean(),
                    additive_density=dd.additive_density.mean(),nonadditive_density=dd.nonadditive_density.mean()))
    for name,rows in [('financing_family_exposures',es),('financing_cell_return_attribution',cs),('financing_joint_weight_profiles',js),
                      ('stock_netting_summary',ns),('financing_factor_attribution',fs)]:save(name,rows)
    if max_error>1e-7:raise ValueError('Financing accounting identity failed')
    (OUT/'audit/financing_checks.json').write_text(json.dumps(dict(passed=True,max_accounting_error=max_error,
        source_sha256=digest(__file__),no_policy_changes=True),indent=2))
    print('Financing follow-up and actual stock-netting analysis complete',flush=True)


if __name__=='__main__':main()
