"""Economic attribution of the EXISTING ridge spectral groups, from stock holdings.

No portfolio is selected or retuned here. Formation-date characteristics describe
the signed holdings; next-month returns only measure their subsequent payoffs.
Stock-level fitted weights remain in the repository's private results directory.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from kernels import FeatureBank
from data_pipeline import digest
from empirical_final.run import ROOT, INPUTS, bind, read_json, clean_manifest, load_managed
from empirical_final.three_experiments import spectral_groups, BLOCKS, block_counts, interval, hac_regression, FACTORS

BASE = ROOT/'outputs/empirical_three_experiments_20261009'
OUT = ROOT/'outputs/spectral_economic_content_20261009'
PRIVATE = ROOT/'results/spectral_economic_content_20261009'

# Positive scores mean MORE of the named attribute, not a presumed profitable side.
# Dictionary definitions determine signs. No return realization determines a name.
FAMILIES = {
    'Value': {'be_me':1, 'ni_me':1, 'fcf_me':1, 'sale_me':1, 'ocf_me':1},
    'Profitability': {'gp_at':1, 'cop_at':1, 'op_at':1, 'ni_be':1},
    'Investment': {'at_gr1':1, 'capx_gr1':1, 'inv_gr1a':1, 'ppeinv_gr1a':1},
    'Momentum': {'ret_12_1':1, 'ret_6_1':1, 'resff3_12_1':1},
    'Recent return': {'ret_1_0':1},
    'Size': {'market_equity':1},
    'Liquidity': {'dolvol_126d':1, 'ami_126d':-1, 'bidaskhl_21d':-1, 'zero_trades_126d':-1},
    'Volatility': {'rvol_21d':1, 'ivol_capm_21d':1, 'ivol_capm_252d':1},
    'Issuance': {'chcsho_12m':1, 'eqnpo_12m':-1},
    'Accruals': {'oaccruals_at':1, 'taccruals_at':1},
    'Quality': {'qmj':1, 'qmj_prof':1, 'qmj_safety':1},
    'Leverage': {'debt_me':1, 'netdebt_me':1},
    'Seasonality': {'seas_1_1an':1, 'seas_2_5an':1},
}
PAIRS = [('Value','Profitability'), ('Momentum','Recent return'),
         ('Size','Liquidity'), ('Volatility','Issuance')]


def save(name, rows):
    f = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(OUT/'tables'/f'{name}.csv', index=False, float_format='%.10g')
    return f


def exposures(x, weights):
    """Actual signed leg sizes, normalized leg characteristics and net exposures."""
    long, short = np.maximum(weights, 0), np.maximum(-weights, 0)
    l, s = long.sum(axis=0), short.sum(axis=0)
    lm = np.divide(long.T@x, l[:,None], out=np.full((4,x.shape[1]),np.nan), where=l[:,None]>0)
    sm = np.divide(short.T@x, s[:,None], out=np.full((4,x.shape[1]),np.nan), where=s[:,None]>0)
    return l, s, lm, sm, weights.T@x


def terciles(values):
    """Contemporaneous cross-sectional ranks; average ties, never future returns."""
    z = (rankdata(values, method='average')-1)/(len(values)-1)-.5
    return np.digitize(z, [-1/6, 1/6])


def family_matrix(frame):
    return np.column_stack([sum(sign*frame[name].to_numpy() for name,sign in members.items())/len(members)
                            for members in FAMILIES.values()])


def checkpoint_year(year, coefficients, bank, names, clean_item, expected, kappa):
    cp = PRIVATE/f'year_{year}'
    cp.mkdir(exist_ok=True)
    source = bind(ROOT/'data/clean'/clean_item['name'], clean_item['sha256'])
    identity = dict(clean_sha256=clean_item['sha256'], coefficient_sha256=__import__('hashlib').sha256(coefficients.tobytes()).hexdigest(),
                    families=FAMILIES, pairs=PAIRS, kappa=kappa, source_sha256=digest(Path(__file__)))
    identity = json.loads(json.dumps(identity))
    done = cp/'complete.json'
    if done.exists():
        meta = json.loads(done.read_text())
        if meta['identity'] != identity: raise ValueError(f'Checkpoint identity changed: {year}')
        for name, value in meta['files'].items():
            if digest(cp/name) != value: raise ValueError(f'Checkpoint checksum: {year}/{name}')
        print(f'Reused verified holdings {year}', flush=True)
        return meta
    frame = pd.read_parquet(source)
    raw_ids = pd.read_parquet(ROOT/f'data/raw/jkp_{year}.parquet',columns=['id','permno','eom'])
    ids = frame[['id','eom']].merge(raw_ids,on=['id','eom'],validate='one_to_one')
    if ids.permno.isna().any() or not ids.id.eq(ids.permno).all(): raise ValueError('JKP ID/PERMNO identity failed')
    prior = pd.read_parquet(bind(ROOT/f'results/final/private/gaussian/seed_0/p_10000/weights_{year}.parquet'))
    family_rows, char_rows, bins, profiles, holdings, positions, balances = [], [], [], [], [], [], []
    max_weight_error, max_payoff_error, max_partition_error = 0., 0., 0.
    previous_month = None
    for date, panel in frame.groupby('eom',sort=True):
        x = panel[names].to_numpy(float); n = len(panel); ret = panel.r.to_numpy(float)
        # One feature evaluation for all four contributions; exactly the original RFF map.
        w = kappa*bank.scores(x, coefficients)/n
        ref = prior[prior.formation_date.eq(date)].set_index('id').loc[panel.id].raw_weight.to_numpy()*kappa
        weight_error = float(np.max(np.abs(w.sum(axis=1)-ref)))
        np.testing.assert_allclose(w.sum(axis=1),ref,rtol=2e-6,atol=2e-9)
        target = expected[expected.return_date.eq(date+pd.offsets.MonthEnd(1))].sort_values('group').scaled_contribution.to_numpy()
        payoff = w.T@ret; payoff_error = float(np.max(np.abs(payoff-target)))
        np.testing.assert_allclose(payoff,target,rtol=2e-6,atol=2e-9)
        max_weight_error=max(max_weight_error,weight_error);max_payoff_error=max(max_payoff_error,payoff_error)
        l,s,lm,sm,net = exposures(x,w)
        for j in range(4):
            for k,name in enumerate(names):
                char_rows.append(dict(formation_date=date,return_date=date+pd.offsets.MonthEnd(1),decision_year=year,
                    group=j+1,characteristic=name,long_mean=lm[j,k],short_mean=sm[j,k],long_minus_short=lm[j,k]-sm[j,k],net_exposure=net[j,k]))
            balances.append(dict(formation_date=date,return_date=date+pd.offsets.MonthEnd(1),decision_year=year,group=j+1,
                long_notional=l[j],short_notional=s[j],gross=l[j]+s[j],net=l[j]-s[j],
                long_count=int((w[:,j]>0).sum()),short_count=int((w[:,j]<0).sum()),
                long_payoff=float(np.maximum(w[:,j],0)@ret),short_payoff=float(np.minimum(w[:,j],0)@ret),
                total_payoff=payoff[j],top10_gross_share=float(np.sort(np.abs(w[:,j]))[-10:].sum()/(l[j]+s[j]))))
            for side, sign in [('long',1),('short',-1)]:
                candidates=np.flatnonzero(w[:,j]*sign>0)
                chosen=candidates[np.argsort(-sign*w[candidates,j])[:20]]
                for rank,i in enumerate(chosen,1):
                    positions.append(dict(formation_date=date,return_date=date+pd.offsets.MonthEnd(1),group=j+1,side=side,
                        position_rank=rank,permno=int(panel.id.iloc[i]),weight=w[i,j],weight_pct=100*w[i,j],
                        subsequent_payoff=w[i,j]*ret[i]))
        hh=panel[['id','eom','return_date']].rename(columns={'id':'permno','eom':'formation_date'}).copy()
        for j in range(4): hh[f'group_{j+1}_weight']=w[:,j]
        hh['total_weight']=w.sum(axis=1);holdings.append(hh)
        scores=family_matrix(panel)
        _,_,fl,fs,fn=exposures(scores,w)
        family_names=list(FAMILIES); assignments={name:terciles(scores[:,k]) for k,name in enumerate(family_names)}
        for k,name in enumerate(family_names):
            for j in range(4):
                family_rows.append(dict(formation_date=date,return_date=date+pd.offsets.MonthEnd(1),decision_year=year,group=j+1,
                    family=name,long_mean=fl[j,k],short_mean=fs[j,k],long_minus_short=fl[j,k]-fs[j,k],net_exposure=fn[j,k]))
        partitions=[(name,assignments[name],3) for name in family_names]
        partitions += [(a+' x '+b,3*assignments[a]+assignments[b],9) for a,b in PAIRS]
        for partition, labels, count in partitions:
            part_return=np.zeros(4)
            for cell in range(count):
                mask=labels==cell; cw=w[mask];r=ret[mask]
                cell_payoff=cw.T@r;part_return+=cell_payoff
                for j in range(4):
                    bins.append(dict(formation_date=date,return_date=date+pd.offsets.MonthEnd(1),decision_year=year,group=j+1,
                        partition=partition,cell=cell,stock_count=int(mask.sum()),stock_fraction=float(mask.mean()),
                        long_notional=float(np.maximum(cw[:,j],0).sum()),short_notional=float(np.maximum(-cw[:,j],0).sum()),
                        net_notional=float(cw[:,j].sum()),payoff=cell_payoff[j],
                        long_payoff=float(np.maximum(cw[:,j],0)@r),short_payoff=float(np.minimum(cw[:,j],0)@r)))
            max_partition_error=max(max_partition_error,float(np.max(np.abs(part_return-payoff))))
            np.testing.assert_allclose(part_return,payoff,atol=1e-12,rtol=1e-10)
            if count==9:
                counts=np.bincount(labels,minlength=9)
                design=np.column_stack([np.ones(9),np.arange(9)//3==1,np.arange(9)//3==2,np.arange(9)%3==1,np.arange(9)%3==2])
                valid=counts>0
                for j in range(4):
                    sums=np.bincount(labels,weights=w[:,j],minlength=9)
                    density=np.divide(n*sums,counts*(l[j]+s[j]),out=np.zeros(9),where=valid)
                    d=design[valid]*np.sqrt(counts[valid,None]);y=density[valid]*np.sqrt(counts[valid])
                    fit=design@np.linalg.lstsq(d,y,rcond=None)[0]
                    for cell in range(9):
                        profiles.append(dict(formation_date=date,decision_year=year,group=j+1,partition=partition,cell=cell,
                            count=counts[cell],relative_weight_density=density[cell],additive_density=fit[cell],
                            nonadditive_density=density[cell]-fit[cell]))
    tables={'holdings':pd.concat(holdings,ignore_index=True),'top_positions':pd.DataFrame(positions),
            'characteristics':pd.DataFrame(char_rows),'families':pd.DataFrame(family_rows),
            'cells':pd.DataFrame(bins),'joint_profiles':pd.DataFrame(profiles),'balances':pd.DataFrame(balances)}
    for name,table in tables.items():table.to_parquet(cp/f'{name}.parquet',index=False,compression='zstd')
    meta=dict(identity=identity,year=year,stock_months=len(frame),max_weight_error=max_weight_error,
              max_payoff_error=max_payoff_error,max_partition_error=max_partition_error,
              files={p.name:digest(p) for p in cp.glob('*.parquet')})
    done.write_text(json.dumps(meta,indent=2))
    print(f'Holdings {year}: {len(frame):,} stock-months; weight error {max_weight_error:.2g}; payoff error {max_payoff_error:.2g}',flush=True)
    return meta


def reconstruct():
    started=time.monotonic()
    for p in [PRIVATE,OUT/'audit',OUT/'tables',OUT/'figures',OUT/'publication']:p.mkdir(parents=True,exist_ok=True)
    protocol=dict(families=FAMILIES,pairs=PAIRS,groups=['0-10%','10-50%','50-90%','90-100%'],
        grouping='existing training spectral ranks, unchanged ridge-selected policy contributions',
        exposures='long and absolute-short weighted means of formation ranks; difference; actual notional exposures also retained',
        partitions='contemporaneous average-rank terciles of equally weighted signed family scores; all cells retained',
        interaction='formation cross-section only: weighted additive row/column fit to nine cell weight densities; no return fit or new portfolio',
        attribution='each family/pair separately partitions actual group holdings; alternative overlapping views are never added',
        chronology='known returns through January close; formation characteristics through each month; next-month payoff',
        source_dictionary='https://jkpfactors-data.s3.amazonaws.com/documents/Documentation.pdf',
        private_holdings=str(PRIVATE.relative_to(ROOT)),bootstrap='5000 paired circular draws; 12 monthly observations per block',
        no_new_learning_algorithm=True,no_discovery_date_analysis=True)
    fp=OUT/'audit/economic_protocol.json'
    if fp.exists() and json.loads(fp.read_text())!=protocol:raise ValueError('Economic interpretation protocol changed')
    fp.write_text(json.dumps(protocol,indent=2))
    clean=clean_manifest();g,dates=load_managed('gaussian',clean);names=clean['characteristics']
    source=read_json(ROOT/'results/final/public/gaussian/seed_0/source.json')['feature_bank']
    bank=FeatureBank('gaussian',len(names),10000,source['ell'],source['seed'])
    if bank.metadata()!=source:raise ValueError('Feature map changed')
    kappa=read_json(ROOT/'results/final/public/linear/seed_0/calibration.json')['scale']
    annual=pd.read_csv(bind(BASE/'tables/e1_annual_paths.csv'));annual=annual[annual.selected].set_index('decision_year')
    expected=pd.read_csv(bind(BASE/'tables/e3_monthly.csv'),parse_dates=['return_date'])
    records=[]
    for year in range(1978,2025):
        h=np.flatnonzero(dates.year<year)
        assert (dates[h]+pd.offsets.MonthEnd(1)).max()<=pd.Timestamp(year,1,31)
        _,coefficients,_=spectral_groups(np.asarray(g[h]),float(annual.loc[year,'penalty']))
        item=next(i for i in clean['files'] if i['name']==f'jkp_{year}.parquet')
        records.append(checkpoint_year(year,coefficients,bank,names,item,expected,kappa))
    for name in ['characteristics','families','cells','joint_profiles','balances']:
        data=pd.concat([pd.read_parquet(PRIVATE/f'year_{y}'/f'{name}.parquet') for y in range(1978,2025)],ignore_index=True)
        data.to_parquet(OUT/'tables'/f'monthly_{name}.parquet',index=False,compression='zstd')
        if name!='characteristics':save('monthly_'+name,data)
        if name=='characteristics':
            save('annual_characteristics',data.groupby(['decision_year','group','characteristic'])[['long_mean','short_mean','long_minus_short','net_exposure']].mean().reset_index())
    tops=pd.concat([pd.read_parquet(PRIVATE/f'year_{y}/top_positions.parquet') for y in range(1978,2025)],ignore_index=True)
    tops.to_csv(PRIVATE/'top_positions_all_months.csv',index=False)
    save('reconstruction_checks',[{k:v for k,v in r.items() if k not in ['identity','files']} for r in records])
    audit=dict(passed=True,elapsed_seconds=time.monotonic()-started,stock_months=sum(r['stock_months'] for r in records),
        annual_refits=len(records),source_sha256=digest(Path(__file__)),inputs=INPUTS,kappa=kappa,
        dictionary_sha256=digest(ROOT/'data/economic_dictionary/Documentation.pdf'),
        max_weight_error=max(r['max_weight_error'] for r in records),max_payoff_error=max(r['max_payoff_error'] for r in records),
        max_partition_error=max(r['max_partition_error'] for r in records),
        private_checkpoint_hashes={str((PRIVATE/f'year_{r["year"]}/complete.json').relative_to(ROOT)):digest(PRIVATE/f'year_{r["year"]}/complete.json') for r in records})
    (OUT/'audit/holdings_execution.json').write_text(json.dumps(audit,indent=2));print('Holdings reconstruction complete',flush=True)


def summarize():
    families=pd.read_parquet(OUT/'tables/monthly_families.parquet')
    chars=pd.read_parquet(OUT/'tables/monthly_characteristics.parquet')
    cells=pd.read_parquet(OUT/'tables/monthly_cells.parquet')
    balances=pd.read_parquet(OUT/'tables/monthly_balances.parquet')
    profiles=pd.read_parquet(OUT/'tables/monthly_joint_profiles.parquet')
    factors=pd.read_csv(BASE/'tables/french_factors.csv',parse_dates=['return_date']).set_index('return_date')
    group_returns=balances.pivot(index='return_date',columns='group',values='total_payoff').sort_index()
    kappa=json.loads((OUT/'audit/holdings_execution.json').read_text())['kappa']
    summaries,dominant,balance_summary,attribution,factor_rows,stability,joint_summary=[],[],[],[],[],[],[]
    periods=[('all',1978,2024)]+[(f'{lo}-{hi}',lo,hi) for lo,hi in BLOCKS]
    for label,lo,hi in periods:
        ff=families[families.decision_year.between(lo,hi)]
        cc=chars[chars.decision_year.between(lo,hi)]
        bb=balances[balances.decision_year.between(lo,hi)]
        pp=profiles[profiles.decision_year.between(lo,hi)]
        dates=sorted(ff.return_date.unique());counts=block_counts(len(dates)); f=factors.loc[dates,FACTORS].to_numpy()
        for group in range(1,5):
            selected=ff[ff.group.eq(group)];b=bb[bb.group.eq(group)].sort_values('return_date')
            for family in FAMILIES:
                ss=selected[selected.family.eq(family)].sort_values('return_date')
                z=ss.long_minus_short.to_numpy();ci=interval(counts@z/len(z))
                summaries.append(dict(period=label,group=group,family=family,months=len(z),long_mean=ss.long_mean.mean(),
                    short_mean=ss.short_mean.mean(),long_minus_short=z.mean(),low=ci[0],high=ci[1],
                    fraction_positive=float(np.mean(z>0)),net_exposure=ss.net_exposure.mean()))
            means=cc[cc.group.eq(group)].groupby('characteristic')[['long_mean','short_mean','long_minus_short','net_exposure']].mean()
            for direction,ascending in [('positive tilt',False),('negative tilt',True)]:
                for rank,(name,row) in enumerate(means.sort_values('long_minus_short',ascending=ascending).head(6).iterrows(),1):
                    dominant.append(dict(period=label,group=group,tilt=direction,rank=rank,characteristic=name,**row.to_dict()))
            balance_summary.append(dict(period=label,group=group,mean_long=b.long_notional.mean(),mean_short=b.short_notional.mean(),
                mean_gross=b.gross.mean(),mean_net=b.net.mean(),mean_long_count=b.long_count.mean(),mean_short_count=b.short_count.mean(),
                annual_long_payoff=12*b.long_payoff.mean(),annual_short_payoff=12*b.short_payoff.mean(),annual_mean=12*b.total_payoff.mean(),
                annual_volatility=np.sqrt(12)*b.total_payoff.std(ddof=1),top10_share=b.top10_gross_share.mean()))
            for kind,y in [('group',b.total_payoff.to_numpy()),('long',b.long_payoff.to_numpy()),('short',b.short_payoff.to_numpy())]:
                beta,se,r2=hac_regression(y,f)
                for name,value,error in zip(['alpha_monthly']+FACTORS,beta,se):
                    factor_rows.append(dict(period=label,group=group,partition=kind,cell=-1,factor=name,coefficient=value,
                        se=error,low=value-1.96*error,high=value+1.96*error,r_squared=r2))
            group_series=group_returns.loc[dates,group].to_numpy();previous=group_returns.loc[dates].iloc[:,:group-1].sum(axis=1).to_numpy()
            raw_q=group_series/kappa;raw_p=previous/kappa
            for partition,data in cells[cells.group.eq(group)&cells.decision_year.between(lo,hi)].groupby('partition'):
                wide=data.pivot(index='return_date',columns='cell',values='payoff').loc[dates]
                np.testing.assert_allclose(wide.sum(axis=1),group_series,atol=1e-12)
                contribution_boot=12*counts@wide.to_numpy()/len(dates);cis=interval(contribution_boot)
                for j,cell in enumerate(wide.columns):
                    r=wide[cell].to_numpy();raw=r/kappa
                    loss_terms=-2*(1-raw_p)*raw+raw*raw_q
                    loss_ci=interval(counts@loss_terms/len(dates))
                    d=data[data.cell.eq(cell)]
                    cov=12*(2*np.cov(previous,r,ddof=1)[0,1]+np.cov(group_series,r,ddof=1)[0,1])
                    attribution.append(dict(period=label,group=group,partition=partition,cell=cell,
                        annual_mean=12*r.mean(),mean_low=cis[0,j],mean_high=cis[1,j],
                        annual_long_payoff=12*d.long_payoff.mean(),annual_short_payoff=12*d.short_payoff.mean(),
                        mean_long_notional=d.long_notional.mean(),mean_short_notional=d.short_notional.mean(),
                        allocated_incremental_loss=loss_terms.mean(),loss_low=loss_ci[0],loss_high=loss_ci[1],
                        allocated_incremental_variance=cov))
                    beta,se,r2=hac_regression(r,f)
                    for name,value,error in zip(['alpha_monthly']+FACTORS,beta,se):
                        factor_rows.append(dict(period=label,group=group,partition=partition,cell=cell,factor=name,
                            coefficient=value,se=error,low=value-1.96*error,high=value+1.96*error,r_squared=r2))
            for (partition,cell),d in pp[pp.group.eq(group)].groupby(['partition','cell']):
                joint_summary.append(dict(period=label,group=group,partition=partition,cell=cell,
                    weight_density=d.relative_weight_density.mean(),additive_density=d.additive_density.mean(),
                    nonadditive_density=d.nonadditive_density.mean()))
        print('Economic summaries '+label,flush=True)
    summary=save('family_exposures',summaries);save('dominant_characteristics',dominant);save('group_holdings_summary',balance_summary)
    save('cell_return_attribution',attribution);save('factor_attribution',factor_rows);save('joint_weight_profiles',joint_summary)
    for group in range(1,5):
        s=summary[(summary.group==group)&(summary.period!='all')].pivot(index='period',columns='family',values='long_minus_short')
        for a in s.index:
            for b in s.index:
                x,y=s.loc[a].to_numpy(),s.loc[b].to_numpy()
                stability.append(dict(group=group,period_a=a,period_b=b,cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
                    sign_agreement=float(np.mean(np.sign(x)==np.sign(y)))))
    save('economic_stability',stability)
    # Exact additive identities: mean, full-sample OLS slopes, risk and response-one loss allocations.
    att=pd.DataFrame(attribution);fac=pd.DataFrame(factor_rows)
    old=pd.read_csv(BASE/'tables/e3_nested_summary.csv')
    max_loss,max_var,max_factor=0.,0.,0.
    for (period,group,partition),part in att.groupby(['period','group','partition']):
        r=old[(old.period==period)&(old.group==group)].iloc[0]
        max_loss=max(max_loss,abs(part.allocated_incremental_loss.sum()-r.delta_loss))
        max_var=max(max_var,abs(part.allocated_incremental_variance.sum()-r.delta_variance))
        f=fac[(fac.period==period)&(fac.group==group)]
        summed=f[f.partition.eq(partition)].groupby('factor').coefficient.sum()
        target=f[f.partition.eq('group')].set_index('factor').coefficient
        max_factor=max(max_factor,float(np.max(np.abs(summed-target))))
    if max(max_loss,max_var,max_factor)>1e-7:raise ValueError('Economic attribution identity failed')
    (OUT/'audit/attribution_checks.json').write_text(json.dumps(dict(passed=True,max_loss_allocation_error=max_loss,
        max_variance_allocation_error=max_var,max_factor_additivity_error=max_factor,source_sha256=digest(Path(__file__)),
        files={p.name:digest(p) for p in (OUT/'tables').iterdir() if p.is_file()}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['holdings','summaries','all'],default='all')
    args=parser.parse_args()
    if args.phase in ('holdings','all'):reconstruct()
    if args.phase in ('summaries','all'):summarize()
