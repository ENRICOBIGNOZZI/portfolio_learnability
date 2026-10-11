"""Finalize only Section VI from verified paths, holdings and fixed Gaussian cache.

Private-data reproduction: python3 -m empirical_final.section_vi_finalize
Public reproduction: python3 -m empirical_final.section_vi_finalize --public-only
Figure 6 only: python3 -m empirical_final.section_vi_finalize --figure-six-only
No theory-guided experiments or manuscript sections are read or changed.
"""
import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_pipeline import digest
from kernels import FeatureBank
from portfolio import annual_splits, dual_path, sharpe
from empirical_final.run import ROOT, bind, clean_manifest, load_managed, public
from empirical_final.compact_common import account_month, summarize_accounts, SCALE
from empirical_final.three_experiments import block_counts, boot_metrics, interval

BASE = ROOT/'outputs/empirical_compact_20261010'
OUT = ROOT/'outputs/section_vi_final_20261011'
BLOCKS = [(1978,1982),(1983,1987),(1988,1992),(1993,1997),(1998,2002),
          (2003,2007),(2008,2012),(2013,2017),(2018,2024)]
SCENARIOS = [('gross',0,0),('trade25',25,0),('trade25_borrow30',25,30)]
INPUTS = {}
LABELS = {'linear':'Linear', 'gaussian_tuned':'Gaussian (joint validation)',
          'matern32_tuned':'Matérn-3/2 (joint validation)'}
COLORS = ['#60656b','#236d9f','#b46a36']


def frozen(path, expected=None):
    path=Path(path); actual=digest(path)
    if expected is not None and actual!=expected: raise ValueError(f'Changed input: {path}')
    INPUTS[str(path.relative_to(ROOT))]=actual
    return path


def read(name): return pd.read_csv(OUT/'tables'/f'{name}.csv')
def save(name, frame):
    frame=pd.DataFrame(frame)
    frame.to_csv(OUT/'tables'/f'{name}.csv',index=False,float_format='%.15g')
    return frame
def audit(name, obj):
    (OUT/'audit'/f'{name}.json').write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')


def representation_accounts():
    """Verify saved joint selections, all stock payoffs, drift and costs."""
    clean=clean_manifest(); frozen(ROOT/'data/clean/manifest.json')
    frozen(ROOT/'empirical_final/compact_protocol.json')
    export=ROOT/'outputs/bandwidth_tuning'
    export_manifest=json.loads(frozen(export/'status.json').read_text())
    assert export_manifest['status']=='complete'
    for name,expected in export_manifest['outputs_sha256'].items():
        frozen(export/name,expected)
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    rf=pd.read_csv(frozen(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    accounts=pd.read_csv(frozen(BASE/'tables/monthly_accounting.csv'))
    rows=[accounts[accounts.policy.eq('linear')]];checks=[];selections=[];paths=[]
    for kernel in ['gaussian','matern32']:
        cache=ROOT/f'results/bandwidth_tuning/{kernel}/cache'
        meta=json.loads(frozen(cache/'progress.json').read_text()); identity=meta['identity']
        assert meta['status']=='complete' and identity['clean_manifest_sha256']==digest(ROOT/'data/clean/manifest.json')
        g=np.load(frozen(cache/'managed.npy',meta['sha256']),mmap_mode='r')
        source=json.loads(frozen(ROOT/f'results/final/public/{kernel}/seed_0/source.json').read_text())
        assert identity['feature_bank']==source['feature_bank'] and pd.DatetimeIndex(identity['dates']).equals(dates)
        folder=ROOT/f'outputs/bandwidth_tuning/{kernel}'
        status=json.loads(frozen(folder/'status.json').read_text())
        # The completed source export has its own frozen file hashes.
        for name,expected in status.get('files',{}).items(): frozen(folder/name,expected)
        selection=pd.read_csv(frozen(folder/'selections.csv'))
        monthly=pd.read_csv(frozen(folder/'monthly.csv'),parse_dates=['formation_date','return_date'])
        surface=pd.read_csv(frozen(folder/'validation_surface.csv'))
        chosen=selection[selection['mode'].eq('tuned')].set_index('year')
        assert len(chosen)==47
        for year, choice in chosen.iterrows():
            v=surface[surface.year.eq(year)]
            selected=v[v.joint_selected]
            assert len(selected)==1 and selected.validation_loss.iloc[0]==v.validation_loss.min()
            np.testing.assert_allclose([choice['lambda'],choice.bandwidth_multiplier],
                selected[['lambda','bandwidth_multiplier']].iloc[0].to_numpy(),rtol=1e-12)
        selections.append(selection.assign(representation=kernel));paths.append(monthly.assign(representation=kernel))
        states={sc:pd.Series(dtype=float) for sc,_,_ in SCENARIOS}; previous=pd.Series(dtype=float)
        account=[]
        for year,train,val,test in annual_splits(dates):
            choice=chosen.loc[year]; b=identity['multipliers'].index(choice.bandwidth_multiplier);h=np.r_[train,val]
            gg=np.asarray(g[b]);alpha,_,_=dual_path(gg[h]@gg[h].T,[choice['lambda']]);coef=gg[h].T@alpha[:,0]
            ref=monthly[monthly['mode'].eq('tuned') & monthly.year.eq(year)]
            np.testing.assert_allclose(gg[test]@coef,ref.raw_excess_return,rtol=2e-7,atol=2e-9)
            item=next(i for i in clean['files'] if i['name']==f'jkp_{year}.parquet')
            data=pd.read_parquet(frozen(ROOT/'data/clean'/item['name'],item['sha256']))
            wp=ROOT/f'results/empirical_compact_20261010/bandwidth_{kernel}_weights_{year}.parquet'
            cached=pd.read_parquet(frozen(wp)) if wp.exists() else None
            bank=FeatureBank(kernel,130,10000,choice.bandwidth,0)
            max_payoff_error=max_sample_error=0.
            for date,f in data.groupby('eom',sort=True):
                retdate=date+pd.offsets.MonthEnd(1);x=f[clean['characteristics']].to_numpy()
                if cached is None: w=SCALE*bank.scores(x,coef)/len(f)
                else:
                    cw=cached[cached.formation_date.eq(date)].set_index('id')
                    assert set(cw.index)==set(f.id)
                    w=cw.loc[f.id].weight.to_numpy()
                    # Spread checks through the whole cross section, not just its first names.
                    sample=np.unique(np.linspace(0,len(f)-1,32,dtype=int))
                    expected=SCALE*bank.scores(x[sample],coef)/len(f)
                    np.testing.assert_allclose(w[sample],expected,rtol=2e-6,atol=2e-9)
                    max_sample_error=max(max_sample_error,float(np.max(np.abs(w[sample]-expected))))
                target=pd.Series(w,index=f.id);returns=pd.Series(f.r.to_numpy(),index=f.id)
                expected=SCALE*ref.set_index('formation_date').loc[date].raw_excess_return
                np.testing.assert_allclose(target@returns,expected,rtol=2e-6,atol=2e-8)
                max_payoff_error=max(max_payoff_error,abs(float(target@returns)-expected))
                for sc,tb,bb in SCENARIOS:
                    step,states[sc]=account_month(target,returns,float(rf.loc[retdate]),states[sc],previous,tb,bb)
                    account.append(dict(policy=kernel+'_tuned',scenario=sc,decision_year=year,
                        formation_date=date,return_date=retdate,**step))
                previous=target
            checks.append(dict(policy=kernel+'_tuned',year=year,payoff_error=max_payoff_error,weight_sample_error=max_sample_error))
            if year%5==0: print(f'Verified {kernel} tuned stock costs through {year}',flush=True)
        rows.append(pd.DataFrame(account))
        if kernel=='gaussian':
            existing=pd.read_csv(frozen(BASE/'tables/bandwidth_gaussian_accounting.csv'))
            np.testing.assert_allclose(pd.DataFrame(account)[['excess_return','total_return','turnover']],
                existing[['excess_return','total_return','turnover']],rtol=2e-7,atol=2e-9)
    combined=save('representation_accounts',pd.concat(rows))
    combined['return_date']=pd.to_datetime(combined.return_date)
    combined['formation_date']=pd.to_datetime(combined.formation_date)
    # All scenarios and all policies must have exactly the same ordered months.
    expected=pd.date_range('1978-02-28','2025-01-31',freq='ME')
    for _,f in combined.groupby(['policy','scenario']):
        assert pd.DatetimeIndex(pd.to_datetime(f.sort_values('return_date').return_date)).equals(expected)
    save('representation_performance',summarize_accounts(combined))
    save('joint_selections',pd.concat(selections));save('joint_monthly_raw',pd.concat(paths));save('stock_cost_checks',checks)
    contrasts=[]
    for scenario in ['gross','trade25_borrow30']:
        p=combined[combined.scenario.eq(scenario)].pivot(index='return_date',columns='policy',values='excess_return').sort_index()
        for length in [6,12,24]:
            boot=boot_metrics(p[list(LABELS)].to_numpy(),block_counts(len(p),length,20261010,5000))['sharpe']
            for j,k in enumerate(list(LABELS)[1:],1):
                ci=interval(boot[:,j]-boot[:,0])
                contrasts.append(dict(scenario=scenario,policy=k,block_length=length,
                    delta_sharpe=float(sharpe(p[k])-sharpe(p.linear)),low=ci[0],high=ci[1]))
    save('representation_paired',contrasts)


def ridge_evaluation(gram, cross, penalties):
    """Symmetric eigensystem pseudoinverse, shared rank at all penalties.

    The cutoff is on Gram eigenvalues (1e-12 relative), matching the existing
    fixed representation's reported effective rank. No inverse is formed.
    """
    t=len(gram);values,u=np.linalg.eigh((gram+gram.T)/2)
    if values.min() < -values.max()*1e-10: raise ValueError('Indefinite Gram')
    active=values>values.max()*1e-12; values=values[active];u=u[:,active]
    rhs=u.T@np.ones(t);factors=rhs[:,None]/(values[:,None]+t*penalties[None,:])
    alpha=u@factors
    returns=cross@alpha
    C=np.sum(values[:,None]/(values[:,None]+t*penalties[None,:]),axis=0)
    history=u@(values[:,None]*factors)
    return returns,C,history,alpha[:,-1],len(values),values.min()/t


def extend_profiles():
    clean=clean_manifest();g,dates=load_managed('gaussian',clean)
    # load_managed checks cache SHA, sample and feature-bank identity.
    meta=json.loads((ROOT/'results/bandwidth_tuning/gaussian/cache/progress.json').read_text())
    frozen(ROOT/'results/bandwidth_tuning/gaussian/cache/managed.npy',meta['sha256'])
    _,manifest=public('gaussian');original=np.array(manifest['lambda_grid'])
    initial=np.linalg.eigvalsh(np.asarray(g[:120])@np.asarray(g[:120]).T)/120
    # Diagnostic extension frozen before any subsequent Sharpe is evaluated.
    positive=np.geomspace(max(original.max(),initial.max()*1e5),initial[initial>initial.max()*1e-12].min()*1e-14,1401)
    grid=np.r_[np.sort(np.unique(np.r_[original,positive]))[::-1],0.]
    audit('ridge_protocol',dict(penalties=grid.tolist(),rank_relative_cutoff=1e-12,
        selected_policy='Unchanged original 120-candidate past-validation policy; extension is retrospective diagnostics only',
        lower_penalty_rule='Initial historical smallest retained eigenvalue times 1e-14',
        representation=meta['identity']['feature_bank'],bootstrap_replicates=5000,bootstrap_seed=20261010))
    gram=np.asarray(g)@np.asarray(g).T
    old=pd.read_csv(frozen(BASE/'tables/e1_monthly_paths.csv'))
    annual_old=pd.read_csv(frozen(BASE/'tables/kernel_annual_paths.csv')).query("policy=='gaussian'")
    selected=pd.read_csv(frozen(BASE/'tables/kernel_selected_monthly.csv')).query("policy=='gaussian'")
    paths=[];annual=[];checks=[]
    for year,train,val,test in annual_splits(dates):
        h=np.r_[train,val];r,C,hist,zero,rank,mumin=ridge_evaluation(gram[np.ix_(h,h)],gram[np.ix_(test,h)],grid)
        original_indices=[int(np.argmin(abs(grid-v))) for v in original]
        assert np.array_equal(grid[original_indices],original)
        np.testing.assert_allclose(r[:,original_indices],old[old.decision_year.eq(year)].filter(like='lambda_'),rtol=2e-6,atol=2e-7)
        np.testing.assert_allclose(C[original_indices],annual_old[annual_old.decision_year.eq(year)].C,rtol=2e-6,atol=2e-7)
        assert C.max()<=rank+1e-8 and C[-1]==rank
        np.testing.assert_allclose(r[:,-2],r[:,-1],rtol=1e-7,atol=1e-8)
        # Direct SVD of G provides an independent feature-space minimum-norm check.
        # Compare all 47 endpoints via the symmetric Moore-Penrose residual;
        # feature-space SVD additionally verifies first and last histories.
        K=gram[np.ix_(h,h)]
        residual=float(np.linalg.norm(K@zero-np.ones(len(h)))/np.sqrt(len(h)))
        assert residual<1e-6
        primal_error=None
        if year in [1978,2024]:
            beta=np.linalg.lstsq(np.asarray(g[h]),np.ones(len(h)),rcond=1e-6)[0]
            primal_error=float(np.max(np.abs(np.asarray(g[test])@beta-r[:,-1])))
            np.testing.assert_allclose(np.asarray(g[test])@beta,r[:,-1],rtol=2e-6,atol=2e-7)
        choice=annual_old[(annual_old.decision_year.eq(year)) & annual_old.selected].iloc[0]
        # An interpolated response-one history is constant, so historical
        # Sharpe at exact zero is unbounded, not a finite rounding artefact.
        sd=hist.std(axis=0,ddof=1)
        srhist=np.divide(np.sqrt(12)*hist.mean(axis=0),sd,out=np.full(len(grid),np.nan),where=sd>1e-10)
        srhist[-1]=np.nan;sroos=sharpe(r)
        for j,penalty in enumerate(grid):
            annual.append(dict(decision_year=year,candidate=j,penalty=penalty,C=C[j],rank=rank,
                historical_sharpe=srhist[j],oos_sharpe=sroos[j],
                selected=bool(np.isclose(penalty,choice.penalty,rtol=1e-12,atol=0)),zero_penalty=penalty==0))
        paths.append(pd.DataFrame(r,columns=[f'lambda_{j:04}' for j in range(len(grid))]).assign(
            decision_year=year,return_date=dates[test]+pd.offsets.MonthEnd(1)))
        checks.append(dict(year=year,rank=rank,refit_months=len(h),minimum_eigenvalue=mumin,
            zero_training_relative_residual=residual,primal_endpoint_error=primal_error,
            near_zero_endpoint_error=float(np.max(np.abs(r[:,-2]-r[:,-1])))))
        if year%5==0:print(f'Extended fixed Gaussian profiles through {year}',flush=True)
    p=save('ridge_monthly_paths',pd.concat(paths));a=save('ridge_annual',annual);save('ridge_endpoint_checks',checks)
    curves=[];summary=[];ids=[];cols=[c for c in p if c.startswith('lambda_')]
    for lo,hi in BLOCKS:
        label=f'{lo}-{hi}';pp=p[p.decision_year.between(lo,hi)];aa=a[a.decision_year.between(lo,hi)]
        ss=selected[selected.decision_year.between(lo,hi)]
        r=pp[cols].to_numpy();sr=sharpe(r);peak=int(sr.argmax());counts=block_counts(len(pp),12,20261010,5000)
        ci=interval(boot_metrics(np.column_stack([r,ss.raw_return]),counts)['sharpe'])
        sizes=pp.groupby('decision_year').size()
        C=aa.assign(n=aa.decision_year.map(sizes)).groupby('candidate').apply(
            lambda f:np.average(f.C,weights=f.n),include_groups=False).to_numpy()
        rank=np.average(aa.groupby('decision_year')['rank'].first(),weights=sizes)
        assert C[-1]==rank and (C<=rank+1e-8).all()
        for j in range(len(grid)):
            curves.append(dict(period=label,candidate=j,penalty=grid[j],mean_C=C[j],oos_sharpe=sr[j],
                sr_low=ci[0,j],sr_high=ci[1,j],retrospective_maximum=j==peak,zero_penalty=j==len(grid)-1,
                months=len(pp),first_return=pp.return_date.min(),last_return=pp.return_date.max()))
        summary.append(dict(period=label,months=len(pp),first_return=pp.return_date.min(),last_return=pp.return_date.max(),
            selected_mean_C=ss.C.mean(),selected_sharpe=float(sharpe(ss.raw_return)),
            selected_low=ci[0,-1],selected_high=ci[1,-1],maximum_C=C[peak],maximum_sharpe=sr[peak],
            maximum_candidate=peak,boundary_maximum=peak in [0,len(grid)-1],
            effective_rank=rank,zero_sharpe=sr[-1],peak_to_zero_decline=sr[peak]-sr[-1]))
        ids.append(pp[['decision_year','return_date']].assign(period=label))
    save('nine_panel_curves',curves);save('nine_panel_summary',summary);save('nine_panel_month_identifiers',pd.concat(ids))
    save('single_window_complexity',a[a.decision_year.eq(2024)])


def retained_economics():
    for name in ['managed_spectra','e3_monthly','e3_annual_groups','spectral_value','paired_contrasts',
                 'economic_exposure_summary','monthly_economic_exposures','monthly_accounting',
                 'dominant_long_short_exposures_named','group_oos_covariance','stock_reconstruction_checks',
                 'formation_timing_by_month','group_factor_regressions','group_past_only_hedged_monthly']:
        shutil.copyfile(frozen(BASE/'tables'/f'{name}.csv'),OUT/'tables'/f'{name}.csv')
    shutil.copyfile(frozen(BASE/'audit/formation_timing_audit.json'),OUT/'audit/formation_timing_audit.json')
    # Reconcile existing nested account paths with every reported economic row.
    m=read('monthly_accounting'); perf=summarize_accounts(m)
    spectral=read('spectral_value')
    for row in spectral.itertuples():
        for sc,col in [('gross','cumulative_gross_sharpe'),('trade25_borrow30','cumulative_net_sharpe')]:
            value=perf[(perf.policy==f'nested_{row.group}') & (perf.scenario==sc)].sharpe.iloc[0]
            np.testing.assert_allclose(value,getattr(row,col),rtol=1e-10)
    group=read('e3_monthly').pivot(index='return_date',columns='group',values='raw_contribution').sort_index()
    contributions=[]
    previous=np.zeros(len(group))
    for j in range(1,5):
        added=group[j].to_numpy();new=previous+added
        cross=-2*np.mean((1-previous)*added);square=np.mean(added**2)
        delta=np.mean((1-new)**2)-np.mean((1-previous)**2)
        np.testing.assert_allclose(delta,cross+square,atol=1e-12)
        # Compute paired losses directly: null-payoff Sharpe is undefined and
        # irrelevant to this response-one loss contrast.
        counts=block_counts(len(group),12,20261010,5000)
        boot_loss=counts@np.column_stack([(1-new)**2,(1-previous)**2])/len(group)
        ci=interval(boot_loss[:,0]-boot_loss[:,1])
        contributions.append(dict(group=j,cumulative_Q=np.mean((1-new)**2),incremental_Q=delta,
            cross_term=cross,own_square=square,delta_Q_low=ci[0],delta_Q_high=ci[1]))
        previous=new
    save('incremental_spectral_loss',contributions)


def finish(fig,name):
    for ext in ['pdf','png']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def figure_six():
    """Draw fixed-penalty OOS paths; annual CV is a distinct dynamic policy."""
    curves=pd.read_csv(OUT/'tables/nine_panel_curves.csv',float_precision='round_trip')
    summary=read('nine_panel_summary').set_index('period')
    grid=np.array(json.loads((OUT/'audit/ridge_protocol.json').read_text())['penalties'])
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':13,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'savefig.facecolor':'white','grid.alpha':.22,'path.simplify':False}):
        fig,axs=plt.subplots(3,3,figsize=(10.5,8.7),sharex=True,sharey=True,layout='constrained')
        plotted=[]
        for ax,(lo,hi) in zip(axs.flat,BLOCKS):
            label=f'{lo}-{hi}';f=curves[curves.period.eq(label)]
            by_candidate=f.sort_values('candidate')
            assert np.array_equal(by_candidate.candidate,np.arange(len(grid)))
            np.testing.assert_allclose(by_candidate.penalty,grid,rtol=1e-13,atol=0)
            assert np.isfinite(f[['mean_C','oos_sharpe']].to_numpy()).all()
            # Retain every vertex, including nearly coincident near-zero policies.
            f=f.sort_values(['mean_C','candidate'],kind='stable');s=summary.loc[label]
            line,=ax.plot(f.mean_C,f.oos_sharpe,color=COLORS[1],lw=1.3)
            assert len(line.get_xydata())==len(grid)
            np.testing.assert_array_equal(line.get_xydata(),f[['mean_C','oos_sharpe']].to_numpy())
            peak=f.loc[f.oos_sharpe.idxmax()];end=f[f.zero_penalty].iloc[0]
            assert peak.retrospective_maximum and end.penalty==0 and end.mean_C==s.effective_rank
            assert int(peak.candidate)==int(s.maximum_candidate)
            np.testing.assert_allclose([peak.mean_C,peak.oos_sharpe],
                [s.maximum_C,s.maximum_sharpe],rtol=1e-12,atol=1e-12)
            ax.scatter(peak.mean_C,peak.oos_sharpe,color=COLORS[2],marker='*',s=105,zorder=5)
            ax.scatter(end.mean_C,end.oos_sharpe,facecolors='white',edgecolors=COLORS[1],
                s=32,zorder=6,clip_on=False)
            ax.set_title(f'{lo}–{hi} · n = {int(s.months)}',fontsize=15);ax.grid(True)
            ax.set_xticks([0,200,400,600]);ax.tick_params(labelsize=13)
            plotted.append(dict(period=label,vertices=len(line.get_xydata()),
                maximum_candidate=int(peak.candidate),zero_candidate=int(end.candidate),
                observed_complexity_endpoint=float(end.mean_C)))
        for ax in axs[2]:ax.set_xlabel('Mean effective complexity',fontsize=13)
        for ax in axs[:,0]:ax.set_ylabel('OOS Sharpe',fontsize=13)
        xmax=float(summary.effective_rank.max())
        ymin=float(min(0,curves.oos_sharpe.min(),summary.selected_sharpe.min()))-.12
        ymax=float(max(curves.oos_sharpe.max(),summary.selected_sharpe.max()))+.25
        axs[0,0].set_xlim(0,xmax);axs[0,0].set_ylim(ymin,ymax)
        handles=[plt.Line2D([0],[0],color=COLORS[1],lw=1.3,label='Evaluated λ path'),
            plt.Line2D([0],[0],color=COLORS[2],marker='*',lw=0,ms=11,label='Retrospective maximum'),
            plt.Line2D([0],[0],color=COLORS[1],marker='o',mfc='white',lw=0,ms=5.5,label='Exact λ = 0')]
        fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,fontsize=12)
        assert len(fig.axes)==9 and len(fig.legends)==1
        assert all(np.array_equal(ax.get_xlim(),[0,xmax]) for ax in axs.flat)
        assert all(np.array_equal(ax.get_ylim(),[ymin,ymax]) for ax in axs.flat)
        finish(fig,'fig06_nine_complete_profiles')
    audit('figure6_render',dict(passed=True,panels=plotted,policies_per_panel=len(grid),
        common_xlim=[0,xmax],common_ylim=[ymin,ymax],confidence_bands=False,
        historical_fit_panels=False,all_evaluated_vertices_retained=True,
        path_simplification=False,smoothing=False,extrapolation=False,
        cv_strategy_plotted=False,markers_per_panel=['retrospective_maximum','exact_zero_endpoint'],
        legend=[handle.get_label() for handle in handles],
        curves_sha256=digest(OUT/'tables/nine_panel_curves.csv')))


def figures():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white','grid.alpha':.25})
    m=read('representation_accounts');wealth=[]
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    for ax,sc,title in zip(axs,['gross','trade25_borrow30'],['Gross','Net: 25 bps trades + 30 bps/year shorts']):
        for (policy,label),color in zip(LABELS.items(),COLORS):
            f=m[(m.policy==policy)&(m.scenario==sc)].sort_values('return_date')
            w=np.cumprod(1+f.total_return);dates=pd.to_datetime(f.return_date)
            ax.plot(pd.DatetimeIndex([pd.Timestamp('1978-01-31'),*dates]),np.r_[1,w],color=color,label=label,lw=1.7)
            wealth.append(pd.DataFrame(dict(policy=policy,scenario=sc,return_date=dates,total_wealth=w)))
        ax.set_yscale('log');ax.set_title(title,fontsize=10);ax.set_xlabel('Payoff date');ax.grid(True)
    axs[0].set_ylabel('Total wealth, initial NAV = 1');axs[0].legend(frameon=False,fontsize=8.5)
    finish(fig,'fig03_representation_wealth');save('figure_wealth',pd.concat(wealth))
    s=read('managed_spectra').query("decision_year==2024 and policy=='gaussian'").copy()
    s=s[s.eigenvalue>s.eigenvalue.max()*1e-12];s['mass_share']=s.eigenvalue/s.eigenvalue.sum()
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.7),layout='constrained')
    for ax,col,title in zip(axs,['eigenvalue','mass_share'],['Raw managed-payoff units','Share of second-moment mass']):
        ax.loglog(s['rank'],s[col],color=COLORS[1],lw=1.8);ax.set_title(title);ax.set_xlabel('Positive spectral rank');ax.grid(True)
    axs[0].set_ylabel('Uncentered second-moment eigenvalue');axs[1].set_ylabel('Eigenvalue / trace')
    finish(fig,'fig04_fixed_gaussian_spectrum');save('figure_spectrum',s)
    c=read('single_window_complexity').sort_values('C');peak=c.loc[c.oos_sharpe.idxmax()]
    # Match selected penalty numerically across public CSV roundtrips.
    original=read('monthly_accounting').query("policy=='gaussian' and decision_year==2024 and scenario=='gross'")
    selected=c[c.selected].iloc[0]
    fig,axs=plt.subplots(2,1,figsize=(9.5,5.8),sharex=True,layout='constrained')
    for ax,col,title in zip(axs,['historical_sharpe','oos_sharpe'],['January 2024 historical fit','February 2024–January 2025: 12 OOS months']):
        ax.plot(c.C,c[col],color=COLORS[1],lw=1.7);ax.set_ylabel('Sharpe');ax.set_title(title);ax.grid(True)
        endpoint=c[c.zero_penalty].iloc[0]
        if col=='oos_sharpe':ax.scatter(endpoint.C,endpoint[col],facecolors='white',edgecolors=COLORS[1],s=38,zorder=5,clip_on=False)
    axs[0].set_yscale('log');axs[0].set_ylabel('Historical Sharpe (log)')
    axs[0].text(.02,.90,'At λ = 0: constant historical payoff, unbounded Sharpe',transform=axs[0].transAxes,fontsize=11)
    axs[1].scatter(peak.C,peak.oos_sharpe,marker='*',s=130,color=COLORS[2],label='Retrospective maximum',zorder=5)
    axs[1].scatter(selected.C,sharpe(original.excess_return),marker='D',s=40,color='black',label='Past-validation selection',zorder=5)
    axs[1].set_xlabel('Effective complexity, fixed Gaussian');axs[1].set_xlim(0,c['rank'].iloc[0]);axs[1].legend(frameon=False,fontsize=9)
    finish(fig,'fig05_fixed_gaussian_2024')
    figure_six()
    e=read('economic_exposure_summary');names=['Value','Profitability','Investment','Momentum','Recent return','Size','Liquidity','Issuance','Debt financing']
    fig,axs=plt.subplots(1,2,figsize=(10,4.7),sharey=True,layout='constrained')
    for ax,col,title in zip(axs,['long_minus_short','signed_notional'],['Long-minus-short rank profile','Signed notional rank exposure']):
        values=e.pivot(index='family',columns='group',values=col).loc[names].to_numpy();v=np.abs(values).max()
        im=ax.imshow(values,cmap='RdBu_r',vmin=-v,vmax=v,aspect='auto');ax.set_title(title,fontsize=10)
        ax.set_xticks(range(4),['0–10%','10–50%','50–90%','90–100%'],fontsize=9);ax.set_yticks(range(len(names)),names,fontsize=10)
        ax.set_xlabel('Positive-rank group')
        for i in range(len(names)):
            for j in range(4):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=9,
                color='white' if abs(values[i,j])>.6*v else '#202020')
        fig.colorbar(im,ax=ax,shrink=.8,pad=.03)
    finish(fig,'fig07_fixed_gaussian_holdings')


def main():
    global OUT
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--public-only',action='store_true')
    modes.add_argument('--figure-six-only',action='store_true',help='Redraw Figure 6 and rebuild the section without changing results or other figures')
    parser.add_argument('--output-dir',type=Path,help='Use a versioned copy of the verified Section VI outputs')
    args=parser.parse_args()
    if args.output_dir:OUT=args.output_dir.resolve()
    for folder in ['tables','audit','figures','publication']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    if not (args.public_only or args.figure_six_only):
        audit('scope',dict(section='VI only',representation_comparison='Linear and joint-validation Gaussian/Matérn-3/2',
            mechanism='Existing fixed-lengthscale seed-0 Gaussian only',theory_modified=False,
            theory_guided_experiments_included=False,appendix=False))
        representation_accounts();extend_profiles();retained_economics()
        audit('inputs',INPUTS)
    else:
        manifest=json.loads((OUT/'audit/results_manifest.json').read_text())
        for name,expected in manifest['tables'].items():
            if digest(OUT/name)!=expected:raise ValueError('Changed public aggregate: '+name)
    if args.figure_six_only:figure_six()
    else:figures()
    from empirical_final import section_vi_publication
    section_vi_publication.OUT=OUT
    section_vi_publication.publication();section_vi_publication.verify()
    result_manifest=dict(tables={str(p.relative_to(OUT)):digest(p) for p in sorted((OUT/'tables').glob('*.csv'))},
        code={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'empirical_final/section_vi_publication.py',ROOT/'empirical_final/templates/final_section_vi.tex']},
        figures={p.name:digest(p) for p in sorted((OUT/'figures').glob('*.pdf'))})
    if args.figure_six_only:
        result_manifest['test_source_sha256']=manifest.get('test_source_sha256')
        result_manifest['delivered_pdf_sha256']=digest(OUT/'publication/Section_VI_Empirical_Portfolio_Learnability.pdf')
        result_manifest['delivered_latex_zip_sha256']=digest(OUT/'publication/Section_VI_Complete_LaTeX_Source.zip')
    audit('results_manifest',result_manifest)


if __name__=='__main__':main()
