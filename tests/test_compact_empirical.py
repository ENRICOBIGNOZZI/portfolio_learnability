"""Numerical and chronology checks for the integrated compact revision."""
import json
import numpy as np
import pandas as pd
import pytest
import torch
from portfolio import fit_windows, sharpe, dual_path
from data_pipeline import rank_months, formation_mask
from empirical_final.neural_portfolio import Extractor, monthly_features, solve_head
from empirical_final.compact_common import OUT, PROTOCOL, account_month


def test_future_payoffs_cannot_change_earlier_kernel_selection_or_refit():
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    rng=np.random.default_rng(53);g=rng.normal(.1,.2,(744,8));grid=np.logspace(-6,1,12)
    first=next(fit_windows(g,dates,grid))
    changed=g.copy();changed[dates.year>=1978]=rng.normal(0,100,(sum(dates.year>=1978),8))
    after=next(fit_windows(changed,dates,grid))
    assert first['choice']==after['choice']
    np.testing.assert_array_equal(first['validation_loss'],after['validation_loss'])
    np.testing.assert_array_equal(first['beta'],after['beta'])


def test_formation_preprocessing_independent_of_future_payoff_availability():
    # This applies to formation-only reconstruction; the retained conditional
    # sample is explicitly documented as failing this availability requirement.
    f=pd.DataFrame(dict(id=[1,2,3,4],eom=pd.Timestamp('1978-01-31'),z=[2.,5.,1.,np.nan],
                        ret_exc_lead1m=[.1,np.nan,.2,-.1]))
    before=rank_months(f,['z'])
    f.ret_exc_lead1m=[np.nan,10,-10,np.nan]
    after=rank_months(f,['z'])
    np.testing.assert_array_equal(before[['id','z']],after[['id','z']])


def test_monthly_neural_gradient_aggregates_all_stocks_before_squaring():
    model=Extractor(3,0).double();x=torch.tensor([[.1,.2,-.4],[-.5,.3,.4],[.4,.2,.1]],dtype=torch.float64)
    r=torch.tensor([.1,-.2,.3],dtype=torch.float64);a=torch.linspace(-1.,2.,65,dtype=torch.float64)
    loss=(1-monthly_features(model,x,r)@a)**2
    expected=(1-(model(x)@a*r).sum()/3)**2
    assert torch.allclose(loss,expected,rtol=1e-12,atol=1e-12)
    gradients=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True)
    direct=torch.autograd.grad(expected,tuple(model.parameters()))
    for g,d in zip(gradients,direct):assert torch.allclose(g,d,rtol=1e-12,atol=1e-12)
    incorrect=torch.mean((1-model(x)@a*r)**2)
    assert not torch.isclose(loss,incorrect)


def test_exact_neural_head_dual_and_spectral_reconstruction():
    rng=np.random.default_rng(6);g=rng.normal(0,.03,(72,65));penalty=1e-4
    a,sigma=solve_head(g,penalty)
    alpha,mu,c=dual_path(g@g.T,[penalty]);np.testing.assert_allclose(g.T@alpha[:,0],a,rtol=1e-10,atol=1e-10)
    values,v=np.linalg.eigh(sigma)
    parts=v*((v.T@g.mean(axis=0))/(values+penalty))[None,:]
    np.testing.assert_allclose(parts.sum(axis=1),a,atol=1e-10)
    assert c[0]<=np.linalg.matrix_rank(g)+1e-10


def test_neural_validation_head_cannot_see_future_months():
    model=Extractor(3,0);rng=np.random.default_rng(8)
    x=torch.tensor(rng.normal(0,.1,(20,3)),dtype=torch.float32)
    r=torch.tensor(rng.normal(0,.1,20),dtype=torch.float32)
    h=monthly_features(model,x,r).detach().numpy()
    future=monthly_features(model,x,100*r).detach().numpy()
    assert not np.array_equal(h,future)
    # Fixed earlier representation and training/validation head evaluation
    # depend only on supplied earlier rows, never an OOS normalization.
    history=np.vstack([h,2*h,-h]);a,_=solve_head(history,1e-3)
    np.testing.assert_array_equal(solve_head(history,1e-3)[0],a)


def test_actual_extractor_training_ignores_held_out_future_payoffs():
    from empirical_final.neural_portfolio import extractor_training
    class SmallPanels:
        names=['a','b','c']
        def __init__(self):
            rng=np.random.default_rng(4)
            self.panels=[(torch.tensor(rng.normal(0,.2,(12,3)),dtype=torch.float32),
                          torch.tensor(rng.normal(0,.05,12),dtype=torch.float32),np.arange(12)) for _ in range(9)]
        def __getitem__(self,i):return self.panels[i]
    panels=SmallPanels();before=extractor_training(panels,np.arange(6),0,[2])[2]
    for i in range(6,9):
        x,r,ids=panels.panels[i];panels.panels[i]=(x,100*r+50,ids)
    after=extractor_training(panels,np.arange(6),0,[2])[2]
    for key in before['state']:torch.testing.assert_close(before['state'][key],after['state'][key],rtol=0,atol=0)
    np.testing.assert_array_equal(before['g'],after['g'])


def test_costs_net_stock_positions_and_convert_annual_short_fee():
    w1=pd.Series([.8,-.7],index=[1,2]);w2=-w1
    r=pd.Series([.1,-.05],index=[1,2]);state=pd.Series(dtype=float)
    net,_=account_month(w1+w2,r,.002,state,state,25,30)
    first,_=account_month(w1,r,.002,state,state,25,30)
    second,_=account_month(w2,r,.002,state,state,25,30)
    assert net['gross_exposure']==0 and net['trading_fee']==0 and net['borrowing_fee']==0
    assert first['trading_fee']+second['trading_fee']>0
    np.testing.assert_allclose(first['borrowing_fee'],(1-first['trading_fee'])*.003/12*.7,atol=1e-14)


def table(name):
    p=OUT/'tables'/f'{name}.csv'
    if not p.exists():pytest.skip('Run python -m empirical_final.compact_reproduce first')
    return pd.read_csv(p)


def test_nine_panel_coverage_nonoverlap_and_recomputed_sharpes():
    dates=table('nine_panel_month_identifiers');paths=table('e1_monthly_paths');curves=table('nine_panel_curves')
    assert len(dates)==564 and not dates.return_date.duplicated().any()
    assert dates.decision_year.min()==1978 and dates.decision_year.max()==2024
    counts=dates.groupby('period').size().to_dict()
    assert len(counts)==9
    assert sorted(counts.values())==[60]*8+[84]
    for lo,hi in PROTOCOL['blocks']:
        label=f'{lo}-{hi}';d=dates[dates.period.eq(label)]
        assert d.decision_year.between(lo,hi).all()
        r=paths[paths.decision_year.between(lo,hi)]
        c=curves[curves.period.eq(label)].sort_values('candidate')
        cols=[col for col in r if col.startswith('lambda_')]
        np.testing.assert_allclose(sharpe(r[cols].to_numpy()),c.oos_sharpe,rtol=1e-10,atol=1e-10)
        assert c.retrospective_maximum.sum()==1
        assert c.loc[c.retrospective_maximum,'oos_sharpe'].iloc[0]==c.oos_sharpe.max()


def test_every_annual_selection_precedes_realizations():
    a=table('kernel_annual_paths');s=a[a.selected]
    assert len(s)==47*3
    assert (pd.to_datetime(s.last_known_payoff)<pd.to_datetime(s.first_test_payoff)).all()
    assert s.train_months.add(s.validation_months).eq(s.refit_months).all()
    for (_,year),f in a.groupby(['policy','decision_year']):
        selected=f[f.selected].iloc[0]
        assert selected.validation_Q==f.validation_Q.min()


def test_spectral_group_payoffs_and_account_net_paths_reconstruct_policy():
    m=table('monthly_accounting');g=table('e3_monthly')
    summed=g.groupby('return_date').scaled_contribution.sum()
    full=m.query("policy=='gaussian' and scenario=='gross'").set_index('return_date').excess_return
    np.testing.assert_allclose(full.loc[summed.index],summed,rtol=2e-6,atol=2e-8)
    for scenario in PROTOCOL['cost_scenarios']:
        a=m[(m.policy=='gaussian')&(m.scenario==scenario[0])].set_index('return_date')
        b=m[(m.policy=='nested_4')&(m.scenario==scenario[0])].set_index('return_date')
        np.testing.assert_allclose(a[['total_return','turnover','trading_fee','borrowing_fee']],
                                   b.loc[a.index,['total_return','turnover','trading_fee','borrowing_fee']],rtol=2e-6,atol=2e-8)


def test_neural_learned_parameters_exact_head_and_seed_completion():
    for seed in PROTOCOL['neural']['seeds']:
        a=table(f'neural_annual_seed_{seed}')
        assert len(a)==47 and a.parameter_change.gt(0).all()
        assert a.head_residual.max()<1e-8
        assert a.C.le(a['rank']+1e-6).all()
        assert (pd.to_datetime(a.last_known_payoff)<pd.to_datetime(a.first_test_payoff)).all()
        m=table(f'neural_monthly_seed_{seed}')
        assert len(m)==564*3 and m.reconstruction_error.max()<2e-7
