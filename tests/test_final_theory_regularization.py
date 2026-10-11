"""Math, adversarial timing, holdings and exported-study reconciliation."""
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from complexity_schedule import spectral_fit, calibrate
from empirical_final.final_theory_regularization import OUT,ROOT,PRIVATE,slope,penalty,tie_choice,decompose
from empirical_final.core import accounting_step
from kernels import FeatureBank
from portfolio import annual_splits,dual_path,ridge_path,sharpe


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):yield


def test_unclipped_rank_estimator_matches_original_and_threshold():
    mu=np.r_[3*np.arange(1,121,dtype=float)**-1.7,0,1e-14]
    actual=slope(mu);original=spectral_fit(mu)
    for key in original:assert actual[key]==pytest.approx(original[key])
    assert actual['positive_rank']==120 and actual['rank_low']==12 and actual['rank_high']==72
    invalid=slope(np.arange(1,121,dtype=float)**-.9)
    assert invalid['b']==pytest.approx(.9) and not invalid['admissible']
    with pytest.raises(ValueError):penalty(1e-6,180,invalid['b'])
    with pytest.raises(ValueError):spectral_fit(np.arange(1,121,dtype=float)**-.9)


def test_scaling_calibration_and_exact_three_component_decomposition():
    c0,b0,T=2e-6,1.3,240
    lam0=penalty(c0,120,b0)
    assert lam0*120**(b0/(b0+1))==pytest.approx(c0)
    assert penalty(c0,2*T,b0)/penalty(c0,T,b0)==pytest.approx(2**(-b0/(b0+1)))
    for c,b_init,b in ((c0,b0,b0),(c0,b0,1.4),(3e-6,1.6,1.7)):
        result=decompose(c,b_init,b,T,c0,b0,lam0)
        assert result['log_lambda_relative_initial']==pytest.approx(sum(result[k] for k in (
            'log_sample_size_contribution','log_representation_contribution','log_b_update_contribution')))


def test_ties_and_candidate_exclusions_do_not_clip_or_replace():
    assert tie_choice([1]*5,range(5))==2
    assert tie_choice([1]*5,[0,1,3,4])==1
    assert tie_choice([0,.1,.2,.3,.4],[2,3,4])==2
    with pytest.raises(ValueError):tie_choice([1]*5,[])


def test_primal_dual_predictions_and_spectrum_match():
    rng=np.random.default_rng(7);g=rng.normal(size=(24,13));test=rng.normal(size=(8,13));lam=.02
    beta,_,C=ridge_path(g,[lam]);alpha,mu,CD=dual_path(g@g.T,[lam])
    np.testing.assert_allclose(test@beta,(test@g.T)@alpha,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(C,CD,rtol=1e-10)
    np.testing.assert_allclose(mu[:13],np.linalg.eigvalsh(g.T@g/24)[::-1],atol=1e-12)


def test_real_initial_calibration_has_no_oos_information():
    g=np.load(ROOT/'results/schedule_cache/matern32_managed.npy',mmap_mode='r')
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    original=calibrate(g,dates);changed=np.array(g);changed[180:]*=-19
    alternate=calibrate(changed,dates)
    assert original['choice']==alternate['choice']
    for key in ('c0','lambda','loss'):np.testing.assert_array_equal(original[key],alternate[key])


@pytest.mark.parametrize('decision',[1978,1999,2024])
def test_real_inner_spectrum_and_lambda_ignore_validation_and_test(decision):
    g=np.load(ROOT/'results/schedule_cache/matern32_managed.npy',mmap_mode='r')
    dates=pd.date_range('1963-01-31','2024-12-31',freq='ME')
    split=next(s for s in annual_splits(dates) if s[0]==decision);_,tr,va,te=split
    changed=np.array(g);changed[np.r_[va,te]]*=17
    for data in (g,changed):
        _,mu,_=dual_path(data[tr]@data[tr].T,[1e-7]);fit=slope(mu)
        value=penalty(2.2657790177294398e-6,len(tr),fit['b'])
        if data is g:expected=(fit,value)
        else:assert (fit,value)==expected
    h=np.r_[tr,va];changed=np.array(g);changed[te]*=-23
    np.testing.assert_array_equal(g[h]@g[h].T,changed[h]@changed[h].T)


def test_frozen_constants_correct_dates_admissibility_and_annual_slices():
    c=pd.read_csv(OUT/'tables/frozen_constants.csv');paths=pd.read_csv(OUT/'tables/annual_policies.csv')
    candidates=pd.read_csv(OUT/'tables/candidate_losses.csv',parse_dates=['inner_last_known_payoff','validation_first_formation','validation_last_formation','decision_date'])
    assert c.training_months.eq(120).all() and c.validation_months.eq(60).all()
    assert c.admissible.tolist()==[False,False,True,True,True]
    for row in paths.itertuples():
        const=c[c.candidate.eq(row.candidate)].iloc[0]
        assert row.C0==pytest.approx(const.C0,rel=1e-13)
        if row.method in ('A','B','C','D'):
            assert row.penalty==pytest.approx(penalty(row.C0,row.T,row.schedule_b),rel=1e-13)
            assert const.admissible and row.schedule_b>1
    a=paths[paths.method.eq('A')].reset_index(drop=True);b=paths[paths.method.eq('B')].reset_index(drop=True)
    np.testing.assert_array_equal(a[['year','T','candidate','lengthscale']],b[['year','T','candidate','lengthscale']])
    assert a.schedule_b.nunique()==1 and a.C0.nunique()==1 and b.C0.nunique()==1
    assert (candidates.inner_last_known_payoff<=candidates.validation_first_formation).all()
    assert (candidates.validation_last_formation<candidates.decision_date).all()
    assert (candidates.full_T-candidates.inner_T).eq(60).all()
    assert not candidates[candidates.candidate.isin([0,1])].D_eligible.any()
    for year,d in candidates.groupby('year'):
        for method,col in (('C','C_validation_loss'),('D','D_validation_loss')):
            eligible=d[d[f'{method}_eligible']]
            expected=tie_choice(d.sort_values('candidate')[col].tolist(),eligible.candidate.tolist())
            actual=paths[paths.method.eq(method)&paths.year.eq(year)].iloc[0]
            assert actual.candidate==expected or actual.full_refit_fallback


def test_all_baselines_and_dates_reproduced_on_common_sample():
    m=pd.read_csv(OUT/'tables/monthly_payoffs.csv',parse_dates=['return_date']);base=pd.read_csv(ROOT/'paper/complexity_schedule/selected_monthly.csv')
    groups=[d.sort_values('return_date') for _,d in m.groupby('method')]
    for d in groups:
        assert len(d)==564 and d.return_date.is_unique
        np.testing.assert_array_equal(d.return_date,groups[0].return_date)
    b=m[m.method.eq('B')].sort_values('return_date')
    np.testing.assert_allclose(b.raw_return,base.raw_excess_return,rtol=1e-7,atol=1e-9)
    assert sharpe(b.raw_return)==pytest.approx(3.56133432162478,rel=1e-9)
    assert sharpe(m[m.method.eq('annual_cv')].raw_return)==pytest.approx(3.460982160688556,rel=1e-7)
    assert sharpe(m[m.method.eq('joint_cv')].raw_return)==pytest.approx(3.58686631197,rel=1e-7)


def test_exported_counterfactual_penalties_and_cumulative_statistics():
    d=pd.read_csv(OUT/'tables/lambda_decomposition.csv')
    c=pd.read_csv(OUT/'tables/frozen_constants.csv').iloc[2]
    np.testing.assert_allclose(d.selected_lambda,c.initial_lambda*d.sample_size_factor*d.representation_factor*d.b_update_factor,rtol=1e-12)
    np.testing.assert_allclose(d[d.method.eq('A')].b_update_factor,1,rtol=1e-12)
    for method in ('A','B','C','D'):
        m=pd.read_csv(OUT/'tables/monthly_accounts.csv')
        q=m[m.method.eq(method)&m.scenario.eq('trade25_borrow30')]
        r=pd.read_csv(OUT/'tables/cumulative_performance.csv')
        last=r[r.method.eq(method)&r.scenario.eq('trade25_borrow30')].iloc[-1]
        assert last.expanding_mean_leverage==pytest.approx(q.gross_exposure.mean(),rel=1e-12)
        assert last.expanding_mean_monthly_turnover==pytest.approx(q.turnover.mean(),rel=1e-12)
        assert last.expanding_sharpe==pytest.approx(float(sharpe(q.excess_return)),rel=1e-12)


def test_actual_saved_holdings_reconcile_existing_accounting_step():
    accounts=pd.read_csv(OUT/'tables/monthly_accounts.csv',parse_dates=['formation_date'])
    holdings=pd.read_parquet(PRIVATE/'holdings_1978.parquet');data=pd.read_parquet(ROOT/'data/clean/jkp_1978.parquet')
    date=pd.Timestamp('1978-01-31');h=holdings[holdings.formation_date.eq(date)].set_index('id')
    ret=data[data.eom.eq(date)].set_index('id').reindex(h.index).r.to_numpy()
    for method in ('A','B','C','D','annual_cv','joint_cv','fixed_lambda','one_anchor'):
        row=accounts[accounts.method.eq(method)&accounts.formation_date.eq(date)&accounts.scenario.eq('trade25_borrow30')].iloc[0]
        target=h[method].to_numpy();step=accounting_step(target,ret,row.rf,np.zeros(len(target)),.0025,.003)
        assert row.excess_return==pytest.approx(step['excess_return'],abs=1e-12)
        assert row.gross_exposure==pytest.approx(np.abs(target).sum(),abs=1e-12)
        assert row.turnover==pytest.approx(step['turnover'],abs=1e-12)
    costs=json.loads((OUT/'audit/costs.json').read_text());assert costs['original_cv_net_reproduced']
    assert costs['maximum_accounting_residual']<1e-12


def test_rolling_translation_and_published_paths_reconcile():
    c=pd.read_csv(OUT/'tables/rolling_constants.csv');p=pd.read_csv(OUT/'tables/rolling_monthly_diagnostics.csv',float_precision='round_trip')
    for row in c.itertuples():
        assert penalty(row.C0_W,row.W,row.anchor_b)==pytest.approx(row.anchor_lambda,rel=1e-12)
    assert p.admissible.all() and p.groupby(['W','rank_fraction_low','rank_fraction_high']).size().eq(493).all()
    assert len(p[p.rank_fraction_low.eq(.1)&p.rank_fraction_high.eq(.6)])==986
    assert json.loads((OUT/'audit/figure_inputs.json').read_text())['rolling_primary_rows']==986
    first=p.groupby(['W','rank_fraction_low','rank_fraction_high']).first()
    np.testing.assert_allclose(first.penalty,c.anchor_lambda.iloc[0],rtol=1e-12)
    fig=pd.read_csv(OUT/'tables/performance_figure_data.csv');acc=pd.read_csv(OUT/'tables/monthly_accounts.csv')
    for (method,scenario),d in fig.groupby(['method','scenario']):
        actual=acc[acc.method.eq(method)&acc.scenario.eq(scenario)&acc.year.ge(1979)].sort_values('return_date')
        np.testing.assert_allclose(d.sort_values('return_date').wealth,np.cumprod(1+actual.total_return),rtol=1e-12)
