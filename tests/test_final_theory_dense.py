"""Verify the exploratory extension shares data, draws, timing and accounts."""
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from empirical_final.final_theory_dense import DENSE,EXTRA,scores
from empirical_final.final_theory_regularization import ROOT,OUT,PRIVATE,penalty
from empirical_final.core import accounting_step
from kernels import FeatureBank
from portfolio import managed_matrix,sharpe
from data_pipeline import digest

DOUT=OUT/'dense_lengthscale'
DPRIVATE=PRIVATE/'dense_lengthscale'


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):yield


def test_dense_scores_use_original_shared_draws():
    bank=FeatureBank('matern32',3,10000,4.392107784616423,0)
    x=np.random.default_rng(4).normal(size=(8,3));beta=np.random.default_rng(3).normal(size=(10000,9))
    actual=scores(x,bank,beta,list(range(9)))
    for j,m in enumerate(DENSE):
        scaled=FeatureBank('matern32',3,10000,bank.ell*m,0)
        np.testing.assert_array_equal(bank.phases,scaled.phases)
        np.testing.assert_allclose(actual[:,j],scaled.features(x)@beta[:,j]/len(x),rtol=1e-9,atol=1e-11)


def test_real_new_training_cache_reconciles_independent_feature_reduction():
    meta=json.loads((DOUT/'audit/cache.json').read_text());clean=json.loads((ROOT/'data/clean/manifest.json').read_text())
    assert digest(DPRIVATE/'extra_managed.npy')==meta['sha256']
    g=np.load(DPRIVATE/'extra_managed.npy',mmap_mode='r')
    data=pd.read_parquet(ROOT/'data/clean/jkp_1963.parquet');data=data[data.eom.eq(pd.Timestamp('1963-01-31'))].sort_values('id')
    panel=dict(x=data[clean['characteristics']].to_numpy(float),r=data.r.to_numpy(float))
    for j,m in enumerate(EXTRA):
        bank=FeatureBank('matern32',130,10000,meta['feature_bank']['ell']*m,0)
        actual=managed_matrix([panel],bank)[0]
        np.testing.assert_allclose(actual,g[j,0],rtol=1e-8,atol=1e-13)


def test_dense_calibrations_preserve_originals_and_frozen_scaling():
    c=pd.read_csv(DOUT/'tables/frozen_constants.csv');original=pd.read_csv(OUT/'tables/frozen_constants.csv')
    assert c.training_months.eq(120).all() and c.validation_months.eq(60).all()
    for row in original.itertuples():
        d=c[c.multiplier.eq(row.multiplier)].iloc[0]
        np.testing.assert_allclose([d.C0,d.b,d.initial_lambda],[row.C0,row.b,row.initial_lambda],rtol=1e-10)
    a=pd.read_csv(DOUT/'tables/annual_policies.csv')
    for row in a.itertuples():
        const=c[c.candidate.eq(row.candidate)].iloc[0]
        assert const.admissible and row.b>1
        assert row.C0==pytest.approx(const.C0,rel=1e-13)
        assert row.penalty==pytest.approx(penalty(row.C0,row.T,row.b),rel=1e-13)
    p=json.loads((DOUT/'audit/protocol.json').read_text());assert p['annual_lambda_grid']==0


def test_dense_selection_is_chronological_and_minimizes_available_losses():
    candidates=pd.read_csv(DOUT/'tables/candidate_losses.csv',parse_dates=['inner_last_formation','validation_first_formation','decision_date'])
    a=pd.read_csv(DOUT/'tables/annual_policies.csv')
    assert (candidates.inner_last_formation<candidates.validation_first_formation).all()
    assert (candidates.validation_first_formation<candidates.decision_date).all()
    assert (candidates.full_T-candidates.inner_T).eq(60).all()
    order=sorted(range(9),key=lambda j:(abs(np.log2(DENSE[j])),DENSE[j]))
    for (year,method),d in candidates.groupby(['year','method']):
        eligible=d[d.eligible].set_index('candidate')
        expected=min([j for j in order if j in eligible.index],key=lambda j:eligible.loc[j].validation_loss)
        actual=a[a.year.eq(year)&a.method.eq(method)].iloc[0]
        assert actual.candidate==expected or actual.fallback


def test_actual_dense_holdings_reconcile_first_and_carried_account_states():
    accounts=pd.read_csv(DOUT/'tables/monthly_accounts.csv',parse_dates=['formation_date'])
    data=pd.read_parquet(ROOT/'data/clean/jkp_1978.parquet');h=pd.read_parquet(DPRIVATE/'dense_holdings_1978.parquet')
    for method in ('C_dense','D_dense'):
        state=pd.Series(dtype=float)
        for date in pd.to_datetime(['1978-01-31','1978-02-28']):
            group=data[data.eom.eq(date)].set_index('id');target=h[h.formation_date.eq(date)].set_index('id')[method]
            ids=target.index.union(state.index);weights=target.reindex(ids,fill_value=0).to_numpy()
            ref=accounts[accounts.method.eq(method)&accounts.formation_date.eq(date)&accounts.scenario.eq('trade25_borrow30')].iloc[0]
            step=accounting_step(weights,group.r.reindex(ids).to_numpy(),ref.rf,state.reindex(ids,fill_value=0).to_numpy(),.0025,.003)
            state=pd.Series(step['next_drift'],index=ids)[weights!=0]
            for key in ('excess_return','turnover','trading_fee','borrowing_fee','accounting_residual'):
                assert ref[key]==pytest.approx(step[key],abs=1e-12)
    assert json.loads((DOUT/'audit/costs.json').read_text())['actual_weights']


def test_dense_summary_uses_identical_common_dates_and_actual_monthly_sharpe():
    accounts=pd.read_csv(DOUT/'tables/monthly_accounts.csv',parse_dates=['return_date'])
    original=pd.read_csv(OUT/'tables/monthly_accounts.csv',parse_dates=['return_date']);summary=pd.read_csv(DOUT/'tables/performance_summary.csv')
    for method,d in accounts[accounts.scenario.eq('trade25_borrow30')].groupby('method'):
        assert len(d)==564
        np.testing.assert_array_equal(d.sort_values('return_date').return_date,
            original[original.method.eq('D')&original.scenario.eq('trade25_borrow30')].sort_values('return_date').return_date)
        q=d[d.year.ge(1979)];r=summary[summary.method.eq(method)&summary.scenario.eq('trade25_borrow30')&summary.period.eq('1979-2024')].iloc[0]
        assert r.sharpe==pytest.approx(float(sharpe(q.excess_return)),rel=1e-12)
