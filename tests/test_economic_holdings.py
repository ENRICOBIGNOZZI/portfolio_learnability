"""Identity, chronology and economic-attribution checks on actual stock holdings."""
import json
import numpy as np
import pandas as pd
import pytest
from empirical_final.economic_holdings import OUT,PRIVATE,BASE,FAMILIES,terciles,exposures


def table(name):
    path=OUT/'tables'/f'{name}.parquet'
    if not path.exists():pytest.skip('Run real-data holdings reconstruction first')
    return pd.read_parquet(path)


def test_stock_weights_reconstruct_existing_policy_month_by_month():
    checks=pd.read_csv(OUT/'tables/reconstruction_checks.csv') if (OUT/'tables/reconstruction_checks.csv').exists() else None
    if checks is None:pytest.skip('Run holdings reconstruction first')
    assert len(checks)==47
    assert checks.max_weight_error.max()<2e-9
    assert checks.max_payoff_error.max()<2e-9
    assert checks.max_partition_error.max()<1e-12


def test_same_month_characteristics_and_next_month_returns():
    f=table('monthly_families')
    assert (pd.to_datetime(f.formation_date)+pd.offsets.MonthEnd(1)==pd.to_datetime(f.return_date)).all()
    assert f.groupby(['formation_date','group']).family.nunique().eq(len(FAMILIES)).all()
    assert f.long_minus_short.between(-1,1).all()
    np.testing.assert_allclose(f.long_minus_short,f.long_mean-f.short_mean,atol=1e-12)


def test_unscaled_leg_characteristics_preserve_actual_signed_exposure():
    f=table('monthly_families');b=table('monthly_balances')
    m=f.merge(b[['formation_date','group','long_notional','short_notional']],on=['formation_date','group'],validate='many_to_one')
    np.testing.assert_allclose(m.net_exposure,m.long_notional*m.long_mean-m.short_notional*m.short_mean,atol=1e-12)


def test_every_characteristic_partition_recovers_group_holdings_and_payoff():
    c=table('monthly_cells');b=table('monthly_balances')
    for column,target in [('long_notional','long_notional'),('short_notional','short_notional'),('payoff','total_payoff')]:
        summed=c.groupby(['formation_date','group','partition'])[column].sum().reset_index()
        merged=summed.merge(b[['formation_date','group',target]].rename(columns={target:'target'}),on=['formation_date','group'],validate='many_to_one')
        np.testing.assert_allclose(merged[column],merged.target,atol=1e-11)


def test_attribution_retains_covariances_and_factor_linearity():
    p=OUT/'audit/attribution_checks.json'
    if not p.exists():pytest.skip('Run economic summaries first')
    a=json.loads(p.read_text());assert a['passed']
    assert a['max_loss_allocation_error']<1e-7
    assert a['max_variance_allocation_error']<1e-7
    assert a['max_factor_additivity_error']<1e-7


def test_characteristic_assignments_do_not_use_future_payoffs():
    from empirical_final.economic_holdings import ROOT,family_matrix
    frame=pd.read_parquet(ROOT/'data/clean/jkp_1978.parquet')
    frame=frame[frame.eom.eq(frame.eom.min())].copy()
    before=family_matrix(frame)
    # Change the realized return and label date, leaving formation information fixed.
    frame['r']=-frame.r.iloc[::-1].to_numpy();frame['return_date']=pd.Timestamp('2099-12-31')
    after=family_matrix(frame)
    np.testing.assert_array_equal(before,after)
    for j in range(before.shape[1]):np.testing.assert_array_equal(terciles(before[:,j]),terciles(after[:,j]))


def test_group_holdings_are_exact_signed_positions_not_risk_normalized_baskets():
    p=PRIVATE/'year_1978/holdings.parquet'
    if not p.exists():pytest.skip('Run holdings reconstruction first')
    h=pd.read_parquet(p);cols=[f'group_{i}_weight' for i in range(1,5)]
    np.testing.assert_allclose(h[cols].sum(axis=1),h.total_weight,atol=1e-12)
    assert h.permno.notna().all() and not h.duplicated(['formation_date','permno']).any()


def test_empty_cells_have_no_observed_density():
    p=table('monthly_joint_profiles')
    assert p.loc[p['count'].eq(0),'relative_weight_density'].isna().all()
    assert p.loc[p['count'].eq(0),'nonadditive_density'].isna().all()
    assert p.loc[p['count'].gt(0),'relative_weight_density'].notna().all()


def test_financing_cells_recover_returns_and_factor_slopes():
    c=table('financing_monthly_cells');b=table('monthly_balances')
    assert c.stock_count.gt(0).all()
    sums=c.groupby(['formation_date','group','partition']).payoff.sum().reset_index()
    m=sums.merge(b[['formation_date','group','total_payoff']],on=['formation_date','group'])
    np.testing.assert_allclose(m.payoff,m.total_payoff,atol=1e-12)
    f=pd.read_csv(OUT/'tables/financing_factor_attribution.csv')
    target=pd.read_csv(OUT/'tables/factor_attribution.csv').query("partition=='group'")
    summed=f.groupby(['period','group','partition','factor']).coefficient.sum().reset_index()
    m=summed.merge(target[['period','group','factor','coefficient']],on=['period','group','factor'])
    np.testing.assert_allclose(m.coefficient_x,m.coefficient_y,atol=1e-10)


def test_account_gross_nets_positions_before_taking_absolute_values():
    n=table('monthly_stock_netting')
    assert n.cancellation.ge(-1e-10).all()
    for date,d in n.groupby('formation_date'):
        d=d.sort_values('group')
        np.testing.assert_allclose(d.incremental_gross.cumsum(),d.cumulative_gross,atol=1e-10)
        np.testing.assert_allclose(d.component_gross.cumsum()-d.cumulative_gross,d.cancellation,atol=1e-10)
    h=pd.read_parquet(PRIVATE/'year_1978/holdings.parquet')
    for date,d in h.groupby('formation_date'):
        expected=np.abs(np.cumsum(d[[f'group_{j}_weight' for j in range(1,5)]].to_numpy(),axis=1)).sum(axis=0)
        np.testing.assert_allclose(n[n.formation_date.eq(date)].sort_values('group').cumulative_gross,expected,atol=1e-10)
