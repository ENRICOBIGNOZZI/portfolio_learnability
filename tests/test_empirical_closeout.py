import numpy as np
import pandas as pd
from empirical_closeout import validation_diagnostics,paired_bootstrap
from data_pipeline import rank_months


def test_ex_post_oracle_is_diagnostic_not_validation_selection():
    d=pd.DataFrame(dict(T_months=[60]*6,decision_year=[2000]*3+[2001]*3,
        oos_loss=[.4,.2,.3,.1,.4,.3],complexity=[0,30,60]*2,
        relative_complexity=[0,.5,1]*2,**{'lambda':[10,.1,0]*2},
        selected=[False,False,True,True,False,False]))
    yearly,summary=validation_diagnostics(d)
    np.testing.assert_allclose(yearly.validation_regret,[.1,0])
    np.testing.assert_allclose(yearly.relative_validation_regret,[.5,0])
    assert summary.within_5_percent.iloc[0]==.5
    assert summary.fraction_near_one.iloc[0]==.5
    assert summary.fraction_near_zero.iloc[0]==.5
    assert yearly['lambda'].tolist()==[0,10]


def test_block_bootstrap_keeps_comparisons_paired():
    rng=np.random.default_rng(5);x=rng.normal(size=48)
    # Identical time series must have exactly zero paired difference and interval.
    wide=pd.DataFrame({60:x,120:x})
    d,_=paired_bootstrap(wide,replicates=50,block=12)
    np.testing.assert_allclose(d[['loss_a_minus_b','ci_low','ci_high']].to_numpy(),0,atol=0)


def test_formation_rank_is_invariant_to_future_payoff_availability():
    d=pd.DataFrame(dict(id=[1,2,3],eom=pd.to_datetime(['2000-01-31']*3),
        signal=[.2,.8,.6],ret_exc_lead1m=[np.nan,.1,-.2]))
    before=rank_months(d,['signal'])
    d.ret_exc_lead1m=[10,np.nan,np.nan]
    after=rank_months(d,['signal'])
    pd.testing.assert_frame_equal(before[['id','signal']],after[['id','signal']])
    assert len(after)==3
