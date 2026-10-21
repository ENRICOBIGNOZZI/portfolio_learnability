import pandas as pd
import pytest

from simulations.paper_density_report import verify_anchors


def old_frame():
    return pd.DataFrame([dict(trial=0,T_train=t,lambda_multiplier=1.,
                              complexity=c,SR_hat=s,payoff_mean=.3,payoff_second=.4)
                         for t,c,s in [(800,41.,2.86),(1600,53.,2.91)]])


def test_anchors_align_by_trial_and_training_time_despite_row_order():
    old=old_frame()
    dense=pd.concat([old.iloc[::-1],pd.DataFrame([dict(trial=0,T_train=919,lambda_multiplier=1.,
                complexity=44.,SR_hat=2.87,payoff_mean=.3,payoff_second=.4)])],ignore_index=True)
    assert verify_anchors(dense,old)['paired_anchor_policies']==2
    dense.loc[dense.T_train==800,'SR_hat']=2.91
    with pytest.raises(AssertionError):verify_anchors(dense,old)


def test_missing_old_anchor_is_detected():
    old=old_frame()
    with pytest.raises(ValueError,match='anchor is missing'):verify_anchors(old.iloc[:1],old)
