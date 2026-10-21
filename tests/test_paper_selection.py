import pandas as pd
import pytest

from simulations.paper_selection import TARGETS,rank_candidates,choose_window


def candidate(cutoff,error,r2):
    return [dict(study='precision8',scenario='rank8192',rank=8192,N=600,rho=.5,
                 T_train_min=cutoff,T_train_max=25600,metric=m,
                 expected_exponent=t,exponent=t*(1+error),r_squared=r2)
            for m,t in TARGETS.items()]


def test_closest_exponents_do_not_override_four_rate_fit_quality():
    rows=candidate(400,.001,.999)+candidate(800,.02,.999)
    rows[0]['r_squared']=.98
    selected=choose_window(rank_candidates(pd.DataFrame(rows)))
    assert selected['T_train_min']==800
    assert selected['max_relative_exponent_error']==pytest.approx(.02)


def test_four_rates_from_different_windows_cannot_be_assembled_as_one_candidate():
    rows=candidate(800,.01,.999)
    rows[0]['T_train_min']=400
    with pytest.raises(ValueError,match='all four rates'):
        rank_candidates(pd.DataFrame(rows))


def test_lambda_window_with_a_different_endpoint_is_not_a_common_window():
    rows=candidate(400,.001,.999)+candidate(800,.02,.999)
    rows[3]['T_train_max']=12800
    ranked=rank_candidates(pd.DataFrame(rows))
    assert not bool(ranked[ranked.T_train_min==400].iloc[0].common_window)
    assert choose_window(ranked)['T_train_min']==800
