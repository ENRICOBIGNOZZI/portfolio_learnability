import json
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

from simulations.asymptotic import (Basis, draw_blocks, evaluate_coefficients,
    fit_statistics, managed_block, matern_kernel, sufficient_path)
from simulations.model import calibrate, fit_path, moments, psi


def config():
    return json.loads((Path(__file__).parents[1]/'simulations/config/asymptotic.json').read_text())


def test_nystrom_is_projection_of_the_original_matern_kernel():
    c=config();b=Basis('nystrom',32,c)
    z=np.random.default_rng(19).uniform(-1,1,(17,3))
    raw=b.raw(z)
    features=b.whiten_rows(raw)
    exact=matern_kernel(z,z)
    remainder=exact-features@features.T
    assert np.linalg.eigvalsh((remainder+remainder.T)/2).min()>-1e-10
    np.testing.assert_allclose(b.raw(b.centers),matern_kernel(b.centers,b.centers),atol=1e-14)
    np.testing.assert_array_equal(Basis('nystrom',16,c).centers,b.centers[:16])


def test_longer_horizons_have_identical_data_prefixes():
    short=list(draw_blocks(9,.95,3,719,13,8))
    long=list(draw_blocks(9,.95,3,719,29,8))
    for component in [1,2]:
        a=np.concatenate([p[component] for p in short])
        b=np.concatenate([p[component] for p in long])[:13]
        np.testing.assert_array_equal(a,b)


def test_streamed_statistics_and_regularization_match_direct_rows():
    c=config();c['stream_block']=8
    basis=Basis('nystrom',16,c)
    times=[7,8,15,16,31]
    path=list(sufficient_path(c,9,.5,1,49,times,basis))
    assert [p[0] for p in path]==times
    raw=np.concatenate([managed_block(basis,z,r) for _,z,r in draw_blocks(9,.5,1,49,31,8)])
    for n,mean,second in path:
        np.testing.assert_allclose(mean,raw[:n].mean(axis=0),atol=1e-17)
        np.testing.assert_allclose(second,raw[:n].T@raw[:n]/n,atol=1e-17)
        penalties=np.array([1e-6,1e-4])
        coeff,C,_=fit_statistics(basis,mean,second,penalties)
        theta,expected_C,_=fit_path(basis.whiten_rows(raw[:n]),penalties)
        np.testing.assert_allclose(coeff,basis.raw_coefficients(theta),rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose(C,expected_C,rtol=1e-9,atol=1e-9)


def test_crossfit_known_oracle_control_variate_is_exact_for_oracle_policies():
    N=12;cal=calibrate(N);rng=np.random.default_rng(17)
    means=[];seconds=[];oracles=[]
    coefficient=N*cal['A_alpha']/cal['kappa_0']*np.r_[1,cal['lambda_factor']]
    for half in range(2):
        z=rng.uniform(-1,1,(37,N,3))
        _,B,mu,h=moments(z,cal)
        k=np.concatenate([np.sqrt(2)*np.cos(np.pi*z.sum(axis=-1))[...,None],psi(z)],axis=-1)
        m=np.einsum('bnr,bn->br',k,mu)/N
        common=np.einsum('bnr,bn->br',k,h)/N
        f=np.einsum('bnr,bnk->brk',k,B)/N
        s=(m.T@m+common.T@common+cal['sigma_eps']**2*np.einsum('bnr,bns->rs',k,k)/N**2
           +np.einsum('brk,km,bsm->rs',f,cal['Sigma_factor'],f))/len(z)
        means.append(m.mean(axis=0));seconds.append(s)
        oracles.append(np.sum(mu**2)/len(z)/cal['kappa_0'])
    evaluation=dict(means=np.array(means),seconds=np.array(seconds),oracles=np.array(oracles))
    scales=np.array([.3,1,2.])
    out=evaluate_coefficients(coefficient[:,None]*scales,evaluation)
    np.testing.assert_allclose(out['SR_hat'],3,atol=1e-12)
    np.testing.assert_allclose(out['payoff_mean'],cal['p_star']*scales,atol=1e-13)
    np.testing.assert_allclose(out['payoff_second'],cal['p_star']*scales**2,atol=1e-13)


def test_result_band_is_one_standard_deviation_not_standard_error():
    import pandas as pd
    import matplotlib.pyplot as plt
    from simulations.report import line
    from simulations.asymptotic_report import draw
    frame=pd.DataFrame([dict(trial=j,T_train=t,value=v) for t in [100,200]
                        for j,v in enumerate([1.,2.,3.,4.])])
    for helper in [line,draw]:
        fig,ax=plt.subplots()
        helper(ax,frame,'T_train','value','test','blue')
        vertices=ax.collections[0].get_paths()[0].vertices[:,1]
        sd=np.std([1.,2.,3.,4.],ddof=1)
        np.testing.assert_allclose(vertices.min(),2.5-sd)
        np.testing.assert_allclose(vertices.max(),2.5+sd)
        percentile=[artist for artist in ax.lines if artist.get_gid()=='trial_percentile_95']
        assert len(percentile)==1
        np.testing.assert_allclose(percentile[0].get_ydata(),[3.85,3.85])
        assert not np.isclose(3.85,2.5+1.96*sd/2)
        plt.close(fig)


def test_dense_design_retains_anchors_and_has_many_actual_tail_points():
    parent=config()
    dense=json.loads((Path(__file__).parents[1]/'simulations/config/asymptotic_dense.json').read_text())
    assert {k for k in parent if parent[k]!=dense[k]}=={'version','T_train','T_train_reference'}
    assert len(dense['T_train'])==46
    assert len(dense['T_train_reference'])==54
    for key in ['T_train','T_train_reference']:
        assert dense[key]==sorted(set(dense[key]))
        assert set(parent[key])<=set(dense[key])
    assert sum(t>=dense['asymptotic_T_min'] for t in dense['T_train_reference'])==28


def test_pilot_reuse_rejects_scientific_changes_and_missing_training_anchors():
    from simulations.asymptotic_dense import validate_upgrade
    parent=config()
    dense=json.loads((Path(__file__).parents[1]/'simulations/config/asymptotic_dense.json').read_text())
    assert set(validate_upgrade(parent,dense))=={'version','T_train','T_train_reference'}
    wrong=dict(dense,confirmation_seed=dense['confirmation_seed']+1)
    with pytest.raises(ValueError,match='confirmation_seed'):
        validate_upgrade(parent,wrong)
    wrong=dict(dense,T_train=[t for t in dense['T_train'] if t!=1600])
    with pytest.raises(ValueError,match='supersets'):
        validate_upgrade(parent,wrong)
