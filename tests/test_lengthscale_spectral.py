"""Exact math, chronology, actual caches/accounts and publication reconstruction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

from data_pipeline import digest
from kernels import FeatureBank
from portfolio import dual_path,ridge_path,sharpe
from empirical_final.lengthscale_spectral import (OUT,ROOT,BASE,MULTIPLIERS,account,bound_penalty,
    choose_from_oos,log_amplitude,log_kappa,rank_fit,shared_scores,transfer_penalty)
from empirical_final.theory_guided_lambda import choose_lambda,complexity,spectrum


@pytest.fixture(autouse=True)
def cpu_threads():
    with threadpool_limits(limits=1):yield


def test_asymptotic_scaling_and_corrected_ratio():
    c,b,T,rho=2.,2.,100.,.1
    lam=bound_penalty(c,b,T,rho)
    assert bound_penalty(8*c,b,T,rho)/lam==pytest.approx(8**(1/(b+1)))
    assert bound_penalty(c,b,2*T,rho)/lam==pytest.approx(2**(-b/(b+1)))
    assert bound_penalty(8*c,b,T,3*rho)/lam==pytest.approx(8**(1/(b+1))*3**(b/(b+1)))
    transfer=transfer_penalty(lam,T,2*T,np.log(8*c),np.log(c),b,3.)
    assert transfer==pytest.approx(bound_penalty(8*c,b,2*T,3*rho))
    kappa=np.exp(log_kappa(b))
    derivative=1-rho/T*kappa*c**(1/b)/b*lam**(-1-1/b)
    assert abs(derivative)<1e-12
    C=kappa*(c/lam)**(1/b)
    c2=kappa*(8*c/bound_penalty(8*c,b,T,rho))**(1/b)
    assert c2/C==pytest.approx(8**(1/(b+1)))


def test_near_one_stability_and_no_clipping_invalid_exponents():
    for b in [1+1e-10,1+3/130,2.,100.]:
        assert np.isfinite(log_kappa(b)) and np.isfinite(bound_penalty(1e-7,b,180,1e-6))
    for b in [.8,1.]:
        with pytest.raises(ValueError):bound_penalty(1e-7,b,180,1e-6)
    with pytest.raises(ValueError,match='Variable b'):
        transfer_penalty(.01,180,240,-4,-4,2.,b_current=3.)


def test_rank_fit_and_fixed_common_amplitude_are_exact_on_power_law():
    mu=3*np.arange(1,181,dtype=float)**-2.5
    f=rank_fit(mu,11,60)
    assert f['b']==pytest.approx(2.5) and f['log_c']==pytest.approx(np.log(3))
    assert f['r_squared']==pytest.approx(1.)
    assert log_amplitude(mu,2.5)==pytest.approx(np.log(3))
    assert log_amplitude(mu,2.5,estimator='mean')==pytest.approx(np.log(3))
    with pytest.raises(ValueError):rank_fit(np.array([1.,0.]),1,2)


def test_controlled_primal_dual_psd_complexity_and_sign_invariance():
    g=np.array([[2.,.1],[1.,.3],[.5,-.2],[.4,.6]])
    penalties=[.02,.1]
    b,_,C=ridge_path(g,penalties);a,mu,dual_C=dual_path(g@g.T,penalties)
    np.testing.assert_allclose(g.T@a,b,atol=1e-12)
    np.testing.assert_allclose(C,dual_C,atol=1e-12)
    np.testing.assert_allclose(mu[:2],np.linalg.eigvalsh(g.T@g/len(g))[::-1],atol=1e-12)
    assert complexity(mu,.02)==pytest.approx(C[0])
    values,u=np.linalg.eigh(g@g.T);signs=np.array([-1.,1.,-1.,1.]);u2=u*signs
    expected=u@((u.T@np.ones(4))/(values+4*.02))
    changed=u2@((u2.T@np.ones(4))/(values+4*.02))
    np.testing.assert_allclose(expected,changed,atol=1e-12)
    controlled,psd=spectrum(np.diag([1.,.1,-1e-12]))
    assert psd['clipped_eigenvalues']==1 and controlled.min()==0
    with pytest.raises(ValueError):spectrum(np.diag([1.,-.01]))
    root=choose_lambda(mu,4,1.,1e-6,10.)
    assert root['kkt_residual']<1e-10
    more=choose_lambda(mu,4,2.,1e-6,10.)
    assert more['C']<=root['C']


@pytest.mark.parametrize('kernel',['gaussian','matern32'])
def test_shared_scores_and_normalized_feature_prefixes(kernel):
    bank=FeatureBank(kernel,2,10000,1.7,0)
    x=np.array([[.1,.2],[.3,-.1],[-.1,.4]])
    betas=np.random.default_rng(42).normal(size=(10000,3));bands=np.array([0,2,4])
    actual=shared_scores(x,bank,betas,bands)
    for j,bi in enumerate(bands):
        other=FeatureBank(kernel,2,10000,1.7*MULTIPLIERS[bi],0)
        np.testing.assert_allclose(actual[:,j],other.features(x)@betas[:,j]/len(x),rtol=1e-10,atol=1e-12)
    full=bank.features(x)
    for P in [1000,5000]:
        np.testing.assert_allclose(bank.features(x,P),full[:,:P]*np.sqrt(10000/P),atol=1e-14)


def test_chronological_oos_selection_excludes_anchor_validation_and_future():
    dates=pd.date_range('1978-02-28','1985-01-31',freq='ME')
    years=np.repeat(np.arange(1978,1985),12)
    r=np.full((len(dates),5),.3);r[:,3]=.9
    choice,info=choose_from_oos(r,dates,'1984-01-31',60,years)
    assert choice==3 and info['selection_first_return']=='1979-02-28' and info['selection_last_return']=='1984-01-31'
    changed=r.copy();changed[dates>'1984-01-31']=1e9;changed[years==1978]=1e8
    assert choose_from_oos(changed,dates,'1984-01-31',60,years)==(choice,info)
    with pytest.raises(ValueError,match='Insufficient'):
        choose_from_oos(r,dates,'1983-01-31',60,years)
    tied=np.ones_like(r)*.9
    assert choose_from_oos(tied,dates,'1984-01-31',60,years)[0]==BASE


def test_same_second_moment_different_exact_ridge_risk_optima():
    # Actual response-one ridge on iid X in {-1,+1}: sample second moment is exactly one.
    T=10
    for m in [.5,.8]:
        variance_mean=(1-m*m)/T
        risk=lambda lam:1-2*m*m/(1+lam)+(m*m+variance_mean)/(1+lam)**2
        optimum=(1-m*m)/(T*m*m)
        numerical=minimize_scalar(risk,bounds=(0,1),method='bounded',options={'xatol':1e-12}).x
        assert numerical==pytest.approx(optimum,rel=1e-5)
    assert (1-.5**2)/(T*.5**2)!=(1-.8**2)/(T*.8**2)


def test_exact_sharpe_oracles_differ_with_identical_second_moment_laws():
    # Exhaustive finite-law expectation, not a Monte Carlo portfolio experiment.
    states=np.array([[np.sqrt(2),0],[-np.sqrt(2),0],[0,2*np.sqrt(2)],[0,-2*np.sqrt(2)]])
    Sigma=np.diag([1.,4.]);h=.6
    for economy in ['A','B']:
        probabilities=np.array([.25*(1+h),.25*(1-h),.25,.25]) if economy=='A' else np.array([.25,.25,.25*(1+h),.25*(1-h)])
        mean=probabilities@states
        np.testing.assert_allclose((states.T*probabilities)@states,Sigma,atol=1e-14)
        expected=[]
        for penalty in [.01,.1,1.,10.]:
            value=0.
            for i in range(4):
                for j in range(4):
                    beta=ridge_path(states[[i,j]],[penalty])[0][:,0]
                    numerator=float(mean@beta);var=float(beta@Sigma@beta-numerator**2)
                    sr=numerator/np.sqrt(var) if np.linalg.norm(beta)>1e-12 else 0.
                    value+=probabilities[i]*probabilities[j]*sr
            r=2*(1+penalty)/(4+penalty)
            m1,m2=h/np.sqrt(2),h*np.sqrt(2)
            exact=h*m1/(4*np.sqrt(1-m1*m1))+h*m1/(2*np.sqrt(1-m1*m1+4*r*r)) if economy=='A' else h*m2/(4*np.sqrt(4-m2*m2))+h*m2/(2*np.sqrt(4-m2*m2+1/(r*r)))
            assert value==pytest.approx(exact,abs=1e-12)
            expected.append(value)
        assert (np.diff(expected)<0).all() if economy=='A' else (np.diff(expected)>0).all()


def test_exact_stock_accounting_nets_signed_weights_and_charges_exits():
    target=pd.Series([.8,-.4],index=[1,2]);returns=pd.Series([.1,-.05],index=[1,2])
    rows,drift=account(target,returns,.01,pd.Series([.7,.2],index=[1,3]),pd.Series([.6,.3],index=[1,3]))
    by={r['scenario']:r for r in rows}
    assert by['gross']['excess_return']==pytest.approx(.1)
    assert by['target_trade25_borrow30']['excess_return']==pytest.approx(.1-.0025*.9-.003/12*.4)
    assert by['trade25_borrow30']['accounting_residual']<1e-12 and set(drift.index)=={1,2}


def read(name):return pd.read_csv(OUT/'tables'/f'{name}.csv')


def test_real_anchors_baselines_and_prior_gaussian_spectral_path():
    anchors=json.loads((OUT/'audit/anchors.json').read_text());m=read('monthly_policies')
    prior=pd.read_csv(ROOT/'outputs/bandwidth_tuning/monthly.csv')
    for kernel in ['matern32','gaussian']:
        a=anchors[kernel];assert a['T0']==180 and a['interior']
        annual=read('annual_policies').query('kernel==@kernel and year==1978').set_index('method')
        assert annual.loc['annual_cv','penalty']==pytest.approx(a['lam0'],rel=1e-10)
        assert annual.loc['one_anchor','penalty']==pytest.approx(a['lam0'],rel=1e-10)
        for method,mode in [('annual_cv','fixed'),('joint_cv','tuned')]:
            actual=m[m.kernel.eq(kernel)&m.method.eq(method)].sort_values('return_date')
            expected=prior[prior.kernel.eq(kernel)&prior['mode'].eq(mode)].sort_values('return_date')
            assert len(actual)==564
            np.testing.assert_array_equal(actual.return_date,expected.return_date)
            np.testing.assert_allclose(actual.raw_return,expected.raw_excess_return,rtol=2e-6,atol=2e-7)
    old=pd.read_csv(ROOT/'outputs/theory_guided_lambda_20261010/tables/monthly_oos.csv')
    actual=m[m.kernel.eq('gaussian')&m.method.eq('one_anchor')].sort_values('return_date')
    np.testing.assert_allclose(actual.raw_return,old.spectral_raw,rtol=1e-7,atol=1e-9)


def test_real_matern_anchor_cannot_read_future_managed_payoffs():
    from empirical_final.run import clean_manifest
    from empirical_final.lengthscale_spectral import load_cache
    from empirical_final.theory_guided_lambda import anchor_from_history
    gs,dates,_=load_cache('matern32',clean_manifest())
    grids=json.loads((OUT/'audit/penalty_grids.json').read_text())['matern32'][BASE]['penalties']
    original=anchor_from_history(gs[BASE],dates,np.asarray(grids))
    changed=np.asarray(gs[BASE]).copy();changed[180:]*=-10000
    assert anchor_from_history(changed,dates,np.asarray(grids))==original


def test_real_adaptive_choices_recompute_from_genuinely_prior_oos_paths():
    a=read('annual_policies');c=read('candidate_policies');m=read('monthly_policies')
    for row in a[a.method.isin(['adaptive_spectral','spectral_transfer'])].itertuples():
        rule='spectral' if row.method=='adaptive_spectral' else 'transfer'
        past=c[c.kernel.eq(row.kernel)&c.rule.eq(rule)&c.year.between(1979,row.year-1)]
        wide=past.pivot(index='return_date',columns='multiplier',values='raw_return').sort_index()
        choice,info=choose_from_oos(wide.to_numpy(),wide.index,f'{row.year}-01-31',60)
        assert row.multiplier==MULTIPLIERS[choice]
        assert row.selection_last_return==info['selection_last_return']
        assert pd.Timestamp(row.selection_last_return)<=pd.Timestamp(row.last_known_payoff)<pd.Timestamp(row.first_test_return)
    for (kernel,method),f in m[m.method.isin(['adaptive_spectral','spectral_transfer'])].groupby(['kernel','method']):
        assert len(f)==492 and f.year.min()==1984 and f.return_date.nunique()==492
    gaussian=m[(m.kernel=='gaussian')&(m.method=='spectral_transfer')]
    assert len(gaussian)==492 and gaussian.year.min()==1984


def test_real_fixed_b_and_transfer_anchor_scaling_from_saved_spectra():
    a=read('annual_policies');s=read('empirical_spectra')
    anchors=json.loads((OUT/'audit/anchors.json').read_text())
    for kernel in ['matern32','gaussian']:
        b=anchors[kernel]['finite_range_b']
        mu0=s[(s.kernel==kernel)&(s.year==1978)&(s.multiplier==1)].sort_values('rank').mu.to_numpy()
        for row in a[(a.kernel==kernel)&(a.method=='spectral_transfer')].itertuples():
            mu=s[(s.kernel==kernel)&(s.year==row.year)&(s.multiplier==row.multiplier)].sort_values('rank').mu.to_numpy()
            expected=transfer_penalty(anchors[kernel]['lam0'],180,row.T,log_amplitude(mu,b),log_amplitude(mu0,b),b)
            assert row.unconstrained_penalty==pytest.approx(expected,rel=1e-9)
            assert row.fixed_b==pytest.approx(b)


def test_real_stock_accounts_and_plot_tables_reconstruct_all_primary_statistics():
    accounts=read('monthly_accounts');summary=read('performance_summary');m=read('monthly_policies')
    cv=json.loads((OUT/'audit/cost_verification.json').read_text())
    assert cv['passed'] and cv['actual_weights'] and cv['original_cv_net_reproduced']
    assert cv['maximum_accounting_residual']<1e-10
    for row in summary[summary.period.eq('1984-2024')].itertuples():
        if row.scenario=='gross':
            f=m[m.kernel.eq(row.kernel)&m.method.eq(row.method)&m.year.ge(1984)]
        else:
            f=accounts[accounts.kernel.eq(row.kernel)&accounts.method.eq(row.method)&accounts.window.eq('common_1984')&accounts.scenario.eq(row.scenario)]
        assert len(f)==492
        assert row.sharpe==pytest.approx(sharpe(f.excess_return))
    for (kernel,method),f in accounts[accounts.window.eq('common_1984')&accounts.scenario.eq('gross')].groupby(['kernel','method']):
        old=m[m.kernel.eq(kernel)&m.method.eq(method)&m.year.ge(1984)].sort_values('return_date')
        np.testing.assert_allclose(f.sort_values('return_date').excess_return,old.excess_return,rtol=2e-6,atol=2e-8)


def test_real_frozen_inputs_sources_and_finite_primary_results():
    for name,expected in json.loads((OUT/'audit/input_hashes.json').read_text()).items():assert digest(ROOT/name)==expected
    v=json.loads((OUT/'audit/estimation.json').read_text())
    assert digest(ROOT/'empirical_final/lengthscale_spectral.py')==v['module_sha256']
    publication=json.loads((OUT/'audit/publication_verification.json').read_text())
    assert digest(ROOT/'empirical_final/lengthscale_publication.py')==publication['publication_module_sha256']
    assert digest(ROOT/'empirical_final/templates/lengthscale_mathematical_note.tex')==publication['mathematical_template_sha256']
    for name in ['monthly_policies','monthly_accounts']:
        assert np.isfinite(read(name).select_dtypes('number')).all().all()
    for kernel,f in read('annual_policies').groupby('kernel'):assert f.rho0.nunique()==1


def test_real_exported_plot_values_reconstruct_spectra_ratios_and_wealth():
    spectra=read('empirical_spectra');plotted=read('figure_spectral_values')
    base=spectra[spectra.multiplier.eq(1)].rename(columns={'mu':'base_mu'})
    joined=plotted.merge(spectra,on=['kernel','year','multiplier','rank']).merge(base[['kernel','year','rank','base_mu']],on=['kernel','year','rank'])
    assert len(joined)==len(plotted)
    np.testing.assert_allclose(joined.eigenvalue,joined.mu,rtol=1e-12,atol=0)
    np.testing.assert_allclose(joined.log_ratio,np.log(joined.mu/joined.base_mu),rtol=1e-10,atol=1e-12)
    wealth=read('figure_wealth_values');accounts=read('monthly_accounts')
    for (kernel,method,scenario),f in wealth.groupby(['kernel','method','scenario']):
        source=accounts[accounts.kernel.eq(kernel)&accounts.method.eq(method)&accounts.scenario.eq(scenario)&accounts.window.eq('common_1984')].sort_values('return_date')
        f=f.sort_values('return_date')
        np.testing.assert_array_equal(f.return_date,source.return_date)
        np.testing.assert_allclose(f.wealth,np.cumprod(1+source.total_return),rtol=1e-10,atol=1e-12)
    contrasts=read('representation_penalty_contrasts')
    np.testing.assert_allclose(contrasts.representation_shapley_contrast+contrasts.penalty_shapley_contrast,
        contrasts.joint_length_joint_penalty-contrasts.base_length_base_penalty,atol=1e-12)
