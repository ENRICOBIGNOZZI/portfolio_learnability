import json
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cholesky, solve_triangular
from simulations.closeout.metrics import financial_decomposition, payoff_moments, ridge_decomposition, normal_residual
from simulations.closeout.results import oracle_indices
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.extended.statistics import gap_slopes


def test_optimum_positive_scaling_and_signed_or_null_policies():
    q=.4
    c=np.array([1.,.25,2.,-1.,0.])
    d=financial_decomposition(c*q,c*c*q,q)
    np.testing.assert_allclose(d['regret'],q*(c-1)**2)
    np.testing.assert_allclose(d['monthly_SR'][:3],np.sqrt(q/(1-q)))
    np.testing.assert_allclose(d['annual_SR'][:3],np.sqrt(12*q/(1-q)))
    np.testing.assert_allclose(d['ray_gap'][:3],0,atol=1e-15)
    assert d['moment_status'][-2]=='negative_mean'
    assert d['monthly_SR'][-2]<0 and np.isnan(d['ray_gap'][-2])
    assert d['moment_status'][-1]=='zero_variance'
    assert np.isnan(d['monthly_SR'][-1]) and d['regret'][-1]==q
    assert np.isnan(financial_decomposition(0,.3,q)['normalized_directional_error'])


def test_directional_error_is_normalized_and_policy_identities_not_mean_identities():
    q=.4;u=np.array([.2,.3]);v=u*u/q+np.array([.05,.02])
    d=financial_decomposition(u,v,q)
    scaled=financial_decomposition(3*u,9*v,q)
    np.testing.assert_allclose(scaled['d'],9*d['d'])
    np.testing.assert_allclose(scaled['normalized_directional_error'],d['normalized_directional_error'])
    np.testing.assert_allclose(scaled['ray_gap'],d['ray_gap'])
    np.testing.assert_allclose(d['regret'],d['ray_gap']+d['scale_gap'])
    assert not np.isclose(d['radial_gap'].mean(),q*(d['c'].mean()-1)**2)
    with pytest.raises(ValueError): financial_decomposition(.4,.1,q)
    # Roundoff-sized negatives are exported, never hidden by clipping d.
    assert financial_decomposition(.4,.4-1e-12,q)['d']<0


def test_population_ridge_decomposition_preserves_signed_cross_and_legacy_api():
    S=np.array([[1.,.2],[.2,.6]]);m=np.array([.2,.1]);q=.4
    penalties=np.array([.01,.3]);rng=np.random.default_rng(3)
    pop=np.column_stack([np.linalg.solve(S+lam*np.eye(2),m) for lam in penalties])
    a=pop+rng.normal(size=(2,2))*.08
    terms,population,optimum=ridge_decomposition(a,m,S,q,penalties)
    np.testing.assert_allclose(population,pop)
    u,v=payoff_moments(a,m,S)
    np.testing.assert_allclose(sum(terms.values()),q-2*u+v)
    expected=2*np.sum((a-pop)*(S@(pop-optimum[:,None])),axis=0)
    np.testing.assert_allclose(terms['signed_cross_term'],expected)
    assert np.any(expected<0)
    L,SR=moment_metrics(a,m,S)
    np.testing.assert_allclose(L,1-2*u+v)
    np.testing.assert_allclose(SR,u/np.sqrt(v-u*u))


@pytest.mark.parametrize('T,P',[(5,7),(7,7),(9,7)])
def test_primal_dual_and_small_lambda_without_jitter_at_regime_switch(T,P):
    x=np.random.default_rng(T).normal(size=(T,P))+.1
    penalties=np.array([1e-10,.03])
    a,C=ridge_path(x,penalties)
    for j,lam in enumerate(penalties):
        dual=x.T@np.linalg.solve(x@x.T/T+lam*np.eye(T),np.ones(T))/T
        primal=np.linalg.solve(x.T@x/T+lam*np.eye(P),x.mean(axis=0))
        np.testing.assert_allclose(x@a[:,j],x@dual,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(x@a[:,j],x@primal,atol=2e-6,rtol=2e-6)
    absolute,relative=normal_residual(x,a,penalties)
    assert absolute<1e-8 and relative<1e-8


def test_oracle_Q_and_SR_choose_one_penalty_from_means():
    regret=np.array([[[1.,5.,2.]],[[3.,0.,2.]]])
    sr=np.array([[[4.,1.,2.]],[[0.,4.,2.]]])
    cq,cs=oracle_indices(regret,sr)
    assert cq.tolist()==[0] and cs.tolist()==[1]
    assert regret[:,0,cq[0]].mean()==2
    assert sr[:,0,cs[0]].mean()==2.5
    assert sr[:,0,cs[0]].mean()!=sr.max(axis=2).mean()


def test_archived_regret_slope_and_full_covariance_jackknife():
    from simulations.closeout.results import CANONICAL
    with np.load(CANONICAL/'distributions/baseline.npz') as z:
        times=z['T'];regret=z['regret'][:,:,0];gap=z['annual_gap'][:,:,0]
    for values,target in [(regret,-.4378196),(gap,-.3986480)]:
        rows=gap_slopes(times,values)
        assert rows[0]['slope']==pytest.approx(target,abs=1e-6)
        for row in rows:
            keep=times>=row['T_min']
            if not row['available']: continue
            assert row['slope']==pytest.approx(np.polyfit(np.log(times[keep]),np.log(values[:,keep].mean(axis=0)),1)[0])
            assert row['MCSE_delta']==pytest.approx(row['MCSE_jackknife'],rel=.03)


def test_original_protocol_and_checkpoint_manifest_corruption_rejected(tmp_path,monkeypatch):
    from simulations.initial_cv import design
    from simulations.provenance import digest
    (tmp_path/'reference_manifest.json').write_text('[]')
    frozen={'design':{'source_hashes':{}},'run_hash':'altered','references_sha256':'wrong'}
    (tmp_path/'protocol.json').write_text(json.dumps(frozen))
    monkeypatch.setattr(design,'CANONICAL',tmp_path)
    with pytest.raises(ValueError,match='Protocol identity'):design.protocol()
    frozen['run_hash']=digest(frozen['design'])
    (tmp_path/'protocol.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='Reference manifest'):design.protocol()


def test_single_history_rank_comparison_never_invents_MCSE():
    from simulations.closeout.audit_results import paired_bound
    mean,mcse,bound=paired_bound([.01])
    assert mean==.01 and np.isnan(mcse) and np.isnan(bound)
    mean,mcse,bound=paired_bound([.01,.01,.01])
    assert mean==pytest.approx(.01) and mcse==pytest.approx(0) and bound==pytest.approx(.01)


def test_altered_audit_checkpoint_or_receipt_rejected(tmp_path,monkeypatch):
    from simulations.closeout import numerical
    from simulations.provenance import file_hash,digest
    plan={'representative_indices':[0],'stress_indices_by_economy':{},'ranks':[4096,8192]}
    monkeypatch.setattr(numerical,'DEST',tmp_path)
    monkeypatch.setattr(numerical,'freeze',lambda:plan)
    (tmp_path/'audit_fits').mkdir()
    path=tmp_path/'audit_fits/rep_000_short.npz'
    np.savez(path,index=0)
    receipt={'sha256':'corrupted','protocol_hash':digest(plan)}
    path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='Altered fit'):numerical.fit(0,short=True)
    receipt['sha256']=file_hash(path);receipt['protocol_hash']='corrupted'
    path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='Altered fit'):numerical.fit(0,short=True)


def test_zero_managed_payoff_has_zero_normal_equation_residual():
    absolute,relative=normal_residual(np.zeros((20,4)),np.zeros((4,2)),[1e-8,.1])
    assert absolute==0 and relative==0


def test_ray_scale_identification_needs_no_ambiguous_moment_root():
    from simulations.closeout.metrics import identified_components
    q=.4
    first=financial_decomposition(.25,.20,q)
    second=financial_decomposition(.375,.45,q)
    # Two distinct, economically admissible moment pairs have identical Q/SR.
    assert first['u']!=second['u'] and first['d']!=second['d']
    assert first['regret']==pytest.approx(second['regret'])
    assert first['monthly_SR']==pytest.approx(second['monthly_SR'])
    identified=identified_components(first['regret'],first['monthly_SR'],q)
    for key in ('ray_gap','scale_gap','normalized_directional_error'):
        assert identified[key]==pytest.approx(first[key])
        assert identified[key]==pytest.approx(second[key])
    assert np.isnan(identified_components(.4,-.2,q)['ray_gap'])


def test_annual_gap_is_monthly_gap_times_sqrt_twelve():
    d=financial_decomposition(np.array([.25,-.25]),np.array([.20,.20]),.4)
    np.testing.assert_allclose(d['annual_gap'],np.sqrt(12)*d['monthly_gap'])


def test_compaction_retains_every_policy_field_and_records_prefix_cache(tmp_path,monkeypatch):
    from simulations.closeout import numerical
    from simulations.provenance import file_hash
    monkeypatch.setattr(numerical,'DEST',tmp_path)
    monkeypatch.setattr(numerical,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(numerical,'freeze',lambda:{})
    (tmp_path/'audit_fits').mkdir()
    path=tmp_path/'audit_fits/rep_001_short.npz'
    coefficients=np.arange(16.).reshape(4,4)
    prefix=np.ones((60,4))
    np.savez(path,index=1,raw_coefficients=coefficients,managed_prefix_low=prefix,protocol_hash='same-policy')
    path.with_suffix('.json').write_text(json.dumps({'sha256':file_hash(path)}))
    numerical.compact_fits()
    with np.load(path) as z:
        np.testing.assert_array_equal(z['raw_coefficients'],coefficients)
        assert 'managed_prefix_low' not in z.files
    cache=tmp_path/'cache/training_prefixes'/path.name
    with np.load(cache) as z:np.testing.assert_array_equal(z['managed_prefix_low'],prefix)
    receipt=json.loads(path.with_suffix('.json').read_text())
    assert receipt['sha256']==file_hash(path)
    assert receipt['training_prefix_cache_sha256']==file_hash(cache)


def test_additional_selected_stress_cases_do_not_enter_representative_MCSE(tmp_path,monkeypatch):
    from simulations.closeout import audit_results
    monkeypatch.setattr(audit_results,'DEST',tmp_path)
    rows=[]
    for i in [0,1,33,50]:
        cohort='representative' if i<12 else 'additional_stress'
        for rank in [4096,8192]:
            for quad in ['primary','independent']:
                value=1. if rank==4096 else (1.01 if cohort=='representative' else 80.)
                rows.append(dict(environment='baseline',index=i,T=60,audit='A',rank=rank,quadrature=quad,cohort=cohort,sample_scope='short',
                    regret=value,u=value,v=value,annual_gap=value,normalized_directional_error=value))
    pd.DataFrame(rows).to_csv(tmp_path/'audit_policy_moments.csv',index=False)
    (tmp_path/'audit_evaluation.json').write_text(json.dumps({'full_representative_indices':[],'short_representative_indices':[0,1,33,50]}))
    (tmp_path/'audit_protocol.json').write_text(json.dumps({'design':{'representative_indices':list(range(12)),'stress_indices_by_economy':{'baseline':33}}}))
    (tmp_path/'audit_fits').mkdir()
    for i in [0,1,33,50]:
        np.savez(tmp_path/'audit_fits'/f'rep_{i:03d}_short.npz',index=i,sample_scope='short',
            **{n+'_selected_a':np.array([1.,1.]) for n in ['baseline','N300','N1200','rho000','rho075']})
    audit_results.summarize_audit()
    compared=pd.read_csv(tmp_path/'audit_paired_comparisons.csv')
    representative=compared[(compared.cohort=='representative')&(compared.comparison=='rank8192_minus4096')]
    assert (representative.histories==2).all()
    np.testing.assert_allclose(representative.paired_mean,.01)
    np.testing.assert_allclose(representative.paired_mcse,0)
    stress=compared[compared.cohort=='additional_stress']
    assert stress.paired_mcse.isna().all() and (stress.status=='descriptive_stress').all()
