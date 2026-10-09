"""Cross-check reused numerical integrals and paired economic innovations."""
from dataclasses import replace
import numpy as np

from simulations.dgp.balanced import BalancedFactorDGP
from simulations.estimator.kernel import NystromBasis
from simulations.diagnostics.low_memory import population_moments_batched
from simulations.extended.design import parameters
from simulations.extended.population import raw_integrals, operators, spectral_path
from simulations.extended.histories import PairedHistory, raw_managed_history
from simulations.run_rough import seed


def test_paired_baseline_is_existing_simulation():
    legacy=BalancedFactorDGP(parameters(),seed('train',7))
    paired=PairedHistory(seed('train',7),7)
    for t in range(12):
        old=legacy.step()
        got=paired.step(robustness=t<8)
        np.testing.assert_array_equal(got['baseline'][0],old.z)
        np.testing.assert_array_equal(got['baseline'][1],old.returns)
        if t<8:
            np.testing.assert_array_equal(got['N300'][1],old.returns[:300])
            np.testing.assert_array_equal(got['N1200'][1][:600],old.returns)


def test_raw_integrals_match_canonical_population_for_every_N():
    basis=NystromBasis(rank=32,seed=4)
    stats=raw_integrals(parameters(),basis,127,27,batch_groups=19)
    for name in ('baseline','N300','N1200','rho000','rho075'):
        p=parameters(name)
        m,S,parts=operators(p,basis,stats,components=True)
        expected=population_moments_batched(p,basis,127,27,batch_groups=19)
        np.testing.assert_allclose(m,expected[0],rtol=2e-13,atol=1e-16)
        np.testing.assert_allclose(S,expected[1],rtol=2e-12,atol=1e-18)
        np.testing.assert_allclose(parts['factor']+parts['idiosyncratic'],S,atol=1e-18)


def test_nested_prefixes_match_separate_bases():
    large=NystromBasis(rank=64,seed=9)
    small=NystromBasis(rank=32,seed=9)
    a=raw_integrals(parameters(),large,99,29,batch_groups=17)
    b=raw_integrals(parameters(),small,99,29,batch_groups=17)
    for got,want in zip(operators(parameters(),small,a),operators(parameters(),small,b)):
        np.testing.assert_allclose(got,want,rtol=1e-11,atol=1e-16)


def test_population_elasticity_exact_limits_and_identity():
    values=np.geomspace(1e-8,1e-3,100)
    C,E,local=spectral_path(values,np.array([1e-13,1e-6,100.]))
    assert C[0]>99.99 and E[0]<1e-5
    assert E[-1]>.9999
    np.testing.assert_allclose(local,.6*E)


def test_raw_managed_baseline_matches_canonical():
    basis=NystromBasis(rank=16,seed=21)
    arrays,_=raw_managed_history(basis,seed('train',4),4,6,4)
    old=BalancedFactorDGP(parameters(),seed('train',4))
    reference=np.array([basis.managed(d.z,d.returns) for d in old.simulate(6)])
    np.testing.assert_allclose(arrays['baseline']@basis.inverse_root,reference,atol=1e-16)


def test_slope_uncertainty_respects_common_multiplicative_shock():
    from simulations.extended.statistics import gap_slopes
    T=np.array([60,120,360,720,1440,2160,3240,4860])
    rng=np.random.default_rng(91)
    gap=np.exp(rng.normal(size=(50,1)))*T[None,:]**(-.6)
    for row in gap_slopes(T,gap):
        if row['available']:
            np.testing.assert_allclose(row['slope'],-.6,atol=1e-14)
            assert row['MCSE_delta']<1e-8
            assert row['MCSE_jackknife']<1e-12


def test_population_rho_invariance_is_exact():
    basis=NystromBasis(rank=16,seed=13)
    stats=raw_integrals(parameters(),basis,97,57)
    m,S=operators(parameters(),basis,stats)
    for name in ('rho000','rho075'):
        m2,S2=operators(parameters(name),basis,stats)
        np.testing.assert_array_equal(m,m2)
        np.testing.assert_array_equal(S,S2)
        assert parameters(name).sr_star==parameters().sr_star


def test_population_archive_detects_corruption(tmp_path,monkeypatch):
    import pytest
    from simulations.extended import archive
    from simulations.provenance import npz_write,json_write,file_hash
    monkeypatch.setattr(archive,'OUTPUT',tmp_path)
    folder=archive.archive_folder('baseline',8,99,17)
    S=np.arange(64,dtype=float).reshape(8,8)
    m=np.arange(8,dtype=float)
    chunks=[]
    for start in (0,4):
        path=folder/f'rows_{start}.npz'
        npz_write(path,second_rows=S[start:start+4])
        chunks.append(dict(file=path.name,first_row=start,rows=4,sha256=file_hash(path)))
    npz_write(folder/'moments.npz',mean=m,floor=.01)
    json_write(folder/'index.json',dict(rank=8,groups=99,seed=17,chunks=chunks,
        moments_sha256=file_hash(folder/'moments.npz')))
    got=archive.read_archived_operator('baseline',8,99,17)
    np.testing.assert_array_equal(got[0],m)
    np.testing.assert_array_equal(got[1],S)
    assert got[2]==.01
    npz_write(folder/'rows_0.npz',second_rows=np.zeros((4,8)))
    with pytest.raises(ValueError,match='integrity'):
        archive.read_archived_operator('baseline',8,99,17)


def test_production_checkpoint_pairing_decomposition_and_identity(tmp_path,monkeypatch):
    import json
    import pytest
    from simulations.extended import compute
    from simulations.extended.design import ENVIRONMENTS,BASIS_SEED,BASE_A
    from simulations.extended.population import population_reference
    from simulations.provenance import npz_write
    monkeypatch.setattr(compute,'OUTPUT',tmp_path)
    monkeypatch.setattr(compute,'calibrate',lambda: {'a':{name:BASE_A for name in ENVIRONMENTS}})
    basis=NystromBasis(rank=32,seed=BASIS_SEED)
    stats=raw_integrals(parameters(),basis,257,91)
    T=np.array([12,24]);penalties=BASE_A*T.astype(float)**(-.6)
    operators_by_name={}
    for name in ENVIRONMENTS:
        reference=population_reference(parameters(name),basis,stats,penalties)
        operators_by_name[name]=(reference['mean'],reference['second'],reference['floor'])
        npz_write(tmp_path/'population'/f'{name}_theory.npz',T=T,
            coefficients=reference['coefficients'].T,bias=reference['bias'])
    monkeypatch.setattr(compute,'load_operator',lambda name,*args:operators_by_name[name])
    compute.run_path(0,[32],T.tolist(),[],stage='production_baseline',
        run_hash='isolated-test',environment_names=['baseline'])
    compute.run_path(0,[32],T.tolist(),T.tolist(),stage='production_robustness',
        run_hash='isolated-test',environment_names=[n for n in ENVIRONMENTS if n!='baseline'])
    with np.load(tmp_path/'production_baseline/P32/rep_000.npz') as a,np.load(tmp_path/'production_robustness/P32/rep_000.npz') as b:
        assert json.loads(str(a['returns_hashes']))['baseline']==json.loads(str(b['returns_hashes']))['baseline']
        for z,names in ((a,['baseline']),(b,[n for n in ENVIRONMENTS if n!='baseline'])):
            assert float(z['maximum_normal_equation_error'])<1e-8
            for name in names:
                assert z[name+'_sr'].shape==(2,97)
                np.testing.assert_allclose(z[name+'_loss'][:,-1]-(1-parameters(name).q_star),
                    z[name+'_theory_decomposition'].sum(axis=1),atol=1e-10)
    with pytest.raises(ValueError,match='identity mismatch'):
        compute.run_path(0,[32],T.tolist(),[],stage='production_baseline',
            run_hash='different-science',environment_names=['baseline'])


def test_pilot_cleanup_requires_frozen_audits_and_preserves_deliverables(tmp_path,monkeypatch):
    import json
    import pytest
    from simulations.extended import advance
    from simulations.provenance import file_hash,json_write
    monkeypatch.setattr(advance,'OUTPUT',tmp_path)
    pilot=tmp_path/'pilot';pilot.mkdir()
    retained=['basis_P4096_seed7.npz','operator_baseline_P512_B7_Q32768_S9.npz',
        'spectra_baseline_P512.npz','basis_user_notes.npz']
    retained += [f'operator_{name}_P4096_B7_Q32768_S9.npz' for name in ('baseline','N300','N1200')]
    obsolete=['basis_P1024_seed7.npz','operator_N300_P1024_B7_Q32768_S9.npz',
        'operator_baseline_P4096_B7_Q131072_S10.npz']
    for name in retained+obsolete:
        (pilot/name).write_bytes(b'unchanged')
    checkpoint=tmp_path/'rep_000.npz';checkpoint.write_bytes(b'actual outcomes')
    with pytest.raises(FileNotFoundError):
        advance.cleanup_pilot_caches()
    assert all((pilot/name).exists() for name in obsolete)
    audit=tmp_path/'quadrature_audit.csv';audit.write_text('verified audit\n')
    json_write(tmp_path/'protocol.json',dict(rank=4096,basis_seed=7,population_groups=32768,
        population_seed=9,run_hash='frozen-study',preproduction_files={audit.name:file_hash(audit)}))
    advance.cleanup_pilot_caches()
    assert all((pilot/name).read_bytes()==b'unchanged' for name in retained)
    assert not any((pilot/name).exists() for name in obsolete)
    assert checkpoint.read_bytes()==b'actual outcomes'
    record=tmp_path/'preproduction_cache_cleanup.json'
    assert json.loads(record.read_text())['events'][0]['bytes_released']==3*len(b'unchanged')
    advance.cleanup_pilot_caches()
    assert len(json.loads(record.read_text())['events'])==1
    (pilot/obsolete[0]).write_bytes(b'regenerated')
    audit.write_text('changed audit\n')
    with pytest.raises(ValueError,match='evidence changed'):
        advance.cleanup_pilot_caches()
    assert (pilot/obsolete[0]).read_bytes()==b'regenerated'
