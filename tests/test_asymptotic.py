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


def test_high_precision_direct_evaluation_matches_the_independent_operator(tmp_path):
    from simulations.asymptotic import evaluation_job
    from simulations.asymptotic_precision import direct_evaluation
    c=config();c.update(evaluation_states=64,evaluation_block=4,precision_evaluation_block=2)
    N=7;rank=8;b=Basis('nystrom',rank,c)
    coeff=np.random.default_rng(291).normal(size=(rank,3))*100
    path=evaluation_job((c,N,'nystrom',rank,str(tmp_path)))
    with np.load(path) as data:
        expected=evaluate_coefficients(coeff,{k:data[k] for k in data.files})
    actual=direct_evaluation(c,N,b,coeff)
    for key in expected:
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-10,atol=1e-12)


def test_memory_bounded_solver_and_certificate_agree_with_full_kernel(tmp_path):
    from simulations.asymptotic_precision import training_job,trace_certificate
    c=config();c.update(fixed_N=5,fixed_rho=.5,T_train=[3,8,17],
        spectrum_T_train=17,precision_frozen_C0=1e-5,stream_block=8)
    rank=8;b=Basis('nystrom',rank,c)
    output=training_job((c,rank,0,str(tmp_path)))
    blocks=list(draw_blocks(5,.5,0,c['fidelity_seed'],17,8))
    z=np.concatenate([p[1] for p in blocks]);r=np.concatenate([p[2] for p in blocks])
    raw=managed_block(b,z,r)
    full=matern_kernel(z.reshape(-1,3),z.reshape(-1,3)).reshape(17,5,17,5)
    managed=np.einsum('in,injm,jm->ij',r,full,r)/25
    with np.load(output) as a:
        for i,n in enumerate(c['T_train']):
            penalties=c['precision_frozen_C0']*n**(-2/3)*np.array(c['lambda_multipliers'])
            coeff,C,_=fit_statistics(b,raw[:n].mean(axis=0),raw[:n].T@raw[:n]/n,penalties)
            np.testing.assert_allclose(a['theta'][:,i*3:(i+1)*3],coeff,rtol=1e-8,atol=1e-8)
            np.testing.assert_allclose(a['complexity'][i],C,rtol=1e-9)
            np.testing.assert_allclose(a['full_trace'][i],np.trace(managed[:n,:n])/n,atol=1e-16)
            eig=np.maximum(np.linalg.eigvalsh(managed[:n,:n]/n),0)
            exact=np.sum(eig[:,None]/(eig[:,None]+penalties),axis=0)
            upper,_=trace_certificate(a['full_trace'][i],a['projected_trace'][i],penalties,C)
            assert (exact-C>=-1e-9).all()
            assert (exact-C<=upper+1e-9).all()
    assert not list(tmp_path.glob('*.dat'))


def test_small_cutoff_analysis_keeps_the_frozen_primary_and_every_window(tmp_path):
    import pandas as pd
    from simulations.asymptotic_analysis import analysis_design
    from simulations.asymptotic_report import tail_fits
    c=config();original=json.loads(json.dumps(c))
    spec=dict(primary_T_train_min=1600,sensitivity_T_min=[100,200,400,800,1600,3200,6400],
              reason='User requested smaller secondary cutoffs')
    (tmp_path/'analysis_spec.json').write_text(json.dumps(spec))
    effective,provenance=analysis_design(c,tmp_path)
    assert c==original and effective['asymptotic_T_min']==1600
    assert provenance['sensitivity_T_min']==spec['sensitivity_T_min']
    effective['bootstrap_resamples']=20
    times=[100,200,400,800,1600,3200,6400,12800,25600]
    frame=pd.DataFrame([dict(N=600,rho=.5,trial=j,lambda_multiplier=1,C_lambda=1e-6,
        T_train=t,SR_gap=(j+1)*t**(-2/3),complexity=(j+1)*t**(1/3))
        for j in range(3) for t in times])
    penalties=np.geomspace(1e-6*25600**(-2/3),1e-6*100**(-2/3),65)
    spectrum=pd.DataFrame([dict(N=600,rho=.5,trial=j,lambda_reg=lam,
        complexity=(j+1)*lam**(-.5)) for j in range(3) for lam in penalties])
    fits,windows,_=tail_fits(effective,frame,spectrum)
    for cutoff in spec['sensitivity_T_min']:
        assert set(windows[windows.T_train_min==cutoff].metric)=={'SR_T','C_T','SR_C','C_lambda'}
        for name,power in [('SR_T',-2/3),('C_T',1/3),('SR_C',-2)]:
            fit=fits[f'N600_rho0.5_{name}_Tmin{cutoff}']
            assert fit['T_train_fit_grid']==[t for t in times if t>=cutoff]
            np.testing.assert_allclose(fit['exponent'],power,atol=1e-8)
        np.testing.assert_allclose(fits[f'C_lambda_Tmin{cutoff}']['exponent'],-.5,atol=1e-8)
    spec['primary_T_train_min']=200
    (tmp_path/'analysis_spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='primary cutoff'):
        analysis_design(c,tmp_path)


def test_precision_selection_reports_failure_instead_of_selecting_a_good_slope():
    import pandas as pd
    from simulations.precision_report import select_rank
    c=config();c.update(precision_ranks=[4096,8192],precision_frozen_C0=1e-6,
        precision_relative_C_error_tolerance=.02)
    lam=1e-6*3200**(-2/3)
    certificates=pd.DataFrame([dict(rank=r,lambda_multiplier=1,relative_C_error_bound=e)
        for r,e in [(4096,.021),(8192,.019)]])
    spectra=pd.DataFrame([dict(rank=r,lambda_reg=lam,relative_C_error_bound=e)
        for r,e in [(4096,.018),(8192,.019)]])
    selected=select_rank(c,certificates,spectra)
    assert selected['selected_rank']==8192
    assert selected['selection_uses_slopes_or_R_squared'] is False
    spectra['relative_C_error_bound']=.03
    failed=select_rank(c,certificates,spectra)
    assert failed['selected_rank'] is None
    assert failed['benchmark_curve_rank']==8192


def test_precision_sharpe_complexity_percentile_keeps_the_balanced_pairing():
    import pandas as pd
    import matplotlib.pyplot as plt
    from simulations.precision_report import balanced
    frame=pd.DataFrame([dict(trial=j,T_train=t,complexity=(j+1)*t,
        SR_hat=(j+1)/t) for t in [100,200] for j in range(4)])
    fig,ax=plt.subplots()
    balanced(ax,frame,'SR_hat','paired','blue')
    np.testing.assert_allclose(ax.lines[0].get_xdata(),[250,500])
    np.testing.assert_allclose(ax.lines[0].get_ydata(),[.025,.0125])
    percentile=next(line for line in ax.lines if line.get_gid()=='trial_percentile_95')
    np.testing.assert_allclose(percentile.get_xdata(),[250,500])
    np.testing.assert_allclose(percentile.get_ydata(),[.0385,.01925])
    band=ax.collections[0].get_paths()[0].vertices[:,1]
    sd=np.std([.01,.02,.03,.04],ddof=1)
    np.testing.assert_allclose(band.max(),.025+sd)
    plt.close(fig)


def test_prefetch_handoff_is_atomic_and_preserves_existing_main_outputs(tmp_path):
    from simulations.asymptotic_prefetch import handoff
    staged=tmp_path/'staged';target=tmp_path/'target';marker=tmp_path/'completed.json'
    staged.write_bytes(b'completed policy')
    assert handoff(staged,target,marker)=='published'
    assert target.read_bytes()==b'completed policy' and not staged.exists()
    staged.write_bytes(b'duplicate computation')
    assert handoff(staged,target,marker)=='main_cache_already_present'
    assert target.read_bytes()==b'completed policy' and not staged.exists()
    target.unlink();marker.write_text('{}');staged.write_bytes(b'late policy')
    assert handoff(staged,target,marker)=='cell_already_exported'
    assert not target.exists() and not staged.exists()


def test_prefetched_training_is_identical_to_the_frozen_main_engine(tmp_path):
    from simulations import asymptotic as engine
    from simulations.asymptotic_prefetch import prefetch_job
    c=config();c.update(fixed_N=5,fixed_rho=.5,cells=[[5,.5]],T_train=[3,8,17],
        T_train_reference=[3,8,17],spectrum_T_train=17,stream_block=8)
    cache=tmp_path/'out/cache';cache.mkdir(parents=True)
    direct=tmp_path/'direct';direct.mkdir()
    rank=8;constant=1e-5
    result=prefetch_job((c,5,.5,0,rank,constant,str(cache),engine.scientific_hash(c)))
    assert result['state']=='published'
    expected=engine.training_job((c,5,.5,0,c['confirmation_seed'],c['T_train'],
        'nystrom',rank,constant,str(direct),'confirmation'))
    with np.load(expected) as a,np.load(cache/Path(expected).name) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key],b[key])
    with pytest.raises(RuntimeError,match='source changed'):
        prefetch_job((c,5,.5,0,rank,constant,str(cache),'wrong scientific signature'))


def test_precision_recovery_only_retries_explicit_disk_errors():
    from simulations.asymptotic_precision_pipeline import disk_failure
    assert disk_failure('RuntimeError: Insufficient free disk for bounded-memory managed-feature scratch')
    assert disk_failure('OSError: [Errno 28] No space left on device')
    assert not disk_failure('RuntimeError: Precision code or design changed; use a new output directory')
    assert not disk_failure('ArithmeticError: Nonpositive evaluation variance')


def test_precision_resource_wait_needs_two_consecutive_sufficient_measurements(tmp_path):
    from unittest.mock import patch
    from types import SimpleNamespace
    from simulations.asymptotic_precision_pipeline import disk_requirement,wait_for_disk
    c=config();c.update(T_train=[100],trials=2)
    required=disk_requirement(c,8,tmp_path)
    states=[];state={}
    samples=[0,required+1024**3,0,required+1024**3,required+1024**3]
    with patch('simulations.asymptotic_precision_pipeline.shutil.disk_usage',
        side_effect=[SimpleNamespace(free=v) for v in samples]),patch('simulations.asymptotic_precision_pipeline.time.sleep') as sleep:
        wait_for_disk(c,8,tmp_path,state,lambda:states.append(dict(state)))
    assert len(states)==5 and sleep.call_count==4
    assert states[-1]['free_disk_bytes']>=states[-1]['required_disk_bytes']
    for j in range(2):
        (tmp_path/f'precision_nystrom8_trial{j:03d}.npz').write_bytes(b'atomic cache placeholder')
    assert disk_requirement(c,8,tmp_path)<required


def test_rho_constants_use_only_shared_times_and_keep_bootstrap_pairing():
    import pandas as pd
    from simulations.asymptotic_report import rho_rate_constants
    from simulations.asymptotic_validate import validate_rho_constants
    c=config();c.update(trials=4,asymptotic_T_min=200,bootstrap_resamples=200)
    scales=np.array([.8,1.,1.2,1.4]);amplitudes={0.:2.,.5:3.,.95:5.}
    rows=[]
    for rho,a in amplitudes.items():
        times=[100,200,400,800,1600]+([3200] if rho==.5 else [])
        for j,scale in enumerate(scales):
            for t in times:
                # A distorted reference-only extension must not bias comparison.
                gap=a*scale*t**(-2/3)*(10 if t==3200 else 1)
                rows.append(dict(N=600,rho=rho,trial=j,T_train=t,lambda_multiplier=1,SR_gap=gap))
    table,audit=rho_rate_constants(c,pd.DataFrame(rows))
    assert validate_rho_constants(c,pd.DataFrame(rows),table,audit)==[200,400,800,1600]
    assert audit['common_T_train_grid']==[200,400,800,1600]
    assert audit['primary_fit_unchanged'] and audit['positive_bootstrap_reference_fraction']==1
    for rho,a in amplitudes.items():
        row=table[table.rho==rho].iloc[0]
        np.testing.assert_allclose(row.amplitude_mean,a*scales.mean(),atol=1e-12)
        np.testing.assert_allclose(row.amplitude_std,np.std(a*scales,ddof=1),atol=1e-12)
        np.testing.assert_allclose(row.amplitude_p95,np.quantile(a*scales,.95),atol=1e-12)
        # Shared trial shocks make this ratio constant in every paired resample.
        np.testing.assert_allclose([row.ratio_to_reference,row.ratio_ci_low,row.ratio_ci_high],a/3,atol=1e-12)
    tampered=table.copy();tampered.loc[tampered.rho==.5,'amplitude_std']/=np.sqrt(c['trials'])
    with pytest.raises(AssertionError):
        validate_rho_constants(c,pd.DataFrame(rows),tampered,audit)


def test_rho_constants_leave_zero_reference_ratios_undefined():
    import pandas as pd
    from simulations.asymptotic_report import rho_rate_constants
    from simulations.asymptotic_validate import validate_rho_constants
    c=config();c.update(trials=2,asymptotic_T_min=100,bootstrap_resamples=20)
    frame=pd.DataFrame([dict(N=600,rho=rho,trial=j,T_train=t,lambda_multiplier=1,
        SR_gap=0. if rho==.5 else t**(-2/3))
        for rho in c['rho_values'] for j in range(2) for t in [100,200,400]])
    table,audit=rho_rate_constants(c,frame)
    assert validate_rho_constants(c,frame,table,audit)==[100,200,400]
    assert audit['positive_bootstrap_reference_fraction']==0
    assert table[['ratio_to_reference','ratio_ci_low','ratio_ci_high']].isna().all().all()


def test_precision_completed_rank_audit_does_not_require_or_select_other_ranks(tmp_path):
    import hashlib
    from simulations.asymptotic_precision import training_job,collect
    from simulations.precision_validate import validate
    c=config();c.update(fixed_N=9,fixed_rho=.5,T_train=[8,12,17,24],
        spectrum_T_train=24,spectrum_fit_T_min=8,spectrum_fit_T_max=24,
        asymptotic_T_min=12,sensitivity_T_min=[8,12],trials=2,
        precision_ranks=[8,16],precision_frozen_C0=1e-5,
        precision_relative_C_error_tolerance=.02,evaluation_states=64,
        precision_evaluation_block=2,stream_block=8,bootstrap_resamples=40)
    (tmp_path/'config.json').write_text(json.dumps(c))
    cache=tmp_path/'cache';cache.mkdir()
    for j in range(c['trials']):training_job((c,8,j,str(cache)))
    collect(c,8,tmp_path)
    result=validate(tmp_path,rank=8)
    assert result['passed'] and result['rank_replications']==2
    assert not result['full_precision_study_complete'] and not result['report_verified']
    assert not (tmp_path/'precision_selection.json').exists()
    assert not (tmp_path/'precision_nystrom16_completed.json').exists()
    # Even an internally rehashed export cannot substitute unpaired/fabricated CIs.
    path=tmp_path/'precision_nystrom8_fits.json';fits=json.loads(path.read_text())
    fit=fits['SR_C_Tmin8'];fit['exponent_ci95']=[-99.,99.]
    path.write_text(json.dumps(fits))
    marker=tmp_path/'precision_nystrom8_completed.json';record=json.loads(marker.read_text())
    record['files_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    marker.write_text(json.dumps(record))
    with pytest.raises(AssertionError):validate(tmp_path,rank=8)
