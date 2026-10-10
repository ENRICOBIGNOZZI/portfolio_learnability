"""Requirement-oriented numerical and artifact verification for the extended study."""
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT,ROOT,REFERENCE,ENVIRONMENTS,BASE_A,parameters
from simulations.extended.freeze import science_hashes
from simulations.extended.statistics import gap_slopes
from simulations.extended.population import spectral_path
from simulations.extended.verify_summaries import verify as verify_all_summaries
from simulations.provenance import digest,file_hash,json_write,utc_now

FIGURES=['Figure0_economic_spectrum','Figure1_learnability','Figure2_complexity',
    'Figure3_sharpe_complexity','FigureR1_N_learnability','FigureR2_rho_learnability',
    'FigureR3_N_complexity','FigureR4_rho_complexity','FigureR5_N_sharpe_complexity',
    'FigureR6_rho_sharpe_complexity']


def verify(require_figures=True):
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    scientific={k:v for k,v in protocol.items() if k not in ('run_hash','frozen_before_production_utc')}
    if digest(scientific)!=protocol['run_hash']:
        raise ValueError('Frozen protocol hash does not match its scientific contents.')
    if science_hashes()!=protocol['source_hashes']:
        raise ValueError('Frozen scientific source identity changed.')
    assert protocol['replications']==300 and protocol['one_common_rank']
    assert protocol['a']['baseline']==BASE_A
    assert set([2160,3240,4860]).issubset(protocol['baseline_T'])
    assert 2160 in protocol['robustness_T'] and 3240 in protocol['robustness_T']
    assert len(protocol['penalties'])==96
    seeds=json.loads((OUTPUT/'seed_manifest.json').read_text())['replications']
    assert [row['index'] for row in seeds]==list(range(300))
    for key in ('training_seed','extra_groups_seed','extra_noise_seed'):
        assert len({row[key] for row in seeds})==300
    checks=[]
    for name in ENVIRONMENTS:
        p=parameters(name)
        # The economy is fixed explicitly; changing N never recalibrates mu.
        assert p.mu_F==parameters().mu_F
        np.testing.assert_array_equal(p.factor_covariance,parameters().factor_covariance)
        stage='production_baseline' if name=='baseline' else 'production_robustness'
        folder=OUTPUT/stage/f'P{protocol["rank"]}'
        assert len(list(folder.glob('rep_*.npz')))==300
        with np.load(OUTPUT/'distributions'/f'{name}.npz') as z:
            data={k:z[k] for k in z.files}
        with np.load(OUTPUT/'population'/f'{name}_theory.npz') as z:
            pop={k:z[k] for k in z.files}
        T=data['T'];expected=protocol['baseline_T'] if name=='baseline' else protocol['robustness_T']
        np.testing.assert_array_equal(T,expected)
        assert data['annual_SR'].shape==(300,len(T),97)
        for key in ('annual_SR','annual_gap','regret','empirical_complexity','relative_complexity'):
            assert np.isfinite(data[key]).all()
        assert np.max(data['annual_SR'])<=p.sr_star*np.sqrt(12)+1e-8
        assert np.min(data['regret'])>=float(pop['projection_floor'])-1e-8
        C=data['empirical_complexity']
        assert (C>=-1e-10).all()
        assert (C<=np.minimum(T,protocol['rank'])[None,:,None]+1e-7).all()
        np.testing.assert_allclose(data['relative_complexity'],C/T[None,:,None],atol=1e-14)
        np.testing.assert_allclose(data['annual_gap'],p.sr_star*np.sqrt(12)-data['annual_SR'],atol=1e-14)
        np.testing.assert_allclose(data['regret'][:,:,-1],data['theory_population_bias']+
            data['theory_estimation_norm']+data['theory_cross_term'],atol=1e-10,rtol=1e-8)
        popC,E,local=spectral_path(pop['eigenvalues'],pop['penalties'])
        np.testing.assert_allclose(popC,pop['complexity'],rtol=1e-12)
        np.testing.assert_allclose(local,pop['local_T_elasticity'],rtol=1e-12)
        np.testing.assert_allclose(pop['penalties'],protocol['a'][name]*T.astype(float)**(-.6),rtol=1e-14)
        history_hashes=set()
        for index in range(300):
            with np.load(folder/f'rep_{index:03d}.npz') as z:
                assert int(z['index'])==index
                assert int(z['training_seed'])==seeds[index]['training_seed']
                assert int(z['rank'])==protocol['rank']
                description=json.loads(str(z['description']))
                assert description['run_hash']==protocol['run_hash']
                assert str(z['identity'])==digest(description)
                assert float(z['maximum_normal_equation_error'])<=1e-8
                history_hashes.add(json.loads(str(z['returns_hashes']))[name])
                np.testing.assert_array_equal(z[name+'_sr']*np.sqrt(12),data['annual_SR'][index])
                np.testing.assert_array_equal(z[name+'_empirical_complexity'],C[index])
        assert len(history_hashes)==300
        table=pd.read_csv(OUTPUT/'theory_distributions.csv')
        for metric in ('annual_SR','annual_gap','empirical_complexity'):
            rows=table[(table.environment==name)&(table.metric==metric)].sort_values('T')
            values=data[metric][:,:,-1]
            np.testing.assert_allclose(rows['mean'],values.mean(axis=0),atol=1e-12)
            np.testing.assert_allclose(rows['median'],np.median(values,axis=0),atol=1e-12)
            np.testing.assert_allclose(rows['sd'],values.std(axis=0,ddof=1),atol=1e-12)
            np.testing.assert_allclose(rows['p025'],np.percentile(values,2.5,axis=0),atol=1e-12)
            np.testing.assert_allclose(rows['p975'],np.percentile(values,97.5,axis=0),atol=1e-12)
            np.testing.assert_allclose(rows['mcse'],values.std(axis=0,ddof=1)/np.sqrt(300),atol=1e-12)
        checks.append(dict(environment=name,replications=300,annual_SR_star=p.sr_star*np.sqrt(12),
            full_penalty_distribution_shape=list(data['annual_SR'].shape),
            economic_ceiling_pass=True,regret_floor_pass=True,normal_equations_pass=True,
            percentiles_independently_recomputed=True,penalty_rule_verified=True))
    for name in ('rho000','rho075'):
        with np.load(OUTPUT/'population'/'baseline_theory.npz') as base,np.load(OUTPUT/'population'/f'{name}_theory.npz') as variant:
            np.testing.assert_array_equal(base['eigenvalues'],variant['eigenvalues'])
            np.testing.assert_array_equal(base['complexity'][:len(variant['T'])],variant['complexity'])
    complete_summary_audit=verify_all_summaries()
    if require_figures:
        for suffix in ('png','pdf'):
            paths=list((OUTPUT/'figures').glob('*.'+suffix))
            assert sorted(p.stem for p in paths)==sorted(FIGURES)
            assert all(p.stat().st_size>10000 for p in paths)
    result=dict(passed=True,run_hash=protocol['run_hash'],checks=checks,
        complete_summary_audit=complete_summary_audit,
        expected_total_economic_replications=1500,ten_figures_present=require_figures,
        visual_review='Must be recorded separately after inspecting the actual rendered images.',
        numerical_resolution='See production_resolution.csv; passing this verifier does not override failed 5% approximation gates.',
        verified_utc=utc_now())
    json_write(OUTPUT/'verification.json',result)
    return result


def manifest():
    files={}
    for path in sorted(OUTPUT.rglob('*')):
        if not path.is_file() or path.name in ('manifest.json','.run.lock'):
            continue
        if path.parent.name=='pilot' and path.name.startswith(('basis_','integrals_','operator_')):
            continue
        files[str(path.relative_to(OUTPUT))]=dict(sha256=file_hash(path),bytes=path.stat().st_size)
    import scipy
    import matplotlib
    analysis_sources={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'simulations'/'extended').glob('*.py')}
    json_write(OUTPUT/'manifest.json',dict(schema='rich6d-extended-artifacts/1',files=files,analysis_sources=analysis_sources,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,platform=platform.platform(),created_utc=utc_now(),
        caches='Large pilot basis/integral/operator caches are regenerable and excluded; seeds and all intended outcomes are retained.'))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only',action='store_true')
    args=parser.parse_args()
    verify(require_figures=not args.data_only)
    manifest()
