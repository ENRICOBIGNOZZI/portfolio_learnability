"""Independent arithmetic and provenance checks on the complete final tables."""
import json
import numpy as np
import pandas as pd

from simulations.extended.design import ENVIRONMENTS, PENALTIES, parameters
from simulations.initial_cv.design import OUTPUT, REFERENCE, protocol
from simulations.initial_cv.summarize import SPECIFICATIONS
from simulations.provenance import file_hash, json_write, utc_now


def audit():
    frozen=protocol()
    checkpoint_records=[]
    for stage in ('pilot','production'):
        record=json.loads((OUTPUT/f'collection_{stage}.json').read_text())
        assert record['status']=='completed' and record['conclusion']=='success'
        assert record['run_hash']==frozen['run_hash']
        checkpoint_records.extend(record['checkpoints'])
    assert sorted(r['index'] for r in checkpoint_records)==list(range(300))
    for r in checkpoint_records:
        assert file_hash(OUTPUT/r['path'])==r['sha256']
    verification=json.loads((OUTPUT/'summary_verification.json').read_text())
    assert verification['passed'] and verification['run_hash']==frozen['run_hash']
    assert verification['checkpoint_sha256']=={r['path']:r['sha256'] for r in checkpoint_records}
    for path,sha in verification['original_oracle_input_sha256'].items():
        assert file_hash(REFERENCE/path)==sha
    assert file_hash(REFERENCE/'production_resolution_final.csv')==verification['original_warning_sha256']
    gate=json.loads((OUTPUT/'pilot_gate.json').read_text())
    allocation=json.loads((OUTPUT/'production_allocation.json').read_text())
    assert gate['passed'] and gate['indices']==list(range(12))
    assert allocation['pilot_gate_sha256']==file_hash(OUTPUT/'pilot_gate.json')
    for path,sha in gate['checkpoint_sha256'].items():
        assert file_hash(OUTPUT/path)==sha
    summary=pd.read_csv(OUTPUT/'summary.csv')
    slopes=pd.read_csv(OUTPUT/'slopes_all_windows.csv')
    selected=pd.read_csv(OUTPUT/'selected_constants.csv')
    frequencies=pd.read_csv(OUTPUT/'selected_constant_frequencies.csv')
    paired=pd.read_csv(OUTPUT/'paired_cv_vs_fixed.csv')
    robust=pd.read_csv(OUTPUT/'paired_robustness.csv')
    warnings=pd.read_csv(OUTPUT/'numerical_warnings.csv')
    original_warnings=pd.read_csv(REFERENCE/'production_resolution_final.csv')
    expected_rows=0;checked_slopes=0;data={}
    def close(a,b):
        np.testing.assert_allclose(a,b,atol=2e-11,rtol=2e-9)
    for name in ENVIRONMENTS:
        with np.load(OUTPUT/'distributions'/f'{name}.npz') as z:
            data[name]={k:z[k] for k in z.files}
        d=data[name];times=d['T'];nT=len(times)
        assert str(d['run_hash'])==frozen['run_hash'] and int(d['replications'])==300
        np.testing.assert_array_equal(d['specifications'],SPECIFICATIONS)
        assert d['annual_SR'].shape==(300,nT,3)
        close(d['annual_gap']+d['annual_SR'],np.sqrt(12)*parameters(name).sr_star)
        close(d['relative_complexity']*times[None,:,None],d['empirical_complexity'])
        close(d['population_relative_complexity']*times[None,:,None],d['population_complexity'])
        choice=selected[selected.environment==name].sort_values('index')
        assert choice['index'].tolist()==list(range(300))
        close(choice.selected_a,d['selected_a'])
        close(d['penalties'][:,:,0],d['selected_a'][:,None]*times[None,:]**(-.6))
        close(d['penalties'][:,:,1],np.broadcast_to(frozen['design']['a_old'][name]*times**(-.6),(300,nT)))
        freq=frequencies[frequencies.environment==name]
        assert freq['count'].sum()==300
        for r in freq.itertuples(index=False):
            assert r.count==int((choice.selected_index==r.index).sum())
        with np.load(REFERENCE/'distributions'/f'{name}.npz') as old:
            optimum=old['annual_SR'][:,:,:96].mean(axis=0).argmax(axis=1)
            np.testing.assert_array_equal(optimum,d['oracle_penalty_index'])
            for t,j in enumerate(optimum):
                close(d['annual_SR'][:,t,2],old['annual_SR'][:,t,j])
                close(d['penalties'][:,t,2],PENALTIES[j])
        for metric in ('annual_SR','annual_gap','regret','empirical_complexity',
                       'population_complexity','relative_complexity','population_relative_complexity'):
            expected_rows+=nT*3
            for s,spec in enumerate(SPECIFICATIONS):
                rows=summary[(summary.environment==name)&(summary.metric==metric)&(summary.specification==spec)].sort_values('T')
                np.testing.assert_array_equal(rows['T'],times)
                values=d[metric][:,:,s]
                assert np.isfinite(values).all()
                for key,expected in [('mean',values.mean(axis=0)),('median',np.median(values,axis=0)),
                    ('p025',np.quantile(values,.025,axis=0)),('p975',np.quantile(values,.975,axis=0)),
                    ('sd',values.std(axis=0,ddof=1)),('mcse',values.std(axis=0,ddof=1)/np.sqrt(300))]:
                    close(rows[key],expected)
                for r in slopes[(slopes.environment==name)&(slopes.quantity==metric)&(slopes.specification==spec)].itertuples(index=False):
                    keep=(times>=r.T_min)&((times<=3240) if r.grid=='robustness_common' else True)
                    assert r.observations==keep.sum()
                    if r.available:
                        close(r.slope,np.polyfit(np.log(times[keep]),np.log(values[:,keep].mean(axis=0)),1)[0])
                        checked_slopes+=1
                        if spec=='population_oracle':
                            assert pd.isna(r.CI_low) and pd.isna(r.CI_high)
                    else:
                        assert keep.sum()<3
        for r in paired[paired.environment==name].itertuples(index=False):
            t=int(np.flatnonzero(times==r.T)[0]);delta=d[r.metric][:,t,0]-d[r.metric][:,t,1]
            close(r.mean,delta.mean());se=delta.std(ddof=1)/np.sqrt(300)
            close(r.mean_CI_low,delta.mean()-1.96*se);close(r.mean_CI_high,delta.mean()+1.96*se)
        for r in warnings[warnings.environment==name].itertuples(index=False):
            source=original_warnings[(original_warnings.environment==name)&(original_warnings['T']==r.T)].iloc[0]
            for key in ('floor_pass','paired_pass','forward_reference_available','certified'):
                assert getattr(r,'original_'+key)==bool(source[key])
            assert not r.strategy_specific_forward_rank_certificate
            assert r.floor_pass==(r.projection_floor<=.05*r.mean_regret)
            close(r.floor_fraction,r.projection_floor/r.mean_regret)
            q=parameters(name).q_star-r.projection_floor
            close(r.subspace_annual_SR,np.sqrt(12*q/(1-q)))
            close(r.representation_annual_gap,np.sqrt(12)*parameters(name).sr_star-r.subspace_annual_SR)
            j=SPECIFICATIONS.index(r.specification);t=int(np.flatnonzero(times==r.T)[0])
            close(r.fraction_mean_Sharpe_gap_numerical,r.representation_annual_gap/d['annual_gap'][:,t,j].mean())
    assert len(summary)==expected_rows
    for r in robust.itertuples(index=False):
        j=SPECIFICATIONS.index(r.specification);t=int(np.flatnonzero(data[r.environment]['T']==r.T)[0])
        delta=data[r.environment][r.metric][:,t,j]-data['baseline'][r.metric][:,t,j]
        close(r.mean,delta.mean());se=delta.std(ddof=1)/np.sqrt(300)
        close(r.mean_CI_low,delta.mean()-1.96*se);close(r.mean_CI_high,delta.mean()+1.96*se)
    json_write(OUTPUT/'arithmetic_audit.json',dict(passed=True,verified_utc=utc_now(),
        run_hash=frozen['run_hash'],replications_per_economy=300,economies=5,
        summary_rows_checked=expected_rows,slopes_checked=checked_slopes,
        paired_comparisons_checked=len(paired)+len(robust),original_warning_rows_checked=len(warnings),
        source_summary_sha256=file_hash(OUTPUT/'summary.csv'),
        scope='Independent table arithmetic, penalty invariance, oracle extraction, replication allocation, transfer hashes and preservation of original warnings. Not a new rank certificate.'))


if __name__=='__main__':
    audit()
