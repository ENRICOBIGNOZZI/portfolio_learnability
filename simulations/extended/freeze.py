"""Freeze one common production rank after completed numerical/resource pilots."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from simulations.extended.design import (OUTPUT,ROOT,REFERENCE,ENVIRONMENTS,PENALTIES,
    T_REQUIRED,T_CANDIDATE,T_ROBUSTNESS,BASE_A,BASIS_SEED,WINDOWS,parameters,seed)
from simulations.extended.calibration import calibrate
from simulations.extended.compute import load_operator
from simulations.extended.population import spectral_path
from simulations.provenance import file_hash,json_write,npz_write,digest,utc_now
from simulations.run_rough import seed as legacy_seed


def science_hashes():
    names=['simulations/extended/'+name+'.py' for name in
           ('design','population','histories','calibration','compute','freeze','production')]
    names += ['simulations/dgp/balanced.py','simulations/dgp/rough.py',
              'simulations/estimator/kernel.py','simulations/estimator/ridge.py',
              'simulations/provenance.py','simulations/run_rough.py']
    return {name:file_hash(ROOT/name) for name in names}


def freeze(rank=4096,include_7290=False):
    if (OUTPUT/'protocol.json').exists():
        raise ValueError('Production is already frozen; do not overwrite its scientific identity.')
    status=json.loads((OUTPUT/'preproduction_audit_status.json').read_text())
    if status['pilot_replications']!=12:
        raise ValueError('Complete all 12 predeclared rank pilots first.')
    if not status['quadrature_all_pass']:
        raise ValueError('Resolve failed quadrature cells before freezing production.')
    rank_table=pd.read_csv(OUTPUT/'rank_audit_theory.csv')
    times=list(T_CANDIDATE if include_7290 else T_REQUIRED)
    forward=rank_table[rank_table.lower_rank==rank]
    backward=rank_table[rank_table.higher_rank==rank]
    # Highest rank has no independent higher-rank reference. Never label its
    # backward difference as a successful forward numerical certificate.
    has_forward=len(forward)>0
    resolution=[]
    for name in ENVIRONMENTS:
        grid=times if name=='baseline' else list(T_ROBUSTNESS)
        source=(forward if has_forward else backward)
        source=source[source.environment==name]
        for T in grid:
            row=source[source['T']==T].iloc[0]
            floor=row.projection_floor if has_forward else row.higher_projection_floor
            regret=row.mean_regret_lower if has_forward else row.mean_regret_higher
            floor_pass=bool(floor<=.05*regret)
            paired_pass=bool(row.paired_upper_bound<=.05*regret)
            resolution.append(dict(environment=name,T=T,rank=rank,projection_floor=floor,
                mean_pilot_regret=regret,floor_fraction=floor/regret,floor_pass=floor_pass,
                paired_upper_bound=row.paired_upper_bound,paired_pass=paired_pass,
                forward_reference_available=has_forward,
                certified=bool(floor_pass and paired_pass and has_forward)))
    resolved=pd.DataFrame(resolution)
    if include_7290 and not resolved[(resolved.environment=='baseline')&(resolved['T']==7290)]['certified'].all():
        raise ValueError('Optional T=7290 requires numerical feasibility; it is not certified.')
    calibration=calibrate()
    spec=dict(schema='rich6d-extended-production/1',reference_commit='2d59c3324569a0e58557ffa03a2c2c0319d65650',
        rank=rank,baseline_T=times,robustness_T=list(T_ROBUSTNESS),replications=300,
        environments={name:parameters(name).to_dict() for name in ENVIRONMENTS},
        a=calibration['a'],baseline_a_fixed=BASE_A,basis_seed=BASIS_SEED,
        population_groups=32768,population_seed=seed('population'),
        penalties=PENALTIES.tolist(),exact_theory_choice='a*T**(-0.6), appended to the full 96-point diagnostic grid',
        annualization='sqrt(12) times marginal monthly Sharpe; not compounded annual-return Sharpe',
        periods_per_year=12,primary_bands='Central 95% replication percentiles, never confidence intervals for the mean',
        slope_windows=WINDOWS,slope_uncertainty='Full cross-T covariance delta method, checked by replication jackknife',
        main_path_T=[60,240,720,1440],robustness_path_T=[60,1440],
        omitted_optional_T7290=not include_7290,
        optional_horizon_rule='Include 7290 only after numerical and resource feasibility; never by fitted slope.',
        one_common_rank=True,numerical_tolerance=.05,
        all_required_cells_certified=bool(resolved.certified.all()),
        numerical_limitation=('None identified at checked cells.' if resolved.certified.all() else
            'The highest investigated rank is a fixed computational approximation, not a resolved infinite-dimensional benchmark. Failed floor cells and the absence of a higher-rank certificate remain explicit; high-T rate claims are conditional.'),
        selection='Fixed rank chosen from preproduction approximation and resource evidence, never from favorable slopes or Sharpe maximization.',
        execution_order=['300 baseline replications','300 paired replications for each of four robustness environments'],
        independent_replications=True,paired_innovations=True,
        source_hashes=science_hashes(),
        preproduction_files={name:file_hash(OUTPUT/name) for name in
            ('pilot_design.json','calibration.json','rank_audit.csv','quadrature_audit.csv','preproduction_audit_status.json')})
    # Record all deterministic production quantities before the first fit.
    population_rows=[]
    for name in ENVIRONMENTS:
        grid=np.asarray(times if name=='baseline' else T_ROBUSTNESS,dtype=int)
        m,S,floor=load_operator(name,rank,spec['population_groups'],spec['population_seed'])
        values,vectors=eigh(S)
        projection=vectors.T@m
        lam=calibration['a'][name]*grid.astype(float)**(-.6)
        coefficients=(vectors@(projection[:,None]/(values[:,None]+lam))).T
        bias=parameters(name).q_star-2*coefficients@m+np.sum(coefficients*(coefficients@S),axis=1)
        C,E,local=spectral_path(values,lam)
        gridC,gridE,_=spectral_path(values,PENALTIES)
        npz_write(OUTPUT/'population'/f'{name}_theory.npz',T=grid,penalties=lam,
            coefficients=coefficients,bias=bias,complexity=C,elasticity=E,local_T_elasticity=local,
            eigenvalues=values,projection_floor=floor,diagnostic_grid_complexity=gridC,
            diagnostic_grid_penalties=PENALTIES)
        for k,T in enumerate(grid):
            population_rows.append(dict(environment=name,T=int(T),prescribed_lambda=lam[k],
                complexity=C[k],relative_complexity=C[k]/T,lambda_elasticity=E[k],
                local_T_elasticity=local[k],regularization_bias=bias[k],projection_floor=floor,
                rank=rank,annual_SR_star=parameters(name).sr_star*np.sqrt(12),
                finite_rank_annual_SR_ceiling=np.sqrt(12*(parameters(name).q_star-floor)/(1-parameters(name).q_star+floor))))
    pd.DataFrame(population_rows).to_csv(OUTPUT/'population_summary.csv',index=False)
    resolved.to_csv(OUTPUT/'production_resolution.csv',index=False)
    seeds=[dict(index=i,training_seed=legacy_seed('train',i),pairing_index=i,
                extra_groups_seed=seed('extra_groups',i),extra_noise_seed=seed('extra_noise',i)) for i in range(300)]
    json_write(OUTPUT/'seed_manifest.json',dict(replications=seeds,
        population_seed=spec['population_seed'],basis_seed=BASIS_SEED,
        calibration_seed=calibration['seed'],families_independent=True))
    run_hash=digest(spec)
    json_write(OUTPUT/'protocol.json',dict(spec,run_hash=run_hash,frozen_before_production_utc=utc_now()))
    print('Frozen production',run_hash,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,default=4096,choices=[512,1024,2048,4096])
    parser.add_argument('--include-7290',action='store_true')
    args=parser.parse_args()
    freeze(args.rank,args.include_7290)

if __name__=='__main__':
    main()
