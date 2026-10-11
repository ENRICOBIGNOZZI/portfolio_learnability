"""Uniform Fourier perturbation bounds at every ACTUAL archived CV penalty.

For any fixed RKHS policy w, ||delta X|| <= delta_beta ||F|| because K(z,z)=1.
E||F||²=3, E||X||²<=c_beta+sigma_eps²/N, hence
||delta m||<=delta_beta||mu_F|| and ||delta S||<=2 A D+D².
This bound also holds after an orthogonal Nyström projection. At population
ridge, ||delta w_lambda||<=(delta_m+delta_S||w_lambda||)/lambda.
For fitted policies the empirical analogue is exported from paired histories.
These are bounds on sensitivity, not claims of equality or operator-source norms.
"""
import numpy as np
import pandas as pd
from simulations.dgp.rough import uniform_tail_bound, normalization
from simulations.extended.design import parameters,ENVIRONMENTS
from simulations.closeout.results import DEST
from simulations.provenance import json_write


def bounds():
    rows=[]
    policies=pd.read_csv(DEST/'policy_metrics.csv')
    policies=policies[policies.specification=='initial_cv']
    for M in (128,512,2048):
        for name in ENVIRONMENTS:
            p=parameters(name)
            tail=uniform_tail_bound(M)
            db=np.sqrt(2*p.c_beta/3)*5*abs(p.eta)*tail
            D=np.sqrt(3)*db;A=np.sqrt(p.c_beta+p.sigma_eps**2/p.N)
            dm=db*np.linalg.norm(p.mu_F);ds=2*A*D+D*D
            subset=policies[policies.environment==name]
            # Population ridge minimizes response-one regularized loss <=1.
            # lambda||w||² <=1; using 1 rather than q is a universal bound.
            spectral_path=DEST/'population_spectral'/f'{"baseline" if name.startswith("rho") else name}.npz'
            if spectral_path.exists():
                with np.load(spectral_path) as z:
                    values,projection=z['values'],z['mean_projection']
                normbound=np.sqrt(np.sum((projection[:,None]/(values[:,None]+subset.penalty.to_numpy()))**2,axis=0))
                norm_source='Exact numerical population-ridge norm from the same archived operator'
            else:
                normbound=1/np.sqrt(subset.penalty.to_numpy())
                norm_source='Conservative universal bound from regularized objective <=1'
            for j,(_,r) in enumerate(subset.iterrows()):
                delta_coef=(dm+ds*normbound[j])/r.penalty
                rows.append(dict(environment=name,index=int(r['index']),T=int(r['T']),M=M,penalty=r.penalty,
                    psi_tail_bound=tail,beta_uniform_error_bound=db,optimal_policy_uniform_error_bound=db*np.linalg.norm(p.mu_F)/(p.c_beta/3+p.sigma_eps**2/p.N),
                    population_mean_operator_error_bound=dm,population_second_operator_error_bound=ds,
                    relative_second_perturbation_over_lambda=ds/r.penalty,
                    population_ridge_coefficient_error_bound=delta_coef,population_ridge_norm=normbound[j],norm_source=norm_source,
                    bound_is_informative=bool(ds/r.penalty<.05),
                    scope='Conservative operator bound relative to ideal infinite series; not empirical CV-choice stability.'))
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'fourier_penalty_bounds.csv',index=False)
    json_write(DEST/'fourier_bound_status.json',dict(normalization=normalization(),bound_128=uniform_tail_bound(128),
        M128_operator_ratio_max=float(frame[frame.M==128].relative_second_perturbation_over_lambda.max()),
        M128_fraction_over_5_percent=float(np.mean(frame[frame.M==128].relative_second_perturbation_over_lambda>=.05)),
        coefficient_bounds_conservative=True,empirical_fits_and_CV_choice_status='not_run',
        scope='An operator bound at actual lambdas does not by itself certify fitted policies or argmin stability.'))

if __name__=='__main__':bounds()
