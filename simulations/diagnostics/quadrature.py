"""Predeclared sensitivity of evaluated headline risks to population quadrature.

Use the first 20 independent headline training paths, fixed learned policies,
and 8192 versus 32768 population groups. This never selects a penalty.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.config.design import design_for
from simulations.estimator.kernel import NystromBasis
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.experiments.monte_carlo import managed_path
from simulations.experiments.population import population_moments


_CONTEXT = None


def initialize(context):
    global _CONTEXT
    _CONTEXT = context


def evaluate(index):
    c = _CONTEXT
    d, env, basis = c['design'], c['environment'], c['basis']
    p = env.parameters()
    with np.load(c['directory']/'replications'/f'{index:04d}.npz') as saved:
        seed = int(saved['train_seed'])
        choice = saved['choice']
        old = saved['regret']
    train, _, _ = managed_path(p,basis,'baseline',seed,max(d.T))
    result=[]
    for t,T in enumerate(d.T):
        coefficients, _ = ridge_path(train[:T],np.asarray(d.penalties))
        risk, _ = moment_metrics(coefficients,c['mean'],c['second'])
        regret = risk-(1-p.q_star)
        for label,j in [('oracle',c['oracle'][t]),('selected',choice[t])]:
            result.append({'replication':index,'T':T,'policy':label,'original_regret':old[t,j],
                           'larger_quadrature_regret':regret[j],'difference':regret[j]-old[t,j]})
    return result


def quadrature_audit(output, workers=2):
    design=design_for('paper');env=design.cases[0]
    directory=Path(output)/'data/baseline'
    main=pd.read_csv(directory/'main_results.csv')
    penalties=np.asarray(design.penalties)
    oracle=np.array([np.argmin(abs(penalties-value)) for value in main.oracle_lambda])
    basis=NystromBasis(rank=env.rank,nu=env.nu,seed=design.basis_seed)
    mean,second=population_moments(env.parameters(),basis,groups=32768,seed=design.population_seed)
    context={'design':design,'environment':env,'basis':basis,'directory':directory,'oracle':oracle,'mean':mean,'second':second}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(context,)) as pool:
        rows=[row for batch in pool.map(evaluate,range(20)) for row in batch]
    frame=pd.DataFrame(rows)
    aggregate=frame.groupby(['T','policy'])[['original_regret','larger_quadrature_regret','difference']].mean().reset_index()
    aggregate['relative_difference']=aggregate.difference/aggregate.original_regret
    destination=Path(output)/'audit';destination.mkdir(parents=True,exist_ok=True)
    frame.to_csv(destination/'quadrature_path_sensitivity.csv',index=False)
    aggregate.to_csv(destination/'quadrature_sensitivity.csv',index=False)
    report={'passed':bool((aggregate.relative_difference.abs()<=.05).all()),'replications':20,
            'population_groups':[8192,32768],'relative_tolerance':.05,
            'max_absolute_relative_difference':float(aggregate.relative_difference.abs().max()),
            'selection_repeated':False,'reason':'Fixed policies and penalties; population evaluation only.'}
    (destination/'quadrature_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    report=quadrature_audit(Path('simulations/outputs/paper'))
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
