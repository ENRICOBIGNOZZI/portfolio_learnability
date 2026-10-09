"""Preproduction gates; the original 5% rule is never relaxed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.extended.design import OUTPUT,ENVIRONMENTS,RANKS,PILOT_PATHS,parameters,seed
from simulations.extended.compute import load_operator
from simulations.estimator.ridge import moment_metrics
from simulations.provenance import json_write,utc_now


def load_pilots(rank,count=PILOT_PATHS):
    records=[]
    for index in range(count):
        path=OUTPUT/'rank_pilot'/f'P{rank}'/f'rep_{index:03d}.npz'
        with np.load(path) as z:
            records.append({key:z[key] for key in z.files})
    return records


def rank_audit(count=PILOT_PATHS):
    rows=[]
    records={rank:load_pilots(rank,count) for rank in RANKS}
    for lower,higher in zip(RANKS[:-1],RANKS[1:]):
        for name in ENVIRONMENTS:
            low=np.stack([z[name+'_loss'] for z in records[lower]])-(1-parameters(name).q_star)
            high=np.stack([z[name+'_loss'] for z in records[higher]])-(1-parameters(name).q_star)
            diff=low-high
            se=diff.std(axis=0,ddof=1)/np.sqrt(count)
            floor=float(records[lower][0][name+'_floor'])
            higher_floor=float(records[higher][0][name+'_floor'])
            T=records[lower][0][name+'_T']
            for t,horizon in enumerate(T):
                for penalty in range(low.shape[-1]):
                    reference=low[:,t,penalty].mean()
                    discrepancy=abs(diff[:,t,penalty].mean())+1.96*se[t,penalty]
                    rows.append(dict(environment=name,T=int(horizon),penalty_index=penalty,
                        exact_theory_choice=penalty==low.shape[-1]-1,lower_rank=lower,higher_rank=higher,
                        replications=count,mean_regret_lower=reference,
                        mean_regret_higher=high[:,t,penalty].mean(),projection_floor=floor,
                        higher_projection_floor=higher_floor,floor_fraction=floor/reference,
                        paired_difference=diff[:,t,penalty].mean(),paired_MCSE=se[t,penalty],
                        paired_upper_bound=discrepancy,paired_fraction=discrepancy/reference,
                        floor_pass=bool(floor<=.05*reference),
                        paired_pass=bool(discrepancy<=.05*reference),
                        lower_rank_certified=bool(floor<=.05*reference and discrepancy<=.05*reference)))
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUT/'rank_audit.csv',index=False)
    table[table.exact_theory_choice].to_csv(OUTPUT/'rank_audit_theory.csv',index=False)
    return table


def quadrature_audit(rank=4096,count=PILOT_PATHS,groups=131072,quadrature_seed=None):
    quadrature_seed=seed('quadrature',1) if quadrature_seed is None else quadrature_seed
    records=load_pilots(rank,count)
    rows=[]
    diagnostic=[]
    for name in ENVIRONMENTS:
        m,S,floor=load_operator(name,rank,groups,quadrature_seed)
        coefficients=np.stack([z[name+'_theory_coefficients'] for z in records])
        n,nt,P=coefficients.shape
        loss,sr=moment_metrics(coefficients.reshape(n*nt,P).T,m,S)
        loss=loss.reshape(n,nt);sr=sr.reshape(n,nt)
        reference=np.stack([z[name+'_loss'][:,-1] for z in records])
        diff=reference-loss
        mean_regret=loss.mean(axis=0)-(1-parameters(name).q_star)
        se=diff.std(axis=0,ddof=1)/np.sqrt(n) if n>1 else np.full(nt,np.nan)
        bound=abs(diff.mean(axis=0))+1.96*se
        for t,T in enumerate(records[0][name+'_T']):
            rows.append(dict(environment=name,rank=rank,T=int(T),replications=count,
                alternate_groups=groups,alternate_seed=quadrature_seed,alternate_floor=floor,
                mean_regret_alternate=mean_regret[t],paired_mean_difference=diff[:,t].mean(),
                paired_MCSE=se[t],uncertainty_available=bool(n>1),
                discrepancy_upper_bound=bound[t],fraction=bound[t]/mean_regret[t],
                quadrature_pass=bool(bound[t]<=.05*mean_regret[t]),
                mean_SR_difference_annualized=float(np.sqrt(12)*(np.stack([z[name+'_sr'][:,-1] for z in records])[:,t]-sr[:,t]).mean())))
        r=records[0]
        if name+'_diagnostic_coefficients' in r:
            for k,T in enumerate(r[name+'_diagnostic_times']):
                l,s=moment_metrics(r[name+'_diagnostic_coefficients'][k],m,S)
                t=int(np.flatnonzero(r[name+'_T']==T)[0])
                for j in range(len(l)):
                    regret=l[j]-(1-parameters(name).q_star)
                    diagnostic.append(dict(environment=name,T=int(T),penalty_index=j,
                        replication=0,absolute_loss_difference=abs(r[name+'_loss'][t,j]-l[j]),
                        reference_regret=regret,fraction=abs(r[name+'_loss'][t,j]-l[j])/regret,
                        scope='Single independent pilot policy; exploratory full-path quadrature check'))
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUT/'quadrature_audit.csv',index=False)
    pd.DataFrame(diagnostic).to_csv(OUTPUT/'quadrature_full_path_diagnostic.csv',index=False)
    return table


def spectral_summary():
    records=[]
    for path in sorted((OUTPUT/'pilot').glob('spectra_*.npz')):
        # Keep actual unsmoothed eigenvalues and adjacent-eigenvalue gaps.
        with np.load(path) as z:
            for operator in z.files:
                values=z[operator]
                ratio=values[:-1]/np.maximum(values[1:],np.finfo(float).tiny)
                for j in np.argsort(ratio[:min(255,len(ratio))])[-12:]:
                    records.append(dict(source=path.name,operator=operator,j=int(j+1),
                        eigenvalue=float(values[j]),next_eigenvalue=float(values[j+1]),
                        adjacent_ratio=float(ratio[j])))
    pd.DataFrame(records).to_csv(OUTPUT/'spectral_clusters.csv',index=False)
    audits=[pd.read_csv(p) for p in sorted((OUTPUT/'pilot').glob('population_audit_*.csv'))]
    pd.concat(audits,ignore_index=True).drop_duplicates(
        ['environment','rank','groups','basis_seed','quadrature_seed','T']).to_csv(
            OUTPUT/'population_rank_seed_elasticity_audit.csv',index=False)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=PILOT_PATHS)
    parser.add_argument('--rank',type=int,default=4096)
    args=parser.parse_args()
    rank=rank_audit(args.count)
    quad=quadrature_audit(args.rank,args.count)
    spectral_summary()
    json_write(OUTPUT/'preproduction_audit_status.json',dict(completed_utc=utc_now(),
        pilot_replications=args.count,quadrature_all_pass=bool(quad.quadrature_pass.all()),
        tolerance=.05,highest_rank_has_no_higher_rank_certificate=True,
        note='A highest-rank backward comparison is not an independent upper-rank certificate. Projection floors always use the analytical economic optimum.'))

if __name__=='__main__':
    main()
