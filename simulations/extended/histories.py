"""Nested economic histories paired across N and persistence.

The baseline stream is bit-for-bit the existing 100-replication stream. Larger
N adds independent groups; smaller N takes whole triplets. Persistence variants
share innovations but each follows its own stationary AR(1). No economy uses
estimated cross-sectional ranks or a simulated out-of-sample evaluation panel.
"""
import hashlib
import numpy as np

from simulations.dgp.balanced import (BalancedFactorDGP, balanced_triplets,
    population_ranks, beta)
from simulations.extended.design import parameters, seed


class PairedHistory:
    def __init__(self,training_seed,pairing_index):
        self.base=BalancedFactorDGP(parameters(),training_seed)
        self.extra_rng=np.random.default_rng(seed('extra_groups',pairing_index))
        self.noise_rng=np.random.default_rng(seed('extra_noise',pairing_index))
        self.extra=self.extra_rng.standard_normal((200,6))
        self.states={'rho000':self.base.state.copy(),'rho075':self.base.state.copy()}
        self.hashers={name:hashlib.sha256() for name in
                      ('baseline','N300','N1200','rho000','rho075')}

    def step(self,robustness=True):
        b=self.base
        state=b.state.copy()
        p=b.parameters
        # Keep the original baseline RNG calls and arithmetic exactly.
        z=balanced_triplets(population_ranks(state))
        loadings=beta(z,p)
        factors=np.asarray(p.mu_F)+b._factor_root@b.factor_rng.standard_normal(3)
        epsilon=p.sigma_eps*b.epsilon_rng.standard_normal(p.N)
        returns=loadings@factors+epsilon
        result={'baseline':(z,returns)}
        if robustness:
            result['N300']=(z[:300],returns[:300])
            ze=balanced_triplets(population_ranks(self.extra))
            re=beta(ze,p)@factors+p.sigma_eps*self.noise_rng.standard_normal(600)
            result['N1200']=(np.vstack((z,ze)),np.r_[returns,re])
            for name,s in self.states.items():
                zr=balanced_triplets(population_ranks(s))
                result[name]=(zr,beta(zr,p)@factors+epsilon)
        innovation=b.characteristic_rng.standard_normal(state.shape)
        b.state=b._rho*state+b._innovation_scale*innovation
        b.t+=1
        if robustness:
            self.extra=b._rho*self.extra+b._innovation_scale*self.extra_rng.standard_normal(self.extra.shape)
            for name in self.states:
                rho=parameters(name).rho[0]
                self.states[name]=rho*self.states[name]+np.sqrt(1-rho*rho)*innovation
        for name,(_,r) in result.items():
            self.hashers[name].update(r.tobytes())
        return result

    def hashes(self):
        return {name:h.hexdigest() for name,h in self.hashers.items()}


def raw_managed_history(basis,training_seed,pairing_index,maximum_T,robustness_T,
                        progress=None):
    sim=PairedHistory(training_seed,pairing_index)
    arrays={name:np.empty((maximum_T if name=='baseline' else robustness_T,basis.rank))
            for name in sim.hashers}
    checkpoints={}
    for t in range(maximum_T):
        robust=t<robustness_T
        observations=sim.step(robust)
        if robust:
            z,r=observations['N1200']
            kernel=basis.raw(z)
            for name,N in (('N300',300),('baseline',600),('N1200',1200)):
                arrays[name][t]=r[:N]@kernel[:N]/N
            for name in ('rho000','rho075'):
                z,r=observations[name]
                arrays[name][t]=r@basis.raw(z)/len(r)
        else:
            z,r=observations['baseline']
            arrays['baseline'][t]=r@basis.raw(z)/len(r)
        if t+1==1440:
            checkpoints={name+'_T1440':h for name,h in sim.hashes().items()}
        if progress is not None and (t+1)%240==0:
            progress(t+1)
    return arrays,{**sim.hashes(),**checkpoints}
