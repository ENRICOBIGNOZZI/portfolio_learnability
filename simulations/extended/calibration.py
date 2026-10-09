"""The unchanged population-only pilot convention for the ridge constant."""
import json
import numpy as np
from scipy.linalg import eigh

from simulations.extended.design import OUTPUT, BASIS_SEED, BASE_A, parameters
from simulations.extended.population import basis_cached, integrals_cached, operators
from simulations.run_rough import seed as legacy_seed
from simulations.provenance import json_write, utc_now


def calibrate():
    destination=OUTPUT/'calibration.json'
    if destination.exists():
        return json.loads(destination.read_text())
    basis=basis_cached(OUTPUT/'pilot',512,BASIS_SEED)
    stats=integrals_cached(OUTPUT/'pilot',parameters(),basis,8192,legacy_seed('pilot'))
    values={}
    for name in ('baseline','N300','N1200'):
        p=parameters(name)
        _,S=operators(p,basis,stats)
        values[name]=float(eigh(S,eigvals_only=True)[-1])
    np.testing.assert_allclose(values['baseline'],BASE_A,rtol=1e-12)
    values['baseline']=BASE_A
    values['rho000']=values['rho075']=BASE_A
    result={'a':values,'rule':'Largest rank-512 population second-moment eigenvalue; same legacy pilot rule across N; original baseline a retained exactly.',
            'groups':8192,'seed':legacy_seed('pilot'),'basis_seed':BASIS_SEED,
            'independent_of_training_and_evaluation':True,'frozen_utc':utc_now(),
            'annual_SR_star':{n:float(parameters(n).sr_star*np.sqrt(12)) for n in values},
            'N_convention':'Factor means and covariance fixed; no Sharpe recalibration across N.',
            'rho_convention':'Stationary marginal operator and a identical across rho.'}
    json_write(destination,result)
    return result

if __name__=='__main__':
    print(calibrate())
