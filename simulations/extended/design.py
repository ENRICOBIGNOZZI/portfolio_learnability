"""Predetermined pilot design. Production is frozen only after resolution gates."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np

from simulations.dgp.balanced import DGPParameters

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'simulations/outputs/rich6d_rough_sr3_annual_monthly_v1'
OUTPUT = ROOT/'simulations/outputs/rich6d_rough_sr3_extended_v2'
T_ORIGINAL = (60,90,120,180,240,360,540,720,1080,1440)
T_REQUIRED = T_ORIGINAL + (2160,3240,4860)
T_CANDIDATE = T_REQUIRED + (7290,)
T_ROBUSTNESS = T_ORIGINAL + (2160,3240)
PENALTIES = np.geomspace(1e-12,1e-2,96)
RANKS = (512,1024,2048,4096)
WINDOWS = {'full':0,'T_ge_360':360,'T_ge_720':720,'T_ge_1440':1440}
BASE = json.loads((REFERENCE/'population.json').read_text())
BASE_PARAMETERS = DGPParameters(**BASE['parameters'])
BASE_A = BASE['s_ref']
BASIS_SEED = BASE['basis_seed']
PERIODS_PER_YEAR = 12
REPLICATIONS = 300
PILOT_PATHS = 12
ENVIRONMENTS = {'baseline':(600,.95),'N300':(300,.95),'N1200':(1200,.95),
                'rho000':(600,0.),'rho075':(600,.75)}


def parameters(name='baseline'):
    N,rho = ENVIRONMENTS[name]
    return replace(BASE_PARAMETERS,N=N,rho=(rho,)*6)


def seed(family,index=0):
    families = {'rank_training':1,'population':2,'spectrum_basis':3,
                'spectrum_quadrature':4,'quadrature':5,'extra_groups':6,
                'extra_noise':7,'N_pilot':8}
    return int(np.random.SeedSequence([2026101001,families[family],index]).generate_state(1)[0])


def pilot_design():
    return {'schema':'rich6d-extended-preproduction/1','baseline_parameters':BASE_PARAMETERS.to_dict(),
            'baseline_a_fixed':BASE_A,'baseline_basis_seed':BASIS_SEED,'ranks':list(RANKS),
            'T_required':list(T_REQUIRED),'T_candidate':list(T_CANDIDATE),
            'robustness_T_predetermined':list(T_ROBUSTNESS), 'replications_per_environment':300,
            'rank_pilot_paths':PILOT_PATHS,'rank_tolerance':.05,'quadrature_tolerance':.05,
            'population_groups_candidates':[32768,131072], 'rank_pilot_T':list(T_CANDIDATE),
            'spectral_audit_basis_seeds':[BASIS_SEED,seed('spectrum_basis',1)],
            'spectral_audit_quadrature_seeds':[seed('population'),seed('spectrum_quadrature',1)],
            'environments':{name:parameters(name).to_dict() for name in ENVIRONMENTS},
            'penalties':PENALTIES.tolist(),'slope_windows':WINDOWS,'primary_bands':'2.5th and 97.5th replication percentiles; no bootstrap',
            'slope_uncertainty':'delta method using complete replication-level cross-T covariance',
            'selection':'Lowest common rank passing the unchanged two-part 5% criterion across required horizons and environments; no rate or Sharpe maximization. Investigate 4096 and 7290 with explicit resource measurements.',
            'scope':'One fixed production rank. Unresolved cells remain visible. No manuscript or empirical writes.'}
