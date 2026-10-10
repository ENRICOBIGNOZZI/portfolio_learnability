"""Visual QA using actual pilot paths, outside the final deliverable directory.

This does not duplicate pilot rows to pretend there are 300 replications. Every
preview is visibly watermarked and never enters the final artifact manifest.
"""
import argparse
import json
from pathlib import Path
import tempfile
import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulations.extended.design import OUTPUT,ENVIRONMENTS,PENALTIES,BASIS_SEED,parameters,seed
from simulations.extended.calibration import calibrate
from simulations.extended.population import spectral_path
from simulations.extended.audits import rank_audit
from simulations.extended.figures import Figures


def preview(count=2):
    folder=Path(tempfile.mkdtemp(prefix='rich6d_actual_pilot_preview_'))
    f=object.__new__(Figures)
    f.data={};f.pop={};f.folder=folder;f.captions=[]
    protocol_path=OUTPUT/'protocol.json'
    f.protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    rank=f.protocol['rank'] if f.protocol is not None else 4096
    a=calibrate()['a']
    for name in ENVIRONMENTS:
        records=[]
        for i in range(count):
            with np.load(OUTPUT/'rank_pilot'/f'P{rank}'/f'rep_{i:03d}.npz') as z:
                records.append({key:z[key] for key in (name+'_T',name+'_sr',name+'_empirical_complexity')})
        T=records[0][name+'_T']
        sr=np.stack([r[name+'_sr'] for r in records])*np.sqrt(12)
        C=np.stack([r[name+'_empirical_complexity'] for r in records])
        if f.protocol is not None:
            grid=f.protocol['baseline_T' if name=='baseline' else 'robustness_T']
            keep=np.isin(T,grid)
            T=T[keep];sr=sr[:,keep];C=C[:,keep]
        f.data[name]=dict(T=T,annual_SR=sr,annual_gap=parameters(name).sr_star*np.sqrt(12)-sr,
                          empirical_complexity=C,relative_complexity=C/T[None,:,None])
        if f.protocol is not None:
            with np.load(OUTPUT/'population'/f'{name}_theory.npz') as z:
                f.pop[name]={key:z[key] for key in z.files}
            continue
        economic_name='baseline' if name.startswith('rho') else name
        path=OUTPUT/'pilot'/f'spectra_{economic_name}_P4096_B{BASIS_SEED}_Q32768_S{seed("population")}.npz'
        with np.load(path) as z:
            values=z['managed'][::-1]
        lam=a[name]*T.astype(float)**(-.6)
        popC,_,_=spectral_path(values,lam)
        gridC,_,_=spectral_path(values,PENALTIES)
        f.pop[name]=dict(T=T,eigenvalues=values,penalties=lam,complexity=popC,
                         diagnostic_grid_complexity=gridC)
    if f.protocol is None:
        f.protocol=dict(rank=4096,a=a,baseline_T=f.data['baseline']['T'].tolist(),
            robustness_T=f.data['N300']['T'].tolist(),main_path_T=[60,240,720,1440],
            robustness_path_T=[60,1440])
        f.ranks=rank_audit(count,write=False)
        selected=f.ranks[(f.ranks.higher_rank==4096)&f.ranks.exact_theory_choice]
        f.resolution=pd.DataFrame(dict(environment=selected.environment,T=selected['T'],
            floor_pass=selected.higher_projection_floor<=.05*selected.mean_regret_higher,
            paired_pass=selected.paired_upper_bound<=.05*selected.mean_regret_higher))
    else:
        f.ranks=pd.read_csv(OUTPUT/'rank_audit.csv')
        f.resolution=pd.read_csv(OUTPUT/'production_resolution.csv')
    f.set_style()
    def save(self,fig,name,caption):
        fig.text(.5,1.035,f'PREPRODUCTION QA: {count} actual pilot paths — not the 300-replication study',
                 ha='center',color='#8b2525',fontsize=9)
        fig.savefig(self.folder/(name+'.png'),bbox_inches='tight',dpi=140)
        plt.close(fig)
        self.captions.append((name,'PREPRODUCTION ONLY. '+caption.replace('300 independent','pilot independent')))
    f.save=types.MethodType(save,f)
    # Call the individual renderers so the production caption path is untouched.
    from simulations.extended.figures import N_NAMES,RHO_NAMES
    f.spectrum();f.learnability(['baseline'],'main','Figure1_learnability')
    f.main_complexity();f.main_path()
    f.learnability(N_NAMES,'N','FigureR1_N_learnability')
    f.learnability(RHO_NAMES,'rho','FigureR2_rho_learnability')
    f.robustness_complexity(N_NAMES,'N','FigureR3_N_complexity')
    f.robustness_complexity(RHO_NAMES,'rho','FigureR4_rho_complexity')
    f.robustness_path(N_NAMES,'N','FigureR5_N_sharpe_complexity')
    f.robustness_path(RHO_NAMES,'rho','FigureR6_rho_sharpe_complexity')
    assert len(list(folder.glob('*.png')))==10
    print(folder,flush=True)
    return folder

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=2)
    args=parser.parse_args()
    preview(args.count)
