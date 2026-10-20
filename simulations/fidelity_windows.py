"""Publish every estimable window of the completed eight-trial pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .asymptotic_dense import write_json
from .asymptotic_report import tail_fits


def build(output):
    out=Path(output)
    config=json.loads((out/'config.json').read_text())
    frame=pd.read_csv(out/'fidelity_trials.csv')
    times=sorted(frame.T_train.unique().tolist())
    # Every distinct, observable temporal window with at least three points.
    cutoffs=[t for t in times if sum(x>=t for x in times)>=3]
    config=dict(config,sensitivity_T_min=cutoffs)
    constant=frame.C_lambda.iloc[0]
    grid=constant*config['spectrum_T_train']**(-2/3)*np.geomspace(.01,100,65)
    spectra=[];cache_hashes={};windows=[];all_fits={}
    for (kind,rank),group in frame.groupby(['representation','rank']):
        spectral=[]
        for trial in range(config['fidelity_trials']):
            path=out/'cache'/f'fidelity_{kind}{rank}_N{config["fixed_N"]}_rho{config["fixed_rho"]:g}_trial{trial:03d}.npz'
            cache_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as saved:
                assert saved['T_train'].tolist()==times
                eig=saved['eigenvalues']
                cs=np.sum(eig[:,None]/(eig[:,None]+grid),axis=0)
            spectral.extend(dict(N=config['fixed_N'],rho=config['fixed_rho'],trial=trial,
                representation=kind,rank=rank,T_train=config['spectrum_T_train'],
                lambda_reg=lam,complexity=C) for lam,C in zip(grid,cs))
        spectrum=pd.DataFrame(spectral)
        fits,table,_=tail_fits(config,group,spectrum)
        table['representation']=kind;table['rank']=rank
        windows.append(table);spectra.append(spectrum)
        all_fits[f'{kind}{rank}']=fits
    windows=pd.concat(windows,ignore_index=True)
    pd.concat(spectra,ignore_index=True).to_csv(out/'fidelity_spectrum.csv',index=False)
    windows.to_csv(out/'fidelity_window_sensitivity.csv',index=False)
    write_json(out/'fidelity_window_fits.json',all_fits)
    source=hashlib.sha256((out/'fidelity_trials.csv').read_bytes()).hexdigest()
    table=windows.pivot(index=['representation','rank','T_train_min'],columns='metric',values='exponent').reset_index()
    table.to_csv(out/'fidelity_window_comparison.csv',index=False)
    (out/'fidelity_windows.html').write_text('''<!doctype html><meta charset="utf-8">
<title>Pilot preliminare: confronto dei tagli</title>
<style>body{font:16px Georgia,serif;max-width:1000px;margin:35px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}</style>
<h1>Pilot preliminare: sensibilita ai tagli iniziali</h1>
<p>Otto repliche appaiate per rappresentazione, N=600, rho=0.5.
Tutti i dati osservati: T_train=800,1600,3200,6400,12800.
Il taglio da 800 usa tutti e cinque i punti; da 1600 quattro, da 3200 tre.
I tagli ancora piu piccoli, 100,200,400, richiedono i nuovi dati in corso.
Il fit principale resta da 1600. La selezione numerica del rango resta 1024;
si mostrano anche il riferimento 2048 e gli altri esiti, senza scegliere
il rango o il taglio dalle pendenze.</p>
<p>Benchmark: gap(T)=-2/3; C(T)=+1/3; C(lambda)=-1/2; gap(C)=-2.
Per C(lambda), lo spettro e fisso a T_train=12800; il taglio cambia soltanto
l'intervallo di lambda corrispondente al percorso bilanciato.</p>
<p><a href="fidelity_window_sensitivity.csv">Esponenti, bootstrap CI e R2 di tutti i fit</a>;
<a href="fidelity_spectrum.csv">Spettri ricostruiti dai cache, per ogni replica</a>.</p>'''+
        table.to_html(index=False,float_format=lambda x:f'{x:.6f}'))
    files=['fidelity_spectrum.csv','fidelity_window_sensitivity.csv','fidelity_window_fits.json',
           'fidelity_window_comparison.csv','fidelity_windows.html']
    write_json(out/'fidelity_window_design.json',dict(replications=config['fidelity_trials'],
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        available_T_train=times,secondary_cutoffs=cutoffs,primary_T_train_min=config['asymptotic_T_min'],
        earlier_cutoffs_unobserved=[100,200,400],source_fidelity_trials_sha256=source,
        source_training_cache_sha256=cache_hashes,
        files_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files},
        note='Secondary analysis requested after the preliminary table; every estimable cutoff and representation is retained. The original primary fit and numerical rank selection are unchanged. T=100,200,400 require the new simulation, not relabeling the pilot.'))
    print(table.to_string(index=False,float_format=lambda x:f'{x:.6f}'))


def validate(output):
    """Check the published CSVs directly, without the fitting helper or cache."""
    out=Path(output)
    design=json.loads((out/'fidelity_window_design.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert design['analysis_source_sha256']==sha(Path(__file__))
    assert design['source_fidelity_trials_sha256']==sha(out/'fidelity_trials.csv')
    assert all(sha(out/name)==value for name,value in design['files_sha256'].items())
    frame=pd.read_csv(out/'fidelity_trials.csv');frame=frame[frame.lambda_multiplier==1]
    spectrum=pd.read_csv(out/'fidelity_spectrum.csv')
    windows=pd.read_csv(out/'fidelity_window_sensitivity.csv')
    old=json.loads((out/'fidelity_rate_diagnostics.json').read_text())['fits']
    config=json.loads((out/'config.json').read_text())
    assert len(windows)==len(old)*len(design['secondary_cutoffs'])*4
    assert not windows.duplicated(['representation','rank','metric','T_train_min']).any()
    checked=0
    for (kind,rank),f in frame.groupby(['representation','rank']):
        spectral=spectrum[(spectrum.representation==kind)&(spectrum['rank']==rank)]
        assert set(f.trial)==set(range(design['replications']))
        for _,g in spectral.groupby('trial'):
            assert len(g)==65 and (np.diff(g.sort_values('lambda_reg').complexity)<0).all()
        for cutoff in design['secondary_cutoffs']:
            g=f[f.T_train>=cutoff].groupby('T_train')[['SR_gap','complexity']].mean().sort_index()
            T=g.index.to_numpy(float)
            constant=f.C_lambda.iloc[0]
            low=constant*config['spectrum_fit_T_max']**(-2/3);high=constant*cutoff**(-2/3)
            s=spectral[(spectral.lambda_reg>=low*(1-1e-10))&
                       (spectral.lambda_reg<=high*(1+1e-10))].groupby('lambda_reg').complexity.mean().sort_index()
            for name,x,y,power in [('SR_T',T,g.SR_gap.to_numpy(),-2/3),
                ('C_T',T,g.complexity.to_numpy(),1/3),('SR_C',g.complexity.to_numpy(),g.SR_gap.to_numpy(),-2),
                ('C_lambda',s.index.to_numpy(float),s.to_numpy(),-.5)]:
                row=windows[(windows.representation==kind)&(windows['rank']==rank)&
                    (windows.T_train_min==cutoff)&(windows.metric==name)]
                assert len(row)==1;row=row.iloc[0]
                assert row.points==len(x) and row.expected_exponent==power
                exponent=np.polyfit(np.log(x),np.log(y),1)[0]
                np.testing.assert_allclose(row.exponent,exponent,atol=1e-10)
                basis=x**power;scale=max(0,float(basis@y/(basis@basis)))
                r2=1-np.sum((y-scale*basis)**2)/np.sum((y-y.mean())**2)
                np.testing.assert_allclose(row.r_squared,r2,atol=1e-10)
                if cutoff==design['primary_T_train_min']:
                    np.testing.assert_allclose(row.exponent,old[f'{kind}{rank}'][name]['exponent'],atol=1e-10)
                checked+=1
    result=dict(passed=True,checked_fits=checked,replications=design['replications'],
        all_estimable_windows_retained=True,original_primary_exponents_reproduced=True,
        method='Direct CSV group means, log-log least squares and fixed-power R2; no fitting helper or cache used',
        files_sha256={name:sha(out/name) for name in [*design['files_sha256'],'fidelity_window_design.json']})
    write_json(out/'fidelity_window_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    validate(args.output) if args.validate else build(args.output)
