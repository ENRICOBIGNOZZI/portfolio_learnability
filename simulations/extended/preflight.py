"""Sequential preproduction supervisor; never launches production implicitly."""
import argparse
import json
import os
import subprocess
import sys
import time

from simulations.extended.design import OUTPUT,BASIS_SEED,PILOT_PATHS,seed
from simulations.extended.audits import rank_audit,quadrature_audit,spectral_summary
from simulations.provenance import json_write,utc_now,output_lock


def execute(module,args):
    command=[sys.executable,'-m',module,*map(str,args)]
    print('Executing',command,flush=True)
    subprocess.run(command,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    with output_lock(OUTPUT/'preflight_supervisor'):
        if args.wait_pid:
            while True:
                try:
                    os.kill(args.wait_pid,0)
                except ProcessLookupError:
                    break
                time.sleep(30)
        # A disappeared process is not success: verify every expected checkpoint.
        for rank in (512,1024,2048,4096):
            if not (OUTPUT/'rank_pilot'/f'P{rank}'/'rep_000.npz').exists():
                raise RuntimeError('Initial resource/rank pilot did not finish; inspect its actual failure before resuming.')
        population_cases=[
            (4096,131072,BASIS_SEED,seed('quadrature',1),False),
            (2048,32768,seed('spectrum_basis',1),seed('population'),True),
            (2048,32768,seed('spectrum_basis',1),seed('spectrum_quadrature',1),True),
            (2048,32768,BASIS_SEED,seed('spectrum_quadrature',1),True),
        ]
        for rank,groups,basis_seed,qseed,spectra_only in population_cases:
            marker=OUTPUT/'pilot'/f'resources_P{rank}_B{basis_seed}_Q{groups}_S{qseed}.json'
            if marker.exists():
                continue
            options=['--max-rank',rank,'--groups',groups,'--basis-seed',basis_seed,'--quadrature-seed',qseed]
            if spectra_only:
                options.append('--spectra-only')
            execute('simulations.extended.pilot_population',options)
        # A first-path check can identify a failed quadrature before further fits.
        quadrature_audit(4096,1)
        for index in range(1,PILOT_PATHS):
            execute('simulations.extended.compute',['--index',index])
            json_write(OUTPUT/'preflight_status.json',dict(stage='rank_pilots',
                completed=index+1,total=PILOT_PATHS,updated_utc=utc_now()))
        rank=rank_audit()
        quad=quadrature_audit()
        spectral_summary()
        json_write(OUTPUT/'preproduction_audit_status.json',dict(completed_utc=utc_now(),
            pilot_replications=PILOT_PATHS,quadrature_all_pass=bool(quad.quadrature_pass.all()),
            tolerance=.05,highest_rank_has_no_higher_rank_certificate=True,
            note='Preproduction evidence complete. Review gates and resource measurements, then freeze production. No production launched by this supervisor.'))

if __name__=='__main__':
    main()
