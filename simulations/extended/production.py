"""Resume verified baseline production, then the four paired robustness economies."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time

from simulations.extended.design import OUTPUT,ENVIRONMENTS
from simulations.extended.compute import run_path
from simulations.extended.freeze import science_hashes
from simulations.provenance import output_lock,json_write,utc_now


def execute(index,protocol,stage):
    if stage=='production_baseline':
        names=['baseline'];baseline=protocol['baseline_T'];robustness=[]
    else:
        names=[name for name in ENVIRONMENTS if name!='baseline']
        baseline=protocol['robustness_T'];robustness=protocol['robustness_T']
    return run_path(index,[protocol['rank']],baseline,robustness,
                    protocol['population_groups'],protocol['population_seed'],
                    stage,protocol['run_hash'],names)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    protocol=json.loads((OUTPUT/'protocol.json').read_text())
    if science_hashes()!=protocol['source_hashes']:
        raise ValueError('Scientific source changed after freeze; do not mix production identities.')
    start=time.perf_counter()
    with output_lock(OUTPUT/'production_supervisor'):
        for stage in ('production_baseline','production_robustness'):
            with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
                futures=[pool.submit(execute,index,protocol,stage) for index in range(300)]
                for completed,future in enumerate(as_completed(futures),1):
                    result=future.result()
                    json_write(OUTPUT/'production_status.json',dict(stage=stage,
                        completed=completed,total=300,last_index=result['index'],
                        elapsed_seconds=time.perf_counter()-start,updated_utc=utc_now(),
                        run_hash=protocol['run_hash']))
                    print(f'{stage}: {completed}/300 verified checkpoints',flush=True)
        json_write(OUTPUT/'production_status.json',dict(stage='all_replications_complete',
            replications_per_environment=300,environments=5,elapsed_seconds=time.perf_counter()-start,
            completed_utc=utc_now(),run_hash=protocol['run_hash'],
            note='Statistical summaries, figure rendering and final verification are separate required steps.'))

if __name__=='__main__':
    main()
