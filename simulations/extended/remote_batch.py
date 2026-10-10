"""Execute an explicitly allocated part of the unchanged frozen production queue."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback

import numpy as np

from simulations.extended.design import OUTPUT, ENVIRONMENTS
from simulations.extended.remote_compatibility import protocol_checked
from simulations.extended.remote_inputs import download
from simulations.extended.remote_reproduction import CANONICAL, bootstrap
from simulations.provenance import digest, file_hash, json_write, utc_now


STAGES = ('production_baseline', 'production_robustness')


def validate_allocation(allocation, protocol, stage):
    if stage not in STAGES or allocation['stage'] != stage or allocation['run_hash'] != protocol['run_hash']:
        raise ValueError('Allocation does not belong to the requested frozen stage.')
    batches = allocation['batches']
    if [row['id'] for row in batches] != list(range(len(batches))):
        raise ValueError('Batch IDs must be unique and consecutive.')
    assigned = [i for row in batches for i in row['indices']]
    completed = allocation['already_completed_indices']
    all_indices = assigned + completed
    if any(type(i) is not int for i in all_indices):
        raise ValueError('Replication indices must be integers.')
    if len(all_indices) != protocol['replications'] or sorted(all_indices) != list(range(protocol['replications'])):
        raise ValueError('Allocation must cover every replication exactly once, including completed checkpoints.')
    if any(not row['indices'] for row in batches):
        raise ValueError('Do not dispatch empty batches.')
    return batches


def qualified_allocation(protocol, stage):
    qualification_path = CANONICAL / 'remote_reproduction_results.json'
    qualification = json.loads(qualification_path.read_text())
    result = qualification['result']
    plan = json.loads((CANONICAL / 'remote_reproduction_plan.json').read_text())
    if (qualification['job_conclusion'] != 'success' or not result['passed']
            or result['run_hash'] != protocol['run_hash'] or not result['source_hashes_checked']
            or not result['baseline']['passed'] or not result['paired']['passed']):
        raise ValueError('Complete fitted-policy host qualification before remote production.')
    if (result['plan_sha256'] != file_hash(CANONICAL / 'remote_reproduction_plan.json')
            or result['input_manifest_sha256'] != file_hash(CANONICAL / 'remote_input_manifest.json')):
        raise ValueError('Qualified numerical inputs or reproduction rules changed.')
    if (platform.system() != 'Darwin' or os.environ.get('RICH6D_RUNNER_LABEL') != plan['runner']
            or platform.python_version() != plan['python']
            or version('numpy') != plan['numpy'] or version('scipy') != plan['scipy']):
        raise ValueError('Production requires the qualified host and exact scientific dependency versions.')
    allocation = json.loads((CANONICAL / f'remote_allocation_{stage}.json').read_text())
    validate_allocation(allocation, protocol, stage)
    if (allocation['qualification_sha256'] != file_hash(qualification_path)
            or allocation['input_manifest_sha256'] != result['input_manifest_sha256']):
        raise ValueError('Allocation was not declared against this host qualification.')
    if stage == 'production_robustness':
        proof_path = CANONICAL / 'remote_baseline_completion.json'
        proof = json.loads(proof_path.read_text())
        if (file_hash(proof_path) != allocation['baseline_completion_sha256']
                or not proof['completed_checkpoint_checks_passed']
                or proof['run_hash'] != protocol['run_hash']
                or proof['counts']['baseline'] != protocol['replications']):
            raise ValueError('Complete and verify all baseline paths before robustness production.')
    return allocation


def check_identity_and_grid(path, index, protocol, stage, seeds):
    baseline = stage == 'production_baseline'
    names = ['baseline'] if baseline else [name for name in ENVIRONMENTS if name != 'baseline']
    times = protocol['baseline_T'] if baseline else protocol['robustness_T']
    expected = dict(index=index, ranks=[protocol['rank']], baseline_times=times,
        robustness_times=[] if baseline else times, groups=protocol['population_groups'],
        quadrature_seed=protocol['population_seed'], stage=stage, run_hash=protocol['run_hash'],
        environment_names=names)
    with np.load(path) as z:
        if (json.loads(str(z['description'])) != expected or str(z['identity']) != digest(expected)
                or int(z['index']) != index or int(z['pairing_index']) != index
                or int(z['rank']) != protocol['rank'] or int(z['training_seed']) != seeds[index]['training_seed']):
            raise ValueError('Remote checkpoint identity or seed mismatch.')
        error = float(z['maximum_normal_equation_error'])
        if not np.isfinite(error) or error > 1e-8:
            raise ValueError('Remote ridge normal-equation check failed.')
        for name in names:
            if not np.array_equal(z[name + '_T'], times):
                raise ValueError('Remote checkpoint is missing a frozen horizon.')
            for metric in ('sr', 'loss', 'empirical_complexity', 'coefficient_norm_squared'):
                values = z[name + '_' + metric]
                if values.shape != (len(times), 97) or not np.isfinite(values).all():
                    raise ValueError('Remote checkpoint is missing finite full-penalty outcomes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--batch-id', type=int, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    bootstrap()
    protocol = protocol_checked()
    allocation = qualified_allocation(protocol, args.stage)
    if not 0 <= args.batch_id < len(allocation['batches']):
        raise ValueError('Unknown allocated batch.')
    indices = allocation['batches'][args.batch_id]['indices']
    if args.artifacts.exists():
        raise ValueError('Use a fresh artifact directory.')
    args.artifacts.mkdir(parents=True)
    record = dict(run_hash=protocol['run_hash'], stage=args.stage, batch_id=args.batch_id,
        requested_indices=indices, completed_indices=[], files=[], status='preparing_inputs',
        allocation_sha256=file_hash(CANONICAL / f'remote_allocation_{args.stage}.json'),
        qualification_sha256=allocation['qualification_sha256'],
        input_manifest_sha256=allocation['input_manifest_sha256'], started_utc=utc_now(),
        workflow_run_id=os.environ.get('GITHUB_RUN_ID'), commit=os.environ.get('GITHUB_SHA'),
        host=dict(system=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                  numpy=version('numpy'), scipy=version('scipy'), runner=os.environ.get('RICH6D_RUNNER_LABEL')),
        scope='Frozen core checks plus identity/grid checks; independent complete checkpoint audit is required after retrieval.')
    manifest_path = args.artifacts / 'transfer_manifest.json'
    start = time.perf_counter()
    try:
        json_write(manifest_path, record)
        download(protocol)
        from simulations.extended.production import execute
        seeds = json.loads((OUTPUT / 'seed_manifest.json').read_text())['replications']
        if [row['index'] for row in seeds] != list(range(protocol['replications'])):
            raise ValueError('Invalid canonical replication-seed manifest.')
        for index in indices:
            record.update(status='running', current_index=index)
            json_write(manifest_path, record)
            execute(index, protocol, args.stage)
            relative = Path(args.stage) / f'P{protocol["rank"]}' / f'rep_{index:03d}.npz'
            check_identity_and_grid(OUTPUT / relative, index, protocol, args.stage, seeds)
            resources = Path(args.stage) / f'resources_{index:03d}.json'
            for name in (relative, resources):
                destination = args.artifacts / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(OUTPUT / name, destination)
                checksum = file_hash(OUTPUT / name)
                if file_hash(destination) != checksum:
                    raise ValueError('Artifact copy failed its checksum.')
                record['files'].append(dict(path=str(name), bytes=destination.stat().st_size, sha256=checksum))
            record['completed_indices'].append(index)
            record.update(updated_utc=utc_now(), elapsed_seconds=time.perf_counter() - start)
            json_write(manifest_path, record)
        record.update(status='completed', completed_utc=utc_now())
    except Exception:
        record.update(status='failed', error=traceback.format_exc(), updated_utc=utc_now())
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter() - start
        json_write(manifest_path, record)


if __name__ == '__main__':
    main()
