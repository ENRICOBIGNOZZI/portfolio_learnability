"""Declare disjoint remote batches only after local admission has stopped."""
import argparse
import json
import math

from simulations.extended.design import OUTPUT
from simulations.extended.remote_batch import STAGES, validate_allocation
from simulations.extended.remote_compatibility import protocol_checked
from simulations.extended.verify_checkpoints import verify
from simulations.provenance import file_hash, json_write, output_lock, utc_now


def balanced_batches(indices, maximum_size=20, concurrency=5):
    if maximum_size < 1 or concurrency < 1:
        raise ValueError('Batch size and concurrency must be positive.')
    if not indices:
        return []
    count = min(len(indices), concurrency * math.ceil(math.ceil(len(indices) / maximum_size) / concurrency))
    size, extra = divmod(len(indices), count)
    batches, start = [], 0
    for batch_id in range(count):
        stop = start + size + (batch_id < extra)
        batches.append(dict(id=batch_id, indices=indices[start:stop]))
        start = stop
    return batches


def declare(stage, maximum_size=20):
    protocol = protocol_checked()
    qualification_path = OUTPUT / 'remote_reproduction_results.json'
    qualification = json.loads(qualification_path.read_text())
    result = qualification['result']
    if (qualification['job_conclusion'] != 'success' or not result['passed']
            or result['run_hash'] != protocol['run_hash'] or not result['source_hashes_checked']
            or not result['baseline']['passed'] or not result['paired']['passed']
            or result['plan_sha256'] != file_hash(OUTPUT / 'remote_reproduction_plan.json')
            or result['input_manifest_sha256'] != file_hash(OUTPUT / 'remote_input_manifest.json')):
        raise ValueError('A successful unchanged fitted-policy qualification is required.')
    destination = OUTPUT / f'remote_allocation_{stage}.json'
    if destination.exists():
        raise ValueError('Preserve existing allocation provenance; do not overwrite a dispatched allocation.')
    with output_lock(OUTPUT / 'production_supervisor'), output_lock(OUTPUT / 'remote_ingestion'):
        audit = verify()
        completed = sorted(row['index'] for row in audit['checkpoints'] if row['stage'] == stage)
        missing = sorted(set(range(protocol['replications'])) - set(completed))
        if not missing:
            raise ValueError('Requested stage is already complete.')
        allocation = dict(run_hash=protocol['run_hash'], stage=stage, declared_utc=utc_now(),
            qualification_sha256=file_hash(qualification_path),
            input_manifest_sha256=result['input_manifest_sha256'],
            completed_checkpoint_audit_sha256=file_hash(OUTPUT / 'checkpoint_integrity.json'),
            already_completed_indices=completed, batches=balanced_batches(missing, maximum_size),
            maximum_batch_size=maximum_size, maximum_parallel_jobs=5,
            scheduling_rule='Ascending missing deterministic indices, balanced across a multiple of five standard macOS jobs; no selection by outcomes.')
        if stage == 'production_robustness':
            if audit['counts']['baseline'] != protocol['replications']:
                raise ValueError('Complete and audit all baseline replications before robustness.')
            proof = OUTPUT / 'remote_baseline_completion.json'
            if not proof.exists():
                json_write(proof, audit)
            else:
                old = json.loads(proof.read_text())
                if (old['run_hash'] != protocol['run_hash'] or not old['completed_checkpoint_checks_passed']
                        or old['counts']['baseline'] != protocol['replications']):
                    raise ValueError('Existing baseline completion record is invalid.')
            allocation['baseline_completion_sha256'] = file_hash(proof)
        validate_allocation(allocation, protocol, stage)
        json_write(destination, allocation)
    return allocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--maximum-size', type=int, default=20)
    args = parser.parse_args()
    result = declare(args.stage, args.maximum_size)
    print(json.dumps(dict(stage=args.stage, completed=len(result['already_completed_indices']),
                          batch_sizes=[len(row['indices']) for row in result['batches']])))


if __name__ == '__main__':
    main()
