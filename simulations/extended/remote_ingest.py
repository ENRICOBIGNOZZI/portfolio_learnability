"""Audit downloaded remote checkpoints in isolation before publishing locally."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from simulations.extended.design import OUTPUT, ROOT
from simulations.extended.remote_batch import STAGES, validate_allocation, check_identity_and_grid
from simulations.extended.remote_compatibility import protocol_checked
from simulations.extended.remote_inputs import supporting_paths
from simulations.provenance import atomic_file, file_hash, json_write, output_lock, utc_now


def validate_transfer(directory, protocol, allocation, run_id, commit, seeds):
    manifest = json.loads((directory / 'transfer_manifest.json').read_text())
    stage = allocation['stage']
    batch_id = manifest['batch_id']
    if type(batch_id) is not int or not 0 <= batch_id < len(allocation['batches']):
        raise ValueError('Unexpected artifact batch ID.')
    requested = allocation['batches'][batch_id]['indices']
    done = manifest['completed_indices']
    if (manifest['run_hash'] != protocol['run_hash'] or manifest['stage'] != stage
            or manifest['requested_indices'] != requested or done != requested[:len(done)]
            or str(manifest['workflow_run_id']) != str(run_id) or manifest['commit'] != commit
            or manifest['allocation_sha256'] != file_hash(OUTPUT / f'remote_allocation_{stage}.json')
            or manifest['qualification_sha256'] != allocation['qualification_sha256']
            or manifest['input_manifest_sha256'] != allocation['input_manifest_sha256']):
        raise ValueError('Transferred checkpoint provenance differs from its declared allocation.')
    if manifest['status'] == 'completed' and done != requested:
        raise ValueError('Incomplete artifact claimed batch completion.')
    expected = {f'{stage}/P{protocol["rank"]}/rep_{i:03d}.npz' for i in done}
    expected |= {f'{stage}/resources_{i:03d}.json' for i in done}
    rows = manifest['files']
    if len(rows) != len(expected) or {r['path'] for r in rows} != expected:
        raise ValueError('Artifact file list must exactly match completed checkpoints and resources.')
    for row in rows:
        path = directory / row['path']
        if (path.is_symlink() or directory.resolve() not in path.resolve().parents
                or not path.is_file() or path.stat().st_size != row['bytes']
                or file_hash(path) != row['sha256']):
            raise ValueError('Transferred file failed its path, size or checksum check.')
    for index in done:
        check_identity_and_grid(directory / stage / f'P{protocol["rank"]}' / f'rep_{index:03d}.npz',
                                index, protocol, stage, seeds)
        resources = json.loads((directory / stage / f'resources_{index:03d}.json').read_text())
        if resources['index'] != index or resources['stage'] != stage or resources['run_hash'] != protocol['run_hash']:
            raise ValueError('Resource record identity differs from its checkpoint.')
    return manifest


def isolated_audit(directory):
    env = dict(os.environ, RICH6D_EXTENDED_OUTPUT=str(directory), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPYCACHEPREFIX='/tmp/rich6d_no_bytecode_cache',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    subprocess.run([sys.executable, '-m', 'simulations.extended.verify_checkpoints'],
                   cwd=ROOT, env=env, check=True)
    return json.loads((directory / 'checkpoint_integrity.json').read_text())


def ingest(download_root, stage, run_id, commit):
    protocol = protocol_checked()
    allocation = json.loads((OUTPUT / f'remote_allocation_{stage}.json').read_text())
    validate_allocation(allocation, protocol, stage)
    if (allocation['qualification_sha256'] != file_hash(OUTPUT / 'remote_reproduction_results.json')
            or allocation['input_manifest_sha256'] != file_hash(OUTPUT / 'remote_input_manifest.json')):
        raise ValueError('Canonical qualification or numerical input manifest changed.')
    seeds = json.loads((OUTPUT / 'seed_manifest.json').read_text())['replications']
    manifests = sorted(download_root.rglob('transfer_manifest.json'))
    if not manifests:
        raise ValueError('No transferred batch manifests were found.')
    batches, candidates = [], {}
    for path in manifests:
        manifest = validate_transfer(path.parent, protocol, allocation, run_id, commit, seeds)
        if manifest['batch_id'] in [row['batch_id'] for row in batches]:
            raise ValueError('Repeated batch artifact.')
        batches.append(manifest)
        for row in manifest['files']:
            if row['path'] in candidates:
                raise ValueError('Overlapping remote checkpoint artifacts.')
            candidates[row['path']] = (path.parent / row['path'], row['sha256'])
    # Also acquire the original production lock: ingestion cannot race a local
    # supervisor that is still admitting the same deterministic indices.
    with output_lock(OUTPUT / 'production_supervisor'), output_lock(OUTPUT / 'remote_ingestion'):
        with tempfile.TemporaryDirectory(prefix='rich6d-ingestion-audit-') as scratch:
            isolated = Path(scratch)
            for relative in ['protocol.json', *supporting_paths()]:
                target = isolated / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(OUTPUT / relative, target)
            for existing_stage in STAGES:
                source = OUTPUT / existing_stage / f'P{protocol["rank"]}'
                destination = isolated / existing_stage / source.name
                destination.mkdir(parents=True, exist_ok=True)
                for path in source.glob('rep_*.npz'):
                    shutil.copyfile(path, destination / path.name)
            for relative, (source, checksum) in candidates.items():
                canonical = OUTPUT / relative
                if canonical.exists() and file_hash(canonical) != checksum:
                    raise ValueError('Refusing to overwrite a different local checkpoint or resource record.')
                target = isolated / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if file_hash(target) != checksum:
                    raise ValueError('Staging copy failed checksum.')
            # The independent audit checks population ceilings/floors, exact
            # theory decompositions, old histories, cross-stage pairing and
            # unique economic paths before any canonical output is changed.
            audit = isolated_audit(isolated)
            published = []
            for relative, (source, checksum) in sorted(candidates.items()):
                destination = OUTPUT / relative
                if destination.exists():
                    if file_hash(destination) != checksum:
                        raise ValueError('Local output changed during isolated audit.')
                    continue
                with atomic_file(destination) as handle:
                    with (isolated / relative).open('rb') as incoming:
                        shutil.copyfileobj(incoming, handle)
                if file_hash(destination) != checksum:
                    raise ValueError('Canonical publication failed checksum.')
                published.append(relative)
            final_audit = isolated_audit(OUTPUT)
            if final_audit['counts'] != audit['counts']:
                raise ValueError('Canonical checkpoint counts differ from the audited staged data.')
            result = dict(run_hash=protocol['run_hash'], stage=stage, workflow_run_id=str(run_id),
                commit=commit, ingested_utc=utc_now(), batches=batches, published=published,
                independent_audit_passed=True, counts=final_audit['counts'],
                checkpoint_integrity_sha256=file_hash(OUTPUT / 'checkpoint_integrity.json'))
            json_write(OUTPUT / f'remote_ingestion_{stage}_{run_id}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-root', type=Path, required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    result = ingest(args.download_root, args.stage, args.run_id, args.commit)
    print(json.dumps(dict(counts=result['counts'], files_published=len(result['published']))))


if __name__ == '__main__':
    main()
