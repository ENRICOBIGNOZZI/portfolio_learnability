"""Storage adapter for the frozen SR-target computation on a small local disk.

The original scientific runner is unmodified. Rank-1024 simulations and fits
still run in full. Checkpoints retain the exact nested rank-512 training matrix
needed to reconstruct every reported theory policy, all rank-comparison metrics,
the original return hashes, and seeds. High-rank histories remain reproducible
from those seeds; they are not additionally retained in the checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from simulations import run_sr_target as original
from simulations.provenance import file_hash, json_write, npz_write, output_lock, utc_now


def compact_record(record, embedding):
    if 'managed_training' not in record or int(record['managed_rank']) != 1024:
        return record
    result = dict(record)
    result['managed_training'] = record['managed_training'] @ embedding
    result['managed_rank'] = 512
    result['generated_managed_rank'] = 1024
    x = result['managed_training']
    for i, T in enumerate(record['T']):
        a = record['theory_coefficients'][i]
        residual = x[:T].T @ (x[:T] @ a) / T + record['penalties'][i, -1] * a - x[:T].mean(axis=0)
        assert np.abs(residual).max() < 1e-8
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    folder = args.output
    protocol = json.loads((folder / 'protocol.json').read_text())
    frozen_hash = protocol['source_hashes']['simulations/run_sr_target.py']
    assert file_hash(original.__file__) == frozen_hash
    with np.load(original.OLD / 'audit/rank_paths_low_memory/rich6d/moments.npz') as z:
        embedding = z['embedding512']
    manifest_path = folder / 'storage_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'scientific_run_hash': protocol['run_hash'],
        'unchanged_scientific_runner_sha256': frozen_hash,
        'storage_adapter_sha256': file_hash(__file__),
        'training_storage': 'Float64 rank-512 histories; first 50 generated at rank 1024 and projected with the verified embedding. All 512/1024 fitted-policy comparison metrics retained unchanged. Rank-1024 histories can be regenerated from saved seeds.',
        'first_used_utc': utc_now(), 'converted_checkpoints': {}}
    assert manifest['storage_adapter_sha256'] == file_hash(__file__)
    with output_lock(folder):
        for path in sorted((folder / 'replications').glob('*.npz')):
            with np.load(path) as z:
                if int(z['managed_rank']) != 1024:
                    continue
                record = {k: z[k] for k in z.files}
            old_hash = file_hash(path)
            result = compact_record(record, embedding)
            npz_write(path, **result)
            with np.load(path) as z:
                for key, value in record.items():
                    if key not in ('managed_training', 'managed_rank'):
                        np.testing.assert_array_equal(z[key], value)
                np.testing.assert_array_equal(z['managed_training'], result['managed_training'])
            manifest['converted_checkpoints'][path.name] = {
                'original_sha256': old_hash, 'compact_sha256': file_hash(path),
                'all_policy_and_rank_metrics_bitwise_unchanged': True}
            print('Storage compacted:', path.name, flush=True)
        json_write(manifest_path, manifest)

    def compact_write(path, **record):
        return npz_write(path, **compact_record(record, embedding))

    original.npz_write = compact_write
    original.main()
    figure_folder = folder / 'figures'
    provenance_path = figure_folder / 'provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        provenance['storage_manifest_sha256'] = file_hash(manifest_path)
        provenance['training_storage'] = manifest['training_storage']
        json_write(provenance_path, provenance)


if __name__ == '__main__':
    main()
