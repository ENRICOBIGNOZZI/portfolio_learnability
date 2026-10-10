"""Remote transfers must pass scientific audits before canonical publication."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from simulations.extended import remote_ingest
from simulations.extended.remote_inputs import supporting_paths
from simulations.extended.remote_reproduction import CANONICAL
from simulations.provenance import file_hash


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    output = tmp_path / 'canonical'
    artifact = tmp_path / 'download' / 'batch-0'
    output.mkdir()
    artifact.mkdir(parents=True)
    for relative in ['protocol.json', *supporting_paths()]:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CANONICAL / relative, target)
    protocol = json.loads((output / 'protocol.json').read_text())
    for name in ['remote_reproduction_results.json', 'remote_input_manifest.json']:
        (output / name).write_text('{}')
    allocation = dict(stage='production_baseline', run_hash=protocol['run_hash'],
        already_completed_indices=list(range(1, 300)), batches=[dict(id=0, indices=[0])],
        qualification_sha256=file_hash(output / 'remote_reproduction_results.json'),
        input_manifest_sha256=file_hash(output / 'remote_input_manifest.json'))
    allocation_path = output / 'remote_allocation_production_baseline.json'
    allocation_path.write_text(json.dumps(allocation))
    files = []
    for relative in ['production_baseline/P4096/rep_000.npz', 'production_baseline/resources_000.json']:
        target = artifact / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CANONICAL / relative, target)
        files.append(dict(path=relative, bytes=target.stat().st_size, sha256=file_hash(target)))
    manifest = dict(run_hash=protocol['run_hash'], stage='production_baseline', batch_id=0,
        requested_indices=[0], completed_indices=[0], workflow_run_id='123', commit='abc',
        allocation_sha256=file_hash(allocation_path), status='completed', files=files,
        qualification_sha256=allocation['qualification_sha256'],
        input_manifest_sha256=allocation['input_manifest_sha256'])
    (artifact / 'transfer_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(remote_ingest, 'OUTPUT', output)
    monkeypatch.setattr(remote_ingest, 'protocol_checked', lambda: protocol)
    return output, artifact


def test_valid_actual_checkpoint_is_audited_and_published(transfer):
    output, artifact = transfer
    result = remote_ingest.ingest(artifact.parent, 'production_baseline', '123', 'abc')
    assert result['independent_audit_passed']
    assert result['counts']['baseline'] == 1
    assert len(result['published']) == 2
    assert file_hash(output / 'production_baseline/P4096/rep_000.npz') == file_hash(
        CANONICAL / 'production_baseline/P4096/rep_000.npz')


def test_corrupt_transfer_is_rejected_without_publication(transfer):
    output, artifact = transfer
    (artifact / 'production_baseline/P4096/rep_000.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        remote_ingest.ingest(artifact.parent, 'production_baseline', '123', 'abc')
    assert not (output / 'production_baseline').exists()


def test_wrong_run_is_rejected_without_publication(transfer):
    output, artifact = transfer
    with pytest.raises(ValueError, match='provenance'):
        remote_ingest.ingest(artifact.parent, 'production_baseline', 'different-run', 'abc')
    assert not (output / 'production_baseline').exists()


def test_valid_checksum_cannot_bypass_population_audit(transfer):
    output, artifact = transfer
    path = artifact / 'production_baseline/P4096/rep_000.npz'
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    arrays['baseline_sr'][0, -1] = 10.0
    np.savez_compressed(path, **arrays)
    manifest_path = artifact / 'transfer_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][0].update(bytes=path.stat().st_size, sha256=file_hash(path))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(subprocess.CalledProcessError):
        remote_ingest.ingest(artifact.parent, 'production_baseline', '123', 'abc')
    assert not (output / 'production_baseline').exists()
