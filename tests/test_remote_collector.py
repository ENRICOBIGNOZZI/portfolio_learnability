"""Collector control flow preserves run identity and audits before importing."""
import json
from pathlib import Path
import subprocess

import pytest

from simulations.extended import remote_collect as collector


@pytest.fixture
def setup_collector(tmp_path, monkeypatch):
    output = tmp_path / 'canonical'
    output.mkdir()
    protocol = dict(run_hash='fixture', replications=2)
    allocation = dict(run_hash='fixture', stage='production_baseline',
        already_completed_indices=[], batches=[dict(id=0, indices=[0, 1])])
    (output / 'remote_allocation_production_baseline.json').write_text(json.dumps(allocation))
    (output / 'seed_manifest.json').write_text(json.dumps(dict(replications=[{}, {}])))
    artifact = dict(id=8, name='rich6d-production_baseline-batch-0', expired=False,
                    size_in_bytes=100, digest='sha256:fixture')
    events = []
    monkeypatch.setattr(collector, 'OUTPUT', output)
    monkeypatch.setattr(collector, 'protocol_checked', lambda: protocol)
    monkeypatch.setattr(collector.time, 'sleep', lambda seconds: events.append('wait'))
    def download(command, **kwargs):
        assert command[:3] == ['gh', 'run', 'download']
        assert '--name' in command
        destination = Path(command[command.index('--dir') + 1])
        (destination / 'transfer_manifest.json').write_text('{}')
        events.append('download')
    def validate(directory, *args):
        assert (directory / 'transfer_manifest.json').exists()
        events.append('transfer_validated')
    def ingest(directory, *args):
        receipt = directory / artifact['name'] / 'github_artifact_receipt.json'
        assert json.loads(receipt.read_text()) == artifact
        assert events.index('transfer_validated') > events.index('download')
        events.append('scientifically_audited_import')
        return dict(counts=dict(baseline=2), published=['checkpoint'])
    monkeypatch.setattr(collector.subprocess, 'run', download)
    monkeypatch.setattr(collector, 'validate_transfer', validate)
    monkeypatch.setattr(collector, 'ingest', ingest)
    return output, artifact, events


@pytest.mark.parametrize('initial_timeout', [False, True])
def test_collection_audits_before_import_and_retries_only_observation(setup_collector, tmp_path,
                                                                    monkeypatch, initial_timeout):
    output, artifact, events = setup_collector
    calls = 0
    def gh(*arguments):
        nonlocal calls
        calls += 1
        if initial_timeout and calls == 1:
            raise subprocess.TimeoutExpired(['gh', 'run', 'view'], 120)
        if arguments[0] == 'run':
            assert arguments[1] == 'view'
            return dict(headSha='abc', status='completed', conclusion='success', jobs=[])
        return dict(artifacts=[artifact])
    monkeypatch.setattr(collector, 'gh_json', gh)
    result = collector.collect('production_baseline', '123', 'abc', tmp_path / 'downloads', 1)
    assert result['status'] == 'all_stage_checkpoints_collected'
    assert events.count('download') == 1
    assert events.count('scientifically_audited_import') == 1
    assert result['downloaded_artifacts'][0]['id'] == 8
    assert events.count('wait') == 1 + initial_timeout


def test_wrong_source_commit_stops_before_download(setup_collector, tmp_path, monkeypatch):
    output, artifact, events = setup_collector
    monkeypatch.setattr(collector, 'gh_json', lambda *args:
        dict(headSha='wrong', status='completed', conclusion='success', jobs=[])
        if args[0] == 'run' else dict(artifacts=[artifact]))
    with pytest.raises(ValueError, match='source commit'):
        collector.collect('production_baseline', '123', 'abc', tmp_path / 'downloads', 1)
    assert not events
    state = json.loads((output / 'remote_collection_production_baseline_123.json').read_text())
    assert state['status'] == 'failed'
