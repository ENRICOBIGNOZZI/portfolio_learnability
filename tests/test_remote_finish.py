"""The remote analysis gate must never admit incomplete or unaudited production."""
import json
from types import SimpleNamespace

import pytest

from simulations.extended import remote_finish as finish
from simulations.provenance import file_hash


@pytest.fixture
def completed(tmp_path):
    protocol = {'run_hash': 'frozen', 'replications': 300, 'rank': 4096}
    counts = {name: 300 for name in finish.ENVIRONMENTS}
    allocation = tmp_path / 'remote_allocation_production_robustness.json'
    allocation.write_text('{}')
    audit_path = tmp_path / 'checkpoint_integrity.json'
    audit = dict(run_hash='frozen', counts=counts, completed_checkpoint_checks_passed=True,
                 paired_common_history_indices=list(range(300)), pending_common_history_indices=[])
    audit_path.write_text(json.dumps(audit))
    ingestion = dict(run_hash='frozen', counts=counts, independent_audit_passed=True,
                     commit='commit', workflow_run_id='123',
                     checkpoint_integrity_sha256=file_hash(audit_path))
    (tmp_path / 'remote_ingestion_production_robustness_123.json').write_text(json.dumps(ingestion))
    state = dict(run_hash='frozen', stage='production_robustness', workflow_run_id='123',
                 source_commit='commit', allocation_sha256=file_hash(allocation),
                 status='all_stage_checkpoints_collected', verified_counts=counts, collector_pid=42)
    path = tmp_path / 'remote_collection_production_robustness_123.json'
    path.write_text(json.dumps(state))
    return tmp_path, protocol, path, state


def test_complete_gate_and_stale_audit(completed):
    output, protocol, path, state = completed
    assert finish.ready(output, protocol, '123', 'commit')
    audit = output / 'checkpoint_integrity.json'
    audit.write_text(audit.read_text() + '\n')
    with pytest.raises(ValueError, match='audit'):
        finish.ready(output, protocol, '123', 'commit')


@pytest.mark.parametrize('change', [
    {'verified_counts': {'baseline': 300, 'N300': 100, 'N1200': 100, 'rho000': 100, 'rho075': 100}},
    {'source_commit': 'wrong'},
    {'status': 'terminal_run_incomplete'},
])
def test_reject_incomplete_or_wrong_run(completed, change):
    output, protocol, path, state = completed
    state.update(change)
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        finish.ready(output, protocol, '123', 'commit')


def test_wait_requires_live_matching_collector(completed, monkeypatch):
    output, protocol, path, state = completed
    state['status'] = 'polling'
    path.write_text(json.dumps(state))
    monkeypatch.setattr(finish.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='python -m simulations.extended.remote_collect --run-id 123'))
    assert not finish.ready(output, protocol, '123', 'commit')
    monkeypatch.setattr(finish.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout=''))
    with pytest.raises(RuntimeError, match='missing'):
        finish.ready(output, protocol, '123', 'commit')


def test_failed_verification_stops_before_report_and_figures(completed, monkeypatch):
    output, protocol, path, state = completed
    monkeypatch.setattr(finish, 'OUTPUT', output)
    monkeypatch.setattr(finish, 'protocol_checked', lambda: protocol)
    monkeypatch.setattr(finish, 'check_audited_bytes', lambda *args: None)
    calls = []
    def execute(command, **kwargs):
        calls.append(command[2:])
        if command[2:] == ['simulations.extended.verify', '--data-only']:
            raise finish.subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(finish.subprocess, 'run', execute)
    with pytest.raises(finish.subprocess.CalledProcessError):
        finish.finish('123', 'commit')
    assert calls == [list(command) for command in finish.COMMANDS[:2]]
    record = json.loads((output / 'remote_final_analysis.json').read_text())
    assert record['status'] == 'failed'
    assert record['completed_commands'] == [list(command) for command in finish.COMMANDS[:1]]


def test_changed_checkpoint_rejected(tmp_path):
    protocol = {'replications': 1, 'rank': 4096}
    records = []
    for stage in ('production_baseline', 'production_robustness'):
        path = tmp_path / stage / 'P4096' / 'rep_000.npz'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'original checkpoint bytes')
        records.append(dict(path=str(path.relative_to(tmp_path)), sha256=file_hash(path)))
    (tmp_path / 'checkpoint_integrity.json').write_text(json.dumps({'checkpoints': records}))
    finish.check_audited_bytes(tmp_path, protocol)
    path.write_bytes(b'changed checkpoint bytes')
    with pytest.raises(ValueError, match='bytes changed'):
        finish.check_audited_bytes(tmp_path, protocol)
