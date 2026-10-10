"""Remote admission must reject incomplete or materially changed fitted outcomes."""
import hashlib
import io
import json

import numpy as np
import pytest

from simulations.extended.remote_reproduction import CANONICAL, compare
from simulations.extended import remote_inputs


@pytest.fixture
def canonical():
    reference = CANONICAL / 'production_baseline/P4096/rep_000.npz'
    plan = json.loads((CANONICAL / 'remote_reproduction_plan.json').read_text())
    protocol = json.loads((CANONICAL / 'protocol.json').read_text())
    with np.load(reference) as z:
        arrays = {key: z[key] for key in z.files}
    return reference, plan['comparison'], protocol, arrays


def test_canonical_reproduction_passes(canonical):
    reference, rules, protocol, _ = canonical
    assert compare(reference, reference, ['baseline'], rules, True, protocol)['passed']


def test_missing_horizon_is_rejected(canonical, tmp_path):
    reference, rules, protocol, arrays = canonical
    arrays['baseline_T'] = arrays['baseline_T'][:-1]
    path = tmp_path / 'missing_horizon.npz'
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='every frozen horizon'):
        compare(path, reference, ['baseline'], rules, True, protocol)


def test_changed_sharpe_fails_the_gate(canonical, tmp_path):
    reference, rules, protocol, arrays = canonical
    arrays['baseline_sr'][0, -1] += .01
    path = tmp_path / 'changed_sharpe.npz'
    np.savez_compressed(path, **arrays)
    result = compare(path, reference, ['baseline'], rules, True, protocol)
    assert not result['passed']
    assert not next(row['passed'] for row in result['metric_checks'] if row['metric'] == 'sr')


@pytest.mark.parametrize('missing_reference_hash', [False, True])
def test_legacy_pilot_hashes_are_all_required(tmp_path, missing_reference_hash):
    reference = CANONICAL / 'rank_pilot/P4096/rep_000.npz'
    plan = json.loads((CANONICAL / 'remote_reproduction_plan.json').read_text())
    protocol = json.loads((CANONICAL / 'protocol.json').read_text())
    names = ['baseline', 'N300', 'N1200', 'rho000', 'rho075']
    with np.load(reference) as z:
        arrays = {key: z[key] for key in z.files}
    old_times = arrays['baseline_T']
    positions = [int(np.flatnonzero(old_times == t)[0]) for t in protocol['baseline_T']]
    arrays['baseline_T'] = old_times[positions]
    for metric in plan['comparison']['metrics']:
        arrays['baseline_' + metric] = arrays['baseline_' + metric][positions]
    hashes = json.loads(str(arrays['returns_hashes']))
    hashes['baseline'] = 'different-terminal-horizon'
    hashes['baseline_T3240'] = 'new-prefix-without-saved-reference'
    for name in names[1:]:
        hashes[name + '_T3240'] = hashes[name]
    if missing_reference_hash:
        del hashes['N300']
    arrays['returns_hashes'] = json.dumps(hashes)
    arrays['maximum_normal_equation_error'] = 0.0
    path = tmp_path / 'legacy_pilot.npz'
    np.savez_compressed(path, **arrays)
    result = compare(path, reference, names, plan['comparison'], False, protocol)
    assert result['passed'] is not missing_reference_hash
    assert len(result['history_hash_checks']) == 9
    assert len(result['additional_hashes_without_saved_reference']) == 5


@pytest.mark.parametrize('corrupt', [False, True])
def test_download_publishes_only_hash_verified_bytes(tmp_path, monkeypatch, corrupt):
    expected = b'frozen-cache'
    received = b'broken-cache' if corrupt else expected
    assert len(received) == len(expected)
    manifest = dict(release_tag='test', files=[dict(path='pilot/cache.npz',
        asset='cache.npz', bytes=len(expected), sha256=hashlib.sha256(expected).hexdigest())])
    monkeypatch.setattr(remote_inputs, 'OUTPUT', tmp_path)
    monkeypatch.setattr(remote_inputs, 'validated_manifest', lambda protocol: manifest)
    monkeypatch.setattr(remote_inputs, 'urlopen', lambda *args, **kwargs: io.BytesIO(received))
    monkeypatch.setattr(remote_inputs.time, 'sleep', lambda seconds: None)
    destination = tmp_path / 'pilot/cache.npz'
    if corrupt:
        with pytest.raises(ValueError, match='failed checksum'):
            remote_inputs.download({})
        assert not destination.exists()
        assert not list(destination.parent.glob('.cache.npz.*'))
    else:
        remote_inputs.download({})
        assert destination.read_bytes() == expected
