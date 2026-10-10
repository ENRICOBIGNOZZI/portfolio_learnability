"""Remote admission must reject incomplete or materially changed fitted outcomes."""
import json

import numpy as np
import pytest

from simulations.extended.remote_reproduction import CANONICAL, compare


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
