"""A scheduling change must preserve the exact frozen replication population."""
import copy

import pytest

from simulations.extended.remote_batch import validate_allocation


def allocation():
    return dict(stage='production_baseline', run_hash='frozen', already_completed_indices=[0, 1],
                batches=[dict(id=0, indices=[2, 3]), dict(id=1, indices=[4, 5])])


def test_complete_disjoint_allocation_is_accepted():
    result = validate_allocation(allocation(), dict(run_hash='frozen', replications=6), 'production_baseline')
    assert [row['indices'] for row in result] == [[2, 3], [4, 5]]


@pytest.mark.parametrize('indices', [[4], [3, 5], [4, 6], [4, True]])
def test_missing_duplicate_out_of_range_or_noninteger_index_is_rejected(indices):
    bad = copy.deepcopy(allocation())
    bad['batches'][1]['indices'] = indices
    with pytest.raises(ValueError):
        validate_allocation(bad, dict(run_hash='frozen', replications=6), 'production_baseline')
