import numpy as np
import pytest

from oos_error import empirical_power, quadratic_oos_loss


def test_observed_loss_uses_unit_target_and_no_ridge_penalty():
    np.testing.assert_allclose(quadratic_oos_loss([0, 1, -1, 2]), [1, 0, 4, 1])
    with pytest.raises(ValueError):
        quadratic_oos_loss([np.nan])


def test_recovers_power_from_errors_and_bootstrap_rows():
    n = np.array([36, 60, 120, 240, 360])
    errors = .8 * (n / 36)**(-.4)
    q, a = empirical_power(n, np.vstack([errors, errors * 2]))
    np.testing.assert_allclose(q, [.4, .4], atol=1e-12)
    np.testing.assert_allclose(a, [.8, 1.6], atol=1e-12)


def test_does_not_subtract_an_unknown_floor_from_total_loss():
    n = np.array([36, 60, 120, 240, 360])
    excess = .1 * (n / 36)**(-.7)
    q, _ = empirical_power(n, .6 + excess)
    assert 0 < q < .7
    q_increasing, _ = empirical_power(n, .6 + .1 * (n / 36)**(.7))
    assert q_increasing < 0
