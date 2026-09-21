import numpy as np

from expanding_window_gaussian import complexity


def test_inverse_history_rule():
    lambda0 = 4.0
    T0 = 120
    assert np.isclose(lambda0 * T0 / 120, 4.0)
    assert np.isclose(lambda0 * T0 / 240, 2.0)
    assert np.isclose(lambda0 * T0 / 480, 1.0)


def test_effective_complexity_falls_with_lambda():
    eigenvalues = np.array([5.0, 2.0, 0.5, 0.1])
    low_penalty = complexity(eigenvalues, 0.1)
    high_penalty = complexity(eigenvalues, 1.0)
    assert low_penalty > high_penalty
    assert 0.0 < high_penalty < len(eigenvalues)
    assert 0.0 < low_penalty < len(eigenvalues)


def test_relative_complexity_definition():
    eigenvalues = np.array([3.0, 1.0])
    T = 200
    value = complexity(eigenvalues, 0.5)
    assert np.isclose(value / T, value / 200)
