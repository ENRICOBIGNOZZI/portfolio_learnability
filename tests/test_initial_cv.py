import numpy as np
import pytest

from simulations.initial_cv.selector import calibrate_initial, A_GRID, FOLDS


def test_forward_fold_penalties_and_response_one_loss_match_direct_solve():
    rng = np.random.default_rng(78)
    x = rng.normal(size=(60, 5))+.1
    result = calibrate_initial(x)
    for k, (end, stop) in enumerate(FOLDS):
        expected = []
        for a in A_GRID:
            lam = a*end**(-.6)
            w = np.linalg.solve(x[:end].T@x[:end]/end+lam*np.eye(5), x[:end].mean(axis=0))
            expected.append(np.mean((1-x[end:stop]@w)**2))
        np.testing.assert_allclose(result['fold_losses'][k], expected, atol=1e-12)
        np.testing.assert_array_equal(result['fold_penalties'][k], A_GRID*end**(-.6))
    assert result['selected_a'] == A_GRID[np.argmin(np.mean(result['fold_losses'],axis=0))]


def test_selector_refuses_future_rows_and_breaks_exact_ties_toward_larger_a():
    with pytest.raises(ValueError):
        calibrate_initial(np.zeros((61, 5)))
    with pytest.raises(ValueError):
        calibrate_initial(np.zeros((59, 5)))
    result = calibrate_initial(np.zeros((60, 5)))
    assert result['selected_index'] == 70
    assert result['selected_a'] == .1


def test_all_calibrations_happen_before_observation_61(monkeypatch):
    from simulations.initial_cv import run
    from simulations.extended.design import ENVIRONMENTS
    class History:
        t = 0
        def __init__(self, *args): pass
        def step(self, robust):
            History.t += 1
            assert History.t <= 60 or len(selected_at) == 5
            return {n:(np.zeros((N,6)),np.ones(N)) for n,N in
                    [('baseline',600),('N300',300),('N1200',1200),('rho000',600),('rho075',600)]}
        def hashes(self): return {n:'test' for n in ENVIRONMENTS}
    class Basis:
        rank = 2
        inverse_root = np.eye(2)
        def raw(self,z): return np.ones((len(z),2))
    selected_at = []
    def selector(x):
        assert History.t == 60 and x.shape == (60,2)
        selected_at.append(History.t)
        return {'selected_a':1.}
    monkeypatch.setattr(run,'PairedHistory',History)
    monkeypatch.setattr(run,'calibrate_initial',selector)
    run.generate_and_calibrate(Basis(),0,65,65)
    assert selected_at == [60]*5
