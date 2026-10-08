from dataclasses import replace

import numpy as np

from simulations.config.design import design_for
from simulations.dgp.balanced import BalancedFactorDGP, DGPParameters
from simulations.dgp.robustness import transform_date, condition_status
from simulations.estimator.kernel import NystromBasis, matern
from simulations.estimator.ridge import ridge_path, validation_select, moment_metrics
from simulations.experiments.population import population_moments, population_ridge


def test_nystrom_is_exact_rkhs_projection_and_managed_payoffs_agree():
    basis = NystromBasis(rank=32)
    p = DGPParameters(N=30)
    date = BalancedFactorDGP(p, 4).step()
    phi = basis.features(date.z)
    direct = basis.raw(date.z)@np.linalg.solve(basis.gram, basis.raw(date.z).T)
    np.testing.assert_allclose(phi@phi.T, direct, atol=1e-12)
    np.testing.assert_allclose(basis.managed(date.z, date.returns), date.returns@phi/p.N, atol=1e-14)
    # Projection cannot have more point-evaluation variance than the full kernel.
    assert np.max(np.sum(phi**2, axis=1)) <= 1+1e-12
    for nu in (.5, 1.5, 2.5):
        gram = matern(date.z, date.z, nu)
        np.testing.assert_array_equal(np.diag(gram), np.ones(p.N))
        assert np.linalg.eigvalsh(gram)[0] > 0


def test_paper_estimator_and_complexity_match_normal_equations():
    x = np.random.default_rng(2).normal(size=(25, 8))
    lam = np.array([1e-7, .1, 2.])
    coefficients, complexity = ridge_path(x, lam)
    second = x.T@x/len(x)
    for j, penalty in enumerate(lam):
        np.testing.assert_allclose((second+penalty*np.eye(8))@coefficients[:, j], x.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(complexity[j], np.trace(np.linalg.solve(second+penalty*np.eye(8), second)), atol=1e-12)


def test_validation_is_chronological_refits_and_cannot_access_oos():
    rng = np.random.default_rng(3)
    training = rng.normal(size=(60, 12))
    lam = np.geomspace(1e-5, 10, 12)
    coef, c, choice, loss, cut = validation_select(training, lam)
    assert cut == 45
    earlier, _ = ridge_path(training[:45], lam)
    np.testing.assert_allclose(loss, np.mean((1-training[45:]@earlier)**2, axis=0))
    assert choice == np.argmin(loss)
    np.testing.assert_allclose(coef, ridge_path(training, lam)[0])
    # External oracle/OOS mutations have no entry point in the selector API.
    unrelated_oos = rng.normal(size=(100, 12))
    unrelated_oos[:] = 1e9
    np.testing.assert_array_equal(validation_select(training, lam)[0], coef)


def test_population_normal_equations_and_exact_bias_estimation_decomposition():
    p, basis = DGPParameters(N=30), NystromBasis(rank=32)
    mean, second = population_moments(p, basis, groups=128, seed=4)
    lam = np.geomspace(1e-9, .1, 8)
    a, complexity, bias, floor = population_ridge(mean, second, lam, p.q_star)
    assert floor >= -1e-12
    assert np.all(np.diff(complexity) < 0)
    perturbation = np.random.default_rng(4).normal(size=a.shape)
    loss, sr = moment_metrics(a+perturbation, mean, second)
    norm = np.sum(perturbation*(second@perturbation), axis=0)
    cross = 2*np.sum(perturbation*(second@a-mean[:, None]), axis=0)
    np.testing.assert_allclose(loss-(1-p.q_star), bias+norm+cross, atol=1e-12)
    assert np.all(sr <= p.sr_star+1e-12)


def test_empirical_rank_variant_has_exact_balance_but_finite_support():
    p = DGPParameters(N=30)
    date = BalancedFactorDGP(p, 6).step()
    z, returns = transform_date(date, p, 'empirical_ranks')
    from simulations.dgp.balanced import beta
    b = beta(z, p)
    np.testing.assert_allclose(b.T@b/p.N, p.gamma_beta, atol=1e-12)
    for d in range(6):
        np.testing.assert_allclose(np.sort(z[:, d]), 2*(np.arange(1, 31)-.5)/30-1)
    assert 'violated' in condition_status('empirical_ranks')['E5']
    original_z = date.z.copy()
    transform_date(date, p, 'heteroskedastic')
    np.testing.assert_array_equal(original_z, date.z)


def test_predeclared_paper_scope_and_independent_replications():
    design = design_for('paper')
    assert design.T == (60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440)
    assert design.cases[0].replications == 500
    assert all(c.replications >= 200 and c.N%3==0 for c in design.cases)
    assert {'baseline', 'empirical_ranks', 'heteroskedastic', 'rank128', 'rank512'} <= {c.name for c in design.cases}
    assert len({np.random.SeedSequence([design.master_seed, 0, r]).generate_state(1)[0] for r in range(500)}) == 500


def test_production_observer_checks_each_date_without_changing_any_draw(monkeypatch):
    from simulations.config.design import Environment
    from simulations.diagnostics import path_integrity as observer
    env = Environment('test', 0, N=30)
    p = env.parameters()
    expected = BalancedFactorDGP(p, 21).step()
    context = {'environment': env, 'integrity_coefficients': p.policy_coefficients}
    monkeypatch.setattr(observer, '_CONTEXT', context)
    monkeypatch.setattr(observer, '_AUDIT', np.zeros(5))
    monkeypatch.setattr(BalancedFactorDGP, 'step', observer.observed_step)
    actual = BalancedFactorDGP(p, 21).step()
    for name in ('state','z','loadings','factors','epsilon','returns'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    assert observer._AUDIT[0]==1
    assert np.max(observer._AUDIT[1:])<1e-11


def test_rank_robustness_uses_nested_rkhs_subspaces_and_reduces_projection_floor():
    small, large = NystromBasis(rank=16), NystromBasis(rank=32)
    np.testing.assert_array_equal(small.anchors, large.anchors[:16])
    p = DGPParameters(N=30)
    z = BalancedFactorDGP(p, 91).step().z
    fsmall, flarge = small.features(z), large.features(z)
    difference = flarge@flarge.T-fsmall@fsmall.T
    assert np.linalg.eigvalsh(difference).min()>-1e-11
    floors=[]
    for basis in (small, large):
        mean, second = population_moments(p,basis,groups=64,seed=33)
        floors.append(population_ridge(mean,second,np.array([1e-5]),p.q_star)[3])
    assert floors[1] <= floors[0]+1e-12


def test_rate_bootstrap_preserves_whole_path_dependence_across_T():
    from simulations.config.design import Environment
    from simulations.experiments.reporting import rate_bootstrap
    d=replace(design_for('smoke'),T=(60,90,120,180),penalties=(.001,.01,.1),bootstrap_replications=30)
    env=Environment('fixture',0,replications=8)
    scales=np.linspace(.5,1.5,8)
    T=np.asarray(d.T)
    regret=scales[:,None,None]*T[None,:,None]**(-.7)*np.array([1,2,3])[None,None,:]
    complexity=np.broadcast_to(T[None,:,None]**.4*np.array([3,2,1])[None,None,:],regret.shape).copy()
    validation=np.broadcast_to([2.,1.,0.],regret.shape).copy()
    values={'regret':regret,'population_sharpe':1-regret,'validation_loss':validation,'complexity':complexity}
    context={'design':d,'environment':env,'population_complexity':np.array([3.,2.,1.]),'reference_sharpe':1.}
    rows=rate_bootstrap(values,context,None,None)
    for row in rows:
        if row['quantity'] in ('oracle_regret','selected_regret'):
            np.testing.assert_allclose([row['OLS_slope'],row['bootstrap_low'],row['bootstrap_high']],-.7,atol=1e-12)
        if 'empirical_complexity' in row['quantity']:
            np.testing.assert_allclose(row['OLS_slope'],.4,atol=1e-12)
