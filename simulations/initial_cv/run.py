"""Generate paired unchanged histories, calibrate at t=60, and freeze a forever."""
import argparse
import hashlib
import json
import time
import numpy as np

from simulations.extended.design import ENVIRONMENTS, parameters
from simulations.extended.histories import PairedHistory
from simulations.extended.population import basis_cached
from simulations.extended.compute import load_operator
from simulations.estimator.ridge import ridge_path, moment_metrics
from simulations.initial_cv.selector import calibrate_initial, T0, EXPONENT
from simulations.initial_cv.design import OUTPUT, CANONICAL, protocol
from simulations.extended.design import OUTPUT as INPUTS
from simulations.provenance import npz_write, file_hash, digest
from simulations.run_rough import seed as original_seed


def reference(index):
    manifest = json.loads((CANONICAL/'reference_manifest.json').read_text())
    entry = manifest[index]
    path = CANONICAL/entry['path']
    if entry['index'] != index or file_hash(path) != entry['sha256']:
        raise ValueError('Original-reference checksum mismatch.')
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def generate_and_calibrate(basis, index, baseline_T, robustness_T):
    history = PairedHistory(original_seed('train', index), index)
    arrays = {n:np.empty((baseline_T if n=='baseline' else robustness_T, basis.rank)) for n in ENVIRONMENTS}
    selected, hashes = {}, {}
    for t in range(baseline_T):
        robust = t < robustness_T
        observations = history.step(robust)
        if robust:
            z, r = observations['N1200']
            kernel = basis.raw(z)
            for name, N in (('N300', 300), ('baseline', 600), ('N1200', 1200)):
                arrays[name][t] = r[:N]@kernel[:N]/N
            for name in ('rho000', 'rho075'):
                z, r = observations[name]
                arrays[name][t] = r@basis.raw(z)/len(r)
        else:
            z, r = observations['baseline']
            arrays['baseline'][t] = r@basis.raw(z)/len(r)
        if t+1 == T0:
            for name in ENVIRONMENTS:
                prefix = arrays[name][:T0]@basis.inverse_root
                selected[name] = calibrate_initial(prefix)
                selected[name]['prefix_sha256'] = hashlib.sha256(prefix.tobytes()).hexdigest()
            print(f'path {index}: all five a values frozen at observation 60', flush=True)
        if t+1 in (1440, 3240):
            hashes.update({name+f'_T{t+1}':h for name,h in history.hashes().items()})
        if (t+1) % 1080 == 0:
            print(f'path {index}: generated {t+1}/{baseline_T}', flush=True)
    return arrays, selected, {**history.hashes(), **hashes}


def run(index):
    frozen = protocol()
    d = frozen['design']
    if index not in d['production_indices']:
        raise ValueError('Undeclared replication index.')
    path = OUTPUT/'replications'/f'rep_{index:03d}.npz'
    if path.exists():
        verify(path, index, frozen)
        return path
    start = time.perf_counter()
    basis = basis_cached(INPUTS/'pilot', d['rank'], d['basis_seed'])
    raw, selected, hashes = generate_and_calibrate(basis, index, max(d['baseline_T']), max(d['robustness_T']))
    results = {}
    max_residual = 0.
    for name in ENVIRONMENTS:
        times = np.array(d['baseline_T'] if name=='baseline' else d['robustness_T'])
        a = selected[name]['selected_a']
        X = raw.pop(name)@basis.inverse_root
        m, S, floor = load_operator(name, d['rank'], d['population_groups'], d['population_seed'])
        with np.load(INPUTS/'population'/f'{name}_theory.npz') as pop:
            eigenvalues = pop['eigenvalues']
        losses, sharpes, complexities, population_C, lambdas = [], [], [], [], []
        for T in times:
            penalties = np.array([a, d['a_old'][name]])*float(T)**EXPONENT
            coefficients, C = ridge_path(X[:T], penalties)
            L, SR = moment_metrics(coefficients, m, S)
            residual = X[:T].T@(X[:T]@coefficients)/T+coefficients*penalties-X[:T].mean(axis=0)[:,None]
            max_residual = max(max_residual, float(np.max(np.abs(residual))))
            if max_residual > 1e-8 or not np.isfinite(SR).all():
                raise ValueError('Policy numerical check failed.')
            losses.append(L); sharpes.append(SR); complexities.append(C); lambdas.append(penalties)
            population_C.append(np.sum(eigenvalues[:,None]/(eigenvalues[:,None]+penalties), axis=0))
        results.update({name+'_'+key:np.asarray(value) for key,value in dict(
            T=times, loss=losses, sr=sharpes, empirical_complexity=complexities,
            population_complexity=population_C, penalties=lambdas, floor=floor,
            selected_a=a, selected_index=selected[name]['selected_index'],
            fold_losses=selected[name]['fold_losses'], validation_scores=selected[name]['mean_validation_loss'],
            calibration_prefix_sha256=selected[name]['prefix_sha256']).items()})
        del X, S
        print(f'path {index}: {name} fits complete; initial a={a:.8g}', flush=True)
    npz_write(path, **results, index=index, run_hash=frozen['run_hash'],
        training_seed=original_seed('train',index), returns_hashes=json.dumps(hashes,sort_keys=True),
        maximum_normal_equation_error=max_residual, elapsed_seconds=time.perf_counter()-start)
    verify(path, index, frozen)
    return path


def verify(path, index, frozen=None):
    frozen = protocol() if frozen is None else frozen
    d = frozen['design']
    from simulations.initial_cv.selector import A_GRID, FOLDS
    old = reference(index)
    with np.load(path) as z:
        if int(z['index']) != index or str(z['run_hash']) != frozen['run_hash']:
            raise ValueError('Checkpoint identity mismatch.')
        if int(z['training_seed']) != int(old['training_seed']):
            raise ValueError('Original training seed changed.')
        got_hashes = json.loads(str(z['returns_hashes']))
        for stage in ('production_baseline', 'production_robustness'):
            expected = json.loads(str(old[stage+'_hashes']))
            for name, value in expected.items():
                if stage == 'production_baseline' and not name.startswith('baseline'):
                    continue
                actual = got_hashes['baseline_T3240'] if stage == 'production_robustness' and name=='baseline' else got_hashes[name]
                if actual != value:
                    raise ValueError(f'Original paired history changed: {stage} {name}')
        for name in ENVIRONMENTS:
            times = np.asarray(d['baseline_T'] if name=='baseline' else d['robustness_T'])
            np.testing.assert_array_equal(z[name+'_T'], times)
            score = np.average(z[name+'_fold_losses'], axis=0, weights=[b-a for a,b in FOLDS])
            np.testing.assert_array_equal(score, z[name+'_validation_scores'])
            choice = len(score)-1-int(np.argmin(score[::-1]))
            if choice != int(z[name+'_selected_index']) or float(z[name+'_selected_a']) != A_GRID[choice]:
                raise ValueError('Calibration does not minimize initial forward validation loss.')
            penalties = np.column_stack([A_GRID[choice]*times.astype(float)**EXPONENT,
                                          d['a_old'][name]*times.astype(float)**EXPONENT])
            np.testing.assert_allclose(z[name+'_penalties'], penalties, rtol=2e-15, atol=0)
            for metric in ('sr','loss','empirical_complexity'):
                values = z[name+'_'+metric]
                if values.shape != (len(times),2) or not np.isfinite(values).all():
                    raise ValueError('Incomplete metrics.')
                np.testing.assert_allclose(values[:,1], old[name+'_'+metric], rtol=1e-7, atol=1e-9)
            p = parameters(name)
            if np.any(z[name+'_loss'] < 1-p.q_star+float(z[name+'_floor'])-1e-8):
                raise ValueError('Policy beats the frozen population projection regret floor.')
        if float(z['maximum_normal_equation_error']) > 1e-8:
            raise ValueError('Normal equations failed.')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    run(args.index)
