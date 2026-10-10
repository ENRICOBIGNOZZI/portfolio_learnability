"""Audit completed production checkpoints without claiming study completion."""
import json

import numpy as np

from simulations.extended.design import OUTPUT, REFERENCE, ENVIRONMENTS, parameters
from simulations.extended.freeze import science_hashes
from simulations.provenance import digest, file_hash, json_write, utc_now


def verify():
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    scientific = {k: v for k, v in protocol.items()
                  if k not in ('run_hash', 'frozen_before_production_utc')}
    assert digest(scientific) == protocol['run_hash']
    assert science_hashes() == protocol['source_hashes']
    seeds = json.loads((OUTPUT / 'seed_manifest.json').read_text())['replications']
    assert [row['index'] for row in seeds] == list(range(300))
    population = {}
    for name in ENVIRONMENTS:
        with np.load(OUTPUT / 'population' / f'{name}_theory.npz') as z:
            population[name] = float(z['projection_floor'])
    records = []
    histories = {name: {} for name in ENVIRONMENTS}
    stage_hashes = {}
    for stage in ('production_baseline', 'production_robustness'):
        names = ['baseline'] if stage == 'production_baseline' else [
            name for name in ENVIRONMENTS if name != 'baseline']
        times = protocol['baseline_T'] if stage == 'production_baseline' else protocol['robustness_T']
        for path in sorted((OUTPUT / stage / f'P{protocol["rank"]}').glob('rep_*.npz')):
            index = int(path.stem.split('_')[-1])
            assert 0 <= index < 300
            expected = dict(index=index, ranks=[protocol['rank']], baseline_times=times,
                robustness_times=[] if stage == 'production_baseline' else times,
                groups=protocol['population_groups'], quadrature_seed=protocol['population_seed'],
                stage=stage, run_hash=protocol['run_hash'], environment_names=names)
            with np.load(path) as z:
                assert json.loads(str(z['description'])) == expected
                assert str(z['identity']) == digest(expected)
                assert int(z['index']) == index and int(z['pairing_index']) == index
                assert int(z['training_seed']) == seeds[index]['training_seed']
                assert int(z['rank']) == protocol['rank']
                error = float(z['maximum_normal_equation_error'])
                assert np.isfinite(error) and error <= 1e-8
                hashes = json.loads(str(z['returns_hashes']))
                stage_hashes[stage, index] = hashes
                for name in names:
                    np.testing.assert_array_equal(z[name + '_T'], times)
                    for metric in ('sr', 'loss', 'empirical_complexity', 'coefficient_norm_squared'):
                        values = z[name + '_' + metric]
                        assert values.shape == (len(times), 97) and np.isfinite(values).all()
                    p = parameters(name)
                    assert np.max(z[name + '_sr']) <= p.sr_star + 1e-8
                    regret = z[name + '_loss'] - (1 - p.q_star)
                    assert np.min(regret) >= population[name] - 1e-8
                    np.testing.assert_allclose(z[name + '_floor'], population[name], atol=1e-10)
                    complexity = z[name + '_empirical_complexity']
                    assert (complexity >= -1e-10).all()
                    assert (complexity <= np.minimum(times, protocol['rank'])[:, None] + 1e-7).all()
                    decomposition = z[name + '_theory_decomposition']
                    assert decomposition.shape == (len(times), 3)
                    np.testing.assert_allclose(regret[:, -1], decomposition.sum(axis=1),
                                               rtol=1e-8, atol=1e-10)
                    histories[name][index] = hashes[name]
                if stage == 'production_baseline' and index < 100:
                    with np.load(REFERENCE / 'replications' / f'{index:04d}.npz') as old:
                        assert hashes['baseline_T1440'] == str(old['stock_return_sha256'])
            records.append(dict(path=str(path.relative_to(OUTPUT)), index=index, stage=stage,
                sha256=file_hash(path), maximum_normal_equation_error=error,
                environments=names, shape_per_environment=[len(times), 97]))
    paired = []
    pending = []
    for index in histories['N300']:
        if index not in histories['baseline']:
            pending.append(index)
            continue
        base = stage_hashes['production_baseline', index]
        robust = stage_hashes['production_robustness', index]
        assert robust['baseline_T1440'] == base['baseline_T1440']
        assert robust['baseline'] == base['baseline_T3240']
        paired.append(index)
    for values in histories.values():
        assert len(values) == len(set(values.values())), 'Duplicated economic history.'
    result = dict(run_hash=protocol['run_hash'], verified_utc=utc_now(),
        scope='Integrity of completed checkpoints only; no claim about study completion, numerical resolution, final summaries or figures.',
        completed_checkpoint_checks_passed=True,
        counts={name: len(values) for name, values in histories.items()},
        paired_common_history_indices=paired, pending_common_history_indices=pending,
        checkpoints=records)
    json_write(OUTPUT / 'checkpoint_integrity.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify()['counts'], sort_keys=True))
