"""Exercise spectral cache loading after dependency, runtime and case mutations."""
import json

import numpy as np
import pytest

from simulations.diagnostics import spectrum_v2 as spectrum


@pytest.mark.parametrize('mutation', ['kernel', 'runtime', 'groups'])
def test_cache_loader_rejects_changed_dependency(tmp_path, monkeypatch, mutation):
    # Numerical eigensolvers are checked elsewhere. This fixture isolates the
    # actual cache-loading branch without modifying any production artifacts.
    monkeypatch.setattr(spectrum, 'dense_crosscheck', lambda: [])
    monkeypatch.setattr(spectrum, 'source_hashes', lambda: {'kernel.py': 'original'})
    directory = tmp_path/'audit/spectrum'
    directory.mkdir(parents=True)
    identity = spectrum.cache_fingerprint(1024, 0)
    values = {'cache_identity': identity, 'source_sha256': spectrum.file_hash(spectrum.__file__)}
    for name in ('kernel', 'baseline_original', 'rich6d'):
        values[name+'_values'] = np.arange(1., 801.)**(-1.5)
        values[name+'_residuals'] = np.zeros(800)
    np.savez_compressed(directory/'groups1024_seed0.npz', **values)
    meta = {'groups': 1024, 'seed': spectrum.seed('spectrum', 0), 'cache_identity': identity}
    (directory/'groups1024_seed0.json').write_text(json.dumps(meta))
    spectrum.run(tmp_path, groups_grid=(1024,), seed_indices=(0,))
    requested_groups = 1024
    if mutation == 'kernel':
        monkeypatch.setattr(spectrum, 'source_hashes', lambda: {'kernel.py': 'modified'})
    elif mutation == 'runtime':
        monkeypatch.setattr(spectrum.scipy, '__version__', 'different-runtime')
    else:
        requested_groups = 2048
        (directory/'groups1024_seed0.npz').rename(directory/'groups2048_seed0.npz')
        (directory/'groups1024_seed0.json').rename(directory/'groups2048_seed0.json')
    with pytest.raises(ValueError, match='Spectral cache identity mismatch'):
        spectrum.run(tmp_path, groups_grid=(requested_groups,), seed_indices=(0,))
