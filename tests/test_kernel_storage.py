"""File storage preserves the full kernel operator and controlled eigenpairs."""
from contextlib import closing

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from simulations.dgp.balanced import DGPParameters, beta
from simulations.estimator.kernel import matern
from simulations.diagnostics.kernel_storage import kernel_matrix, FileKernel
from simulations.diagnostics.spectrum_v2 import managed_operator


def test_file_kernel_matches_dense_products_and_eigenpairs():
    # Balanced triplets are required by the managed-operator identity.
    from simulations.dgp.balanced import balanced_triplets
    rng = np.random.default_rng(421)
    z = balanced_triplets(rng.uniform(.001, .999, (32, 6)))
    dense = matern(z, z)
    np.testing.assert_array_equal(kernel_matrix(z), dense)
    with closing(kernel_matrix(z, maximum_dense_bytes=0, block_rows=17)) as stored:
        assert isinstance(stored, FileKernel)
        for shape in ((len(z),), (len(z), 5)):
            right = rng.normal(size=shape)
            np.testing.assert_allclose(stored@right, dense@right, rtol=2e-13, atol=2e-13)
        for eta in (0., .35):
            p = DGPParameters(loading_map='rich6d' if eta else 'baseline_original', eta=eta)
            operator = managed_operator(stored, beta(z, p), p)
            reference = managed_operator(dense, beta(z, p), p)@np.eye(len(z))
            values = eigsh(operator, k=12, which='LA', tol=1e-11, return_eigenvectors=False,
                           v0=np.ones(len(z)))
            np.testing.assert_allclose(np.sort(values), np.linalg.eigvalsh(reference)[-12:], rtol=2e-9, atol=1e-14)
    assert stored._file.closed
