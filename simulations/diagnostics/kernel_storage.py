"""Full kernel matrix with bounded resident storage for spectral matvecs."""
import os
import tempfile

import numpy as np

from simulations.estimator.kernel import matern
from simulations.provenance import require


class FileKernel:
    """Keep exact float64 kernel rows in a local, automatically removed file.

    Read only one row block per multiplication. The operating system can cache
    the file but no process-owned dense n-by-n array is retained in RAM.
    """
    def __init__(self, z, block_rows=256):
        self.shape = (len(z), len(z))
        self.block_rows = block_rows
        self._file = tempfile.TemporaryFile()
        for first in range(0, len(z), block_rows):
            values = np.ascontiguousarray(matern(z[first:first+block_rows], z), dtype=np.float64)
            self._file.write(values.tobytes())
        self._file.flush()

    def __len__(self):
        return self.shape[0]

    def __matmul__(self, right):
        right = np.asarray(right)
        n = len(self)
        require(right.ndim in (1, 2) and right.shape[0] == n, 'Invalid kernel multiplier shape')
        result = np.empty(right.shape, dtype=np.result_type(right, np.float64))
        for first in range(0, n, self.block_rows):
            count = min(self.block_rows, n-first)
            size = count*n*8
            raw = os.pread(self._file.fileno(), size, first*n*8)
            require(len(raw) == size, 'Incomplete temporary kernel read')
            rows = np.frombuffer(raw, dtype=np.float64).reshape(count, n)
            result[first:first+count] = rows@right
        return result

    def close(self):
        self._file.close()


def kernel_matrix(z, maximum_dense_bytes=128*1024*1024, block_rows=256):
    n = len(z)
    if n*n*8 > maximum_dense_bytes:
        return FileKernel(z, block_rows)
    matrix = np.empty((n, n))
    for first in range(0, n, block_rows):
        matrix[first:first+block_rows] = matern(z[first:first+block_rows], z)
    return matrix
