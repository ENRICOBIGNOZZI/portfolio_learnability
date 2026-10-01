"""Linear, Gaussian and Matern-3/2 direct portfolio learning only."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import ctypes
import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist

KERNELS = ('linear', 'gaussian', 'matern32')
FEATURE_COUNT = 10000
STOCK_BATCH_SIZE = 256


# Apple documents vvcos as the double-precision vector cosine:
# https://developer.apple.com/documentation/accelerate/vvcos(_:_:_:)
_VECTOR_COS = None
if sys.platform == 'darwin':
    try:
        _ACCELERATE = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
        _VECTOR_COS = _ACCELERATE.vvcos
        _VECTOR_COS.argtypes = [ctypes.POINTER(ctypes.c_double),
                               ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
        _VECTOR_COS.restype = None
    except (OSError, AttributeError):
        _VECTOR_COS = None


def cosine(values):
    """Float64 vector cosine, with NumPy fallback outside Apple Accelerate."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    if _VECTOR_COS is None or values.size > np.iinfo(np.int32).max:
        return np.cos(values)
    result = np.empty_like(values)
    count = ctypes.c_int(values.size)
    _VECTOR_COS(result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.byref(count))
    return result


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def median_distance(sample):
    sample = np.asarray(sample, float)
    if sample.ndim != 2 or len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError('Need at least two finite characteristic vectors.')
    ell = float(np.median(pdist(sample, metric='euclidean')))
    if not np.isfinite(ell) or ell <= 0:
        raise ValueError('Median distance must be positive.')
    return ell


def exact_kernel(x, y, kernel, ell):
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    if kernel == 'linear':
        return 1.0 + x @ y.T
    if kernel not in KERNELS or not np.isfinite(ell) or ell <= 0:
        raise ValueError('Invalid kernel or bandwidth.')
    q = cdist(x, y, metric='euclidean') / ell
    if kernel == 'gaussian':
        return np.exp(-q*q/2)
    v = np.sqrt(3.0)*q
    return (1+v)*np.exp(-v)


@dataclass
class FeatureBank:
    kernel: str
    dimension: int
    maximum: int = 10000
    ell: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.kernel not in KERNELS or self.dimension < 1 or self.maximum < 1:
            raise ValueError('Invalid feature bank.')
        if not np.isfinite(self.ell) or self.ell <= 0:
            raise ValueError('Bandwidth must be positive.')
        normal_seed, radial_seed, phase_seed = np.random.SeedSequence(self.seed).spawn(3)
        # Rows are frequencies; prefixes are unchanged when maximum changes.
        normal = np.random.default_rng(normal_seed).normal(
            size=(self.maximum, self.dimension))
        radial = np.ones(self.maximum)
        if self.kernel == 'matern32':
            radial = np.sqrt(np.random.default_rng(radial_seed).chisquare(
                3.0, size=self.maximum)/3.0)
        self.frequencies = normal/(self.ell*radial[:, None])
        self.phases = np.random.default_rng(phase_seed).uniform(
            0, 2*np.pi, size=self.maximum)

    def features(self, x, count=None):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError('Invalid characteristic matrix.')
        if self.kernel == 'linear':
            return np.column_stack((np.ones(len(x)), x))
        count = self.maximum if count is None else int(count)
        if not 1 <= count <= self.maximum:
            raise ValueError('Feature count is outside the frozen bank.')
        return np.sqrt(2.0/count)*cosine(
            x @ self.frequencies[:count].T + self.phases[:count])

    def scores(self, x, beta, count=None):
        """Evaluate a policy without materializing all stock-by-RFF values."""
        x = np.asarray(x, float)
        if self.kernel == 'linear':
            return self.features(x, count) @ beta
        return np.concatenate([self.features(x[start:start+STOCK_BATCH_SIZE], count) @ beta
                               for start in range(0, len(x), STOCK_BATCH_SIZE)])

    def metadata(self):
        return {'kernel':self.kernel, 'input_dimension':self.dimension,
                'maximum_features':self.maximum, 'ell':self.ell, 'seed':self.seed,
                'cosine_backend':('unused' if self.kernel=='linear' else
                    'Apple Accelerate vvcos float64' if _VECTOR_COS is not None else 'NumPy float64'),
                'frequencies_sha256':array_hash(self.frequencies),
                'phases_sha256':array_hash(self.phases)}
