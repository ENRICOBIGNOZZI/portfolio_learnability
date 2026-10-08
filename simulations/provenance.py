"""Versioned fingerprints, atomic artifacts and exclusive output ownership.

Execution concurrency and rendering are deliberately absent from the scientific
identity. Changes to orchestration, audit code and kernel implementations are not.
"""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'simulation-provenance/2.0'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes(kind='science', root=ROOT):
    root = Path(root)
    science = {'kernels.py', 'simulations/run.py', 'simulations/provenance.py'}
    for folder in ('config', 'dgp', 'estimator', 'experiments'):
        science.update(str(p.relative_to(root)) for p in (root/'simulations'/folder).rglob('*.py'))
    # Only these diagnostics execute inside economic generation or preflight.
    for name in ('baseline', 'variants', 'path_integrity', 'confirmation_pilot'):
        path = root/'simulations/diagnostics'/f'{name}.py'
        if path.exists():
            science.add(str(path.relative_to(root)))
    science.discard('simulations/experiments/reporting.py')
    rendering = {str(p.relative_to(root)) for p in (root/'simulations').rglob('*.py')} - science
    names = science if kind == 'science' else rendering
    if kind == 'audit':
        names = science | {str(p.relative_to(root)) for p in (root/'simulations/diagnostics').rglob('*.py')}
    if kind not in ('science', 'render', 'audit'):
        raise ValueError('Unknown source dependency class.')
    return {name: file_hash(root/name) for name in sorted(names)}


def scientific_design(design):
    value = asdict(design) if is_dataclass(design) else dict(design)
    value.pop('workers', None)
    return value


def identity(design, environment, kernel, sources=None):
    return digest({'schema': SCHEMA, 'design': scientific_design(design),
                   'environment': asdict(environment), 'kernel': kernel,
                   'sources': sources if sources is not None else source_hashes()})


def require(condition, message='Scientific verification failed.'):
    # Unlike assert, this gate remains active under python -O.
    if not condition:
        raise ValueError(message)


@contextmanager
def atomic_file(path, mode='wb'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, mode) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def json_write(path, value):
    with atomic_file(path, 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def npz_write(path, **values):
    with atomic_file(path) as handle:
        np.savez_compressed(handle, **values)


@contextmanager
def output_lock(output):
    output = Path(output).resolve()
    historical = [ROOT/'simulations/outputs'/name for name in ('paper', 'smoke', 'audit')]
    require(not any(output == p or p in output.parents for p in historical),
            'Historical V1 output is read-only. Choose a new --output directory.')
    output.mkdir(parents=True, exist_ok=True)
    # Never unlink the lock inode: another process may already have opened it.
    with (output/'.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process owns this output directory.') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
