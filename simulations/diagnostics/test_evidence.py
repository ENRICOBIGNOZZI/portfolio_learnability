"""Execute tests and bind the actual transcript to source and runtime hashes."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

from simulations.provenance import ROOT, file_hash, json_write, source_hashes, utc_now


def run(destination, arguments):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pytest', *arguments]
    sources = source_hashes() | source_hashes('render')
    sources.update({str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT/'tests').rglob('*.py')})
    started = utc_now()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log = destination/'test_transcript.txt'
    log.write_text(result.stdout+result.stderr)
    report = {'schema': 'executed-tests/2.0', 'command': command, 'cwd': str(ROOT),
              'started_utc': started, 'completed_utc': utc_now(), 'return_code': result.returncode,
              'passed': result.returncode == 0, 'source_hashes': sources,
              'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'pytest')},
              'environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
              'log': str(log), 'log_sha256': file_hash(log)}
    json_write(destination/'test_verification.json', report)
    print(result.stdout+result.stderr, end='')
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2/audit'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.arguments or ['tests', '-q']))
