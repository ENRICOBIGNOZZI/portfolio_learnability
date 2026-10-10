"""Transport exact frozen numerical caches without adding large cache blobs to Git."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

from simulations.extended.design import OUTPUT, ENVIRONMENTS
from simulations.extended.remote_compatibility import protocol_checked
from simulations.provenance import atomic_file, file_hash, json_write, utc_now


REPOSITORY = 'ENRICOBIGNOZZI/portfolio_learnability'
MANIFEST = OUTPUT / 'remote_input_manifest.json'


def cache_paths(protocol):
    rank, basis = protocol['rank'], protocol['basis_seed']
    groups, seed = protocol['population_groups'], protocol['population_seed']
    paths = [f'pilot/basis_P{rank}_seed{basis}.npz']
    cases = [(name, rank) for name in ('baseline', 'N300', 'N1200')] + [('baseline', 512)]
    paths += [f'pilot/operator_{name}_P{p}_B{basis}_Q{groups}_S{seed}.npz' for name, p in cases]
    return paths


def supporting_paths():
    return ['calibration.json', 'seed_manifest.json'] + [
        f'population/{name}_theory.npz' for name in ENVIRONMENTS]


def prepare(protocol):
    files = []
    for relative in cache_paths(protocol):
        path = OUTPUT / relative
        files.append(dict(path=relative, asset=path.name, bytes=path.stat().st_size,
                          sha256=file_hash(path)))
    supporting = [dict(path=relative, bytes=(OUTPUT / relative).stat().st_size,
                       sha256=file_hash(OUTPUT / relative)) for relative in supporting_paths()]
    if supporting[0]['sha256'] != protocol['preproduction_files']['calibration.json']:
        raise ValueError('Population calibration differs from the frozen preproduction audit.')
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        if (old['run_hash'] != protocol['run_hash'] or old['files'] != files
                or old['supporting_files'] != supporting):
            raise ValueError('Do not overwrite a published input manifest with different caches.')
        return old
    manifest = dict(run_hash=protocol['run_hash'], repository=REPOSITORY,
        release_tag=f'rich6d-extended-inputs-{protocol["run_hash"][:12]}-v1',
        prepared_utc=utc_now(), files=files, supporting_files=supporting,
        scope='Exact auxiliary numerical inputs for the frozen study, not completed Monte Carlo results.')
    json_write(MANIFEST, manifest)
    return manifest


def validated_manifest(protocol):
    manifest = json.loads(MANIFEST.read_text())
    if manifest['run_hash'] != protocol['run_hash'] or manifest['repository'] != REPOSITORY:
        raise ValueError('Remote numerical inputs belong to a different study or repository.')
    if manifest['release_tag'] != f'rich6d-extended-inputs-{protocol["run_hash"][:12]}-v1':
        raise ValueError('Unexpected input release identity.')
    if [row['path'] for row in manifest['files']] != cache_paths(protocol):
        raise ValueError('Unexpected cache paths in the numerical input manifest.')
    if [row['path'] for row in manifest['supporting_files']] != supporting_paths():
        raise ValueError('Unexpected supporting files in the numerical input manifest.')
    for row in manifest['supporting_files']:
        path = OUTPUT / row['path']
        if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
            raise ValueError(f'Canonical supporting input changed: {row["path"]}')
    if file_hash(OUTPUT / 'calibration.json') != protocol['preproduction_files']['calibration.json']:
        raise ValueError('Frozen population calibration changed.')
    for row in manifest['files']:
        if row['asset'] != Path(row['path']).name or row['bytes'] <= 0:
            raise ValueError('Invalid numerical input asset metadata.')
        if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('Invalid numerical input checksum.')
    return manifest


def download(protocol):
    manifest = validated_manifest(protocol)
    for row in manifest['files']:
        path = OUTPUT / row['path']
        if path.exists():
            if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
                raise ValueError(f'Existing cache does not match the frozen input: {path}')
            continue
        url = f'https://github.com/{REPOSITORY}/releases/download/{manifest["release_tag"]}/{row["asset"]}'
        for attempt in range(3):
            try:
                h, size = hashlib.sha256(), 0
                with atomic_file(path) as handle, urlopen(url, timeout=60) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        handle.write(block)
                        h.update(block)
                        size += len(block)
                    if size != row['bytes'] or h.hexdigest() != row['sha256']:
                        raise ValueError(f'Downloaded numerical input failed checksum: {row["asset"]}')
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2)
        print('Verified numerical input', row['asset'], row['sha256'], flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    protocol = protocol_checked()
    result = prepare(protocol) if args.prepare else download(protocol)
    print(json.dumps(dict(manifest=str(MANIFEST), files=len(result['files']),
                         bytes=sum(row['bytes'] for row in result['files']))))


if __name__ == '__main__':
    main()
