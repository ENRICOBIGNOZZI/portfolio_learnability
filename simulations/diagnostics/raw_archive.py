"""Pack, download and verify synthetic checkpoints with per-file SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

from simulations.provenance import ROOT, file_hash, json_write, require, source_hashes, utc_now

REPOSITORY = 'ENRICOBIGNOZZI/portfolio_learnability'
TAG = 'synthetic-checkpoints-cdfbd11-v1'
ASSET = 'synthetic-checkpoints-cdfbd11.tar.gz'
INDEX = ROOT/'simulations/outputs/confirmation_v2/audit/raw_archive.json'


def pack(destination):
    snapshot = json.loads((ROOT/'simulations/outputs/confirmation_v2/audit/historical_snapshot.json').read_text())
    files = {name: row for name, row in snapshot['files'].items() if name.endswith('.npz')}
    require(all(name.startswith('simulations/outputs/') for name in files), 'Only synthetic files are permitted')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/ASSET
    with tarfile.open(archive, 'w:gz', compresslevel=1) as handle:
        for name, row in sorted(files.items()):
            require(file_hash(ROOT/name) == row['sha256'], f'Historical checkpoint changed: {name}')
            info = handle.gettarinfo(str(ROOT/name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            with (ROOT/name).open('rb') as source:
                handle.addfile(info, source)
    index = {'schema': 'synthetic-raw-archive/2.0', 'source_commit': snapshot['git_sha'],
             'created_utc': utc_now(), 'release_tag': TAG, 'repository': REPOSITORY,
             'asset': ASSET, 'sha256': file_hash(archive), 'bytes': archive.stat().st_size,
             'url': f'https://github.com/{REPOSITORY}/releases/download/{TAG}/{ASSET}',
             'files': files, 'file_count': len(files), 'published_and_download_verified': False,
             'scope': 'Synthetic Monte Carlo NPZ checkpoints and population moments only; no empirical inputs.'}
    json_write(INDEX, index)
    json_write(destination/'raw_archive.json', index)
    print(str(archive), flush=True)
    return index


def pack_confirmation(destination):
    output = ROOT/'simulations/outputs/confirmation_v2'
    protocol = json.loads((ROOT/'simulations/config/confirmation_v2.json').read_text())
    manifest = json.loads((output/'run_manifest.json').read_text())
    require(manifest['source_hashes'] == source_hashes(), 'Scientific source changed before packaging')
    for env in manifest['configuration']['cases']:
        names = sorted(p.name for p in (output/'data'/env['name']/'replications').glob('*.npz'))
        require(names == [f'{r:04d}.npz' for r in range(env['replications'])], 'Cannot package an incomplete confirmation cohort')
    for filename in ('confirmation_reconstruction.json', 'numerical_reconstruction.json'):
        evidence = json.loads((output/'audit'/filename).read_text())
        require(evidence['passed'] and not evidence.get('partial') and not evidence.get('raw_only'),
                'Full independent reconstruction required before packaging')
        for name, expected in evidence['input_hashes'].items():
            require(file_hash(output/name) == expected, 'Changed verified archive input: '+name)
        for name, expected in evidence['verifier_sources'].items():
            require(file_hash(ROOT/name) == expected, 'Changed archive verifier: '+name)
    members = list(output.rglob('*.npz'))
    members += [ROOT/'simulations/config/confirmation_v2.json', output/'run_manifest.json',
                output/'audit/protocol_freeze.json']
    members += list((output/'data').glob('*/population.json'))
    files = {str(p.relative_to(ROOT)): {'sha256': file_hash(p), 'bytes': p.stat().st_size}
             for p in sorted(members)}
    require(any(name.endswith('.npz') for name in files), 'No confirmation checkpoint files')
    tag = 'synthetic-confirmation-v2-'+protocol['protocol_hash'][:12]
    asset = 'synthetic-confirmation-v2.tar.gz'
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    archive = destination/asset
    with tarfile.open(archive, 'w:gz', compresslevel=1) as handle:
        for name in files:
            info = handle.gettarinfo(str(ROOT/name), arcname=name)
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
            with (ROOT/name).open('rb') as stream:
                handle.addfile(info, stream)
    index = {'schema': 'synthetic-raw-archive/2.0', 'protocol_hash': protocol['protocol_hash'],
             'scientific_source_hashes': manifest['source_hashes'],
             'configuration_sha256': file_hash(ROOT/'simulations/config/confirmation_v2.json'),
             'created_utc': utc_now(), 'release_tag': tag, 'repository': REPOSITORY, 'asset': asset,
             'sha256': file_hash(archive), 'bytes': archive.stat().st_size, 'files': files, 'file_count': len(files),
             'url': f'https://github.com/{REPOSITORY}/releases/download/{tag}/{asset}',
             'scope': 'Synthetic confirmation paths, selected coefficients/payoffs, quadrature moments, numerical audit checkpoints, frozen configuration and population/production manifests; no licensed empirical data.'}
    json_write(output/'audit/confirmation_raw_archive.json', index)
    json_write(destination/'confirmation_raw_archive.json', index)
    print(archive, flush=True)
    return index


def fetch(destination, index_path=INDEX):
    index = json.loads(Path(index_path).read_text())
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp)/index['asset']
        # Public release: reproduction requires no GitHub authentication.
        urllib.request.urlretrieve(index['url'], archive)
        require(file_hash(archive) == index['sha256'], 'Archive checksum mismatch')
        with tarfile.open(archive, 'r:gz') as handle:
            members = handle.getmembers()
            require(len(members) == len(index['files']), 'Archive member count mismatch')
            require({m.name for m in members} == set(index['files']), 'Unexpected or duplicate archive member')
            for member in members:
                path = destination/member.name
                require(member.isfile() and destination in path.resolve().parents, 'Unsafe archive path')
                contents = handle.extractfile(member).read()
                require(hashlib.sha256(contents).hexdigest() == index['files'][member.name]['sha256'], 'Member checksum mismatch')
                if path.exists():
                    require(file_hash(path) == index['files'][member.name]['sha256'], 'Refusing to overwrite differing local checkpoint')
                else:
                    from simulations.provenance import atomic_file
                    with atomic_file(path) as output:
                        output.write(contents)
    return {'passed': True, 'archive_sha256': index['sha256'], 'files_verified': len(members),
            'downloaded_utc': utc_now(), 'url': index['url']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pack', 'pack-confirmation', 'fetch'])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--index', type=Path, default=INDEX)
    args = parser.parse_args()
    result = pack(args.destination) if args.action == 'pack' else pack_confirmation(args.destination) if args.action == 'pack-confirmation' else fetch(args.destination, args.index)
    if args.report:
        json_write(args.report, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
