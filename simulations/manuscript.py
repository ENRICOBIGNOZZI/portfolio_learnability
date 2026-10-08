"""Compile the integrated manuscript while verifying preserved source bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from simulations.paper import generate_section


ROOT = Path(__file__).resolve().parents[1]


def verify_preservation():
    report = json.loads((ROOT/'simulations/outputs/audit/manuscript_preservation.json').read_text())
    for row in report['protected_files']:
        path = ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Theoretical source changed: {path}')
    empirical = json.loads((ROOT/'simulations/outputs/audit/empirical_preservation.json').read_text())
    for name, expected in empirical.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Protected empirical source or artifact changed: {name}')
    return {'theory_entries_verified': len(report['protected_files']),
            'empirical_entries_verified': len(empirical)}


def compile_manuscript(smoke_layout=False):
    profile = 'smoke' if smoke_layout else 'paper'
    output = ROOT/'simulations/outputs'/profile
    if not smoke_layout:
        from simulations.diagnostics.verify import verify
        verify(output, 'paper')
    preservation = verify_preservation()
    generate_section(output, profile)
    paper = ROOT/'paper'
    main = paper/'main.tex'
    build = ROOT/'tmp/pdfs'/('smoke_manuscript' if smoke_layout else 'manuscript')
    build.mkdir(parents=True, exist_ok=True)
    if smoke_layout:
        main = build/'layout_smoke.tex'
        main.write_text((paper/'main.tex').read_text().replace(
            '../simulations/outputs/paper', '../simulations/outputs/smoke'))
    env = os.environ.copy()
    # Preserve nested input commands inside the original proof byte-for-byte.
    env['TEXINPUTS'] = str(paper)+os.pathsep+str(paper/'theory')+os.pathsep+env.get('TEXINPUTS', '')
    result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                             '-outdir='+str(build), str(main)], cwd=paper, env=env,
                            capture_output=True, text=True)
    (build/'compile_transcript.txt').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Manuscript compilation failed: {build / "compile_transcript.txt"}')
    destination = output/'paper'/('full_layout_smoke.pdf' if smoke_layout else 'The_Law_of_Portfolio_Learnability.pdf')
    shutil.copyfile(build/(main.stem+'.pdf'), destination)
    verification = {**preservation, 'compiled': True, 'profile': profile,
                    'source_main_sha256': hashlib.sha256((paper/'main.tex').read_bytes()).hexdigest(),
                    'pdf_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'pdf': str(destination.relative_to(ROOT)), 'visual_review': 'pending'}
    verification['generated_source_hashes'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (output/'paper/simulation_section.tex', output/'paper/simulation_appendix.tex',
                     ROOT/'simulations/paper.py', ROOT/'simulations/manuscript.py')}
    (output/'paper/manuscript_compilation.json').write_text(json.dumps(verification, indent=2)+'\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-layout', action='store_true',
                        help='Layout preview containing explicitly labeled smoke results; not the final paper.')
    print(compile_manuscript(parser.parse_args().smoke_layout))
