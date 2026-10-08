"""Compile the integrated manuscript while verifying preserved source bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from simulations.paper import generate_section
from simulations.provenance import file_hash, json_write, require


ROOT = Path(__file__).resolve().parents[1]


def compiler_dependencies(recorder, cwd):
    """Hash actual compiler inputs, including nested sources and graphic files."""
    files = {}
    for line in Path(recorder).read_text().splitlines():
        if not line.startswith('INPUT '):
            continue
        path = Path(line[6:])
        path = path.resolve() if path.is_absolute() else (Path(cwd)/path).resolve()
        if path.is_file() and path.suffix.lower() in ('.tex', '.bib', '.bbl', '.pdf', '.png', '.jpg', '.jpeg', '.eps', '.sty', '.cls'):
            name = str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)
            files[name] = file_hash(path)
    require(bool(files), 'Compiler recorder contains no document dependencies.')
    return dict(sorted(files.items()))


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


def compile_manuscript(smoke_layout=False, confirmation=True):
    initial_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    initial_git_status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)
    profile = 'smoke' if smoke_layout else 'confirmation_v2' if confirmation else 'paper'
    output = ROOT/'simulations/outputs'/('smoke_v2' if smoke_layout else profile)
    require(profile != 'paper', 'Historical V1 output is preserved. Compile the versioned confirmation instead.')
    if not smoke_layout:
        from simulations.diagnostics.reconstruct_confirmation import verify
        verify(output)
    preservation = verify_preservation()
    generate_section(output, profile)
    paper = ROOT/'paper'
    main = paper/'main.tex'
    build = ROOT/'tmp/pdfs'/('smoke_manuscript_v2' if smoke_layout else 'confirmation_manuscript')
    build.mkdir(parents=True, exist_ok=True)
    if smoke_layout:
        import re
        main = build/'layout_smoke.tex'
        body = (paper/'main.tex').read_text()
        body = re.sub(r'\\newcommand\{\\simoutput\}\{[^}]+\}',
                      lambda _: r'\newcommand{\simoutput}{../simulations/outputs/smoke_v2}', body)
        main.write_text(body)
    else:
        require('../simulations/outputs/confirmation_v2' in main.read_text(), 'Main manuscript must include the versioned confirmation section.')
    env = os.environ.copy()
    # Preserve nested input commands inside the original proof byte-for-byte.
    env['TEXINPUTS'] = str(paper)+os.pathsep+str(paper/'theory')+os.pathsep+env.get('TEXINPUTS', '')
    result = subprocess.run(['latexmk', '-pdf', '-recorder', '-interaction=nonstopmode', '-halt-on-error',
                             '-outdir='+str(build), str(main)], cwd=paper, env=env,
                            capture_output=True, text=True)
    (build/'compile_transcript.txt').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Manuscript compilation failed: {build / "compile_transcript.txt"}')
    destination = output/'paper'/('full_layout_smoke.pdf' if smoke_layout else 'The_Law_of_Portfolio_Learnability.pdf')
    shutil.copyfile(build/(main.stem+'.pdf'), destination)
    verification = {**preservation, 'compiled': True, 'profile': profile,
                    'checkout_git_sha': initial_git_sha, 'tracked_tree_clean_at_start': not bool(initial_git_status.strip()),
                    'initial_tracked_status': initial_git_status,
                    'command': result.args, 'return_code': result.returncode,
                    'latex_version': subprocess.check_output(['pdflatex', '--version'], text=True).splitlines()[0],
                    'transcript_sha256': file_hash(build/'compile_transcript.txt'),
                    'source_main_sha256': hashlib.sha256((paper/'main.tex').read_bytes()).hexdigest(),
                    'pdf_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'pdf': str(destination.relative_to(ROOT)), 'visual_review': 'pending'}
    verification['generated_source_hashes'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (output/'paper/simulation_section.tex', output/'paper/simulation_appendix.tex',
                     ROOT/'simulations/paper.py', ROOT/'simulations/manuscript.py')}
    if confirmation and not smoke_layout:
        verification['generated_source_hashes'].update({str(path.relative_to(ROOT)): file_hash(path)
                for path in (output/'paper/simulation_numbers.tex', output/'paper/number_lineage.json', ROOT/'simulations/paper_confirmation.py')})
    verification['compiler_input_hashes'] = compiler_dependencies(build/(main.stem+'.fls'), paper)
    (output/'paper/manuscript_compilation.json').write_text(json.dumps(verification, indent=2)+'\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-layout', action='store_true',
                        help='Layout preview containing explicitly labeled smoke results; not the final paper.')
    print(compile_manuscript(parser.parse_args().smoke_layout))
