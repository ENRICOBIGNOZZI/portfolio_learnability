"""Compile the isolated real-data wrapper, preserving all original paper sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication',type=Path,required=True)
    args=parser.parse_args();pub=args.publication.resolve()
    final=pub/'manuscript_real_data.pdf'
    if final.exists():raise FileExistsError('Preserve completed PDF; choose a new publication revision')
    env=os.environ.copy();env['TEXINPUTS']='.:./theory//:'+env.get('TEXINPUTS','')
    command=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(pub/'build'),str(pub/'manuscript_real_data.tex')]
    result=subprocess.run(command,cwd=ROOT/'paper',env=env,capture_output=True,text=True)
    (pub/'build').mkdir(exist_ok=True)
    (pub/'build/compile.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    log=(pub/'build/manuscript_real_data.log').read_text()
    if 'undefined' in log or 'Overfull' in log:raise ValueError('Resolve references or overfull boxes before delivery')
    inputs={}
    for line in (pub/'build/manuscript_real_data.fls').read_text().splitlines():
        if not line.startswith('INPUT '):continue
        path=Path(line[6:]);path=(ROOT/'paper'/path).resolve()
        if path.is_relative_to(ROOT):
            if 'simulations' in path.relative_to(ROOT).parts:raise ValueError('Forbidden manuscript dependency')
            if path.suffix in ('.tex','.bib','.pdf'):
                inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copyfile(pub/'build/manuscript_real_data.pdf',final)
    report={'passed':True,'pdf_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'inputs':inputs,
            'command':command,'working_directory':'paper','TEXINPUTS':env['TEXINPUTS'],
            'layout_scope':'No overfull boxes or unresolved references; inherited font substitutions and underfull introduction paragraphs retained.'}
    with (pub/'build_verification.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(final)

if __name__=='__main__':main()
