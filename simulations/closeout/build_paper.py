"""Compile isolated full manuscript and simulation-only documents before linking."""
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
from simulations.closeout.results import DEST
from simulations.extended.design import ROOT
from simulations.provenance import file_hash,json_write,utc_now

OLD=r'\newcommand{\simoutput}{../simulations/outputs/confirmation_v2}'
NEW=r'\newcommand{\simoutput}{../simulations/outputs/rich6d_initial_cv60_closeout_v1}'


def run():
    tectonic=shutil.which('tectonic') or '/opt/homebrew/bin/tectonic'
    if not Path(tectonic).exists():raise RuntimeError('Tectonic required for staging compilation.')
    target=DEST/'paper';target.mkdir(exist_ok=True)
    protected={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'paper/theory').rglob('*') if p.is_file()}
    protected.update({str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'paper/empirics.tex',*(ROOT/'paper/complexity_schedule').rglob('*.tex')]})
    source=(ROOT/'paper/main.tex').read_text()
    if OLD not in source and NEW not in source:raise ValueError('Unknown simulation link; inspect before replacing.')
    staged=source.replace(OLD,NEW)
    # Verify actual generated inputs and PDFs before compilation.
    inputs=['simulation_section.tex','simulation_appendix.tex','final_table.tex','slope_table.tex','economy_table.tex','recovered_diagnostics.tex','numerical_scope.tex']
    for name in inputs:
        if not (target/name).is_file():raise ValueError('Missing simulation input '+name)
    for name in ['figure_0_spectrum','figure_1_recovery','figure_2_complexity','figure_3_sharpe_complexity','appendix_economies','appendix_normalized']:
        if not (DEST/'figures'/f'{name}.pdf').is_file():raise ValueError('Missing figure '+name)
    with tempfile.TemporaryDirectory(prefix='simulation-closeout-') as scratch:
        staging=Path(scratch)
        shutil.copytree(ROOT/'paper',staging/'paper',ignore=shutil.ignore_patterns('*.aux','*.log','*.out','*.bbl','*.blg','*.fdb_latexmk','*.fls'))
        (staging/'simulations').symlink_to(ROOT/'simulations',target_is_directory=True)
        # Nested theory inputs are written without their directory prefix.
        # Supply the original files through staging-only links, without editing them.
        for theory_input in (staging/'paper/theory').glob('*.tex'):
            alias=staging/'paper'/theory_input.name
            if not alias.exists(): alias.symlink_to(theory_input)
        (staging/'paper/main.tex').write_text(staged)
        result=subprocess.run([tectonic,'--keep-logs','--keep-intermediates','main.tex'],cwd=staging/'paper',capture_output=True,text=True)
        (target/'staging_build.txt').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Staging manuscript build failed; see staging_build.txt')
        shutil.copyfile(staging/'paper/main.pdf',target/'paper_staging.pdf')
        log=(staging/'paper/main.log').read_text()
        (target/'staging_latex.log').write_text(log)
        if 'undefined references' in log.lower() or 'undefined citations' in log.lower():
            raise ValueError('Unresolved manuscript references/citations; inspect staging_latex.log.')
        prefix=source[:source.index(r'\begin{document}')].replace(OLD,NEW)
        standalone=prefix+r'''\begin{document}
\begin{center}{\large\bfseries Rich6D simulation closeout}\\[0.15em]12 October 2026\end{center}
\input{\simoutput/paper/simulation_section.tex}
\clearpage\appendix
\renewcommand{\thesection}{\Alph{section}}
\input{\simoutput/paper/simulation_appendix.tex}
\end{document}
'''
        (staging/'paper/simulation_closeout.tex').write_text(standalone)
        result=subprocess.run([tectonic,'--keep-logs','simulation_closeout.tex'],cwd=staging/'paper',capture_output=True,text=True)
        (target/'standalone_build.txt').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Simulation-only document build failed.')
        shutil.copyfile(staging/'paper/simulation_closeout.pdf',target/'simulation_closeout.pdf')
        (target/'simulation_closeout.tex').write_text(standalone.replace('../simulations/outputs/rich6d_initial_cv60_closeout_v1','..'))
    for name,expected in protected.items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Protected source changed during compilation: '+name)
    # Only after successful isolated build make the authorized one-line link concrete.
    (ROOT/'paper/main.tex').write_text(staged)
    json_write(DEST/'paper_build_verification.json',dict(passed=True,compiled_utc=utc_now(),
        only_manuscript_change='One-line simoutput link',protected_source_sha256=protected,
        manuscript_pdf_sha256=file_hash(target/'paper_staging.pdf'),simulation_pdf_sha256=file_hash(target/'simulation_closeout.pdf'),
        simulation_input_sha256={p.name:file_hash(p) for p in target.glob('*.tex') if p.name!='simulation_closeout.tex'},
        references_resolved=True,visual_review_required=True))

if __name__=='__main__':run()
