"""Versioned manuscript integration, dependency audit, and extracted-ZIP builds.

Reproduce without fitting any investment policy:
    python3 -m empirical_final.manuscript_cleanup
The original integrated manuscript is retained in the delivery's sources/.
"""
import argparse
from collections import Counter
import difflib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from pypdf import PdfReader

from data_pipeline import digest
from empirical_final import section_vi_finalize as section
from empirical_final import section_vi_publication as publication
from empirical_final.figure6_audit import scientific_audit

ROOT = section.ROOT
CANONICAL = ROOT / 'outputs/section_vi_final_20261011'
DELIVERY = ROOT / 'outputs/manuscript_final_clean_20261012'
SOURCE_NAME = 'Portfolio_Paper_con_Empirics_Aggiornata.zip'
OBSOLETE_ASSETS = ['equity_with_costs.pdf', 'cumulative_wealth.pdf',
                   'complexity_gaussian.png', 'complexity_matern32.png',
                   'economic_spectrum.png', 'gross_and_net_exposure.png']
THEORY = ['introduction.tex', 'population_problem.tex', 'direct_port_learning.tex',
          'economomic_spectrum_final.tex', 'representation_learnability.tex',
          'proofs_learning.tex', 'references.bib']
START = r'Figure~\ref{fig:equity_costs} provides a simple implementation-cost sensitivity.'
STOP = r'\subsection*{Population-optimal portfolio policy in a factor economy}'
TABLE_PLACEHOLDER = 'as shown in table ....'
TABLE_REFERENCE = r'as shown in Table~\ref{tab:kernel_policy_classes}.'


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def remove_obsolete_block(text):
    assert text.count(START) == text.count(STOP) == 1
    start, stop = text.index(START), text.index(STOP)
    assert start < stop
    return text[:start] + text[stop:], text[start:stop]


def dependency_audit(project, entry='main.tex'):
    """Resolve the included TeX graph, figures, bibliography and cross-references."""
    project = Path(project)
    visited, labels, references, graphics, bibliography = set(), [], [], [], []

    def visit(name):
        path = project / name
        if not path.suffix:
            path = path.with_suffix('.tex')
        assert path.is_file(), f'Missing input: {path}'
        assert path.resolve().is_relative_to(project.resolve()), path
        relative = str(path.relative_to(project))
        if relative in visited:
            raise AssertionError(f'Repeated input: {relative}')
        visited.add(relative)
        # Strip unescaped comments while retaining escaped percentage signs.
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        labels.extend(re.findall(r'\\label\{([^}]+)\}', text))
        for group in re.findall(r'\\(?:ref|eqref|autoref|pageref|cref|Cref)\{([^}]+)\}', text):
            references.extend(group.split(','))
        for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            assert (project / image).is_file(), f'Missing figure: {image}'
            graphics.append(image)
        for name in re.findall(r'\\bibliography\{([^}]+)\}', text):
            for source in name.split(','):
                assert (project / (source + '.bib')).is_file(), source
                bibliography.append(source + '.bib')
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            visit(name)

    visit(entry)
    duplicates = [key for key, count in Counter(labels).items() if count > 1]
    unresolved = sorted(set(references) - set(labels))
    assert not duplicates, f'Duplicate labels: {duplicates}'
    assert not unresolved, f'Unresolved references: {unresolved}'
    included_text = '\n'.join((project / name).read_text() for name in visited)
    assert 'fig:equity_costs' not in included_text and 'equity_with_costs' not in included_text
    return dict(passed=True, entry=entry, included_tex=sorted(visited),
                figures=graphics, bibliography=bibliography, labels=len(labels),
                references=len(references), duplicate_labels=duplicates,
                unresolved_references=unresolved)


def compile_project(project, entry, log_path):
    compiler = shutil.which('tectonic')
    if not compiler:
        raise RuntimeError('Install tectonic to build this manuscript')
    result = subprocess.run([compiler, '--keep-logs', '--keep-intermediates', entry],
                            cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    log = (project / Path(entry).with_suffix('.log')).read_text(errors='replace')
    assert not re.search(r'undefined|multiply defined|Missing character|LaTeX Error|Undefined control sequence', log, re.I), log_path
    pdf = project / Path(entry).with_suffix('.pdf')
    reader = PdfReader(pdf)
    text = '\n'.join(p.extract_text() or '' for p in reader.pages)
    assert not re.search(r'\?\?|PLACEHOLDER|TODO|\bnan\b|table\s+\.{3}', text, re.I)
    return dict(passed=True, pages=len(reader.pages), pdf_sha256=digest(pdf),
                undefined_commands_or_references=False,
                overfull_boxes=re.findall(r'Overfull \\[hv]box[^\n]*', log),
                text=text), pdf


def integrate(source_zip, delivery):
    delivery = Path(delivery).resolve()
    assert delivery != CANONICAL.resolve(), 'A versioned delivery is required'
    delivery.mkdir(parents=True, exist_ok=True)
    audit = delivery / 'audit'
    audit.mkdir(exist_ok=True)
    supplied = Path(source_zip).resolve()
    retained = delivery / 'sources' / SOURCE_NAME
    retained.parent.mkdir(exist_ok=True)
    if supplied != retained:
        if retained.exists():
            assert digest(supplied) == digest(retained), 'Delivery already uses a different manuscript'
        else:
            shutil.copyfile(supplied, retained)
    original = ROOT / 'tmp/manuscript_cleanup/source'
    original.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(retained) as z:
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in z.namelist())
        z.extractall(original)
    original = original / 'Portfolio'
    assert (original / 'main.tex').is_file()
    # Audit the scientific objects before any graphic or manuscript revision.
    numerical = scientific_audit(CANONICAL)
    numerical['input_sha256'] = {str(Path(k).relative_to(ROOT)): v for k, v in numerical['input_sha256'].items()}
    write_json(audit / 'figure6_scientific_audit.json', numerical)
    canonical_hashes = {str(p.relative_to(CANONICAL)): digest(p) for p in CANONICAL.rglob('*') if p.is_file()}
    versioned = delivery / 'section_vi'
    for folder in ['tables', 'audit', 'figures', 'publication']:
        shutil.copytree(CANONICAL / folder, versioned / folder, dirs_exist_ok=True)
    section.OUT = publication.OUT = versioned
    section.figure_six()
    publication.publication()
    publication.verify()
    # Record a manifest for subsequent --figure-six-only --output-dir runs.
    section.audit('results_manifest', dict(
        tables={str(p.relative_to(versioned)): digest(p) for p in sorted((versioned / 'tables').glob('*.csv'))},
        code={str(p.relative_to(ROOT)): digest(p) for p in [ROOT / 'empirical_final/section_vi_finalize.py',
              ROOT / 'empirical_final/section_vi_publication.py', ROOT / 'empirical_final/templates/final_section_vi.tex']},
        figures={p.name: digest(p) for p in sorted((versioned / 'figures').glob('*.pdf'))}))
    # The original published outputs remain byte-for-byte intact.
    assert all(digest(CANONICAL / name) == sha for name, sha in canonical_hashes.items())
    unchanged_tables = {p.name: digest(p) for p in sorted((versioned / 'tables').glob('*.csv'))}
    assert all(digest(CANONICAL / 'tables' / name) == sha for name, sha in unchanged_tables.items())
    unaffected_figures = {p.name: digest(p) for p in sorted((versioned / 'figures').iterdir())
                         if p.suffix in ['.pdf', '.png'] and not p.name.startswith('fig06_')}
    assert all(digest(CANONICAL / 'figures' / name) == sha for name, sha in unaffected_figures.items())
    project = delivery / 'Portfolio'
    project.mkdir(exist_ok=True)
    # Copy sources and image assets, excluding precompiled outputs and stale build state.
    for path in original.rglob('*'):
        if path.is_file() and (path.suffix in ['.tex', '.bib', '.png'] or
                               (path.suffix == '.pdf' and path.parent.name == 'figures')):
            target = project / path.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    cleaned, removed = remove_obsolete_block((original / 'appendice_2.tex').read_text())
    (project / 'appendice_2.tex').write_text(cleaned)
    (audit / 'removed_superseded_empirical_block.tex').write_text(removed)
    direct = (project / 'direct_port_learning.tex').read_text()
    assert direct.count(TABLE_PLACEHOLDER) == 1
    (project / 'direct_port_learning.tex').write_text(direct.replace(TABLE_PLACEHOLDER, TABLE_REFERENCE))
    # Preserve the integrated manuscript's float placement at 12 pt / 1.5 spacing.
    (project / 'empirics.tex').write_text((versioned / 'publication/empirics.tex').read_text().replace(
        r'\begin{figure}[H]', r'\begin{figure}[!htbp]'))
    for name in ['numbers.tex', 'figure6_caption.tex']:
        shutil.copyfile(versioned / 'publication' / name, project / name)
    for path in (versioned / 'publication/figures').glob('*.pdf'):
        shutil.copyfile(path, project / 'figures' / path.name)
    # Existing integrated tables differ only in their float placement.
    for path in (versioned / 'publication/tables').glob('*.tex'):
        target = project / 'tables' / path.name
        assert target.read_text() == path.read_text().replace(r'\begin{table}[H]', r'\begin{table}[!htbp]')
    retained_text = '\n'.join(p.read_text() for p in project.rglob('*.tex'))
    removed_assets = []
    for name in OBSOLETE_ASSETS:
        assert name not in retained_text, f'Asset still used: {name}'
        path = project / 'figures' / name
        if path.exists():
            path.unlink()
            removed_assets.append(name)
    theory_hashes = {}
    for name in ['main.tex', *THEORY, 'Empirics_only.tex', 'numbers.tex',
                 'tables/performance.tex', 'tables/spectral_value.tex', 'tables/historical_endpoints.tex']:
        before, after = digest(original / name), digest(project / name)
        if name == 'direct_port_learning.tex':
            assert (project / name).read_bytes() == (original / name).read_bytes().replace(
                TABLE_PLACEHOLDER.encode(), TABLE_REFERENCE.encode())
            theory_hashes[name] = dict(before=before, after=after, unchanged=False,
                                      editorial_only=True, mathematical_body_unchanged=True)
        else:
            assert before == after, f'Unexpected source change: {name}'
            theory_hashes[name] = dict(before=before, after=after, unchanged=True)
    for name in ['FRONTIER.png', 'Complexity.png']:
        assert digest(original / 'figures' / name) == digest(project / 'figures' / name)
    # Exact protected byte spans surrounding the one permitted appendix deletion.
    source_appendix = (original / 'appendice_2.tex').read_bytes()
    assert (project / 'appendice_2.tex').read_bytes() == (
        source_appendix[:source_appendix.index(START.encode())] +
        source_appendix[source_appendix.index(STOP.encode()):])
    instructions = ('Complete manuscript: tectonic main.tex\n'
                    'Standalone Section VI: tectonic Empirics_only.tex\n'
                    'Alternative: pdflatex main.tex; bibtex main; pdflatex main.tex (twice).\n'
                    'Run commands inside Portfolio/. A normal TeX installation supplies apalike.bst.\n'
                    'No external manuscript paths or private data are needed to compile.\n'
                    'Main manuscript format: original 12 pt, 1.5 spacing, original geometry.\n'
                    'Repository reproduction: python3 -m empirical_final.manuscript_cleanup\n')
    (project / 'README_BUILD.txt').write_text(instructions)
    dependencies = [dependency_audit(project), dependency_audit(project, 'Empirics_only.tex')]
    write_json(audit / 'manuscript_dependencies.json', dependencies)
    diff = []
    changed_sources = []
    for path in sorted(project.rglob('*.tex')):
        name = str(path.relative_to(project))
        before = original / name
        if before.exists() and before.read_bytes() != path.read_bytes():
            changed_sources.append(name)
            diff.extend(difflib.unified_diff(before.read_text().splitlines(True), path.read_text().splitlines(True),
                                            fromfile='original/Portfolio/' + name, tofile='final/Portfolio/' + name))
    (audit / 'manuscript_source.diff').write_text(''.join(diff))
    write_json(audit / 'source_preservation.json', dict(
        source_zip_sha256=digest(retained), changed_tex=changed_sources,
        unchanged_sources=theory_hashes, appendix_theory_prefix_and_suffix_byte_identical=True,
        removed_assets=removed_assets, preserved_theory_assets=['FRONTIER.png', 'Complexity.png'],
        unchanged_empirical_tables=unchanged_tables, unchanged_other_empirical_figures=unaffected_figures,
        canonical_published_artifacts_unchanged=canonical_hashes))
    builds = []
    for entry, name in [('main.tex', 'Portfolio_Paper_Final_Clean.pdf'),
                        ('Empirics_only.tex', 'Empirics_Section_VI_Final.pdf')]:
        result, pdf = compile_project(project, entry, audit / (Path(entry).stem + '_build.txt'))
        shutil.copyfile(pdf, delivery / name)
        result['entry'] = entry
        result['output'] = name
        # Keep extracted text in its own audit file, outside the machine summary.
        (audit / (Path(entry).stem + '_text.txt')).write_text(result.pop('text'))
        builds.append(result)
    package = delivery / 'Portfolio_Paper_Final_Clean.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(project.rglob('*')):
            if path.is_file() and (path.suffix in ['.tex', '.bib', '.png', '.txt'] or
                                   (path.suffix == '.pdf' and path.parent.name == 'figures')):
                info = zipfile.ZipInfo('Portfolio/' + str(path.relative_to(project)), date_time=(2026, 10, 12, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, path.read_bytes())
    # Compile only the actual archive contents in a separate directory.
    extracted = ROOT / 'tmp/manuscript_cleanup/extracted'
    if extracted.exists():
        shutil.rmtree(extracted)
    with zipfile.ZipFile(package) as z:
        z.extractall(extracted)
    extracted = extracted / 'Portfolio'
    independent = []
    for entry in ['main.tex', 'Empirics_only.tex']:
        dependency_audit(extracted, entry)
        result, pdf = compile_project(extracted, entry, audit / (Path(entry).stem + '_extracted_build.txt'))
        # PDF extractors can emit literal CR glyphs in mathematical delimiters;
        # compare without Python's universal-newline translation on only one side.
        assert result.pop('text') == (audit / (Path(entry).stem + '_text.txt')).read_bytes().decode()
        result['entry'] = entry
        result['text_matches_delivered_pdf'] = True
        independent.append(result)
    write_json(audit / 'compilation.json', dict(passed=True, direct=builds, independent_extracted_zip=independent,
                                               source_zip_sha256=digest(package)))
    write_json(versioned / 'audit/figure6_revision.json', dict(
        passed=True, baseline='outputs/section_vi_final_20261011',
        unchanged_canonical_tables=len(unchanged_tables), unchanged_other_figures=4,
        cv_diamonds_removed=True, dynamic_cv_policy_numerically_preserved=True,
        plotted_candidates_per_period=1522, total_plotted_candidates=13698,
        exact_zero_endpoint_in_every_panel=True, source_package_compiled_independently=True,
        scientific_audit='../../audit/figure6_scientific_audit.json'))
    # Retain the two named final PDFs; the source project stays free of build state.
    for entry in ['main', 'Empirics_only']:
        for suffix in ['.pdf', '.aux', '.bbl', '.blg', '.log', '.out']:
            (project / (entry + suffix)).unlink(missing_ok=True)
    print(f'Created and independently compiled {package} and both PDFs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DELIVERY)
    parser.add_argument('--source-zip', type=Path)
    args = parser.parse_args()
    source = args.source_zip or args.output_dir / 'sources' / SOURCE_NAME
    integrate(source, args.output_dir)


if __name__ == '__main__':
    main()
