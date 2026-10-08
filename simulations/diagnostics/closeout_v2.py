"""Assemble the thirteen requested conditions from current, bound evidence.

This command is deliberately strict about missing execution, source drift and
PDF review. Failed numerical tolerances are retained as scientific limits when
the audits were executed and the manuscript marks those regions unresolved.
"""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from simulations.diagnostics.execution_timing import report_timing
from simulations.manuscript import verify_preservation
from simulations.provenance import ROOT, file_hash, json_write, require, source_hashes, utc_now


def full_suite_command(command):
    """Check pytest arguments separately from Python's module-launch option."""
    if len(command) >= 3 and command[1:3] == ['-m', 'pytest']:
        arguments = command[3:]
    elif command and Path(command[0]).name in ('pytest', 'pytest.exe'):
        arguments = command[1:]
    else:
        return False
    return 'tests' in arguments and not any(
        arg.startswith(('-k', '-m', '--ignore', '--deselect')) for arg in arguments)


def closeout(output):
    output = Path(output).resolve()
    audit = output/'audit'
    conditions = {}

    def read(relative):
        return json.loads((output/relative).read_text())

    def hashes(mapping, base):
        require(bool(mapping), 'Missing evidence hashes')
        for name, expected in mapping.items():
            require(file_hash(base/name) == expected, 'Changed evidence dependency: '+name)

    def condition(name, met, command, evidence, **details):
        conditions[name] = {'condition_met': bool(met), 'command': command,
                            'evidence': {name: file_hash(output/name) for name in evidence}, **details}

    preservation = verify_preservation()
    snapshot = read('audit/historical_snapshot.json')
    for name, original in snapshot['files'].items():
        require(file_hash(ROOT/name) == original['sha256'], 'Historical artifact changed: '+name)
    condition('source_preservation_verified', True, 'python -m simulations.diagnostics.closeout_v2',
              ['audit/source_preservation.json', 'audit/historical_snapshot.json'],
              **preservation, historical_artifacts_verified=len(snapshot['files']),
              authority='Byte comparison with source archive; no inferred personal confirmation.')

    historical = read('audit/historical_reconstruction.json')
    require(historical['passed'] and historical['raw_checkpoint_mode'] and historical['scope'] == 'full',
            'Full historical reconstruction missing')
    require(historical['verifier_sha256'] == file_hash(ROOT/'simulations/diagnostics/reconstruct.py'),
            'Historical reconstruction source is stale')
    count = sum(row['raw_replications'] for row in historical['environments'])
    require(count == 3500, 'Historical evaluation count')
    condition('baseline_reproduced_or_archived_verified', True,
              'python -m simulations.diagnostics.reconstruct --full --report simulations/outputs/confirmation_v2/audit/historical_reconstruction.json',
              ['audit/historical_reconstruction.json'], evaluations=count,
              limitation='Archived raw surfaces to aggregates; no claim of a stock-return fitting replay.')

    tests = read('audit/test_verification.json')
    require(tests['passed'] and tests['return_code'] == 0 and 'tests' in tests['command'],
            'Executed full-suite test evidence missing')
    require(full_suite_command(tests['command']), 'Filtered test suite is not full-suite evidence')
    hashes(tests['source_hashes'], ROOT)
    log = Path(tests['log'])
    if not log.is_absolute():
        log = ROOT/log
    require(file_hash(log) == tests['log_sha256'], 'Test transcript changed')
    for name in ('cache_mutation_tests_passed', 'aggregate_mutation_tests_passed'):
        condition(name, True, tests['command'], ['audit/test_verification.json'],
                  log_sha256=tests['log_sha256'], source_files_bound=len(tests['source_hashes']))

    archive_evidence = []
    for prefix in ('', 'confirmation_'):
        index = read(f'audit/{prefix}raw_archive.json')
        downloaded = read(f'audit/{prefix}raw_archive_download_verification.json')
        require(downloaded['passed'] and downloaded['archive_sha256'] == index['sha256'], 'Raw public download checksum')
        require(downloaded['files_verified'] == len(index['files']), 'Raw public download member count')
        require(downloaded['url'] == index['url'], 'Raw public download URL')
        for name, entry in index['files'].items():
            require(file_hash(ROOT/name) == entry['sha256'], 'Published raw member changed: '+name)
        archive_evidence += [f'audit/{prefix}raw_archive.json', f'audit/{prefix}raw_archive_download_verification.json']
    condition('raw_checkpoint_archive_accessible', True,
              'python -m simulations.diagnostics.raw_archive fetch --destination . [--index confirmation index]', archive_evidence,
              scope='Historical and confirmation synthetic checkpoints, actually downloaded with per-file verification.')

    exact = read('audit/exact_kernel_reference.json')
    require(exact['passed'] and exact['rows'] == 288, 'Exact reference execution incomplete')
    require(exact['source_sha256'] == file_hash(ROOT/'simulations/diagnostics/exact_kernel.py'), 'Stale exact reference')
    require(exact['scientific_source_hashes'] == source_hashes(), 'Exact reference scientific source drift')
    condition('exact_kernel_reference_checked', True, 'python -m simulations.diagnostics.exact_kernel',
              ['audit/exact_kernel_reference.json', 'audit/exact_kernel_reference.csv'],
              comparisons=exact['rows'], limitation=exact['interpretation'])

    numerical = read('audit/numerical_reconstruction.json')
    require(numerical['passed'] and 'spectrum' in numerical['details'], 'Numerical reconstruction incomplete')
    hashes(numerical['input_hashes'], output)
    hashes(numerical['verifier_sources'], ROOT)
    resolved = []
    for name in ('baseline_original', 'rich6d'):
        require('quadrature' in numerical['details'][name], 'Missing quadrature reconstruction')
        rank = pd.read_csv(audit/f'{name}_rank_policy_resolution.csv')
        quad = pd.read_csv(audit/f'{name}_quadrature_resolution.csv')
        merged = rank.merge(quad, on=['environment', 'T', 'method'], suffixes=('_rank', '_quadrature'), validate='one_to_one')
        require(len(merged) == len(rank), 'Missing combined numerical region')
        for _, row in merged.iterrows():
            resolved.append({'environment': name, 'T': int(row['T']), 'method': row.method,
                             'rank_certified': bool(row.certified_rank),
                             'quadrature_certified': bool(row.certified_quadrature),
                             'certified': bool(row.certified_rank and row.certified_quadrature)})
    regions = pd.DataFrame(resolved)
    regions.to_csv(audit/'claimed_resolution_regions.csv', index=False)
    headline = regions[(regions['T'] == 1440)&regions.method.isin(['holdout25', 'rolling3', 'theory_1'])]
    condition('numerical_resolution_certified_for_claimed_region', bool(headline.certified.all()),
              'python -m simulations.diagnostics.reconstruct_resolution', ['audit/numerical_reconstruction.json', 'audit/claimed_resolution_regions.csv'],
              execution_verified=True, headline_cells=int(len(headline)), headline_certified=int(headline.certified.sum()),
              scope='Each listed practical-grid policy cell separately, relative to audited rank 1024 and quadrature; no infinite-kernel equality.',
              limitation='Uncertified cells and extra horizons remain visible and unresolved; a failed tolerance is not an unexecuted audit.')

    gradient = read('audit/rich6d_gradient_audit.json')
    require(gradient['passed'] and gradient['samples'] == 131072, 'Rich-map gradient audit incomplete')
    require(gradient['source_sha256'] == file_hash(ROOT/'simulations/diagnostics/rich6d_audit.py'), 'Stale gradient audit')
    require(gradient['scientific_sources'] == source_hashes(), 'Rich-map scientific source drift')
    condition('rich6d_algebra_and_assumptions_checked', True,
              'python -m simulations.diagnostics.rich6d_audit', ['audit/rich6d_gradient_audit.json', 'audit/test_verification.json'],
              samples=gradient['samples'], limitation='Smooth benchmark; source condition is an analytic assumption, not a minimax hardness claim.')

    manifest = read('run_manifest.json')
    production = read('audit/confirmation_reconstruction.json')
    require(production['passed'] and not production['partial'] and not production['raw_only'], 'Full confirmation reconstruction missing')
    hashes(production['input_hashes'], output)
    hashes(production['verifier_sources'], ROOT)
    inventory = {str(p.relative_to(output)) for p in (output/'data').rglob('*')
                 if p.is_file() and p.suffix in ('.npz', '.csv', '.json')}
    require(inventory == set(production['input_hashes']), 'Confirmation inventory changed after verification')
    require(manifest['source_hashes'] == source_hashes(), 'Confirmation scientific source drift')
    observed = {item['environment']: item['raw_paths'] for item in production['environments']}
    require(observed == {env['name']: env['replications'] for env in manifest['configuration']['cases']}, 'Confirmation cohort mismatch')
    for name in ('theory_path_executed', 'selector_comparison_executed', 'independent_and_forward_OOS_executed'):
        condition(name, True, 'python -m simulations.diagnostics.reconstruct_confirmation',
                  ['audit/confirmation_reconstruction.json', 'run_manifest.json'], environments=observed,
                  evaluations=sum(observed.values()), distinct_innovation_seed_indices=max(observed.values()),
                  limitation='Prospectively reduced 100/50/50 cohorts, not maximum 500/200/200 targets; paired maps reuse innovations.')

    publication = read('audit/publication_reconstruction.json')
    require(publication['passed'] and publication['source_sha256'] == file_hash(ROOT/'simulations/diagnostics/publication_v2.py'), 'Stale publication reconstruction')
    hashes(publication['source_hashes'], output)
    hashes(publication['generated_hashes'], output)
    figures = read('audit/figure_provenance.json')
    require(figures['renderer_sha256'] == file_hash(ROOT/'simulations/plotting/confirmation.py'), 'Stale renderer')
    hashes(figures['sources'], output)
    hashes(figures['figures'], output)
    condition('all_reported_numbers_reconstructed', True,
              'python -m simulations.diagnostics.publication_v2',
              ['audit/publication_reconstruction.json', 'audit/figure_provenance.json', 'audit/confirmation_reconstruction.json', 'audit/numerical_reconstruction.json'],
              numeric_cells=publication['numeric_lineage_cells'], paired_rows=publication['paired_comparison_rows'])

    compilation = read('paper/manuscript_compilation.json')
    require(compilation['compiled'] and compilation['return_code'] == 0
            and compilation['profile'] == 'confirmation_v2' and compilation['tracked_tree_clean_at_start'], 'Clean-checkout compilation missing')
    pdf = ROOT/compilation['pdf']
    require(file_hash(pdf) == compilation['pdf_sha256'], 'Compiled PDF changed')
    require(file_hash(ROOT/'paper/main.tex') == compilation['source_main_sha256'], 'Manuscript include changed')
    hashes(compilation['generated_source_hashes'], ROOT)
    hashes(compilation['compiler_input_hashes'], ROOT)
    visual = read('paper/visual_review.json')
    require(visual['passed'] and visual['pdf_sha256'] == compilation['pdf_sha256'], 'Actual review of final PDF missing')
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    pages = int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))
    require(visual['pages_reviewed'] == pages, 'Incomplete page review')
    condition('paper_compiled_and_render_reviewed', True, compilation['command'],
              ['paper/manuscript_compilation.json', 'paper/visual_review.json'],
              pages=pages, pdf_sha256=compilation['pdf_sha256'], checkout_git_sha=compilation['checkout_git_sha'])
    timing = report_timing(output)
    result = {'schema': 'confirmation-closeout/2.0', 'completed_utc': utc_now(),
              'source_sha256': file_hash(__file__), 'protocol_hash': manifest['configuration']['protocol_hash'],
              'conditions': conditions, 'execution_timing': timing,
              'numerical_tolerance_failures_are_scoped_limits': True,
              'diff_command': 'git diff cdfbd11a5390c15c54cace9b95e7d5e40c2ec35e -- simulations tests .github/workflows .gitignore paper/main.tex'}
    json_write(audit/'final_report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulations/outputs/confirmation_v2'))
    print(json.dumps(closeout(parser.parse_args().output), indent=2))
