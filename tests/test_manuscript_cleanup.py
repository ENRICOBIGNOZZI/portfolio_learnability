"""Scientific objects, actual Matplotlib artists, and final archive acceptance."""
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from data_pipeline import digest
from empirical_final import section_vi_finalize as section
from empirical_final.figure6_audit import scientific_audit
from empirical_final.manuscript_cleanup import (
    CANONICAL, DELIVERY, SOURCE_NAME, START, STOP, THEORY, OBSOLETE_ASSETS,
    TABLE_PLACEHOLDER, TABLE_REFERENCE, dependency_audit, remove_obsolete_block,
)


@pytest.fixture(scope='module')
def numerical_audit():
    return scientific_audit()


def test_original_cv_decisions_and_all_observed_paths(numerical_audit):
    a = numerical_audit
    assert a['passed'] and a['investment_results_modified'] is False
    assert (a['oos_months'], a['annual_refits'], a['candidates_per_refit']) == (564, 47, 1522)
    assert a['historical_candidate_observations'] == 13698
    assert len(a['annual_choices']) == 47 and len(a['blocks']) == 9
    for block in a['blocks']:
        assert block['candidate_observations'] == 1522
        assert block['diamond_matches_actual_cv']
        assert not block['numerical_or_indexing_inconsistency']
        assert block['distinct_selected_penalties'] > 1
        assert not block['matches_fixed_candidate']
    assert a['figure5']['on_own_path'] and a['figure5']['diamond_retained']
    assert a['figure5']['candidate'] == 465
    np.testing.assert_allclose(a['figure5']['curve_sharpe'], a['figure5']['plotted_selected_sharpe'], rtol=1e-12)


def test_figure6_artists_show_complete_observed_paths_without_cv_diamonds(monkeypatch):
    captured = {}
    monkeypatch.setattr(section, 'OUT', CANONICAL)
    monkeypatch.setattr(section, 'finish', lambda fig, name: captured.update(fig=fig, name=name))
    monkeypatch.setattr(section, 'audit', lambda name, obj: captured.update(render=obj))
    section.figure_six()
    fig = captured['fig']
    try:
        assert captured['name'] == 'fig06_nine_complete_profiles'
        assert len(fig.axes) == 9 and len(fig.legends) == 1
        legend = fig.legends[0]
        assert [t.get_text() for t in legend.get_texts()] == [
            'Evaluated λ path', 'Retrospective maximum', 'Exact λ = 0']
        assert [h.get_marker() for h in legend.legend_handles] == ['None', '*', 'o']
        curves = pd.read_csv(CANONICAL / 'tables/nine_panel_curves.csv', float_precision='round_trip')
        summaries = pd.read_csv(CANONICAL / 'tables/nine_panel_summary.csv').set_index('period')
        limits = (fig.axes[0].get_xlim(), fig.axes[0].get_ylim())
        for ax, (lo, hi) in zip(fig.axes, section.BLOCKS):
            f = curves[curves.period.eq(f'{lo}-{hi}')].sort_values(['mean_C', 'candidate'], kind='stable')
            s = summaries.loc[f'{lo}-{hi}']
            assert len(ax.lines) == 1 and len(ax.collections) == 2
            line = ax.lines[0]
            assert len(line.get_xydata()) == 1522
            np.testing.assert_array_equal(line.get_xydata(), f[['mean_C', 'oos_sharpe']])
            assert not line.get_path().should_simplify
            assert (ax.get_xlim(), ax.get_ylim()) == limits
            assert line.get_xdata().max() == s.effective_rank
            for collection, marker, expected in [
                (ax.collections[0], '*', [s.maximum_C, s.maximum_sharpe]),
                (ax.collections[1], 'o', [s.effective_rank, s.zero_sharpe]),
            ]:
                style = MarkerStyle(marker)
                expected_path = style.get_path().transformed(style.get_transform())
                np.testing.assert_array_equal(collection.get_paths()[0].vertices, expected_path.vertices)
                np.testing.assert_allclose(collection.get_offsets(), [expected], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(ax.collections[1].get_facecolors(), [[1, 1, 1, 1]])
        assert captured['render']['cv_strategy_plotted'] is False
        assert captured['render']['path_simplification'] is False
        assert not captured['render']['smoothing'] and not captured['render']['extrapolation']
    finally:
        plt.close(fig)


def test_original_theory_and_protected_appendix_spans_are_byte_identical():
    with zipfile.ZipFile(DELIVERY / 'sources' / SOURCE_NAME) as original:
        with zipfile.ZipFile(DELIVERY / 'Portfolio_Paper_Final_Clean.zip') as final:
            for name in ['main.tex', *THEORY, 'numbers.tex', 'Empirics_only.tex']:
                expected = original.read('Portfolio/' + name)
                if name == 'direct_port_learning.tex':
                    expected = expected.replace(TABLE_PLACEHOLDER.encode(), TABLE_REFERENCE.encode())
                assert expected == final.read('Portfolio/' + name)
            old = original.read('Portfolio/appendice_2.tex')
            new = final.read('Portfolio/appendice_2.tex')
            assert new == old[:old.index(START.encode())] + old[old.index(STOP.encode()):]
            assert b'Population-optimal portfolio policy in a factor economy' in new
            assert b'Portfolio representations generated by different kernels' in new
            assert b'Benchmark portfolio-learning rates' in new
            for name in OBSOLETE_ASSETS:
                assert 'Portfolio/figures/' + name not in final.namelist()
            for name in ['FRONTIER.png', 'Complexity.png']:
                assert original.read('Portfolio/figures/' + name) == final.read('Portfolio/figures/' + name)


def test_archive_dependencies_figures_and_cross_references(tmp_path):
    with zipfile.ZipFile(DELIVERY / 'Portfolio_Paper_Final_Clean.zip') as z:
        z.extractall(tmp_path)
    project = tmp_path / 'Portfolio'
    full = dependency_audit(project)
    standalone = dependency_audit(project, 'Empirics_only.tex')
    assert len(full['figures']) == 7
    assert len(standalone['figures']) == 5
    assert not full['duplicate_labels'] and not full['unresolved_references']
    assert 'tables/performance.tex' in full['included_tex']
    assert 'tables/spectral_value.tex' in full['included_tex']
    assert 'tables/historical_endpoints.tex' in full['included_tex']
    assert 'appendice_2.tex' in full['included_tex']
    empirics = (project / 'empirics.tex').read_text()
    template = (section.ROOT / 'empirical_final/templates/final_section_vi.tex').read_text()
    assert empirics == template.replace(r'\begin{figure}[H]', r'\begin{figure}[!htbp]')
    assert 'strategy is not plotted as a point on these curves' in empirics
    assert 'Diamonds show the annual' not in empirics
    assert (project / 'figure6_caption.tex').read_bytes() == (DELIVERY / 'section_vi/publication/figure6_caption.tex').read_bytes()
    numbers = (project / 'numbers.tex').read_text()
    assert r'\newcommand{\MissingCount}{7,489}' in numbers
    table = (project / 'tables/performance.tex').read_text()
    for gross, net in [('3.091', '2.476'), ('3.646', '2.857'), ('3.587', '2.747')]:
        assert f'{gross} & {net}' in table
    # The scanner must detect real broken references, rather than only report success.
    (project / 'empirics.tex').write_text(empirics + '\n' + r'\ref{missing:test}')
    with pytest.raises(AssertionError, match='Unresolved references'):
        dependency_audit(project)


def test_all_empirical_tables_other_figures_and_historical_artifacts_unchanged():
    preservation = json.loads((DELIVERY / 'audit/source_preservation.json').read_text())
    for name, expected in preservation['unchanged_empirical_tables'].items():
        assert digest(CANONICAL / 'tables' / name) == digest(DELIVERY / 'section_vi/tables' / name) == expected
    for name, expected in preservation['unchanged_other_empirical_figures'].items():
        assert digest(CANONICAL / 'figures' / name) == digest(DELIVERY / 'section_vi/figures' / name) == expected
    for name, expected in preservation['canonical_published_artifacts_unchanged'].items():
        assert digest(CANONICAL / name) == expected


def test_delivered_pdfs_and_actual_extracted_archive_compiled():
    builds = json.loads((DELIVERY / 'audit/compilation.json').read_text())
    assert builds['passed'] and len(builds['independent_extracted_zip']) == 2
    assert builds['source_zip_sha256'] == digest(DELIVERY / 'Portfolio_Paper_Final_Clean.zip')
    for result in builds['direct']:
        pdf = DELIVERY / result['output']
        assert result['pdf_sha256'] == digest(pdf)
        reader = PdfReader(pdf)
        assert len(reader.pages) == result['pages']
        text = '\n'.join(p.extract_text() or '' for p in reader.pages)
        assert '7,489' in text and 'Historical out-of-sample Sharpe' in text
        assert '3.091' in text and '3.646' in text and '3.587' in text
        assert '??' not in text
    for result in builds['independent_extracted_zip']:
        assert result['passed'] and result['text_matches_delivered_pdf']
        assert not result['undefined_commands_or_references']


def test_appendix_cleanup_rejects_ambiguous_boundaries():
    with pytest.raises(AssertionError):
        remove_obsolete_block(START + START + STOP)
