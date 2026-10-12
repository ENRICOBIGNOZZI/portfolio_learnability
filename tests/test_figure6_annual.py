"""Annual return reconstruction, marker geometry and narrow manuscript scope."""
import json
import zipfile

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import pytest

from data_pipeline import digest
from empirical_final.figure6_annual import (
    BASELINE, DELIVERY, FIGURE, START, annual_profiles, plot_profiles,
    revise_text, selected_years, annual_endpoint_table, revise_numbers,
)


@pytest.fixture(scope='module')
def profiles():
    return annual_profiles()


def test_complete_annual_paths_match_monthly_returns_and_original_cv(profiles):
    curves, summary, audit = profiles
    assert summary.decision_year.tolist() == [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2021]
    assert not summary.substituted.any() and summary.months.eq(12).all()
    assert len(curves) == 13698 and audit['candidates_per_year'] == 1522
    assert not audit['backtest_rerun'] and not audit['returns_pooled_across_years']
    assert not audit['complexity_averaged_across_years']
    assert summary.selected_on_annual_curve.all()
    np.testing.assert_allclose(summary.selected_sharpe, summary.deployed_selected_sharpe, rtol=2e-7)
    assert summary.stored_sharpe_maximum_error.max() < 2e-11
    for s in summary.itertuples():
        f = curves[curves.decision_year.eq(s.decision_year)]
        assert len(f) == 1522 and f.zero_penalty.sum() == f.selected.sum() == 1
        assert f.loc[f.oos_sharpe.idxmax(), 'candidate'] == s.maximum_candidate
        assert f[f.zero_penalty].C.iloc[0] == f['rank'].iloc[0]
    for name, sha in audit['input_sha256'].items():
        assert digest(BASELINE.parents[1] / name) == sha


def test_missing_year_fallback_stays_in_block_and_is_deterministic():
    available = set(range(1978, 2025)) - {1980, 2021}
    years = selected_years(available)
    assert years[0] == (1978, 1982, 1980, 1979)
    assert years[-1] == (2018, 2024, 2021, 2020)
    with pytest.raises(ValueError, match='No saved annual results in 1978-1982'):
        selected_years(range(1983, 2025))


def test_all_annual_vertices_and_three_correct_markers_are_visible(profiles):
    curves, summary, _ = profiles
    fig, audit = plot_profiles(curves, summary)
    try:
        assert len(fig.axes) == 9
        assert [h.get_marker() for h in fig.legends[0].legend_handles] == ['None', 'D', '*', 'o']
        for ax, s in zip(fig.axes, summary.itertuples()):
            f = curves[curves.decision_year.eq(s.decision_year)].sort_values('candidate')
            assert len(ax.lines) == 1 and len(ax.collections) == 3
            np.testing.assert_array_equal(ax.lines[0].get_xydata(), f[['C', 'oos_sharpe']])
            assert not ax.lines[0].get_path().should_simplify
            for collection, marker, coordinates in zip(ax.collections, ['D', '*', 'o'], [
                [s.selected_C, s.selected_sharpe], [s.maximum_C, s.maximum_sharpe],
                [s.zero_C, s.zero_sharpe],
            ]):
                style = MarkerStyle(marker)
                np.testing.assert_array_equal(collection.get_paths()[0].vertices,
                    style.get_path().transformed(style.get_transform()).vertices)
                np.testing.assert_allclose(collection.get_offsets(), [coordinates])
            np.testing.assert_array_equal(ax.collections[0].get_facecolors(), [[0, 0, 0, 1]])
            np.testing.assert_array_equal(ax.collections[2].get_facecolors(), [[1, 1, 1, 1]])
            assert str(s.decision_year) in ax.get_title()
            np.testing.assert_array_equal(ax.get_xlim(), audit['common_xlim'])
            np.testing.assert_array_equal(ax.get_ylim(), audit['common_ylim'])
            assert ax.get_xlim()[1] > f.C.max()
            assert ax.get_ylim()[0] < f.oos_sharpe.min() < f.oos_sharpe.max() < ax.get_ylim()[1]
        assert not audit['smoothing'] and not audit['extrapolation']
    finally:
        plt.close(fig)


def test_only_annual_figure_and_related_table_and_prose_change(profiles):
    source = BASELINE / 'Portfolio'
    original = (source / 'empirics.tex').read_text()
    updated = revise_text(original)
    assert updated.startswith(original[:original.index(START)])
    suffix = original[original.index(r'\label{fig:complexity_nine}'):]
    assert updated.endswith(suffix)
    assert 'these same nine selected decision years' in updated
    with zipfile.ZipFile(DELIVERY / 'Portfolio_Paper_Annual_Figure6_LaTeX.zip') as archive:
        changed = {'empirics.tex', 'figure6_caption.tex', f'figures/{FIGURE}.pdf', 'README_BUILD.txt',
                   'tables/historical_endpoints.tex', 'numbers.tex'}
        originals = {str(p.relative_to(source)): p for p in source.rglob('*') if p.is_file()}
        assert set(archive.namelist()) == {'Portfolio/' + name for name in originals}
        for name, path in originals.items():
            content = archive.read('Portfolio/' + name)
            if name not in changed:
                assert content == path.read_bytes(), name
            else:
                assert content != path.read_bytes(), name
        assert archive.read('Portfolio/empirics.tex').decode() == updated
        assert archive.read('Portfolio/tables/historical_endpoints.tex').decode() == annual_endpoint_table(profiles[1])
        assert archive.read('Portfolio/numbers.tex').decode() == revise_numbers(
            (source / 'numbers.tex').read_text(), profiles[1])
    builds = json.loads((DELIVERY / 'audit/compilation.json').read_text())
    assert builds['passed'] and len(builds['independent_extracted_zip']) == 2
    assert all(b['text_matches_delivered_pdf'] for b in builds['independent_extracted_zip'])


def test_table_ii_rows_equal_annual_figure_coordinates(profiles):
    _, summary, _ = profiles
    table = annual_endpoint_table(summary)
    rows = [line for line in table.splitlines() if line[:4].isdigit()]
    assert len(rows) == 9
    for row, s in zip(rows, summary.itertuples()):
        cells = row.removesuffix(r' \\').split(' & ')
        assert int(cells[0]) == s.decision_year
        np.testing.assert_allclose(float(cells[1]), s.maximum_C, atol=.05, rtol=0)
        np.testing.assert_allclose(float(cells[2]), s.maximum_sharpe, atol=.0005, rtol=0)
        assert float(cells[3]) == s.zero_C
        np.testing.assert_allclose(float(cells[4]), s.zero_sharpe, atol=.0005, rtol=0)
        np.testing.assert_allclose(float(cells[5]), s.maximum_sharpe-s.zero_sharpe, atol=.0005, rtol=0)
    original = (BASELINE / 'Portfolio/numbers.tex').read_text()
    updated = revise_numbers(original, summary)
    macro = r'\newcommand{\HistoricalFindings}'
    assert updated[:updated.index(macro)] == original[:original.index(macro)]
    assert 'annual complexity ranges from 20.0 to 291.5' in updated
    assert '0.252 to 8.424' in updated
