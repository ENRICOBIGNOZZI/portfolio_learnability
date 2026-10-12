"""Replace only Figure 6 using saved annual fixed-Gaussian ridge exports.

Reproduce without fitting portfolios: python3 -m empirical_final.figure6_annual
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_pipeline import digest
from portfolio import sharpe
from empirical_final.manuscript_cleanup import compile_project, dependency_audit, write_json
from empirical_final.section_vi_finalize import BLOCKS, BASE, ROOT

BASELINE = ROOT / 'outputs/manuscript_final_clean_20261012'
DELIVERY = ROOT / 'outputs/manuscript_annual_figure6_20261012'
FIGURE = 'fig06_nine_complete_profiles'
START = 'To assess whether the final-year Sharpe--complexity profile is representative'
STOP = r'\begin{figure}'
INTRODUCTION = r'''To assess the historical Sharpe--complexity trade-off using year-specific
effective complexity rather than block-average complexity,
Figure~\ref{fig:complexity_nine} reports one representative annual profile
from each of the nine historical blocks. We select the central calendar
year in each block: 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, and
2021. Holding the Gaussian representation and lengthscale fixed, each
profile connects the full evaluated penalty path using that year's
effective complexity and the annualized Sharpe of its 12 subsequent
monthly OOS returns. The annual past-validation selection is therefore
an actual point on its own annual curve. These one-year profiles are
noisier than pooled-block profiles and their retrospective maxima are
descriptive. The historical table and block summaries below retain the
original pooled-block results.

'''
CAPTION = r'''\caption{Representative annual out-of-sample Sharpe--complexity profiles.
Each panel corresponds to one selected decision year, the central calendar
year of its indicated historical block: 1980, 1985, 1990, 1995, 2000,
2005, 2010, 2015, or 2021. Annual OOS Sharpe is annualized from the 12
subsequent monthly returns (February of the decision year through January
of the following year); effective complexity is measured for that same
year's fitted policy, with no averaging across years or pooling of returns.
The Gaussian representation and lengthscale are fixed. Curves connect all
actual evaluated penalties, including the extended near-zero positive
grid and the exact minimum-norm $\lambda=0$ endpoint, without smoothing
or extrapolation. Black diamonds mark annual past-validation selections
verified to lie on their annual curves; orange stars mark retrospective
maxima over the full annual evaluated grid; open circles mark exact
zero-penalty endpoints. All nine panels share common axes. These profiles
are noisier than the previous pooled-block curves because each uses only
one OOS year. Retrospective maxima do not select deployable policies or
estimate population-optimal complexity.}
'''


def selected_years(available):
    """Use exact central years; break a missing-year distance tie chronologically."""
    available = set(available)
    result = []
    for lo, hi in BLOCKS:
        central = (lo + hi) // 2
        candidates = available.intersection(range(lo, hi + 1))
        if not candidates:
            raise ValueError(f'No saved annual results in {lo}-{hi}')
        year = min(candidates, key=lambda y: (abs(y - central), y))
        result.append((lo, hi, central, year))
    return result


def annual_profiles(baseline=BASELINE):
    """Independently reconstruct all annual Sharpes and past-only CV coordinates."""
    section = Path(baseline) / 'section_vi'
    inputs = {}

    def load(path):
        inputs[str(path.relative_to(ROOT))] = digest(path)
        return pd.read_csv(path, float_precision='round_trip')

    annual = load(section / 'tables/ridge_annual.csv')
    monthly = load(section / 'tables/ridge_monthly_paths.csv')
    accounts = load(section / 'tables/monthly_accounting.csv')
    original_monthly = load(BASE / 'tables/kernel_selected_monthly.csv').query("policy == 'gaussian'")
    original = load(BASE / 'tables/kernel_annual_paths.csv').query("policy == 'gaussian'")
    protocol_path = section / 'audit/ridge_protocol.json'
    inputs[str(protocol_path.relative_to(ROOT))] = digest(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    grid = np.asarray(protocol['penalties'])
    assert grid[-1] == 0 and (grid[:-1] > 0).all()
    columns = [f'lambda_{j:04}' for j in range(len(grid))]
    rows, summaries = [], []
    for lo, hi, central, year in selected_years(annual.decision_year.unique()):
        f = annual[annual.decision_year.eq(year)].sort_values('candidate').copy()
        m = monthly[monthly.decision_year.eq(year)].sort_values('return_date')
        assert len(f) == len(grid) and len(m) == 12
        np.testing.assert_array_equal(f.candidate, np.arange(len(grid)))
        np.testing.assert_allclose(f.penalty, grid, rtol=1e-13, atol=0)
        dates = pd.date_range(f'{year}-02-28', periods=12, freq='ME')
        assert pd.DatetimeIndex(m.return_date).equals(dates)
        returns = m[columns].to_numpy()
        sr = sharpe(returns)
        assert np.isfinite(returns).all() and np.isfinite(sr).all()
        np.testing.assert_allclose(f.oos_sharpe, sr, rtol=2e-11, atol=2e-12)
        assert np.isfinite(f.C).all() and (f.C > 0).all()
        assert f['rank'].nunique() == 1 and (f.C <= f['rank']).all()
        assert (np.diff(f.C) >= -1e-10).all()
        # Compare the exported selected flag against original past validation losses.
        old = original[original.decision_year.eq(year)]
        choice = old[old.selected]
        assert len(choice) == 1
        choice = choice.iloc[0]
        assert choice.candidate == old.loc[old.validation_Q.idxmin(), 'candidate']
        assert pd.Timestamp(choice.last_known_payoff) < pd.Timestamp(choice.first_test_payoff)
        selected = f[f.selected]
        assert len(selected) == 1
        selected = selected.iloc[0]
        np.testing.assert_allclose(selected.penalty, choice.penalty, rtol=1e-12)
        np.testing.assert_allclose(selected.C, choice.C, rtol=2e-6, atol=2e-7)
        deployed = accounts.query("policy == 'gaussian' and scenario == 'gross'")
        deployed = deployed[deployed.decision_year.eq(year)].sort_values('return_date')
        assert len(deployed) == 12
        assert pd.DatetimeIndex(deployed.return_date).equals(dates)
        selected_returns = returns[:, int(selected.candidate)]
        original_returns = original_monthly[original_monthly.decision_year.eq(year)].sort_values('return_date')
        assert len(original_returns) == 12
        assert pd.DatetimeIndex(original_returns.return_date).equals(dates)
        np.testing.assert_allclose(original_returns.selected_penalty, selected.penalty, rtol=1e-12)
        np.testing.assert_allclose(original_returns.C, selected.C, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(selected_returns, original_returns.raw_return, rtol=2e-7, atol=2e-9)
        selected_sr = float(sharpe(deployed.excess_return))
        np.testing.assert_allclose(sr[int(selected.candidate)], selected_sr, rtol=2e-7, atol=2e-9)
        zero = f[f.zero_penalty]
        assert len(zero) == 1
        zero = zero.iloc[0]
        assert zero.penalty == 0 and zero.C == zero['rank']
        peak = int(np.argmax(sr))
        # Use independently reconstructed monthly Sharpe, not rounded stored ratios.
        f['oos_sharpe'] = sr
        f['period'] = f'{lo}-{hi}'
        f['retrospective_maximum'] = f.candidate.eq(peak)
        rows.append(f)
        summaries.append(dict(period=f'{lo}-{hi}', central_year=central, decision_year=year,
            substituted=year != central, months=12, first_return_date=str(dates[0].date()),
            last_return_date=str(dates[-1].date()), candidates=len(grid),
            selected_candidate=int(selected.candidate), selected_penalty=float(selected.penalty),
            selected_C=float(selected.C), selected_sharpe=float(sr[int(selected.candidate)]),
            deployed_selected_sharpe=selected_sr, selected_on_annual_curve=True,
            maximum_candidate=peak, maximum_C=float(f.C.iloc[peak]), maximum_sharpe=float(sr[peak]),
            zero_candidate=int(zero.candidate), zero_C=float(zero.C), zero_sharpe=float(sr[int(zero.candidate)]),
            stored_sharpe_maximum_error=float(np.max(np.abs(annual.loc[f.index, 'oos_sharpe'] - sr)))))
    audit = dict(passed=True, panels=summaries, input_sha256=inputs,
        representation=protocol['representation'], candidates_per_year=len(grid),
        total_plotted_candidates=sum(s['candidates'] for s in summaries),
        sharpe_formula='sqrt(12) * mean(12 monthly OOS returns) / sample_std(ddof=1)',
        complexity_averaged_across_years=False, returns_pooled_across_years=False,
        backtest_rerun=False, substituted_years=[s for s in summaries if s['substituted']])
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries), audit


def plot_profiles(curves, summary):
    """One full annual path per panel, with all observed vertices retained."""
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 13,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'grid.alpha': .22, 'path.simplify': False}):
        fig, axs = plt.subplots(3, 3, figsize=(10.5, 8.7), sharex=True, sharey=True,
                                layout='constrained')
        for ax, s in zip(axs.flat, summary.itertuples()):
            f = curves[curves.decision_year.eq(s.decision_year)].sort_values('candidate')
            line, = ax.plot(f.C, f.oos_sharpe, color='#236d9f', lw=1.3)
            line.get_path().should_simplify = False
            np.testing.assert_array_equal(line.get_xydata(), f[['C', 'oos_sharpe']].to_numpy())
            ax.scatter(s.selected_C, s.selected_sharpe, color='black', marker='D', s=35, zorder=7)
            ax.scatter(s.maximum_C, s.maximum_sharpe, color='#e58a24', marker='*', s=110, zorder=6)
            ax.scatter(s.zero_C, s.zero_sharpe, facecolors='white', edgecolors='#236d9f',
                       s=36, zorder=8)
            ax.set_title(f'{s.decision_year}  ({s.period.replace("-", chr(8211))})', fontsize=14)
            ax.set_xticks([0, 200, 400, 600])
            ax.tick_params(labelsize=13)
            ax.grid(True)
        for ax in axs[2]:
            ax.set_xlabel('Annual effective complexity', fontsize=12)
        for ax in axs[:, 0]:
            ax.set_ylabel('Annual OOS Sharpe', fontsize=13)
        xmax = float(curves.C.max()) * 1.025
        ymin, ymax = float(curves.oos_sharpe.min()), float(curves.oos_sharpe.max())
        margin = .055 * (ymax - ymin)
        limits = dict(common_xlim=[0, xmax], common_ylim=[ymin - margin, ymax + margin])
        axs[0, 0].set_xlim(limits['common_xlim'])
        axs[0, 0].set_ylim(limits['common_ylim'])
        handles = [
            plt.Line2D([0], [0], color='#236d9f', lw=1.3, label='Evaluated annual λ path'),
            plt.Line2D([0], [0], color='black', marker='D', lw=0, ms=5, label='Past-validation selection'),
            plt.Line2D([0], [0], color='#e58a24', marker='*', lw=0, ms=11, label='Retrospective maximum'),
            plt.Line2D([0], [0], color='#236d9f', marker='o', mfc='white', lw=0, ms=5.5, label='Exact λ = 0')]
        fig.legend(handles=handles, loc='outside upper center', ncol=2, frameon=False, fontsize=12)
        assert len(fig.axes) == 9
        assert all(np.array_equal(ax.get_xlim(), limits['common_xlim']) for ax in axs.flat)
        assert all(np.array_equal(ax.get_ylim(), limits['common_ylim']) for ax in axs.flat)
    return fig, dict(passed=True, **limits, panels=9, vertices_per_panel=len(curves) // 9,
        all_evaluated_vertices_retained=True, path_simplification=False,
        smoothing=False, extrapolation=False, cv_strategy_plotted=True,
        markers_per_panel=['annual_past_validation', 'annual_retrospective_maximum', 'exact_zero_endpoint'])


def revise_text(text):
    start = text.index(START)
    stop = text.index(STOP, start)
    updated = text[:start] + INTRODUCTION + text[stop:]
    figure = updated.index(r'\includegraphics[width=\textwidth]{figures/' + FIGURE + '.pdf}')
    caption_start = updated.index(r'\caption{', figure)
    caption_end = updated.index(r'\label{fig:complexity_nine}', caption_start)
    return updated[:caption_start] + CAPTION + updated[caption_end:]


def build(baseline=BASELINE, delivery=DELIVERY):
    baseline, delivery = Path(baseline).resolve(), Path(delivery).resolve()
    assert baseline != delivery
    delivery.mkdir(parents=True, exist_ok=True)
    audit_dir = delivery / 'audit'
    audit_dir.mkdir(exist_ok=True)
    curves, summary, numerical = annual_profiles(baseline)
    write_json(audit_dir / 'annual_profiles.json', numerical)
    for name, frame in [('annual_curves', curves), ('annual_summary', summary)]:
        frame.to_csv(audit_dir / f'{name}.csv', index=False, float_format='%.17g')
    project = delivery / 'Portfolio'
    source = baseline / 'Portfolio'
    before = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source, project, dirs_exist_ok=True)
    (project / 'empirics.tex').write_text(revise_text((source / 'empirics.tex').read_text()))
    (project / 'figure6_caption.tex').write_text(CAPTION)
    (project / 'README_BUILD.txt').write_text((source / 'README_BUILD.txt').read_text().replace(
        'python3 -m empirical_final.manuscript_cleanup', 'python3 -m empirical_final.figure6_annual'))
    fig, rendering = plot_profiles(curves, summary)
    fig.savefig(project / f'figures/{FIGURE}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(audit_dir / 'figure6.png', dpi=180)
    plt.close(fig)
    write_json(audit_dir / 'figure6_render.json', rendering)
    changed = ['empirics.tex', 'figure6_caption.tex', f'figures/{FIGURE}.pdf', 'README_BUILD.txt']
    for name, sha in before.items():
        if name not in changed:
            assert digest(project / name) == sha, f'Unexpected source change: {name}'
        assert digest(source / name) == sha, f'Baseline modified: {name}'
    # Exact prefix/suffix checks enforce the requested empirical prose scope.
    old, new = (source / 'empirics.tex').read_text(), (project / 'empirics.tex').read_text()
    prefix = old[:old.index(START)]
    suffix = old[old.index(r'\label{fig:complexity_nine}'):]
    assert new.startswith(prefix) and new.endswith(suffix)
    (audit_dir / 'manuscript_source.diff').write_text(''.join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile='baseline/Portfolio/empirics.tex',
        tofile='annual/Portfolio/empirics.tex')))
    write_json(audit_dir / 'source_preservation.json', dict(passed=True,
        baseline=str(baseline.relative_to(ROOT)), changed_project_files=changed,
        protected_project_files={name: sha for name, sha in before.items() if name not in changed},
        baseline_project_sha256=before, empirical_text_outside_introduction_and_caption_unchanged=True,
        other_figures_and_tables_unchanged=True, theory_and_proofs_unchanged=True))
    dependencies = [dependency_audit(project, entry) for entry in ['main.tex', 'Empirics_only.tex']]
    write_json(audit_dir / 'dependencies.json', dependencies)
    builds = []
    for entry, filename in [('main.tex', 'Portfolio_Paper_Annual_Figure6.pdf'),
                            ('Empirics_only.tex', 'Empirics_Section_VI_Annual_Figure6.pdf')]:
        result, pdf = compile_project(project, entry, audit_dir / (Path(entry).stem + '_build.txt'))
        shutil.copyfile(pdf, delivery / filename)
        text = result.pop('text')
        (audit_dir / (Path(entry).stem + '_text.txt')).write_text(text)
        result.update(entry=entry, output=filename)
        builds.append(result)
    package = delivery / 'Portfolio_Paper_Annual_Figure6_LaTeX.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(before):
            info = zipfile.ZipInfo('Portfolio/' + name, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (project / name).read_bytes())
    extracted = ROOT / 'tmp/figure6_annual/extracted'
    if extracted.exists():
        shutil.rmtree(extracted)
    with zipfile.ZipFile(package) as z:
        z.extractall(extracted)
    independent = []
    for entry in ['main.tex', 'Empirics_only.tex']:
        dependency_audit(extracted / 'Portfolio', entry)
        result, _ = compile_project(extracted / 'Portfolio', entry,
            audit_dir / (Path(entry).stem + '_extracted_build.txt'))
        assert result.pop('text') == (audit_dir / (Path(entry).stem + '_text.txt')).read_bytes().decode()
        result.update(entry=entry, text_matches_delivered_pdf=True)
        independent.append(result)
    write_json(audit_dir / 'compilation.json', dict(passed=True, direct=builds,
        independent_extracted_zip=independent, source_zip_sha256=digest(package)))
    for entry in ['main', 'Empirics_only']:
        for suffix in ['.pdf', '.aux', '.bbl', '.blg', '.log', '.out']:
            (project / (entry + suffix)).unlink(missing_ok=True)
    print(f'Compiled annual Figure 6 manuscript, empirical section and LaTeX ZIP: {delivery}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DELIVERY)
    args = parser.parse_args()
    build(delivery=args.output_dir)


if __name__ == '__main__':
    main()
