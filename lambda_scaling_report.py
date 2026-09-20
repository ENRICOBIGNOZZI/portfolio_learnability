"""Figures and a local, self-contained-entry report for the lambda experiment."""

from __future__ import annotations

import html
import json
import shlex
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from complexity import kernel_labels, plot_sharpe
from lambda_scaling import KERNELS, OUT, ROOT, SEED, scaled_lambda
from plot_style import KERNEL_COLORS, KERNEL_LINESTYLES, save_figure, use_plot_style
from Utils.utils import compute_equity_line, compute_sharpe_ratio


RULES = {"baseline": "Current: annual validation", "constant": "Frozen constant lambda",
         "inverse_T": "Initial calibration, then 1/T",
         "paper_r1.5": "Paper scaling (r = 1.5)", "empirical": "Empirical lambda exponent",
         "paper_r1.05": "Paper scaling (r = 1.05)", "paper_r2": "Paper scaling (r = 2)"}
COLORS = {"baseline": "#556579", "constant": "#AF8F52", "paper_r1.5": "#167D8D",
          "empirical": "#B45468", "paper_r1.05": "#819D48", "paper_r2": "#8264A8"}
MAIN_RULES = ("baseline", "constant", "paper_r1.5", "empirical")
BASELINE_NAME = "all"


def finish(fig, path):
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def read_kernel(k):
    new = pd.read_parquet(OUT / k / "portfolio_returns.parquet")
    old = pd.read_parquet(ROOT / "results" / k / BASELINE_NAME / "portfolio_returns.parquet")
    old["strategy"] = "baseline"
    original_diag = pd.read_parquet(ROOT / "results" / k / BASELINE_NAME / "lambda_diagnostics.parquet")
    # C and T are from the saved original refits; only visualization metadata is added.
    counts = new.drop_duplicates("test_year").set_index("test_year").n
    original_diag = original_diag.loc[original_diag.selected].set_index("test_year")
    old["test_year"] = old.formation_date.dt.year
    old["n"] = old.test_year.map(counts)
    old["effective_dimension"] = old.test_year.map(original_diag.effective_dimension)
    old["relative_complexity"] = old.effective_dimension / old.n
    return pd.concat([old, new], ignore_index=True).sort_values(["strategy", "return_date"])


def performance(data):
    records = []
    for (k, rule), group in data.groupby(["kernel", "strategy"]):
        group = group.sort_values("return_date")
        wealth = np.r_[1., compute_equity_line(group.portfolio_return)]
        drawdown = wealth / np.maximum.accumulate(wealth) - 1
        records.append({"kernel": k, "strategy": rule, "months": len(group),
                        "sharpe": compute_sharpe_ratio(group.portfolio_return),
                        "raw_sharpe": compute_sharpe_ratio(group.raw_portfolio_return),
                        "terminal_equity": wealth[-1], "max_drawdown": drawdown.min(),
                        "max_gross_exposure": group.gross_exposure.max(),
                        "mean_gross_exposure": group.gross_exposure.mean()})
    return pd.DataFrame(records)


def equity_overview(data):
    for rule, filename in [("baseline", "equities_current.png"), ("paper_r1.5", "equities_paper.png"),
                           ("empirical", "equities_empirical.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in KERNELS:
            part = data.loc[(data.kernel == k) & (data.strategy == rule)].sort_values("return_date")
            ax.plot(part.return_date, compute_equity_line(part.portfolio_return), label=kernel_labels[k],
                    color=KERNEL_COLORS[k], linestyle=KERNEL_LINESTYLES[k])
        ax.set(title=RULES[rule], xlabel="Return date", ylabel="Compounded excess-return series (log scale)", yscale="log")
        ax.legend(ncol=3, loc="upper left")
        fig.text(.98, .02, "Monthly gross exposure <= 2; same saved-data universe", ha="right", fontsize=8)
        fig.subplots_adjust(bottom=.15)
        save_figure(fig, OUT / filename)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(13, 5.2))
    for ax, rule in zip(axes, ["baseline", "paper_r1.5"]):
        for k in KERNELS:
            part = data.loc[(data.kernel == k) & (data.strategy == rule)].sort_values("return_date")
            ax.plot(part.return_date, compute_equity_line(part.portfolio_return), label=kernel_labels[k],
                    color=KERNEL_COLORS[k], linestyle=KERNEL_LINESTYLES[k])
        ax.set(title=RULES[rule], xlabel="Return date", yscale="log")
        ax.legend(ncol=2, loc="upper left")
    axes[0].set_ylabel("Compounded excess-return series (log scale)")
    finish(fig, OUT / "equities_comparison.png")


def trajectory_plots(k, data, params):
    folder = OUT / k
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for rule in MAIN_RULES:
        part = data.loc[data.strategy == rule].sort_values("return_date")
        annual = part.drop_duplicates("test_year")
        axes[0, 0].plot(part.return_date, compute_equity_line(part.portfolio_return), color=COLORS[rule], label=RULES[rule])
        for ax, field in [(axes[0, 1], "lambda"), (axes[1, 0], "effective_dimension"), (axes[1, 1], "relative_complexity")]:
            ax.plot(annual.n, annual[field], color=COLORS[rule], label=RULES[rule])
    axes[0, 0].set(title="Implemented equity", yscale="log", ylabel="Compounded excess returns", xlabel="Return date")
    axes[0, 1].set(title="Actual lambda at refit", yscale="log", xscale="log", xlabel="Refit months T", ylabel="Lambda(T)")
    axes[1, 0].set(title="Effective complexity", xlabel="Refit months T", ylabel="C(T)")
    axes[1, 1].set(title="Relative complexity", xlabel="Refit months T", ylabel="C(T) / T")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle(kernel_labels[k] + " · Same data, different lambda rules")
    fig.tight_layout(rect=(0, .08, 1, .95))
    save_figure(fig, folder / "strategy_comparison.png")
    plt.close(fig)

    # Main-text diagnostic: one initial calibration, then inverse-history shrinkage.
    # This figure deliberately focuses on the three rules in the current Gaussian
    # regularization table and uses the exact refit-specific C/T computed from
    # each year's managed-payoff spectrum.
    focus_rules = ("baseline", "constant", "inverse_T")
    focus_labels = {
        "baseline": "Annual validation",
        "constant": "Fixed initial penalty",
        "inverse_T": r"Initial calibration, then $1/T$",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for rule in focus_rules:
        part = data.loc[data.strategy == rule].sort_values("return_date")
        if part.empty:
            continue
        annual = part.drop_duplicates("test_year")
        axes[0].plot(
            annual.n, annual["lambda"],
            color=COLORS.get(rule, KERNEL_COLORS[k]),
            linewidth=1.8,
            label=focus_labels[rule],
        )
        axes[1].plot(
            annual.n, annual["relative_complexity"],
            color=COLORS.get(rule, KERNEL_COLORS[k]),
            linewidth=1.8,
            label=focus_labels[rule],
        )

    axes[0].set(
        xscale="log", yscale="log",
        xlabel="Refit months $T$",
        ylabel=r"Regularization $\lambda_T$",
        title="Shrinkage path",
    )
    axes[1].set(
        xlabel="Refit months $T$",
        ylabel=r"Relative complexity $C(\lambda_T)/T$",
        title="Learnable complexity per observation",
    )
    for axis in axes:
        axis.grid(False)
        axis.yaxis.grid(True, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.suptitle(kernel_labels[k] + " · History, shrinkage, and relative complexity", fontsize=13)
    fig.text(
        0.5, 0.015,
        "Same expanding refits and managed-payoff spectra; the inverse-history rule is calibrated once and then fixed.",
        ha="center", fontsize=8, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(fig, folder / "lambda_and_relative_complexity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for rule in focus_rules:
        part = data.loc[data.strategy == rule].sort_values("return_date")
        if part.empty:
            continue
        annual = part.drop_duplicates("test_year")
        ax.plot(
            annual.n, annual["relative_complexity"],
            color=COLORS.get(rule, KERNEL_COLORS[k]),
            linewidth=1.9,
            label=focus_labels[rule],
        )
    ax.set(
        xlabel="Refit months $T$",
        ylabel=r"Relative complexity $C(\lambda_T)/T$",
        title=kernel_labels[k] + " · Relative portfolio complexity",
    )
    ax.grid(False)
    ax.yaxis.grid(True, alpha=0.35)
    ax.legend(frameon=False, fontsize=8.5)
    fig.text(
        0.5, 0.015,
        r"$C(\lambda_T)/T$ is computed from the refit-specific managed-payoff eigenvalues; no test return enters this quantity.",
        ha="center", fontsize=8, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, folder / "relative_complexity_over_time.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for rule in ["paper_r1.05", "paper_r1.5", "paper_r2"]:
        part = data.loc[data.strategy == rule].sort_values("return_date")
        axes[0].plot(part.return_date, compute_equity_line(part.portfolio_return), label=RULES[rule], color=COLORS[rule])
        annual = part.drop_duplicates("test_year")
        axes[1].loglog(annual.n, annual["lambda"], label=RULES[rule], color=COLORS[rule])
    axes[0].set(title="Sensitivity to assumed r", yscale="log", xlabel="Return date", ylabel="Compounded excess returns")
    axes[1].set(title="Frozen initial constant, different powers", xlabel="Refit months T", ylabel="Lambda(T)")
    axes[0].legend()
    fig.suptitle(kernel_labels[k])
    finish(fig, folder / "r_sensitivity.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    spectrum = pd.read_parquet(folder / "initial_spectrum.parquet")
    fits = pd.read_csv(folder / "spectral_fits.csv")
    axes[0].loglog(spectrum["rank"], spectrum.eigenvalue / spectrum.eigenvalue.max(), color=KERNEL_COLORS[k])
    for row in fits.itertuples():
        ranks = np.arange(row.rank_start, row.rank_end + 1)
        axes[0].loglog(ranks, np.exp(row.intercept) * ranks**(-row.b) / spectrum.eigenvalue.max(), "--",
                       label=f"Ranks {row.rank_start}-{row.rank_end}: b={row.b:.2f}", linewidth=1)
    if len(fits):
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(.05, .08, f"Rank {len(spectrum)}: no spectral-tail fit", transform=axes[0].transAxes, fontsize=9)
    axes[0].set(title="Initial train spectrum (120 months)", xlabel="Eigenvalue rank", ylabel="Eigenvalue / largest eigenvalue")
    path = pd.read_csv(folder / "empirical_lambda_optima.csv")
    axes[1].loglog(path.n, path.lambda_optimum, "o-", color=COLORS["empirical"], label="Initial validation minima")
    intercept = np.mean(np.log(path.lambda_optimum) + params["alpha_empirical"] * np.log(path.n / params["n0"]))
    axes[1].loglog(path.n, np.exp(intercept) * (path.n / params["n0"])**(-params["alpha_empirical"]), "--",
                  color="#555555", label=f"Fitted alpha = {params['alpha_empirical']:.2f}")
    axes[1].set(title="Empirical lambda exponent: pre-test only", xlabel="Nested trailing train months", ylabel="Validation-optimal lambda")
    axes[1].legend(fontsize=8)
    fig.suptitle(kernel_labels[k])
    finish(fig, folder / "initial_exponents.png")

    profile = pd.read_parquet(folder / "initial_validation_profiles.parquet")
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    for n, group in profile.groupby("n"):
        # Display excess relative to each minimum to reveal flat optima.
        ax.semilogx(group["lambda"], group.validation_loss - group.validation_loss.min(), label=f"T = {n}")
    ax.set(xlabel="Lambda", ylabel="Validation MSE minus minimum for each T", ylim=(0, .1),
           title=kernel_labels[k] + " · Initial validation profiles (zoom)")
    ax.legend(ncol=3)
    finish(fig, folder / "validation_profiles.png")


def scaled_complexity_plots(k):
    folder = OUT / k
    diagnostics = pd.read_parquet(folder / "grid_diagnostics.parquet")
    returns = pd.read_parquet(folder / "grid_returns.parquet")
    sharpe = returns.groupby("lambda0").raw_portfolio_return.agg(compute_sharpe_ratio)
    summaries = {}
    for relative in [False, True]:
        field = "relative_complexity" if relative else "effective_dimension"
        tag = "relative_complexity" if relative else "complexity"
        summary = diagnostics.groupby("lambda0").agg(complexity=(field, "mean"), train_sharpe=("in_sample_sharpe", "mean"))
        summary["test_sharpe"] = sharpe
        summary = summary.reset_index()
        summaries[relative] = summary
        summary.to_csv(folder / f"{tag}_summary.csv", index=False)
        plot_sharpe(summary, k, kernel_labels[k] + " · Paper-scaled lambda",
                    "Average C/T" if relative else "Average effective complexity C",
                    "One point per initial constant; alpha fixed, r = 1.5.",
                    folder / f"sharpe_vs_{tag}.png", normalized=relative)
        last = diagnostics.loc[diagnostics.test_year == diagnostics.test_year.max()]
        one = last.rename(columns={field: "complexity", "out_of_sample_sharpe_raw": "test_sharpe",
                                  "in_sample_sharpe": "train_sharpe"})
        plot_sharpe(one, k, kernel_labels[k] + " · Paper-scaled lambda · Test 2024",
                    "C/T" if relative else "Effective complexity C",
                    "Single 12-month test window; constants fixed at initial calibration.",
                    folder / f"sharpe_vs_{tag}_2024.png", normalized=relative)
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.semilogx(summary.lambda0, summary.complexity, "o-", markersize=3, color=KERNEL_COLORS[k])
        params = json.loads((folder / "parameters.json").read_text())
        ax.axvline(params["lambda0"], color="#333333", linestyle="--", label="Initial validation choice")
        ax.set(xlabel="Initial constant lambda0 (lambda at T0 = 120)", ylabel="Average C/T" if relative else "Average C",
               title=kernel_labels[k] + " · Complexity along scaled paths")
        ax.legend()
        finish(fig, folder / f"{tag}_vs_lambda0.png")
        for pooled in [False, True]:
            table = summary if pooled else one
            plot_3d(k, table, relative, pooled, folder / f"{'pooled_oos_' if pooled else ''}{tag}_3d.png")

    wide = returns.pivot(index="return_date", columns="lambda0", values="raw_portfolio_return").sort_index()
    values = wide.to_numpy().reshape(-1, 12, wide.shape[1])
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(500):
        sample = values[rng.integers(0, len(values), len(values))].reshape(-1, wide.shape[1])
        boot.append(np.sqrt(12) * sample.mean(axis=0) / sample.std(axis=0, ddof=1))
    summary = summaries[True].sort_values("lambda0")
    summary["p025"] = np.quantile(boot, .025, axis=0)
    summary["p975"] = np.quantile(boot, .975, axis=0)
    summary.to_csv(folder / "relative_complexity_uncertainty.csv", index=False)
    summary = summary.sort_values("complexity")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.fill_between(summary.complexity, summary.p025, summary.p975, color=KERNEL_COLORS[k], alpha=.2)
    ax.plot(summary.complexity, summary.test_sharpe, color=KERNEL_COLORS[k])
    ax.set(title=kernel_labels[k] + " · Scaled-path OOS Sharpe", xlabel="Average C/T", ylabel="Pooled raw Sharpe")
    fig.text(.5, .02, "Pointwise 2.5-97.5% percentiles; 500 paired annual-block resamples, without refitting.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    save_figure(fig, folder / "relative_complexity_uncertainty.png")
    plt.close(fig)


def plot_3d(k, data, relative, pooled, path):
    import plotly.graph_objects as go
    data = data.sort_values("lambda0")
    x, y, z = np.log10(data.lambda0), data.complexity, data.test_sharpe
    title = kernel_labels[k] + (" · Pooled OOS" if pooled else " · Test 2024")
    ylabel = "Average C/T" if relative and pooled else "C/T" if relative else "Average C" if pooled else "C"
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot(x, y, z, color=KERNEL_COLORS[k])
    ax.scatter(x, y, z, c=z, cmap="viridis", s=18)
    best = int(np.argmax(z.to_numpy()))
    ax.scatter(x.iloc[best], y.iloc[best], z.iloc[best], marker="*", s=170, color="#E66164")
    ax.set(xlabel="log10(initial constant lambda0)", ylabel=ylabel, title=title)
    ax.text2D(-.10, .56, "Raw OOS Sharpe", transform=ax.transAxes,
              rotation=90, va="center", fontsize=10)
    ax.view_init(elev=28, azim=225)
    ax.set_box_aspect((1.2, 1.1, .85))
    fig.text(.5, .03, "One fixed alpha, one point per lambda0; star chosen ex post. Actual lambda changes with T.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.12, right=.87, bottom=.12, top=.9)
    save_figure(fig, path)
    plt.close(fig)
    interactive = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers", customdata=data.lambda0,
                                         marker={"color": z, "colorscale": "Viridis", "size": 4},
                                         hovertemplate="lambda0: %{customdata:.3e}<br>C: %{y:.4f}<br>Sharpe: %{z:.3f}<extra></extra>"))
    interactive.update_layout(title=title + " · Paper scaling, r=1.5",
                              scene={"xaxis_title": "log10(lambda0)", "yaxis_title": ylabel, "zaxis_title": "Raw OOS Sharpe"},
                              template="plotly_white", height=720)
    interactive.write_html(path.with_suffix(".html"), include_plotlyjs=True, config={"displaylogo": False})


def learning_plots(k):
    folder = OUT / k
    curve = pd.read_csv(folder / "learning_curve.csv")
    profile = pd.read_csv(folder / "learning_rate_profile.csv")
    meta = json.loads((folder / "learning_rate.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.fill_between(curve.n, curve.p025, curve.p975, color=KERNEL_COLORS[k], alpha=.17, label="Annual-block percentiles")
    ax.plot(curve.n, curve.sharpe, "o-", color=KERNEL_COLORS[k], label="Raw OOS Sharpe")
    ax.plot(curve.n, curve.fitted_sharpe, "--", color="#333333", label=f"Profile fit: p={meta['p_hat']:.2f}")
    ax.plot(curve.n, curve.fixed_p_paper_fit, ":", color="#BA7047", label=f"Fixed paper p={meta['p_paper']:.2f}")
    ax.set(xscale="log", xlabel="Nested trailing train months", ylabel="Pooled raw OOS Sharpe",
           title=f"Same test months: {meta['first_test_year']}-{meta['last_test_year']}")
    ax.set_xticks(curve.n, [str(n) for n in curve.n])
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(profile.p, profile.sse - profile.sse.min(), color=KERNEL_COLORS[k])
    ax.axvline(meta["p_paper"], linestyle=":", color="#BA7047", label="Paper exponent")
    ax.axvline(meta["p_hat"], linestyle="--", color="#333333", label="Profile minimum")
    ax.set(xlabel="Candidate learning exponent p", ylabel="SSE minus minimum", title="Unknown Sharpe limit profiled out")
    ax.legend()
    fig.suptitle(kernel_labels[k] + " · Empirical learning-rate diagnostic")
    fig.text(.5, .015, "Fitted curve: SR(T) = S_inf - A (T/36)^(-p), A >= 0. Descriptive, with no observed population gap.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    save_figure(fig, folder / "learning_rate.png")
    plt.close(fig)


def image_tag(path, caption):
    escaped = html.escape(str(path), quote=True)
    return f'<figure><a href="{escaped}"><img loading="lazy" src="{escaped}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'


def build_report(output=None, characteristic_name="all", characteristics=None, *, regenerate_figures=True):
    global OUT, BASELINE_NAME
    OUT = Path(output) if output is not None else ROOT / "results/lambda_scaling"
    BASELINE_NAME = characteristic_name
    use_plot_style()
    datasets = {k: read_kernel(k) for k in KERNELS}
    data = pd.concat(datasets.values(), ignore_index=True)
    scores = performance(data)
    scores.to_csv(OUT / "performance.csv", index=False)
    if regenerate_figures or not (OUT / "oos_error/report.html").exists():
        from oos_error import build_error_report
        build_error_report(OUT)
    if regenerate_figures:
        equity_overview(data)
    parameters = pd.read_csv(OUT / "parameters.csv")
    rates = pd.read_csv(OUT / "learning_rates.csv")
    gallery = []
    for k in KERNELS:
        params = json.loads((OUT / k / "parameters.json").read_text())
        if regenerate_figures:
            trajectory_plots(k, datasets[k], params)
            scaled_complexity_plots(k)
            learning_plots(k)
        graphs = sorted((OUT / k).glob("*.png"))
        cards = []
        for path in graphs:
            card = image_tag(path.relative_to(OUT), path.stem.replace("_", " "))
            if path.with_suffix(".html").exists():
                card += f'<p><a href="{path.with_suffix(".html").relative_to(OUT)}">Apri il grafico 3D interattivo</a></p>'
            cards.append(card)
        old_graphs = sorted((ROOT / "results" / k / BASELINE_NAME).glob("*.png"))
        originals = "".join(image_tag(Path("..") / k / BASELINE_NAME / p.name, "Selezione annuale · " + p.stem.replace("_", " ")) for p in old_graphs)
        gallery.append(f'<details><summary>{kernel_labels[k]} · {len(graphs)} nuovi grafici e tutti gli originali</summary>'
                       f'<div class="gallery">{"".join(cards)}</div><h3>Grafici originali</h3><div class="gallery">{originals}</div></details>')
        print(f"{k}: report figures ready", flush=True)

    score_table = scores.pivot(index="kernel", columns="strategy", values="sharpe").reindex(KERNELS)
    score_table = score_table[list(MAIN_RULES)].rename(columns=RULES)
    parameter_table = parameters.set_index("kernel").reindex(KERNELS)[
        ["lambda0", "b_used", "alpha_paper", "alpha_empirical", "alpha_boot_p025", "alpha_boot_p975", "implied_r"]]
    rate_table = rates.set_index("kernel").reindex(KERNELS)[
        ["p_paper", "p_hat", "p_boot_p025", "p_boot_p975", "boundary_share", "nonmonotone_steps"]]
    best_baseline = scores.loc[scores.strategy == "baseline"].sort_values("sharpe", ascending=False).iloc[0]
    best_scaled = scores.loc[scores.strategy == "paper_r1.5"].sort_values("sharpe", ascending=False).iloc[0]
    run_label = "Tutte le caratteristiche" if BASELINE_NAME == "all" else f"Solo {BASELINE_NAME}"
    analysis_folder = "complexity_analysis" if BASELINE_NAME == "all" else f"complexity_analysis_{BASELINE_NAME}"
    cli_options = ("" if BASELINE_NAME == "all" else
                   " --characteristics " + html.escape(shlex.join(characteristics or [BASELINE_NAME])))
    spectral_rows = []
    for k in KERNELS:
        fits = pd.read_csv(OUT / k / "spectral_fits.csv")
        ranges = ", ".join(f"{int(row.rank_start)}–{int(row.rank_end)}" for row in fits.itertuples()) or "nessun fit: rango insufficiente"
        spectral_rows.append(f"{kernel_labels[k]}: {ranges}")
    spectral_description = "; ".join(spectral_rows)
    alpha_zero_count = int(((parameters.alpha_boot_p025 <= 0) &
                            (parameters.alpha_boot_p975 >= 0)).sum())
    rate_lower_count = int(np.isclose(rates.p_boot_p025, .02).sum())
    rate_upper_count = int(np.isclose(rates.p_boot_p975, 2.).sum())
    content = f'''<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(run_label)} · Lambda che scala con i dati</title>
<style>body{{margin:0;background:#f3f1ec;color:#243343;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1160px;margin:auto;padding:36px 26px 80px}}
h1{{font:44px/1.1 Georgia,serif;max-width:900px}}h2{{font:29px/1.2 Georgia,serif;margin-top:0}}h3{{font-size:20px}}section{{background:white;padding:28px;margin:22px 0;border-radius:12px}}p{{max-width:100ch}}
.eyebrow{{color:#167D8D;text-transform:uppercase;letter-spacing:.13em;font-size:12px}}.formula{{padding:20px;background:#f2f6f5;font:23px/1.7 Georgia,serif}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #ddd}}th:first-child{{text-align:left}}img{{width:100%;height:auto}}figure{{margin:18px 0}}figcaption{{font-size:13px;color:#65717b}}a{{color:#126e7e}}summary{{font-size:20px;cursor:pointer;padding:16px 0}}.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}code{{background:#eef1f3;padding:2px 5px}}.note{{border-left:4px solid #BA7047;padding:12px 20px;background:#fff8ef}}@media(max-width:760px){{.gallery{{grid-template-columns:1fr}}h1{{font-size:34px}}section{{padding:18px}}}}</style>
<main><div class="eyebrow">Portfolio research · {html.escape(run_label)} · 9 settembre 2026</div>
<h1>Una costante iniziale.<br>Lambda segue il numero dei mesi.</h1>
<p><strong>{html.escape(run_label)}</strong>. Sei kernel, lo stesso campione e la baseline con selezione annuale delle stesse caratteristiche. Il nuovo esperimento sceglie λ<sub>0</sub> una volta e applica una potenza fissata, con diagnostiche separate per la potenza empirica e il learning rate.</p>
<p><a href="#errore-oos">Errore OOS vs train</a> · <a href="#equity">Equity</a> · <a href="#parametri">Potenze</a> · <a href="#rate">Diagnostica Sharpe</a> · <a href="#grafici">Tutti i grafici</a> · <a href="../{analysis_folder}/report.html">Report complessità</a></p>
<section id="equity"><h2>Equity attuali e lambda scalata</h2>
<p>Lo Sharpe è annualizzato e calcolato sui medesimi 564 rendimenti mensili, febbraio 1978-gennaio 2025 (finestre di formazione 1978-2024).
Il massimo fra i kernel attuali è {kernel_labels[best_baseline.kernel]} ({best_baseline.sharpe:.3f}); con la regola del paper e r=1,5 è {kernel_labels[best_scaled.kernel]} ({best_scaled.sharpe:.3f}).</p>
{image_tag('equities_comparison.png', 'Stessa scala verticale, stessi rendimenti di partenza e stesso cap lordo a 2. Sinistra: selezione annuale attuale. Destra: costante iniziale e potenza fissate.')}
<div class="table">{score_table.to_html(float_format=lambda x: f'{x:.3f}', border=0)}</div>
<p>Il controllo con λ costante permette di distinguere l'effetto di congelare la selezione da quello di far diminuire λ.
La strategia con potenza empirica è un esperimento aggiuntivo, con potenza stimata solo sul periodo iniziale.
Per tutte le nuove strategie si riottimizzano i coefficienti del portafoglio ogni anno; λ<sub>0</sub>, potenza e lengthscale rimangono fissi.</p>
<details><summary>Grafici equity separati</summary>{image_tag('equities_current.png', 'Caso attuale: rendimenti originali invariati.')}{image_tag('equities_paper.png', 'Caso con scaling del paper e r=1,5.')}{image_tag('equities_empirical.png', 'Caso con esponente empirico congelato prima del test.')}</details></section>
<section id="errore-oos"><h2>Errore OOS vs numero di dati di training</h2>
<p>Qui l'asse y è direttamente la loss quadratica OOS: <strong>MSE(T) = media [1 − payoff(T)]²</strong>.
L'asse x conta i mesi di training. Tutte le lunghezze sono valutate sugli stessi 384 mesi OOS,
con la regola di lambda già fissata. I punti sono gli errori osservati, incluse le risalite.</p>
{image_tag('oos_error/oos_error_vs_train.png', 'Errore OOS osservato rispetto alla lunghezza del training, per tutti i kernel.')}
<p><a href="oos_error/report.html"><strong>Apri tutti i grafici dell'errore, le versioni log-log e le pendenze empiriche</strong></a>.</p>
<p>La pendenza log-log della loss totale è un rate descrittivo sul range osservato.
Il rate dell'excess risk richiederebbe il rischio ottimo di popolazione, che non è osservato e non viene sottratto arbitrariamente.</p></section>
<section id="parametri"><h2>La potenza del paper e ciò che si può stimare</h2>
<div class="formula">λ<sub>T</sub> = λ<sub>0</sub>(T/T<sub>0</sub>)<sup>−a</sup>, &nbsp; a = b/(br+1), &nbsp; T<sub>0</sub> = 120<br>
c = λ<sub>0</sub>T<sub>0</sub><sup>a</sup>, &nbsp; SR* − SR(T) = O<sub>P</sub>(T<sup>−p</sup>), &nbsp; p = br/(br+1) = ra.</div>
<p>Riferimento: <em>Portfolio_Paper.pdf</em>, pp. 10-11, Assunzioni 1-2, Teorema 1 e Corollario 2; pp. 14-15 per C(T).
T conta i payoff mensili aggregati, non le osservazioni azione-mese. Il parametro b descrive lo spettro economico, r l'allineamento della policy ottimale allo spettro: <strong>b da solo non identifica a</strong>.</p>
<p>Scenario principale: <strong>r=1,5 assunto</strong>, con sensibilità a 1,05 e 2, senza scegliere r sulla performance OOS.
Per i kernel con una coda spettrale stimabile uso b dal primo train: il fit principale parte dal rango 5, se ci sono almeno tre autovalori disponibili; altrimenti uso il primo intervallo stimabile.
Intervalli effettivamente disponibili: {html.escape(spectral_description)}.
È una descrizione dello spettro finito, non una stima dimostrata consistente del parametro di popolazione.
Per lineare e gaussiano applico la convenzione b=∞: a=1/r. Per il gaussiano è una rappresentazione della sola potenza; il paper prevede fattori logaritmici nel rate di apprendimento e non una λ ottimale unica senza ulteriori condizioni.
Con una sola caratteristica anche l'NTK senza bias ha rango al massimo due: applico esplicitamente la convenzione di rango finito b=∞, senza stimare una coda da due autovalori.
Anche gli altri kernel sono qui approssimati da un numero finito di feature: il collegamento allo spettro infinito resta un'ipotesi operativa.</p>
<div class="table">{parameter_table.to_html(float_format=lambda x: f'{x:.5g}', border=0)}</div>
<p class="note"><strong>Risultato della stima di a:</strong> {alpha_zero_count} intervalli bootstrap su {len(parameters)} comprendono zero.
La stima puntuale deve essere letta insieme a questa incertezza: includere zero significa che la validation iniziale non identifica con precisione il segno della potenza.</p>
<p>λ<sub>0</sub> minimizza la stessa loss raw (1 − payoff)² del codice attuale, sui 60 mesi di validation dopo il primo train di 120 mesi.
La griglia contiene gli stessi 60 valori della run originale; la lengthscale è quella già scelta nella prima validation.
Il primo refit include train e validation: <strong>180 mesi, quindi λ è già scalata rispetto a T<sub>0</sub>=120</strong>.
In seguito si applica la formula al numero effettivo di mesi del refit, senza arrotondare alla griglia e senza riselezionare λ.</p>
<p>La potenza empirica a deriva dal fit log λ*(n) = intercetta − a log(n/120), con train annidati di 36, 48, 60, 84, 96 e 120 mesi,
tutti terminanti a dicembre 1972 e valutati sulla stessa validation 1973-1977. La griglia esplorativa ha 161 valori log-spaziati ed è estesa di un fattore 100 ai due estremi.
Gli intervalli sono percentili di 500 ricampionamenti appaiati dei cinque blocchi annuali della validation, senza rifare i modelli.
Una potenza negativa viene conservata: segnala che i minimi osservati chiedono più regolarizzazione al crescere di n e non viene forzata a rispettare il paper.
La r implicita, quando a&gt;0, è 1/a − 1/b; valori esterni a (1,2] non supportano le ipotesi del teorema.</p></section>
<section id="rate"><h2>Il tentativo di ottenere il learning rate empirico</h2>
<p>Per ogni kernel stimo modelli con 36, 60, 90, 120, 180, 240 e 360 mesi di training trailing, sempre usando la regola principale e la stessa λ<sub>0</sub> iniziale.
Tutti sono valutati sui medesimi 384 mesi OOS (finestre 1993-2024). Il confronto usa rendimenti <strong>senza cap</strong>, coerenti con il problema ridge del paper.</p>
<div class="formula">SR(T) ≈ S<sub>∞</sub> − A (T/36)<sup>−p</sup>, &nbsp; A ≥ 0.</div>
<p>Per ogni p fra 0,02 e 2 stimo l'asintoto S<sub>∞</sub> e A e salvo l'intero profilo degli errori.
400 ricampionamenti di blocchi annuali, appaiati fra le lunghezze, descrivono la stabilità di p.
I valori ai bordi non sono stime precise: possono indicare una curva troppo piatta, non monotona o incompatibile con la forma imposta.
Le colonne degli intervalli sono percentili descrittivi, non intervalli formalmente validi per il rate asintotico.</p>
<div class="table">{rate_table.to_html(float_format=lambda x: f'{x:.3f}', border=0)}</div>
<p><strong>Risultato del tentativo:</strong> le stime puntuali di p vanno da {rates.p_hat.min():.2f} a {rates.p_hat.max():.2f}.
Il limite inferiore bootstrap raggiunge il bordo 0,02 in {rate_lower_count} kernel su {len(rates)};
quello superiore raggiunge 2 in {rate_upper_count} kernel su {len(rates)}.
Quando i percentili coprono quasi tutta la griglia, il fit produce una potenza descrittiva ma non identifica un rate preciso né valida quello del paper.</p>
<p class="note"><strong>Il rate del paper non è direttamente osservato.</strong> SR* è ignoto; sostituirlo con lo Sharpe migliore del backtest produrrebbe un gap arbitrario.
Il fit sopra stima anche l'asintoto e mostra la sensibilità della potenza. Inoltre un limite O<sub>P</sub> non impone un'uguaglianza esatta a potenza.
Finestre trailing sovrapposte, cambiamenti di regime e una sola storia di mercato impediscono di leggere questo esercizio come una verifica del teorema.</p>
{''.join(image_tag(Path(k) / 'learning_rate.png', kernel_labels[k] + ' · Curva di apprendimento e profilo della potenza.') for k in KERNELS)}</section>
<section><h2>Come leggere i grafici di complexity</h2>
<p>Per ciascun valore iniziale λ<sub>0</sub> della griglia costruisco un'intera traiettoria λ<sub>T</sub> con la stessa potenza principale.
Nei grafici aggregati ogni punto rappresenta quindi una <strong>costante iniziale</strong>, non una λ mantenuta uguale nel tempo.
La complexity è C<sub>T</sub>=Σ μ<sub>j,T</sub>/(μ<sub>j,T</sub>+λ<sub>T</sub>); C/T viene calcolato prima della media sulle finestre.
Gli Sharpe OOS sono calcolati sui rendimenti mensili concatenati. Gli Sharpe IS sono medie fra finestre sovrapposte.
Le stelle ai massimi OOS sono diagnostiche ex post e non determinano la costante della strategia implementata.</p>
<p>Sono disponibili le curve in C e C/T, i grafici rispetto alla costante iniziale, le versioni della finestra 2024,
le curve 3D statiche e interattive, le bande bootstrap, le traiettorie di λ e complexity, la sensibilità a r e le diagnostiche iniziali.</p></section>
<section id="grafici"><h2>Galleria completa</h2>{''.join(gallery)}</section>
<section><h2>Riproducibilità e limiti del confronto</h2>
<p><code>OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 lambda_scaling.py{cli_options}</code><br>
Solo grafici e report: <code>python3 lambda_scaling.py{cli_options} --report-only</code>.</p>
<p>Le feature casuali usano gli stessi seed e dimensioni della run originale (1.000 estrazioni; NTK produce 2.000 coordinate).
Per ogni anno ricostruisco anche il portafoglio originale e verifico i rendimenti raw contro quelli salvati.
I nuovi pesi sono ricalcolati per le λ esatte: il cap non deriva da interpolazioni dei rendimenti o da gross exposure presi dalla baseline.
I file delle baseline con tutte le caratteristiche e di quelle selezionate, inclusi PNG e parquet, vengono verificati con SHA-256 prima e dopo l'esperimento.
I test numerici verificano inoltre la normalizzazione di λ nella ridge, la regola al refit e l'identificazione su curve sintetiche.</p>
<p>Il filtro delle caratteristiche del dataset esistente usa la disponibilità sull'intero periodo 1963-2024: la baseline e il confronto condividono questo limite di preprocessing.
La calibrazione dei nuovi parametri non usa gli anni di test, ma il dataset e il processo di ricerca pregresso non costituiscono una prova di investimento incontaminata.
Le equity capitalizzano meccanicamente rendimenti excess; senza riconciliare cash e finanziamento non sono ricchezza netta investibile.
Costi e turnover non vengono aggiunti in questo esperimento. Il bootstrap ricampiona rendimenti già stimati, preserva la dipendenza entro l'anno e non quella fra anni.</p>
<p>Dati: <a href="performance.csv">performance completa</a> · <a href="parameters.csv">parametri</a> · <a href="learning_rates.csv">rate empirici</a> · <a href="baseline_integrity.json">integrità degli originali</a>.</p></section></main></html>'''
    (OUT / "report.html").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    build_report()
