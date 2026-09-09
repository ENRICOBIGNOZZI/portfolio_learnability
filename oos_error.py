"""Observed OOS quadratic error versus training months, on a common test set."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, NullLocator, StrMethodFormatter

from plot_style import KERNEL_COLORS, save_figure, use_plot_style


ROOT = Path(__file__).resolve().parent
LABELS = {"linear": "Linear", "gaussian": "Gaussian", "ntk": "NTK",
          "matern12": "Matérn 1/2", "matern32": "Matérn 3/2", "matern52": "Matérn 5/2"}


def quadratic_oos_loss(payoffs):
    """Unpenalized out-of-sample loss with the same target (one) as training."""
    values = np.asarray(payoffs, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OOS payoffs must be finite")
    return (1.0 - values)**2


def empirical_power(n, errors):
    """Fit observed error = A * (n / min(n))**(-q), without subtracting a floor.

    Accepts a vector or rows of bootstrap curves. This is a finite-range slope
    of TOTAL loss, not the theoretical exponent of excess population risk.
    """
    n, errors = np.asarray(n, dtype=float), np.asarray(errors, dtype=float)
    if n.ndim != 1 or len(n) < 3 or np.any(n <= 0) or len(np.unique(n)) != len(n):
        raise ValueError("Need at least three distinct positive training sizes")
    if errors.shape[-1] != len(n) or np.any(errors <= 0) or not np.isfinite(errors).all():
        raise ValueError("Need one finite positive error per training size")
    x = np.log(n / n.min())
    xc = x - x.mean()
    y = np.log(errors)
    q = -np.sum(y * xc, axis=-1) / np.sum(xc**2)
    log_a = y.mean(axis=-1) + q * x.mean()
    return q, np.exp(log_a)


def analyze_kernel(source, kernel, bootstraps=1000):
    data = pd.read_parquet(source / kernel / "learning_returns.parquet")
    if data.duplicated(["return_date", "n"]).any():
        raise ValueError("Duplicate OOS observations")
    wide = data.pivot(index="return_date", columns="n", values="raw_portfolio_return").sort_index()
    if wide.isna().any().any():
        raise ValueError("Every train size must have exactly the same OOS dates")
    losses = quadratic_oos_loss(wide.to_numpy())
    sizes = wide.columns.to_numpy()
    observed = losses.mean(axis=0)
    year_for_date = data.drop_duplicates("return_date").set_index("return_date").test_year.reindex(wide.index)
    blocks = [np.flatnonzero(year_for_date.to_numpy() == y) for y in sorted(year_for_date.unique())]
    if any(len(block) != 12 for block in blocks):
        raise ValueError("Expected complete annual test blocks")
    rng = np.random.default_rng(20260909)
    draws = np.array([losses[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean(axis=0)
                      for _ in range(bootstraps)])
    q, a = empirical_power(sizes, observed)
    q_boot, _ = empirical_power(sizes, draws)
    fitted = a * (sizes / sizes.min())**(-q)
    table = pd.DataFrame({"kernel": kernel, "train_months": sizes, "oos_months": len(wide),
                          "oos_mse": observed, "mse_boot_p025": np.quantile(draws, .025, axis=0),
                          "mse_boot_p975": np.quantile(draws, .975, axis=0), "power_fit_mse": fitted})
    summary = {"kernel": kernel, "q_total_mse": float(q), "a_at_smallest_n": float(a),
               "q_boot_p025": float(np.quantile(q_boot, .025)), "q_boot_p975": float(np.quantile(q_boot, .975)),
               "mse_smallest_n": observed[0], "mse_largest_n": observed[-1],
               "nondecreasing_steps": int(np.sum(np.diff(observed) >= 0)),
               "oos_months": len(wide), "first_return_date": str(wide.index.min().date()),
               "last_return_date": str(wide.index.max().date()),
               "first_test_year": int(data.test_year.min()), "last_test_year": int(data.test_year.max())}
    local = pd.DataFrame({"kernel": kernel, "n_from": sizes[:-1], "n_to": sizes[1:],
                          "local_q": -np.diff(np.log(observed)) / np.diff(np.log(sizes))})
    boot = pd.DataFrame(draws, columns=[str(n) for n in sizes])
    boot["q_total_mse"] = q_boot
    return table, summary, local, boot


def plot_curve(ax, curve, meta, *, loglog=False, band=True):
    kernel = meta["kernel"]
    color = KERNEL_COLORS[kernel]
    n, mse = curve.train_months, curve.oos_mse
    if band:
        ax.fill_between(n, curve.mse_boot_p025, curve.mse_boot_p975, color=color, alpha=.16)
    ax.plot(n, mse, "o-", color=color, markersize=4, label="MSE OOS osservata")
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(n, curve.power_fit_mse, "--", color="#454545", linewidth=1.3,
                label=rf"Fit empirico: MSE $\propto T^{{{-meta['q_total_mse']:.3f}}}$")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(n, [str(value) for value in n])
    ax.set_xlabel("Osservazioni di training T (mesi)")
    ax.set_ylabel("Errore quadratico medio OOS")
    ax.set_title(LABELS[kernel])


def make_figures(folder, curves, summaries):
    minimum = min(curve.oos_mse.min() for curve in curves.values()) * .90
    maximum = max(curve.oos_mse.max() for curve in curves.values()) * 1.10
    for kernel, curve in curves.items():
        meta = summaries[kernel]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
        plot_curve(axes[0], curve, meta)
        plot_curve(axes[1], curve, meta, loglog=True)
        axes[0].set_title("Errore OOS vs numero di dati di train")
        axes[1].set_title("Gli stessi errori in scala log-log")
        axes[1].legend(fontsize=8, loc="upper right")
        fig.suptitle(LABELS[kernel] + " · Loss quadratica OOS effettivamente osservata")
        fig.text(.5, .018, f"Stessi {meta['oos_months']} mesi OOS per ogni T. Bande: bootstrap annuale appaiato (95% descrittivo). "
                 "Lambda segue la regola fissata del paper, r = 1.5.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .94))
        save_figure(fig, folder / f"{kernel}_oos_error.png")
        plt.close(fig)
    for loglog, filename in [(False, "oos_error_vs_train.png"), (True, "oos_error_loglog.png")]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.3), sharey=True)
        for ax, (kernel, curve) in zip(axes.flat, curves.items()):
            meta = summaries[kernel]
            # Show exact observed points prominently; per-kernel plots carry uncertainty bands.
            plot_curve(ax, curve, meta, loglog=loglog, band=False)
            if loglog:
                ax.legend(fontsize=7, loc="upper right")
            else:
                ax.set_xticks([36, 120, 240, 360])
            ax.set_ylim(minimum, maximum)
        fig.suptitle("Errore OOS vs numero di osservazioni di training" + (" · Scala log-log" if loglog else ""), fontsize=17)
        fig.text(.5, .015, "Stessi mesi di test e stessa definizione della loss per tutte le lunghezze. Punti osservati, senza smoothing. "
                 "Le linee tratteggiate, dove presenti, sono fit descrittivi.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .95))
        save_figure(fig, folder / filename)
        plt.close(fig)


def build_error_report(source=None, bootstraps=1000):
    source = Path(source) if source is not None else ROOT / "results/lambda_scaling"
    folder = source / "oos_error"
    folder.mkdir(parents=True, exist_ok=True)
    use_plot_style()
    curves, summaries, local_rates, source_hashes = {}, {}, [], {}
    for kernel in LABELS:
        curve, summary, local, boot = analyze_kernel(source, kernel, bootstraps)
        curves[kernel], summaries[kernel] = curve, summary
        local_rates.append(local)
        curve.to_csv(folder / f"{kernel}_oos_error.csv", index=False)
        boot.to_parquet(folder / f"{kernel}_bootstrap.parquet", index=False)
        path = source / kernel / "learning_returns.parquet"
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    all_curves = pd.concat(curves.values(), ignore_index=True)
    summary = pd.DataFrame(summaries.values())
    all_curves.to_csv(folder / "oos_error_vs_train.csv", index=False)
    summary.to_csv(folder / "empirical_error_rates.csv", index=False)
    pd.concat(local_rates, ignore_index=True).to_csv(folder / "local_error_rates.csv", index=False)
    make_figures(folder, curves, summaries)
    scores = all_curves.pivot(index="train_months", columns="kernel", values="oos_mse").reindex(columns=LABELS)
    rates = summary.set_index("kernel")[["q_total_mse", "q_boot_p025", "q_boot_p975", "nondecreasing_steps"]]
    first = next(iter(summaries.values()))
    cards = "".join(f'<h3>{html.escape(label)}</h3><a href="{kernel}_oos_error.png"><img src="{kernel}_oos_error.png" alt="{html.escape(label)}: errore OOS vs train"></a>'
                    for kernel, label in LABELS.items())
    page = f'''<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Errore OOS vs dati di train</title><style>body{{font:16px/1.65 system-ui,sans-serif;color:#253343;background:#f4f3ef;margin:0}}main{{max-width:1200px;margin:auto;padding:32px 24px 70px}}h1{{font:40px/1.15 Georgia,serif}}h2{{font:27px/1.25 Georgia,serif}}section{{padding:25px;background:white;margin:24px 0;border-radius:10px}}img{{width:100%;height:auto}}a{{color:#126f80}}.formula{{padding:18px;background:#eff5f4;font:23px Georgia,serif}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:right;border-bottom:1px solid #ddd}}.table{{overflow-x:auto}}</style><main>
<h1>Errore OOS rispetto al numero di dati di training</h1>
<p>Asse x: osservazioni mensili di training. Asse y: errore quadratico medio sui medesimi {first['oos_months']} mesi di test ({first['first_return_date']}–{first['last_return_date']}). Sei kernel; lambda scalata con la regola principale già fissata, r=1,5.</p>
<section><h2>Errore effettivamente osservato</h2><div class="formula">MSE<sub>OOS</sub>(T) = (1/M) Σ<sub>t nel test</sub> [1 − payoff<sub>T,t</sub>]<sup>2</sup></div>
<p>È la stessa loss quadratica usata dal training, valutata fuori campione. Non include la penalità ridge e usa i payoff senza cap.
Ogni punto aggrega gli stessi mesi OOS; cambia soltanto la lunghezza del training trailing (36, 60, 90, 120, 180, 240, 360 mesi) e la lambda determinata dalla regola già fissata.
Le linee uniscono i punti e mantengono le risalite osservate.</p>
<a href="oos_error_vs_train.png"><img src="oos_error_vs_train.png" alt="MSE OOS vs numero di mesi di training: sei kernel"></a>
<div class="table">{scores.to_html(float_format=lambda x: f'{x:.5f}', border=0)}</div></section>
<section><h2>Il decadimento in scala log-log</h2>
<p>Qui sono rappresentati <strong>gli stessi errori misurati</strong>, con entrambi gli assi logaritmici. La linea tratteggiata stima MSE(T) ≈ A(T/36)<sup>−q</sup> tramite regressione di log MSE su log T.
La potenza q è quindi la pendenza negativa della loss totale nel range osservato. Non è il rate precedentemente stimato dal gap di Sharpe.</p>
<a href="oos_error_loglog.png"><img src="oos_error_loglog.png" alt="Errore OOS in scala log-log e pendenza empirica"></a>
<div class="table">{rates.to_html(float_format=lambda x: f'{x:.4f}', border=0)}</div>
<p>I percentili provengono da {bootstraps} bootstrap appaiati di blocchi annuali, senza rifare il training. Ogni estrazione usa gli stessi anni per tutte le lunghezze e per tutti i kernel.
La colonna nondecreasing_steps conta gli intervalli in cui l'errore non diminuisce.</p>
<p><strong>Loss totale ed excess risk sono diversi.</strong> Il rate teorico riguarda un errore rispetto all'ottimo di popolazione, ignoto.
La MSE totale può tendere a un livello positivo. Qui non sottraggo un minimo o un asintoto stimato per ottenere artificialmente una retta: q descrive solo il decadimento finito della MSE osservata.</p></section>
<section><h2>Grafici individuali con incertezza</h2>{cards}</section>
<section><h2>Dati e riproducibilità</h2><p><a href="oos_error_vs_train.csv">Errori osservati</a> · <a href="empirical_error_rates.csv">Pendenze e bootstrap</a> · <a href="local_error_rates.csv">Pendenze fra punti consecutivi</a> · <a href="../report.html">Report completo</a>.</p>
<p><code>python3 oos_error.py --source {html.escape(str(source.relative_to(ROOT)))}</code></p>
<p>Calcoli ricavati dai rendimenti OOS dei modelli già stimati: nessuna modifica ai portafogli o alla selezione iniziale dei parametri.
Rimangono i limiti del dataset e del bootstrap descritti nel report principale.</p></section></main></html>'''
    (folder / "report.html").write_text(page, encoding="utf-8")
    (folder / "provenance.json").write_text(json.dumps({"definition": "mean((1-raw_portfolio_return)**2)",
        "bootstrap_samples": bootstraps, "seed": 20260909, "source_sha256": source_hashes}, indent=2))
    print(summary[["kernel", "q_total_mse", "q_boot_p025", "q_boot_p975"]].round(4).to_string(index=False))
    return folder / "report.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/lambda_scaling")
    args = parser.parse_args()
    print(build_error_report(args.source.resolve()))
