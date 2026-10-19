"""Rebuild plots and honest, prespecified rate diagnostics from saved trials."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import nnls

COLORS = ["#157A95", "#D06432", "#7756A2"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.dpi": 160, "axes.titleweight": "bold"})


def r_squared(y, prediction):
    den = np.sum((y - np.mean(y))**2)
    return float(1 - np.sum((y - prediction)**2) / den) if den > 0 else float("nan")


def rate_fit(x, values, expected, resamples=1000, x_trials=None):
    """All points, fixed exponent with one positive scale; no fitted endpoint."""
    x = np.asarray(x, float)
    y = values.mean(axis=0)
    basis = x**expected
    amplitude = max(0, float(basis @ y / (basis @ basis)))
    prediction = amplitude * basis
    positive = np.all(y > 0)
    slope = float(np.polyfit(np.log(x), np.log(y), 1)[0]) if positive else None
    slopes = []
    rng = np.random.default_rng(540007)
    if positive:
        for _ in range(resamples):
            indices = rng.integers(0, len(values), len(values))
            avg = values[indices].mean(axis=0)
            bx = x if x_trials is None else x_trials[indices].mean(axis=0)
            if np.all(avg > 0):
                slopes.append(np.polyfit(np.log(bx), np.log(avg), 1)[0])
    interval = np.quantile(slopes, [.025, .975]).tolist() if slopes else [None, None]
    floor_coefs, _ = nnls(np.column_stack([np.ones_like(basis), basis]), y)
    result = dict(expected_exponent=expected, exponent=slope, exponent_ci95=interval,
                  amplitude=amplitude, r_squared=r_squared(y, prediction),
                  rmse=float(np.sqrt(np.mean((y-prediction)**2))),
                  floor=float(floor_coefs[0]), floor_amplitude=float(floor_coefs[1]),
                  floor_r_squared=r_squared(y, floor_coefs[0] + floor_coefs[1] * basis),
                  x=x.tolist(), observed=y.tolist(), prediction=prediction.tolist(),
                  fit_points=len(x), all_points_used=True)
    result["good_rate_fit"] = bool(result["r_squared"] >= .9 and interval[0] is not None
                                   and interval[0] <= expected <= interval[1])
    return result


def matrix(frame, x, y):
    return frame.pivot(index="trial", columns=x, values=y).sort_index(axis=1)


def line(ax, frame, x, y, label, color, log=False):
    tab = matrix(frame, x, y)
    xx = tab.columns.to_numpy(float)
    yy = tab.mean(axis=0).to_numpy()
    se = tab.std(axis=0).to_numpy() / np.sqrt(len(tab))
    ax.plot(xx, yy, "o-", label=label, color=color, markersize=4, linewidth=1.8)
    if not log or np.all(yy - 1.96 * se > 0):
        ax.fill_between(xx, yy-1.96*se, yy+1.96*se, color=color, alpha=.12)
    return xx, yy


def build(out):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    from .model import calibrate
    calibration = []
    for N in config["N_values"]:
        cal = calibrate(N, config["SR_ann"], config["sigma_eps"], terms=config["fourier_terms"])
        calibration.append(dict(N=N, A_alpha=cal["A_alpha"], A_beta=cal["A_beta"],
            kappa_0=cal["kappa_0"], covariance_upper_bound=cal["upper_bound"],
            covariance_margin=cal["kappa_0"]-cal["upper_bound"], p_star=cal["p_star"],
            SR_star=config["SR_ann"], sigma_eps=cal["sigma_eps"],
            factor_means=json.dumps(cal["lambda_factor"].tolist())))
    pd.DataFrame(calibration).to_csv(out / "calibration.csv", index=False)
    data = pd.read_csv(out / "trials.csv")
    spectral = pd.read_csv(out / "complexity_lambda_trials.csv")
    chosen = data[data.rule == "pilot_frozen"]
    expected = config["trials"] * len(config["cells"]) * len(config["T_train"]) * 2
    keys = ["N", "rho", "trial", "T_train", "rule"]
    if len(data) != expected or data.duplicated(keys).any():
        raise ValueError("Missing or duplicated Monte Carlo cells")
    if set(map(tuple, data[["N", "rho"]].drop_duplicates().to_numpy())) != set(map(tuple, config["cells"])):
        raise ValueError("Monte Carlo environment grid differs from frozen design")
    if not np.isfinite(data.select_dtypes("number")).all().all():
        raise ValueError("Nonfinite Monte Carlo output")
    if data.e4_relative_error.max() > 1e-10 or data.regret.min() < -1e-9:
        raise ValueError("Economic identity failed")
    dimensions = ["N", "rho", "T_train", "rule"]
    summary = data.groupby(dimensions).agg(
        SR_mean=("SR_hat", "mean"), SR_std=("SR_hat", "std"),
        gap_mean=("SR_gap", "mean"), gap_std=("SR_gap", "std"),
        complexity_mean=("complexity", "mean"), complexity_std=("complexity", "std"),
        Q_mean=("Q_hat", "mean"), regret_mean=("regret", "mean"),
        lambda_reg=("lambda_reg", "first"), trials=("trial", "count")).reset_index()
    for name in ["SR", "gap", "complexity"]:
        summary[name+"_se"] = summary[name+"_std"] / np.sqrt(summary.trials)
    summary.to_csv(out / "summary.csv", index=False)
    nfix, rhofix = config["fixed_N"], config["fixed_rho"]
    reference = chosen[(chosen.N == nfix) & (chosen.rho == rhofix)]
    T = np.array(config["T_train"], dtype=float)
    gap_values = matrix(reference, "T_train", "SR_gap").to_numpy()
    complexity_values = matrix(reference, "T_train", "complexity").to_numpy()
    fits = {
        "SR_T_N": rate_fit(T, gap_values, -2/3, config["bootstrap_resamples"]),
        "SR_T_rho": rate_fit(T, gap_values, -2/3, config["bootstrap_resamples"]),
        "SR_complexity_N": rate_fit(complexity_values.mean(axis=0), gap_values, -2,
                                     config["bootstrap_resamples"], complexity_values),
        "C_T": rate_fit(T, complexity_values, 1/3, config["bootstrap_resamples"]),
    }
    lambda_ref = spectral[(spectral.N == nfix) & (spectral.rho == rhofix)]
    low, high = reference.lambda_reg.min(), reference.lambda_reg.max()
    fixed_window = lambda_ref[(lambda_ref.lambda_reg >= low*(1-1e-10)) &
                              (lambda_ref.lambda_reg <= high*(1+1e-10))]
    table = matrix(fixed_window, "lambda_reg", "complexity")
    fits["C_lambda"] = rate_fit(table.columns.to_numpy(float), table.to_numpy(), -.5,
                                config["bootstrap_resamples"])
    for rho in config["rho_values"]:
        ref = chosen[(chosen.N == nfix) & (chosen.rho == rho)]
        fits[f"rho_{rho:g}"] = rate_fit(T, matrix(ref, "T_train", "SR_gap").to_numpy(),
                                        -2/3, config["bootstrap_resamples"])
    # Publish every N fit, including failures; never choose N from fit quality.
    for N in config["N_values"]:
        ref = chosen[(chosen.N == N) & (chosen.rho == rhofix)]
        gaps = matrix(ref, "T_train", "SR_gap").to_numpy()
        cs = matrix(ref, "T_train", "complexity").to_numpy()
        fits[f"N_{N}_SR_T"] = rate_fit(T, gaps, -2/3, config["bootstrap_resamples"])
        fits[f"N_{N}_SR_C"] = rate_fit(cs.mean(axis=0), gaps, -2,
                                       config["bootstrap_resamples"], cs)
        fits[f"N_{N}_C_T"] = rate_fit(T, cs, 1/3, config["bootstrap_resamples"])
        lower, upper = ref.lambda_reg.min(), ref.lambda_reg.max()
        f = spectral[(spectral.N == N) & (spectral.rho == rhofix) &
                     (spectral.lambda_reg >= lower*(1-1e-10)) &
                     (spectral.lambda_reg <= upper*(1+1e-10))]
        tab = matrix(f, "lambda_reg", "complexity")
        fits[f"N_{N}_C_lambda"] = rate_fit(tab.columns.to_numpy(float), tab.to_numpy(),
                                          -.5, config["bootstrap_resamples"])
    (out / "rate_fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False)+"\n")
    pd.DataFrame([dict(plot=k, **{q: v for q, v in val.items()
                               if q not in ["x", "observed", "prediction"]})
                  for k, val in fits.items()]).to_csv(out / "rate_fits.csv", index=False)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    pages = PdfPages(out / "requested_plots.pdf")
    files = []

    def save(fig, stem):
        fig.text(.5, .01, f"{config['trials']} replications per cell; bands: 95% Monte Carlo intervals. "
                 "Fits use the full prespecified window.", ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, .055, 1, .97))
        fig.savefig(figures / (stem+".pdf"))
        fig.savefig(figures / (stem+".png"))
        pages.savefig(fig)
        files.append(stem)
        plt.close(fig)

    def label_fit(ax, fit, text, x=None):
        xx = np.array(fit["x"]) if x is None else x
        ax.plot(xx, fit["amplitude"] * xx**fit["expected_exponent"], "k--", linewidth=1.5,
                label=text)
        if fit["expected_exponent"] in [-2/3, -2] and fit["floor"] > 0:
            ax.plot(xx, fit["floor"] + fit["floor_amplitude"] * xx**fit["expected_exponent"],
                    color="#777777", linestyle=":", linewidth=1.7,
                    label=rf"Floor + same rate (diagnostic), $R^2$={fit['floor_r_squared']:.3f}")
        ax.set_title(f"Fixed exponent fit: $R^2$={fit['r_squared']:.3f}")
        ax.legend(fontsize=8)

    for grouping, values, stem, subtitle, fit_key in [
        ("N", config["N_values"], "01_sharpe_T_by_N", rf"$\rho={rhofix}$", "SR_T_N"),
        ("rho", config["rho_values"], "02_sharpe_T_by_rho", rf"$N={nfix}$", "SR_T_rho")]:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        for val, color in zip(values, COLORS):
            frame = chosen[(chosen.N == val) & (chosen.rho == rhofix)] if grouping == "N" else chosen[(chosen.N == nfix) & (chosen.rho == val)]
            label = f"N = {val}" if grouping == "N" else rf"$\rho={val:g}$"
            line(axes[0], frame, "T_train", "SR_hat", label, color)
            line(axes[1], frame, "T_train", "SR_gap", label, color, log=True)
        axes[0].axhline(config["SR_ann"], color="black", linestyle=":", label=rf"$SR^\star={config['SR_ann']:g}$")
        axes[0].set(xlabel="Training periods T", ylabel=r"Annualized $SR(\widehat W)$", title="Sharpe recovery")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(T, [str(int(t)) for t in T])
        axes[0].legend(fontsize=8)
        axes[1].set(xscale="log", yscale="log", xlabel="Training periods T", ylabel=r"$SR^\star-SR(\widehat W)$")
        axes[1].set_xticks(T, [str(int(t)) for t in T])
        label_fit(axes[1], fits[fit_key], rf"$a T^{{-2/3}}$, N={nfix}, $\rho={rhofix}$")
        fig.suptitle("Learning the portfolio Sharpe ratio | " + subtitle)
        save(fig, stem)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for N, color in zip(config["N_values"], COLORS):
        frame = chosen[(chosen.N == N) & (chosen.rho == rhofix)]
        agg = frame.groupby("T_train").agg(C=("complexity", "mean"), SR=("SR_hat", "mean"),
            SR_se=("SR_hat", lambda x: x.std()/np.sqrt(len(x))), gap=("SR_gap", "mean"))
        axes[0].errorbar(agg.C, agg.SR, yerr=1.96*agg.SR_se, color=color, marker="o", label=f"N = {N}")
        axes[1].plot(agg.C, agg.gap, "o-", color=color, label=f"N = {N}")
    axes[0].axhline(3, color="black", linestyle=":")
    axes[0].set(xlabel=r"Mean effective complexity $\widehat C_T(\lambda_T)$",
                ylabel=r"Annualized $SR(\widehat W)$", title="Along the balanced regularization path")
    axes[0].legend(fontsize=8)
    axes[1].set(xscale="log", yscale="log", xlabel="Mean effective complexity C",
                ylabel=r"$SR^\star-SR(\widehat W)$")
    label_fit(axes[1], fits["SR_complexity_N"], rf"$a C^{{-2}}$, N={nfix}")
    fig.suptitle(rf"Sharpe and effective complexity | $\rho={rhofix}$; points increase in T")
    save(fig, "03_sharpe_complexity_by_N")

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for N, color in zip(config["N_values"], COLORS):
        frame = chosen[(chosen.N == N) & (chosen.rho == rhofix)]
        line(ax, frame, "T_train", "complexity", f"N = {N}", color, log=True)
    label_fit(ax, fits["C_T"], rf"$a T^{{1/3}}$, N={nfix}")
    ax.set(xscale="log", yscale="log", xlabel="Training periods T", ylabel=r"$\widehat C_T(\lambda_T)$")
    ax.set_xticks(T, [str(int(t)) for t in T])
    fig.suptitle(rf"Effective complexity and history | $\rho={rhofix}$")
    save(fig, "04_complexity_T")

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for N, color in zip(config["N_values"], COLORS):
        frame = spectral[(spectral.N == N) & (spectral.rho == rhofix)]
        line(ax, frame, "lambda_reg", "complexity", f"N = {N}", color, log=True)
    ax.axvspan(low, high, color="#999999", alpha=.12, label=f"Prespecified fit window, N={nfix}")
    label_fit(ax, fits["C_lambda"], rf"$a \lambda^{{-1/2}}$, N={nfix}")
    ax.set(xscale="log", yscale="log", xlabel=r"Regularization $\lambda$", ylabel=r"$\widehat C_T(\lambda)$")
    fig.suptitle(rf"Effective complexity and ridge penalty | T={int(T.max())}, $\rho={rhofix}$")
    save(fig, "05_complexity_lambda")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for rule, style, label in [("literal_C1", "o--", "Original constant C = 1"),
                               ("pilot_frozen", "o-", "Independent pilot, frozen C")]:
        frame = data[(data.N == nfix) & (data.rho == rhofix) & (data.rule == rule)]
        avg = frame.groupby("T_train")[["SR_hat", "complexity"]].mean()
        axes[0].plot(avg.index, avg.SR_hat, style, label=label)
        axes[1].plot(avg.index, avg.complexity, style, label=label)
    axes[0].axhline(3, color="black", linestyle=":")
    axes[0].set(xscale="log", xlabel="Training periods T", ylabel="Annualized Sharpe")
    axes[1].set(xscale="log", yscale="log", xlabel="Training periods T", ylabel="Effective complexity")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(rf"Regularization scale audit | N={nfix}, $\rho={rhofix}$")
    save(fig, "06_original_constant_audit")
    pages.close()

    checks = dict(completed_replications=len(data[["N", "rho", "trial"]].drop_duplicates()),
                  expected_replications=config["trials"] * len(config["cells"]),
                  max_e4_relative_error=float(data.e4_relative_error.max()),
                  min_regret=float(data.regret.min()),
                  max_half_sample_SR_difference=float(np.max(np.abs(data.SR_half1-data.SR_half2))),
                  mean_half_sample_SR_difference=float(np.mean(np.abs(data.SR_half1-data.SR_half2))),
                  max_oracle_SR_deviation=float(np.max(np.abs(data.SR_oracle_eval-3))),
                  all_finite=bool(np.isfinite(data.select_dtypes("number")).all().all()),
                  figures=files, all_requested_rate_fits_good=all(fits[k]["good_rate_fit"] for k in ["SR_T_N", "SR_T_rho", "SR_complexity_N", "C_T", "C_lambda"]))
    main_keys = ["SR_T_N", "SR_T_rho", "SR_complexity_N", "C_T", "C_lambda"]
    checks["all_requested_fixed_power_fits_R2_above_090"] = all(fits[k]["r_squared"] >= .9 for k in main_keys)
    checks["all_requested_finite_representation_fits_R2_above_090"] = all(
        (max(fits[k]["r_squared"], fits[k]["floor_r_squared"]) if k.startswith("SR_")
         else fits[k]["r_squared"]) >= .9 for k in main_keys)
    (out / "checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    table_html = pd.DataFrame([dict(plot=k, expected=v["expected_exponent"],
        fitted=v["exponent"], R2=v["r_squared"], floor=v["floor"],
        floor_R2=v["floor_r_squared"], good_fit=v["good_rate_fit"])
        for k, v in fits.items()]).to_html(index=False, float_format=lambda x: f"{x:.4f}")
    figures_html = "".join(f'<figure><img src="figures/{s}.png"><figcaption>{html.escape(s)}</figcaption></figure>' for s in files)
    report = f'''<!doctype html><html lang="it"><meta charset="utf-8"><title>Nuove simulazioni D=3</title>
<style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#18313b}}img{{width:100%}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}figure{{margin:30px 0}}code{{background:#eef3f4}}</style>
<h1>Nuove simulazioni: Sharpe e complexity</h1>
<p>{config['trials']} repliche indipendenti per ciascuna delle cinque celle richieste. N=300,600,1200 a rho=0.5; rho=0,0.5,0.95 a N=600.
T indica sempre il numero di osservazioni di training. D=3, s=3, b=2, M=1000.</p>
<p>Le cinque figure richieste sono le prime cinque; la sesta confronta la costante originale con quella scelta su un pilot indipendente.
Il PDF <a href="requested_plots.pdf">requested_plots.pdf</a> contiene tutte le figure.</p>
<h2>Rate: verifica quantitativa</h2>
<p>Si fitta il gap di Sharpe, non il livello. Benchmark: gap(T) ~ T^(-2/3), C(T) ~ T^(1/3), C(lambda) ~ lambda^(-1/2).
Gap(C) ~ C^(-2) riguarda esclusivamente il percorso bilanciato lambda(T), non qualsiasi variazione della penalita a T fissato.
Gli esponenti sono benchmark asintotici: la teoria non garantisce un'equivalenza con costante positiva per questa singola economia.</p>
<p>Fit principale: un solo coefficiente positivo con esponente teorico fissato. Sono riportati anche esponente libero, intervallo bootstrap in rate_fits.json,
e fit diagnostico floor + a*x^p. Tutti i cinque T entrano nel fit. La banda mostra l'incertezza tra repliche, condizionata alla banca RFF, al pilot e al campione di integrazione.</p>
{table_html}<p>good_fit richiede R2 &gt;= 0.90 e che l'intervallo bootstrap al 95% dell'esponente libero includa quello teorico.
Un esito negativo resta visibile; non sono state modificate curve, semi, finestre o dati per migliorarlo.
Questa e una classificazione descrittiva del fit finito, non un test di validita del teorema: una convergenza piu rapida puo rispettare lo stesso upper bound.</p>
<h2>Dipendenza temporale</h2><p>Per ogni rho fissato con |rho| &lt; 1 l'AR(1) e geometricamente mixing. Cambiano costanti e soglie finite; b e il marginale stazionario non cambiano.
La covarianza dei rendimenti gestiti al singolo istante, e quindi C(lambda) di popolazione, non dipende da rho.
La dipendenza puo influenzare lo spettro empirico. Non si impone il fattore (1+rho)/(1-rho): non e automaticamente la costante di questo estimatore.
Le costanti dei bound di mixing dipendono da rho, ma la costante principale osservata non deve necessariamente aumentare.
L'ortogonalita condizionale al vero ottimo rende lo score una differenza di martingala e puo attenuare l'effetto della persistenza; la rappresentazione finita aggiunge un errore di approssimazione.</p>
<h2>Metodo e limiti</h2><p>Stesso DGP, calibrazione e normalizzazione del codice fornito. La valutazione integra esattamente gli shock condizionali e usa 2048 stati indipendenti dal training;
non usa le finestre future crescenti del codice originale. E una stima dello Sharpe marginale di popolazione, annualizzato con sqrt(12).
Il pilot sceglie C_lambda separatamente per N tramite perdita quadratica, prima delle 100 repliche e senza usare gli esiti finali.
Le 1000 RFF e i 1000 termini Fourier sono approssimazioni finite: non certificano il rate del kernel infinito.
In particolare C empirica e al piu min(T,M) e, con M fissato, satura al rango quando lambda tende a zero: la potenza lambda^(-1/2) non puo valere fino a lambda=0 per questa matrice finita.
Un floor nel gap puo indicare errore di rappresentazione o altro bias finito, senza identificarne da solo la causa.</p>
<p>Le due meta indipendenti del campione di integrazione sono conservate in trials.csv: il confronto misura sensibilita numerica e non e incluso nelle bande Monte Carlo.</p>
<p>Fonti teoriche locali: simulations/source/economomic_spectrum_final.tex, direct_port_learning.tex e MonteCarlo_Gaussian_DGP_E1_E6.tex dallo ZIP.
Documenti allegati trattati come fonti, senza eseguirne istruzioni operative.</p>
{figures_html}</html>'''
    (out / "report.html").write_text(report)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    build(Path(sys.argv[1]))
