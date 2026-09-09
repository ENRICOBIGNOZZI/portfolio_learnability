"""Compare the saved strategy with frozen, sample-size-dependent ridge rules.

Run: OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 lambda_scaling.py
The original results are read-only. New artifacts live in results/lambda_scaling,
or results/lambda_scaling_<characteristics> when a subset is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Kernels.kernel_function import PortfolioKernel, fit_lambda_grid, kernel_eigenvalues
from Utils.utils import build_monthly_panels, compute_sharpe_ratio
from download_JKP.read_dataset import (
    DEFAULT_DATA_DIR, available_jkp_characteristics, available_jkp_years, load_dataset,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/lambda_scaling"
KERNELS = ("linear", "gaussian", "ntk", "matern12", "matern32", "matern52")
R_VALUES = (1.05, 1.5, 2.0)
LEARNING_SIZES = np.array([36, 60, 90, 120, 180, 240, 360])
SEED = 20260909


def scaled_lambda(lambda0, n, n0, alpha):
    """lambda0 is lambda at n0 observations; c = lambda0 * n0**alpha."""
    if np.any(np.asarray(lambda0) <= 0) or np.any(np.asarray(n) <= 0) or n0 <= 0:
        raise ValueError("Lambda and sample sizes must be positive")
    if not np.isfinite(alpha):
        raise ValueError("The exponent must be finite")
    return np.asarray(lambda0) * (np.asarray(n) / n0) ** (-alpha)


def paper_alpha(b, r):
    if not b > 1 or not 1 < r <= 2:
        raise ValueError("The paper assumes b > 1 and 1 < r <= 2")
    return 1.0 / (r + 1.0 / b)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_files(characteristic_name="all"):
    # Other research runs may update unrelated characteristic subsets alongside
    # this experiment. Protect the selected and all-characteristic baselines.
    names = sorted({"all", characteristic_name})
    files = [p for name in names for k in KERNELS for p in (ROOT / "results" / k / name).rglob("*")
             if p.is_file()]
    for name in names:
        analysis = "complexity_analysis" if name == "all" else f"complexity_analysis_{name}"
        files.extend(p for p in (ROOT / "results" / analysis).rglob("*") if p.is_file())
        equity = ROOT / "results" / f"equities_{name}.png"
        if equity.exists():
            files.append(equity)
    return sorted(files)


def baseline_spec(kernel, characteristic_name="all"):
    folder = ROOT / "results" / kernel / characteristic_name
    diag = pd.read_parquet(folder / "lambda_diagnostics.parquet")
    first = diag.loc[diag.test_year == diag.test_year.min()]
    return {
        "lengthscale": float(first.lengthscale.iloc[0]),
        "grid": np.sort(first["lambda"].unique()),
        "first_test_year": int(first.test_year.iloc[0]),
        "train_start": int(first.train_start.iloc[0]),
        "train_end": int(first.train_end.iloc[0]),
        "validation_end": int(first.validation_end.iloc[0]),
        "cap": float(first.max_gross_exposure.iloc[0]),
    }


def panels_for_year(year, characteristics):
    return build_monthly_panels(load_dataset(characteristics=characteristics, years=[year]),
                                characteristics=characteristics)


def prepare_matrices(characteristics, years, specs):
    """Stream stock panels: keep only the small managed-payoff matrices in RAM."""
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": 1, "characteristics": characteristics, "years": years,
        "kernel_source": digest(ROOT / "Kernels/kernel_function.py"),
        "utils_source": digest(ROOT / "Utils/utils.py"),
        "loader_source": digest(ROOT / "download_JKP/read_dataset.py"),
        "manifest": digest(ROOT / "data/JKP_USA_clean/cleaning_manifest.json"),
        "clean_files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted((ROOT / "data/JKP_USA_clean").glob("*.parquet"))],
        "lengthscales": {k: s["lengthscale"] for k, s in specs.items()},
        "n_random_features": 1000, "random_state": 0,
    }
    signature = json.loads(json.dumps(signature))
    manifest = cache / "signature.json"
    if (manifest.exists() and json.loads(manifest.read_text()) == signature
            and all((cache / f"{k}.npz").exists() for k in KERNELS)):
        print("Reusing verified managed-payoff cache", flush=True)
        return
    kernels = {k: PortfolioKernel(kernel=k, ell=specs[k]["lengthscale"],
                                  n_random_features=1000) for k in KERNELS}
    rows = {k: [] for k in KERNELS}
    dates = []
    for year in years:
        for month in panels_for_year(year, characteristics):
            dates.append(month["formation_date"].to_datetime64())
            for k, kernel in kernels.items():
                features = kernel.features(month["x"])
                rows[k].append((features.T @ month["r"]) / month["n_assets"])
        if year % 5 == 0 or year == years[-1]:
            print(f"Managed payoffs: through {year}", flush=True)
    for k in KERNELS:
        np.savez_compressed(cache / f"{k}.npz", matrix=np.vstack(rows[k]), dates=np.array(dates))
    manifest.write_text(json.dumps(signature, indent=2))


def spectral_fit(matrix):
    """A descriptive finite-spectrum fit, excluding the noisy bottom tail."""
    eigen = kernel_eigenvalues(matrix)
    eigen = eigen[eigen > eigen.max() * 1e-10]
    ranks = np.arange(1, len(eigen) + 1)
    estimates = []
    for lo, fraction in [(2, .35), (5, .60), (10, .80)]:
        hi = min(len(eigen), max(lo + 3, int(len(eigen) * fraction)))
        take = (ranks >= lo) & (ranks <= hi)
        if take.sum() < 3:
            continue
        slope, intercept = np.polyfit(np.log(ranks[take]), np.log(eigen[take]), 1)
        residual = np.log(eigen[take]) - (intercept + slope * np.log(ranks[take]))
        r2 = 1 - np.sum(residual**2) / np.sum((np.log(eigen[take]) - np.log(eigen[take]).mean())**2)
        estimates.append({"rank_start": lo, "rank_end": hi, "b": -slope,
                          "intercept": intercept, "r_squared": r2})
    return pd.DataFrame({"rank": ranks, "eigenvalue": eigen}), pd.DataFrame(
        estimates, columns=["rank_start", "rank_end", "b", "intercept", "r_squared"])


def empirical_lambda_rate(train, validation, grid, bootstraps=500):
    """Fit log lambda*(n) on log n, all evaluated on the same initial validation.

    Nested trailing training samples share an end date. Annual blocks are
    resampled jointly across sample sizes and lambdas. No test returns are used.
    """
    sizes = np.array([36, 48, 60, 84, 96, len(train)])
    dense_grid = np.geomspace(grid.min() / 100, grid.max() * 100, 161)
    losses = []
    for n in sizes:
        betas, _ = fit_lambda_grid(train[-n:], dense_grid)
        prediction = validation @ np.column_stack(list(betas.values()))
        losses.append((1.0 - prediction)**2)
    losses = np.asarray(losses)
    means = losses.mean(axis=1)
    best = means.argmin(axis=1)
    x = np.log(sizes / len(train))
    centered = x - x.mean()

    def slope(indices):
        return -np.sum(np.log(dense_grid[indices]) * centered, axis=-1) / np.sum(centered**2)

    alpha = float(slope(best))
    rng = np.random.default_rng(SEED)
    blocks = np.arange(len(validation)).reshape(-1, 12)
    draws = []
    for _ in range(bootstraps):
        take = blocks[rng.integers(0, len(blocks), len(blocks))].ravel()
        indices = losses[:, take].mean(axis=1).argmin(axis=1)
        draws.append({"alpha": float(slope(indices)),
                      "boundary_share": float(np.mean((indices == 0) | (indices == len(dense_grid)-1)))})
    path = pd.DataFrame({"n": sizes, "lambda_optimum": dense_grid[best],
                         "validation_loss": means[np.arange(len(sizes)), best],
                         "at_boundary": (best == 0) | (best == len(dense_grid)-1)})
    profile = pd.DataFrame([
        {"n": n, "lambda": lam, "validation_loss": loss}
        for n, row in zip(sizes, means) for lam, loss in zip(dense_grid, row)
    ])
    return alpha, path, pd.DataFrame(draws), profile


def fit_kernel(k, spec, characteristic_name="all"):
    folder = OUT / k
    folder.mkdir(parents=True, exist_ok=True)
    cached = np.load(OUT / "cache" / f"{k}.npz")
    matrix, dates = cached["matrix"], pd.DatetimeIndex(cached["dates"])
    initial = (dates.year >= spec["train_start"]) & (dates.year <= spec["train_end"])
    val = (dates.year > spec["train_end"]) & (dates.year <= spec["validation_end"])
    train, validation = matrix[initial], matrix[val]
    n0 = len(train)
    spectrum, spectral = spectral_fit(train)
    spectrum.to_parquet(folder / "initial_spectrum.parquet", index=False)
    spectral.to_csv(folder / "spectral_fits.csv", index=False)
    primary = spectral.loc[spectral.rank_start == 5]
    if primary.empty:
        primary = spectral.iloc[:1]
    b_fitted = float(primary.iloc[0].b) if len(primary) else None
    # In one dimension the bias-free NTK has only two independent coordinates.
    # Its spectrum has no tail to fit; use the finite-rank convention explicitly.
    finite_rank = k == "linear" or (k == "ntk" and spec.get("input_dimension") == 1)
    b = np.inf if finite_rank or k == "gaussian" else b_fitted
    if b is None:
        raise ValueError(f"{k}: too few positive eigenvalues to estimate spectral decay")
    if not b > 1:
        raise ValueError(f"{k}: spectral estimate b={b} violates the paper; choose an explicit scenario")
    alpha_empirical, path, boot, profile = empirical_lambda_rate(train, validation, spec["grid"])
    path.to_csv(folder / "empirical_lambda_optima.csv", index=False)
    boot.to_csv(folder / "empirical_alpha_bootstrap.csv", index=False)
    profile.to_parquet(folder / "initial_validation_profiles.parquet", index=False)
    betas, _ = fit_lambda_grid(train, spec["grid"])
    losses = np.mean((1 - validation @ np.column_stack(list(betas.values())))**2, axis=0)
    selected_index = int(np.argmin(losses))
    lambda0 = float(spec["grid"][selected_index])
    rules = {f"paper_r{r:g}": paper_alpha(b, r) for r in R_VALUES}
    rules["empirical"] = alpha_empirical
    # A constant-lambda control separates freezing lambda from imposing decay.
    rules["constant"] = 0.0
    metadata = {
        "kernel": k, "n0": n0, "lambda0": lambda0,
        "lambda0_at_boundary": selected_index in (0, len(spec["grid"]) - 1),
        "b_fitted": b_fitted, "b_used": "infinity" if np.isinf(b) else b,
        "b_fit_r2": float(primary.iloc[0].r_squared) if len(primary) else None,
        "spectral_rank": len(spectrum),
        "b_convention": "finite_rank" if finite_rank else ("gaussian_limit" if k == "gaussian" else "spectral_fit"),
        "characteristics": characteristic_name,
        "input_dimension": spec.get("input_dimension"),
        "r_assumed": 1.5, "alpha_paper": rules["paper_r1.5"],
        "p_paper": 1.0 if np.isinf(b) else 1.5 * rules["paper_r1.5"],
        "alpha_empirical": alpha_empirical,
        "alpha_boot_p025": float(boot.alpha.quantile(.025)),
        "alpha_boot_p975": float(boot.alpha.quantile(.975)),
        "alpha_boundary_share": float(path.at_boundary.mean()),
        "implied_r": float(1 / alpha_empirical - 1 / b) if alpha_empirical > 0 else None,
        "lengthscale": spec["lengthscale"], "cap": spec["cap"],
        "calibration_train_end": str(dates[initial][-1].date()),
        "calibration_validation_end": str(dates[val][-1].date()),
        "rules": rules,
    }
    (folder / "parameters.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(f"{k}: b={b:.3g}, lambda0={lambda0:.4g}, alpha={rules['paper_r1.5']:.3f}, "
          f"empirical alpha={alpha_empirical:.3f}", flush=True)

    saved = pd.read_parquet(ROOT / "results" / k / characteristic_name / "portfolio_returns.parquet")
    selected_rows, grid_rows, grid_returns, learning = [], [], [], []
    beta_save = {}
    reproduction_errors = []
    years = sorted(set(dates.year[dates.year >= spec["first_test_year"]]))
    for year in years:
        before, test = dates.year < year, dates.year == year
        refit, test_matrix = matrix[before], matrix[test]
        n = len(refit)
        assert dates[before][-1] + pd.offsets.MonthEnd(1) <= dates[test][0]
        # Evaluate the full initial-constant grid for all complexity plots.
        grid_actual = scaled_lambda(spec["grid"], n, n0, rules["paper_r1.5"])
        actual = {name: float(scaled_lambda(lambda0, n, n0, a)) for name, a in rules.items()}
        old = saved.loc[saved.formation_date.dt.year == year].sort_values("formation_date")
        old_lambda = float(old["lambda"].iloc[0])
        all_lambdas = np.unique(np.r_[grid_actual, list(actual.values()), old_lambda])
        fitted, eigen = fit_lambda_grid(refit, all_lambdas)
        error = np.max(np.abs(test_matrix @ fitted[old_lambda] - old.raw_portfolio_return.to_numpy()))
        reproduction_errors.append(float(error))
        if error > 1e-7:
            raise AssertionError(f"{k}/{year}: saved raw returns differ by {error}")
        for name, lam in actual.items():
            beta_save[f"{year}_{name}"] = fitted[lam]
            raw = test_matrix @ fitted[lam]
            complexity = float(np.sum(eigen / (eigen + lam)))
            for date, value in zip(dates[test], raw):
                selected_rows.append({"kernel": k, "strategy": name, "test_year": year,
                                      "formation_date": date, "return_date": date + pd.offsets.MonthEnd(1),
                                      "n": n, "lambda0": lambda0, "lambda": lam,
                                      "alpha": rules[name], "raw_portfolio_return": value,
                                      "effective_dimension": complexity, "relative_complexity": complexity / n})
        grid_beta = np.column_stack([fitted[lam] for lam in grid_actual])
        insample, raw_test = refit @ grid_beta, test_matrix @ grid_beta
        for column, (constant, lam) in enumerate(zip(spec["grid"], grid_actual)):
            complexity = float(np.sum(eigen / (eigen + lam)))
            grid_rows.append({"kernel": k, "test_year": year, "lambda0": constant, "lambda": lam,
                              "estimation_months": n, "effective_dimension": complexity,
                              "relative_complexity": complexity / n,
                              "in_sample_sharpe": compute_sharpe_ratio(insample[:, column]),
                              "out_of_sample_sharpe_raw": compute_sharpe_ratio(raw_test[:, column]),
                              "selected": column == selected_index})
            for date, value in zip(dates[test], raw_test[:, column]):
                grid_returns.append({"lambda0": constant, "lambda": lam, "test_year": year,
                                     "return_date": date + pd.offsets.MonthEnd(1),
                                     "raw_portfolio_return": value})
        # Common OOS calendar and nested trailing train samples for learning rates.
        if n >= LEARNING_SIZES.max():
            for size in LEARNING_SIZES:
                lam = float(scaled_lambda(lambda0, size, n0, rules["paper_r1.5"]))
                size_betas, _ = fit_lambda_grid(refit[-size:], [lam])
                raw = test_matrix @ size_betas[lam]
                for date, value in zip(dates[test], raw):
                    learning.append({"kernel": k, "n": size, "test_year": year,
                                     "return_date": date + pd.offsets.MonthEnd(1),
                                     "lambda": lam, "raw_portfolio_return": value})
        if year % 10 == 0 or year == years[-1]:
            print(f"{k}: refits and learning curves through {year}", flush=True)
    pd.DataFrame(selected_rows).to_parquet(folder / "raw_selected_returns.parquet", index=False)
    pd.DataFrame(grid_rows).to_parquet(folder / "grid_diagnostics.parquet", index=False)
    pd.DataFrame(grid_returns).to_parquet(folder / "grid_returns.parquet", index=False)
    pd.DataFrame(learning).to_parquet(folder / "learning_returns.parquet", index=False)
    np.savez_compressed(folder / "test_betas.npz", **beta_save)
    (folder / "reproduction.json").write_text(json.dumps({"max_baseline_raw_error": max(reproduction_errors)}))
    return metadata


def evaluate_exposures(characteristics, years, specs):
    """Use actual newly fitted weights to apply the identical monthly gross cap."""
    models = {k: PortfolioKernel(kernel=k, ell=specs[k]["lengthscale"], n_random_features=1000)
              for k in KERNELS}
    beta_files = {k: np.load(OUT / k / "test_betas.npz") for k in KERNELS}
    rules = list(json.loads((OUT / KERNELS[0] / "parameters.json").read_text())["rules"])
    records = {k: [] for k in KERNELS}
    for year in years:
        if year < specs[KERNELS[0]]["first_test_year"]:
            continue
        beta = {k: np.column_stack([beta_files[k][f"{year}_{name}"] for name in rules]) for k in KERNELS}
        for month in panels_for_year(year, characteristics):
            for k in KERNELS:
                weights = models[k].features(month["x"]) @ beta[k] / month["n_assets"]
                gross = np.abs(weights).sum(axis=0)
                scale = np.minimum(1.0, specs[k]["cap"] / np.maximum(gross, 1e-300))
                raw = month["r"] @ weights
                for column, name in enumerate(rules):
                    records[k].append({"formation_date": month["formation_date"], "strategy": name,
                                       "portfolio_return": raw[column] * scale[column],
                                       "raw_return_check": raw[column], "raw_gross_exposure": gross[column],
                                       "gross_exposure": gross[column] * scale[column],
                                       "net_exposure": weights[:, column].sum() * scale[column],
                                       "scale_factor": scale[column]})
        if year % 5 == 0 or year == years[-1]:
            print(f"Actual weights and gross caps: through {year}", flush=True)
    for k in KERNELS:
        raw = pd.read_parquet(OUT / k / "raw_selected_returns.parquet")
        result = raw.merge(pd.DataFrame(records[k]), on=["formation_date", "strategy"], validate="one_to_one")
        assert len(result) == len(raw)
        np.testing.assert_allclose(result.raw_portfolio_return, result.raw_return_check, atol=1e-8, rtol=1e-8)
        assert result.gross_exposure.max() <= specs[k]["cap"] + 1e-10
        assert np.isfinite(result.select_dtypes("number")).all().all()
        result.drop(columns="raw_return_check").to_parquet(OUT / k / "portfolio_returns.parquet", index=False)


def profile_learning_rate(sizes, sharpes, powers=None):
    """Profile SR(n) = S_infinity - A * (n/min(n))**(-p), A >= 0.

    This is a descriptive fit to point estimates, not an observation of the
    population Sharpe gap or a test of the paper's stochastic big-O bound.
    """
    if powers is None:
        powers = np.linspace(.02, 2.0, 199)
    sizes, sharpes = np.asarray(sizes), np.asarray(sharpes)
    x = (sizes[None, :] / sizes.min()) ** (-np.asarray(powers)[:, None])
    xc = x - x.mean(axis=1, keepdims=True)
    amplitude = np.maximum(0, -(xc @ (sharpes - sharpes.mean())) / np.sum(xc**2, axis=1))
    limit = sharpes.mean() + amplitude * x.mean(axis=1)
    predicted = limit[:, None] - amplitude[:, None] * x
    sse = np.sum((predicted - sharpes)**2, axis=1)
    return pd.DataFrame({"p": powers, "sse": sse, "sharpe_limit": limit, "amplitude": amplitude})


def estimate_learning_rates(k, bootstraps=400):
    folder = OUT / k
    data = pd.read_parquet(folder / "learning_returns.parquet")
    wide = data.pivot(index="return_date", columns="n", values="raw_portfolio_return").sort_index()
    assert not wide.isna().any().any()
    values, sizes = wide.to_numpy(), wide.columns.to_numpy()
    sharpes = np.sqrt(12) * values.mean(axis=0) / values.std(axis=0, ddof=1)
    profile = profile_learning_rate(sizes, sharpes)
    best = profile.loc[profile.sse.idxmin()]
    # Whole test years, paired across every history length; model is not refit.
    year_map = data.drop_duplicates("return_date").set_index("return_date").test_year.reindex(wide.index)
    blocks = [np.flatnonzero(year_map.to_numpy() == y) for y in sorted(year_map.unique())]
    assert all(len(b) == 12 for b in blocks)
    rng, draws, sharpe_draws = np.random.default_rng(SEED), [], []
    for _ in range(bootstraps):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        sampled = values[indices]
        sr = np.sqrt(12) * sampled.mean(axis=0) / sampled.std(axis=0, ddof=1)
        fitted = profile_learning_rate(sizes, sr)
        winner = fitted.loc[fitted.sse.idxmin()]
        draws.append(winner.to_dict())
        sharpe_draws.append(sr)
    boot = pd.DataFrame(draws)
    point = pd.DataFrame({"n": sizes, "sharpe": sharpes,
                          "p025": np.quantile(sharpe_draws, .025, axis=0),
                          "p975": np.quantile(sharpe_draws, .975, axis=0),
                          "fitted_sharpe": best.sharpe_limit - best.amplitude * (sizes / sizes.min())**(-best.p)})
    p0 = json.loads((folder / "parameters.json").read_text())["p_paper"]
    theoretical = profile_learning_rate(sizes, sharpes, [p0]).iloc[0]
    point["fixed_p_paper_fit"] = theoretical.sharpe_limit - theoretical.amplitude * (sizes / sizes.min())**(-p0)
    summary = {"kernel": k, "p_hat": best.p, "p_boot_p025": boot.p.quantile(.025),
               "p_boot_p975": boot.p.quantile(.975), "sharpe_limit": best.sharpe_limit,
               "amplitude": best.amplitude, "p_paper": p0,
               "boundary_share": float(((boot.p == profile.p.min()) | (boot.p == profile.p.max())).mean()),
               "nonmonotone_steps": int((np.diff(sharpes) < 0).sum()),
               "first_test_year": int(data.test_year.min()), "last_test_year": int(data.test_year.max()),
               "oos_months": len(wide), "sse": best.sse,
               "sse_fixed_p_paper": theoretical.sse}
    point.to_csv(folder / "learning_curve.csv", index=False)
    profile.to_csv(folder / "learning_rate_profile.csv", index=False)
    boot.to_csv(folder / "learning_rate_bootstrap.csv", index=False)
    (folder / "learning_rate.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--html-only", action="store_true", help="Update report text using existing figures")
    parser.add_argument("--characteristics", nargs="+", default=["all"])
    args = parser.parse_args()
    characteristic_name = "_".join(args.characteristics)
    OUT = ROOT / "results" / ("lambda_scaling" if characteristic_name == "all" else f"lambda_scaling_{characteristic_name}")
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(ROOT)): digest(p) for p in original_files(characteristic_name)}
    (OUT / "baseline_before.json").write_text(json.dumps(before, indent=2))
    if not (args.report_only or args.html_only):
        characteristics = (available_jkp_characteristics(DEFAULT_DATA_DIR)
                           if characteristic_name == "all" else args.characteristics)
        years = available_jkp_years(DEFAULT_DATA_DIR)
        specs = {k: baseline_spec(k, characteristic_name) for k in KERNELS}
        for spec in specs.values():
            spec["input_dimension"] = len(characteristics)
        prepare_matrices(characteristics, years, specs)
        parameters = [fit_kernel(k, specs[k], characteristic_name) for k in KERNELS]
        evaluate_exposures(characteristics, years, specs)
        estimates = [estimate_learning_rates(k) for k in KERNELS]
        pd.DataFrame(parameters).drop(columns="rules").to_csv(OUT / "parameters.csv", index=False)
        pd.DataFrame(estimates).to_csv(OUT / "learning_rates.csv", index=False)
    from lambda_scaling_report import build_report
    build_report(output=OUT, characteristic_name=characteristic_name,
                 characteristics=args.characteristics, regenerate_figures=not args.html_only)
    after = {str(p.relative_to(ROOT)): digest(p) for p in original_files(characteristic_name)}
    assert before == after, "An original result was changed"
    (OUT / "baseline_integrity.json").write_text(json.dumps({"unchanged": True, "sha256": after}, indent=2))
    print(f"Complete: {OUT / 'report.html'}; {len(after)} original files unchanged", flush=True)


if __name__ == "__main__":
    main()
