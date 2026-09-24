"""Final uncapped empirical diagnostics for portfolio learnability.

Focus:
1. lambda and effective complexity through expanding history;
2. complexity versus return, Sharpe, drawdown, and leverage;
3. annual re-optimization versus one initial calibration followed by the
   Gaussian inverse-history rule lambda_T = lambda_0 T_0 / T.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import save_figure, use_plot_style
from Utils.utils import compute_sharpe_ratio


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "learnability_diagnostics"
GAUSSIAN = ROOT / "results" / "expanding_gaussian" / "all"
SEED = 20260923
BOOTSTRAPS = 3000
RESPONSE_TARGET = 0.1
RULES = ("annual_validation", "fixed", "inverse_T")
LABELS = {
    "annual_validation": "Annual validation",
    "fixed": "Fixed initial lambda",
    "inverse_T": "Initial calibration + 1/T",
}


def annual_max_drawdown(returns):
    returns = np.asarray(returns, dtype=float)
    if np.all(1.0 + returns > 0):
        wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        return float(drawdown.min()), "compounded"
    pnl = np.r_[0.0, np.cumsum(returns)]
    drawdown = pnl - np.maximum.accumulate(pnl)
    return float(drawdown.min()), "additive_fallback"


def annual_economics(monthly, annual):
    rows = []
    for (rule, year), group in monthly.groupby(["rule", "test_year"]):
        r = group.sort_values("return_date")["raw_portfolio_return"].to_numpy(dtype=float)
        dd, dd_type = annual_max_drawdown(r)
        row = {
            "rule": rule,
            "rule_label": LABELS[rule],
            "test_year": int(year),
            "mean_return_ann": float(12.0 * r.mean()),
            "volatility_ann": float(np.sqrt(12.0) * r.std(ddof=1)),
            "sharpe": float(compute_sharpe_ratio(r)),
            "quadratic_criterion": float(np.mean((RESPONSE_TARGET - r) ** 2)),
            "max_drawdown": dd,
            "drawdown_definition": dd_type,
            "worst_month": float(r.min()),
        }
        diag = annual.loc[
            annual["rule"].eq(rule) & annual["test_year"].eq(year)
        ].iloc[0]
        row.update({
            "T": int(diag["T"]),
            "lambda": float(diag["lambda"]),
            "C": float(diag["C"]),
            "C_over_T": float(diag["C_over_T"]),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rule", "test_year"])


def rule_summary(monthly, annual):
    rows = []
    for rule in RULES:
        group = monthly.loc[monthly["rule"].eq(rule)].sort_values("return_date")
        r = group["raw_portfolio_return"].to_numpy(dtype=float)
        dd, dd_type = annual_max_drawdown(r)
        diag = annual.loc[annual["rule"].eq(rule)]
        rows.append({
            "rule": rule,
            "label": LABELS[rule],
            "months": len(r),
            "sharpe": float(compute_sharpe_ratio(r)),
            "mean_ann": float(12.0 * r.mean()),
            "vol_ann": float(np.sqrt(12.0) * r.std(ddof=1)),
            "quadratic_criterion": float(np.mean((RESPONSE_TARGET - r) ** 2)),
            "max_drawdown": dd,
            "drawdown_definition": dd_type,
            "mean_C": float(diag["C"].mean()),
            "mean_C_over_T": float(diag["C_over_T"].mean()),
            "lambda_log_std": float(np.log(diag["lambda"]).std(ddof=1)),
            "lambda_total_variation": float(np.abs(np.diff(np.log(diag["lambda"]))).sum()),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(monthly):
    keep = monthly.loc[
        monthly["rule"].isin(["annual_validation", "inverse_T"])
    ].copy()
    wide = keep.pivot(
        index="return_date",
        columns="rule",
        values="raw_portfolio_return",
    ).sort_index()
    year_map = (
        keep.drop_duplicates("return_date")
        .set_index("return_date")["test_year"]
        .reindex(wide.index)
    )
    years = sorted(year_map.unique())
    blocks = [np.flatnonzero(year_map.to_numpy() == year) for year in years]
    if not all(len(block) == 12 for block in blocks):
        raise ValueError("Expected complete 12-month test blocks")

    values = wide[["annual_validation", "inverse_T"]].to_numpy(dtype=float)

    def metrics(x):
        old = x[:, 0]
        rate = x[:, 1]
        return {
            "delta_sharpe_rate_minus_annual": (
                compute_sharpe_ratio(rate) - compute_sharpe_ratio(old)
            ),
            "delta_mean_ann_rate_minus_annual": 12.0 * (rate.mean() - old.mean()),
            "delta_vol_ann_rate_minus_annual": np.sqrt(12.0) * (
                rate.std(ddof=1) - old.std(ddof=1)
            ),
            "delta_Q_rate_minus_annual": np.mean((1.0 - rate) ** 2)
            - np.mean((1.0 - old) ** 2),
        }

    point = metrics(values)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(blocks), len(blocks))
        take = np.concatenate([blocks[i] for i in chosen])
        draws.append(metrics(values[take]))
    boot = pd.DataFrame(draws)

    summary = []
    for name, estimate in point.items():
        summary.append({
            "metric": name,
            "estimate": float(estimate),
            "p025": float(boot[name].quantile(0.025)),
            "p975": float(boot[name].quantile(0.975)),
            "bootstrap_probability_positive": float((boot[name] > 0).mean()),
        })
    return pd.DataFrame(summary), boot


def plot_paths(annual):
    use = annual.loc[annual["rule"].isin(RULES)].copy()

    specs = [
        ("lambda", "Penalty lambda", True, "lambda_over_time.png"),
        ("C", "Effective complexity C", False, "complexity_over_time.png"),
        ("C_over_T", "Relative complexity C/T", False, "relative_complexity_over_time.png"),
    ]
    for field, ylabel, logy, filename in specs:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for rule in RULES:
            part = use.loc[use["rule"].eq(rule)].sort_values("T")
            ax.plot(part["T"], part[field], label=LABELS[rule], linewidth=1.8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Expanding estimation history T (months)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} as market history expands")
        ax.grid(False)
        ax.yaxis.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, OUT / filename)
        plt.close(fig)


def plot_complexity_relationships(econ):
    rules = ("annual_validation", "inverse_T")
    fields = [
        ("mean_return_ann", "Annualized mean OOS return", "complexity_vs_return.png"),
        ("sharpe", "Annual OOS Sharpe", "complexity_vs_sharpe.png"),
        ("max_drawdown", "Annual maximum drawdown", "complexity_vs_drawdown.png"),
        ("volatility_ann", "Annualized OOS volatility", "complexity_vs_volatility.png"),
    ]
    for field, ylabel, filename in fields:
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        for rule in rules:
            part = econ.loc[econ["rule"].eq(rule)].sort_values("C")
            ax.scatter(part["C"], part[field], s=20, alpha=0.7, label=LABELS[rule])
        ax.set_xlabel("Effective complexity C")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} versus effective complexity")
        ax.grid(False)
        ax.yaxis.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, OUT / filename)
        plt.close(fig)


def plot_hyperoptimized_vs_rate(monthly, annual):
    old = monthly.loc[monthly["rule"].eq("annual_validation")].sort_values("return_date")
    rate = monthly.loc[monthly["rule"].eq("inverse_T")].sort_values("return_date")
    merged = old[["return_date", "raw_portfolio_return"]].merge(
        rate[["return_date", "raw_portfolio_return"]],
        on="return_date",
        suffixes=("_annual", "_rate"),
        validate="one_to_one",
    )
    merged["loss_annual"] = (RESPONSE_TARGET - merged["raw_portfolio_return_annual"]) ** 2
    merged["loss_rate"] = (RESPONSE_TARGET - merged["raw_portfolio_return_rate"]) ** 2
    merged["cumulative_loss_advantage_rate"] = (
        merged["loss_annual"] - merged["loss_rate"]
    ).cumsum()

    window = 60
    for suffix in ("annual", "rate"):
        r = merged[f"raw_portfolio_return_{suffix}"]
        mean = r.rolling(window).mean()
        vol = r.rolling(window).std(ddof=1)
        merged[f"rolling_sharpe_{suffix}"] = np.sqrt(12.0) * mean / vol

    a = annual.loc[
        annual["rule"].isin(["annual_validation", "inverse_T"])
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    for rule in ("annual_validation", "inverse_T"):
        part = a.loc[a["rule"].eq(rule)].sort_values("T")
        axes[0, 0].plot(part["T"], part["lambda"], label=LABELS[rule])
        axes[0, 1].plot(part["T"], part["C"], label=LABELS[rule])

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Annual retuning versus deterministic decay")
    axes[0, 0].set_xlabel("T (months)")
    axes[0, 0].set_ylabel("lambda")
    axes[0, 1].set_title("Effective complexity induced by the two rules")
    axes[0, 1].set_xlabel("T (months)")
    axes[0, 1].set_ylabel("C")

    axes[1, 0].plot(
        merged["return_date"],
        merged["rolling_sharpe_annual"],
        label=LABELS["annual_validation"],
    )
    axes[1, 0].plot(
        merged["return_date"],
        merged["rolling_sharpe_rate"],
        label=LABELS["inverse_T"],
    )
    axes[1, 0].set_title("60-month rolling OOS Sharpe")
    axes[1, 0].set_xlabel("Return date")
    axes[1, 0].set_ylabel("Sharpe")

    axes[1, 1].plot(
        merged["return_date"],
        merged["cumulative_loss_advantage_rate"],
    )
    axes[1, 1].axhline(0.0, linewidth=0.8)
    axes[1, 1].set_title("Cumulative response-target loss advantage of 1/T (c=0.1)")
    axes[1, 1].set_xlabel("Return date")
    axes[1, 1].set_ylabel("Cumulative Q_0.1(annual) - Q_0.1(1/T)")

    for ax in axes.flat:
        ax.grid(False)
        ax.yaxis.grid(True, alpha=0.35)
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    fig.tight_layout()
    save_figure(fig, OUT / "hyperoptimized_vs_rate_rule.png")
    plt.close(fig)

    merged.to_parquet(OUT / "hyperoptimized_vs_rate_monthly.parquet", index=False)


def plot_complexity_vs_gross(annual):
    portfolio = pd.read_parquet(
        ROOT / "results" / "gaussian" / "all" / "portfolio_returns.parquet"
    )
    portfolio["test_year"] = pd.to_datetime(portfolio["formation_date"]).dt.year
    gross = portfolio.groupby("test_year").agg(
        mean_gross=("gross_exposure", "mean"),
        p95_gross=("gross_exposure", lambda x: x.quantile(0.95)),
        max_gross=("gross_exposure", "max"),
    ).reset_index()
    complexity = annual.loc[
        annual["rule"].eq("annual_validation"),
        ["test_year", "C"],
    ]
    merged = complexity.merge(gross, on="test_year", validate="one_to_one")
    merged.to_csv(OUT / "complexity_vs_gross.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.scatter(merged["C"], merged["mean_gross"], s=22, label="Mean gross")
    ax.scatter(merged["C"], merged["p95_gross"], s=22, label="95th percentile gross")
    ax.set_xlabel("Effective complexity C")
    ax.set_ylabel("Gross exposure")
    ax.set_title("Uncapped leverage versus effective complexity")
    ax.grid(False)
    ax.yaxis.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, OUT / "complexity_vs_gross_exposure.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    use_plot_style()

    annual = pd.read_parquet(GAUSSIAN / "annual_diagnostics.parquet")
    monthly = pd.read_parquet(GAUSSIAN / "monthly_returns.parquet")
    annual = annual.loc[annual["rule"].isin(RULES)].copy()
    monthly = monthly.loc[monthly["rule"].isin(RULES)].copy()

    econ = annual_economics(monthly, annual)
    econ.to_csv(OUT / "annual_economic_diagnostics.csv", index=False)

    summary = rule_summary(monthly, annual)
    summary.to_csv(OUT / "rule_summary.csv", index=False)

    bootstrap_summary, bootstrap = paired_bootstrap(monthly)
    bootstrap_summary.to_csv(
        OUT / "hyperoptimized_vs_rate_bootstrap_summary.csv",
        index=False,
    )
    bootstrap.to_parquet(
        OUT / "hyperoptimized_vs_rate_bootstrap.parquet",
        index=False,
    )

    annual_pairs = econ.pivot(index="test_year", columns="rule")
    paired = pd.DataFrame({
        "test_year": annual_pairs.index,
        "rate_lower_loss": (
            annual_pairs["quadratic_criterion"]["inverse_T"]
            < annual_pairs["quadratic_criterion"]["annual_validation"]
        ),
        "rate_higher_sharpe": (
            annual_pairs["sharpe"]["inverse_T"]
            > annual_pairs["sharpe"]["annual_validation"]
        ),
        "rate_higher_mean_return": (
            annual_pairs["mean_return_ann"]["inverse_T"]
            > annual_pairs["mean_return_ann"]["annual_validation"]
        ),
    }).reset_index(drop=True)
    paired.to_csv(OUT / "year_by_year_rule_comparison.csv", index=False)

    plot_paths(annual)
    plot_complexity_relationships(econ)
    plot_hyperoptimized_vs_rate(monthly, annual)
    plot_complexity_vs_gross(annual)

    print("Rule summary")
    print(summary.to_string(index=False))
    print("\nPaired annual-block bootstrap")
    print(bootstrap_summary.to_string(index=False))
    print("\nYear-by-year win shares")
    print(paired.drop(columns="test_year").mean().to_string())


if __name__ == "__main__":
    main()
