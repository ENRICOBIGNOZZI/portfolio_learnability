import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from Kernels.kernel_function import PortfolioKernel
from download_JKP.read_dataset import (
    available_jkp_characteristics,
    available_jkp_years,
    load_jkp_value,
)


# ============================================================
# 1. CHARACTERISTICS UTILITY
# ============================================================

def as_characteristic_list(characteristics):
    """
    Converts:
        None              -> []
        "be_me"           -> ["be_me"]
        ["be_me", "mom"]  -> ["be_me", "mom"]
    """

    if characteristics is None:
        return []

    if isinstance(characteristics, str):
        return [characteristics]

    return list(characteristics)


# ============================================================
# 2. CROSS-SECTIONAL RANK TRANSFORMATION
#
# Each characteristic is ranked independently into [-1, 1].
#
# Input:
#       N x K matrix
#
# Output:
#       N x K matrix
# ============================================================

def rank_characteristics(x):

    x = np.asarray(x, dtype=float)

    # If only one characteristic, make it N x 1
    if x.ndim == 1:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            "x must be a vector or a 2D matrix."
        )

    N, K = x.shape

    z = np.empty(
        (N, K),
        dtype=float
    )

    for k in range(K):

        ranks = (
            pd.Series(x[:, k])
            .rank(method="average")
            .to_numpy()
        )

        z[:, k] = (
            2.0
            * ((ranks - 0.5) / N)
            - 1.0
        )

    return z


# ============================================================
# 3. MONTHLY EQUITY CROSS-SECTIONS
#
# For every month:
#
#       Z_t : N_t x K characteristic matrix
#       R_t : N_t future excess returns
#
# characteristics can be:
#
#       None
#       "be_me"
#       ["be_me", "mom12m", "size", ...]
#
# ============================================================

def build_monthly_panels(
    df,
    characteristics=None,
    characteristics_are_ranked=None,
):

    characteristics = as_characteristic_list(
        characteristics
    )

    if characteristics_are_ranked is None:
        characteristics_are_ranked = bool(
            df.attrs.get(
                "characteristics_rank_standardized",
                False,
            )
        )

    panels = []

    for date, cs in df.groupby(
        "eom",
        sort=True
    ):

        # ----------------------------------------------------
        # Columns needed in this cross-section
        # ----------------------------------------------------

        columns = [
            "id",
            "permno",
            "me",
            "ret_exc_lead1m"
        ] + characteristics

        cs = (
            cs[columns]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        # ----------------------------------------------------
        # Drop missing future returns and characteristics
        # ----------------------------------------------------

        drop_columns = [
            "ret_exc_lead1m"
        ] + characteristics

        cs = cs.dropna(
            subset=drop_columns
        )

        if len(cs) == 0:
            continue

        # ----------------------------------------------------
        # Returns
        # ----------------------------------------------------

        r = (
            cs["ret_exc_lead1m"]
            .to_numpy(dtype=float)
        )

        # ----------------------------------------------------
        # Characteristics
        # ----------------------------------------------------

        if len(characteristics) > 0:

            x_raw = (
                cs[characteristics]
                .to_numpy(dtype=float)
            )

            if characteristics_are_ranked:
                z = x_raw.copy()
            else:
                z = rank_characteristics(
                    x_raw
                )

        else:

            # Static portfolio case:
            # no characteristics
            x_raw = np.empty(
                (len(cs), 0)
            )

            z = np.empty(
                (len(cs), 0)
            )

        # ----------------------------------------------------
        # Store monthly cross-section
        # ----------------------------------------------------

        panels.append(
            {
                "date":
                    pd.Timestamp(date),

                "id":
                    cs["id"].to_numpy(),

                "permno":
                    cs["permno"].to_numpy(),

                "me":
                    cs["me"].to_numpy(dtype=float),

                "characteristics":
                    characteristics,

                "x_raw":
                    x_raw,

                # Rank-normalized characteristics, cross-section by cross-section.
                "x":
                    z,

                "z":
                    z,

                "r":
                    r
            }
        )

    return panels


# ============================================================
# 4. PAPER-CONSISTENT FINITE-FEATURE PORTFOLIO TRAINING
#
# For a shared stock-level policy w_it = phi(z_it)' beta, define the monthly
# managed payoff
#
#     g_t = sum_i r^e_{i,t+1} phi(z_it).
#
# Then R^p_{t+1} = g_t' beta and the response-one estimator in the paper is
#
#     min_beta T^{-1} sum_t (1 - g_t' beta)^2 + lambda ||beta||^2.
#
# There is deliberately no division by the number of stocks and no budget or
# leverage constraint. The ridge path is solved in the T x T monthly dual.
# ============================================================


def _annualized_sharpe(returns):
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return np.nan
    volatility = returns.std(ddof=1)
    if np.isclose(volatility, 0.0):
        return np.nan
    return np.sqrt(12.0) * returns.mean() / volatility


def _response_one_loss(returns):
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return np.nan
    return float(np.mean((1.0 - returns) ** 2))


def _annualized_volatility(returns):
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return np.nan
    return float(np.sqrt(12.0) * returns.std(ddof=1))


def _resolve_characteristics(data_dir, characteristics):
    if isinstance(characteristics, str) and characteristics.lower() == "all":
        return available_jkp_characteristics(data_dir)
    resolved = as_characteristic_list(characteristics)
    if not resolved:
        raise ValueError("train_model requires at least one characteristic.")
    return resolved


def _estimate_median_length_scale(
    panels,
    *,
    max_points=2048,
    n_pairs=50000,
    random_state=0,
):
    """Estimate a Gaussian bandwidth from training data only."""
    panels = list(panels)
    if not panels:
        raise ValueError("At least one training panel is required for ell='median'.")

    rng = np.random.default_rng(random_state)
    points_per_panel = max(1, max_points // len(panels))
    samples = []
    for panel in panels:
        x = np.asarray(panel["x"], dtype=float)
        if len(x) > points_per_panel:
            indices = rng.choice(len(x), size=points_per_panel, replace=False)
            x = x[indices]
        samples.append(x)
    sample = np.vstack(samples)
    if len(sample) > max_points:
        sample = sample[rng.choice(len(sample), size=max_points, replace=False)]
    if len(sample) < 2:
        raise ValueError("At least two observations are required to estimate ell.")

    pair_count = min(int(n_pairs), len(sample) * (len(sample) - 1))
    left = rng.integers(0, len(sample), size=pair_count)
    right = rng.integers(0, len(sample) - 1, size=pair_count)
    right += right >= left
    distances = np.linalg.norm(sample[left] - sample[right], axis=1)
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if len(positive) == 0:
        raise ValueError("Cannot estimate a positive Gaussian length scale.")
    return float(np.median(positive))


def _prepare_rff_year(kernel, panels):
    """Compress every stock cross-section into monthly managed payoffs."""
    monthly_statistics = []
    for panel in panels:
        statistics = kernel.managed_feature_statistics(panel["x"], panel["r"])
        monthly_statistics.append(
            {
                "date": pd.Timestamp(panel["date"]),
                "n_assets": statistics["n_observations"],
                "return_loading": statistics["return_loading"],
                "net_loading": statistics["net_loading"],
            }
        )
    if not monthly_statistics:
        raise ValueError("Cannot prepare RFF statistics from an empty year.")
    return monthly_statistics


def _managed_payoff_matrix(monthly_statistics):
    monthly_statistics = list(monthly_statistics)
    if not monthly_statistics:
        raise ValueError("At least one monthly managed payoff is required.")
    return np.vstack([month["return_loading"] for month in monthly_statistics])


def _ridge_path_from_managed_payoffs(
    managed_payoffs,
    *,
    lambda_grid=None,
):
    """Solve the response-one KRR path using the small monthly dual.

    ``lambda_grid`` contains the direct ridge penalties used in every rolling
    window. It is deliberately not rescaled by the managed-payoff spectrum.
    """
    G = np.asarray(managed_payoffs, dtype=float)
    if G.ndim != 2 or len(G) == 0 or not np.isfinite(G).all():
        raise ValueError("managed_payoffs must be a finite, non-empty matrix.")

    n_months = len(G)
    dual_covariance = (G @ G.T) / n_months
    eigenvalues, eigenvectors = np.linalg.eigh(dual_covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    lambda_grid = np.asarray(
        np.logspace(-6.0, 6.0, 37) if lambda_grid is None else lambda_grid,
        dtype=float,
    )
    if lambda_grid.ndim != 1 or len(lambda_grid) == 0 or np.any(lambda_grid <= 0):
        raise ValueError("lambda_grid must contain positive values.")
    lambda_grid = np.unique(lambda_grid)

    response = np.ones(n_months, dtype=float)
    projected_response = eigenvectors.T @ response
    denominators = eigenvalues[:, None] + lambda_grid[None, :]
    dual_coefficients = eigenvectors @ (
        projected_response[:, None] / (n_months * denominators)
    )
    coefficients = G.T @ dual_coefficients
    effective_dimensions = np.sum(
        eigenvalues[:, None] / denominators,
        axis=0,
    )
    return {
        "coefficients": coefficients,
        "dual_coefficients": dual_coefficients,
        "effective_dimensions": effective_dimensions,
        "eigenvalues": eigenvalues,
        "target_alignment": projected_response**2 / n_months,
        "lambda_grid": lambda_grid,
    }


def _portfolio_returns_from_loadings(monthly_statistics, coefficients):
    G = _managed_payoff_matrix(monthly_statistics)
    return G @ coefficients


def _evaluate_oos_panels(
    panels,
    monthly_statistics,
    kernel,
    coefficients,
    lambda_grid,
    test_year,
    selected_index,
    *,
    volatility_scales=None,
    max_gross_exposure=10.0,
    save_weights=False,
):
    """Evaluate raw weights and a proportional monthly gross-exposure cap."""
    records = []
    weight_records = []
    if len(panels) != len(monthly_statistics):
        raise ValueError("Panels and monthly statistics are not aligned.")

    if max_gross_exposure is not None and max_gross_exposure <= 0:
        raise ValueError("max_gross_exposure must be positive or None.")
    if volatility_scales is None:
        volatility_scales = np.ones(coefficients.shape[1], dtype=float)
    volatility_scales = np.asarray(volatility_scales, dtype=float)
    if volatility_scales.shape != (coefficients.shape[1],):
        raise ValueError("volatility_scales must have one value per coefficient path.")
    if np.any(~np.isfinite(volatility_scales)) or np.any(volatility_scales <= 0):
        raise ValueError("volatility_scales must be finite and positive.")

    for panel, month in zip(panels, monthly_statistics, strict=True):
        if pd.Timestamp(panel["date"]) != month["date"]:
            raise ValueError("Panels and monthly statistics have different dates.")

        n_assets = len(panel["r"])
        native_weights = np.empty((n_assets, coefficients.shape[1]), dtype=float)
        for start in range(0, n_assets, kernel.batch_size):
            stop = min(start + kernel.batch_size, n_assets)
            features = kernel.feature_map(panel["x"][start:stop])
            native_weights[start:stop] = features @ coefficients

        native_portfolio_returns = panel["r"] @ native_weights
        loading_returns = month["return_loading"] @ coefficients
        if not np.allclose(
            native_portfolio_returns, loading_returns, rtol=1e-9, atol=1e-11
        ):
            raise AssertionError("Managed-payoff and stock-level returns disagree.")
        raw_weights = native_weights * volatility_scales[None, :]
        raw_portfolio_returns = native_portfolio_returns * volatility_scales
        raw_net_exposures = raw_weights.sum(axis=0)
        raw_gross_exposures = np.abs(raw_weights).sum(axis=0)
        if max_gross_exposure is None:
            leverage_scales = np.ones(coefficients.shape[1], dtype=float)
        else:
            leverage_scales = np.minimum(
                1.0,
                np.divide(
                    max_gross_exposure,
                    raw_gross_exposures,
                    out=np.ones_like(raw_gross_exposures),
                    where=raw_gross_exposures > 0.0,
                ),
            )
        portfolio_returns = raw_portfolio_returns * leverage_scales
        net_exposures = raw_net_exposures * leverage_scales
        gross_exposures = raw_gross_exposures * leverage_scales
        selected_weights = (
            raw_weights[:, selected_index] * leverage_scales[selected_index]
        )

        if save_weights:
            for index in range(n_assets):
                weight_records.append(
                    {
                        "test_year": test_year,
                        "date": pd.Timestamp(panel["date"]),
                        "id": panel["id"][index],
                        "permno": panel["permno"][index],
                        "weight": float(selected_weights[index]),
                        "raw_weight": float(raw_weights[index, selected_index]),
                        "native_weight": float(native_weights[index, selected_index]),
                        "return": float(panel["r"][index]),
                    }
                )

        for index, lambda_ in enumerate(lambda_grid):
            record = {
                    "test_year": test_year,
                    "date": pd.Timestamp(panel["date"]),
                    "penalty_index": index,
                    "lambda": float(lambda_),
                    "portfolio_return": float(portfolio_returns[index]),
                    "portfolio_return_raw": float(raw_portfolio_returns[index]),
                    "portfolio_return_native": float(native_portfolio_returns[index]),
                    "sum_weights": float(net_exposures[index]),
                    "sum_absolute_weights": float(gross_exposures[index]),
                    "raw_sum_weights": float(raw_net_exposures[index]),
                    "raw_sum_absolute_weights": float(raw_gross_exposures[index]),
                    "leverage_scale": float(leverage_scales[index]),
                    "volatility_scale": float(volatility_scales[index]),
                    "max_gross_exposure": max_gross_exposure,
                    "n_assets": n_assets,
                    "selected_by_validation": is_selected,
                }
            records.append(record)
    return records, weight_records


def _evaluate_native_oos_months(
    monthly_statistics,
    coefficients,
    lambda_grid,
    test_year,
    selected_index,
):
    """Evaluate only native returns from compressed monthly payoff loadings.

    This path is intended for auxiliary exercises that only study statistical
    loss and effective dimension.  It avoids reconstructing stock-level weights
    for every value of lambda, so long-history experiments remain fast and
    memory-bounded.  Headline economic results continue to use
    ``_evaluate_oos_panels`` and therefore retain weights and exposure data.
    """
    native_returns = _managed_payoff_matrix(monthly_statistics) @ coefficients
    records = []
    for month_index, month in enumerate(monthly_statistics):
        for index, lambda_ in enumerate(lambda_grid):
            records.append(
                {
                    "test_year": test_year,
                    "date": pd.Timestamp(month["date"]),
                    "penalty_index": index,
                    "lambda": float(lambda_),
                    "portfolio_return": np.nan,
                    "portfolio_return_raw": np.nan,
                    "portfolio_return_native": float(native_returns[month_index, index]),
                    "sum_weights": np.nan,
                    "sum_absolute_weights": np.nan,
                    "raw_sum_weights": np.nan,
                    "raw_sum_absolute_weights": np.nan,
                    "leverage_scale": np.nan,
                    "volatility_scale": np.nan,
                    "max_gross_exposure": np.nan,
                    "n_assets": np.nan,
                    "selected_by_validation": index == selected_index,
                }
            )
    return records


def _pad_vectors(vectors):
    """Stack possibly unequal monthly spectra, padding missing entries with NaN."""
    width = max(len(vector) for vector in vectors)
    result = np.full((len(vectors), width), np.nan, dtype=float)
    for row, vector in enumerate(vectors):
        result[row, : len(vector)] = vector
    return result


def train_model(
    data_dir,
    characteristics="all",
    *,
    kernel="gaussian",
    kernel_parameters=None,
    lambda_grid=None,
    train_years=10,
    validation_years=5,
    test_years=1,
    test_year_start=None,
    test_year_end=None,
    output_dir="results/gaussian_rff_all_characteristics",
    max_windows=None,
    target_volatility=0.10,
    max_gross_exposure=10.0,
    save_weights=False,
    native_oos_only=False,
):
    """Train a rolling finite-feature response-one portfolio.

    The policy is shared across stocks, ``w_it = phi(z_it)' beta``. Estimation
    weights are free and are never divided by N_t. Lambda is chosen only with
    response-one validation loss, then the chosen lambda is refitted on train + validation.
    In the held-out implementation, an optional proportional monthly overlay
    enforces ``sum_i abs(w_it) <= max_gross_exposure``.  Auxiliary statistical
    exercises may set ``native_oos_only=True`` to skip stock-level weight
    reconstruction and evaluate directly from compressed monthly loadings.
    """
    started_at = time.perf_counter()
    if kernel not in {"gaussian", "matern32", "linear"}:
        raise NotImplementedError(
            "train_model supports Gaussian RFF, Matérn-3/2 RFF, and linear ridge."
        )

    parameters = {"batch_size": 4096}
    if kernel in {"gaussian", "matern32"}:
        parameters.update(
            {
                "ell": "median",
                "n_random_features": 1024,
                "random_state": 0,
            }
        )
    if kernel_parameters:
        parameters.update(kernel_parameters)
    if kernel in {"gaussian", "matern32"} and parameters["n_random_features"] is None:
        raise ValueError("Stationary RFF training requires n_random_features.")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    characteristics = _resolve_characteristics(data_dir, characteristics)
    available_years = available_jkp_years(data_dir)
    available_year_set = set(available_years)
    first_test_year = available_years[0] + train_years + validation_years
    candidate_windows = []
    for test_year in available_years:
        if test_year < first_test_year:
            continue
        if test_year_start is not None and test_year < int(test_year_start):
            continue
        if test_year_end is not None and test_year > int(test_year_end):
            continue
        train_year_list = list(
            range(test_year - train_years - validation_years, test_year - validation_years)
        )
        validation_year_list = list(range(test_year - validation_years, test_year))
        test_year_list = list(range(test_year, test_year + test_years))
        if set(train_year_list + validation_year_list + test_year_list).issubset(
            available_year_set
        ):
            candidate_windows.append(
                (test_year, train_year_list, validation_year_list, test_year_list)
            )
    if max_windows is not None:
        candidate_windows = candidate_windows[: int(max_windows)]
    if not candidate_windows:
        raise ValueError("No complete rolling windows were available.")

    panels_by_year = {}

    def load_year(year):
        if year not in panels_by_year:
            year_df = load_jkp_value(
                data_dir,
                CHARACTERISTIC=characteristics,
                years=[year],
            )
            panels_by_year[year] = build_monthly_panels(
                year_df,
                characteristics=characteristics,
            )
        return panels_by_year[year]

    first_training_years = candidate_windows[0][1]
    first_training_panels = [
        panel for year in first_training_years for panel in load_year(year)
    ]
    if kernel in {"gaussian", "matern32"}:
        requested_ell = parameters["ell"]
        if requested_ell is None or (
            isinstance(requested_ell, str) and requested_ell.lower() == "median"
        ):
            resolved_ell = _estimate_median_length_scale(
                first_training_panels,
                random_state=int(parameters["random_state"]),
            )
            ell_method = "median pairwise distance in the first training window"
        else:
            resolved_ell = float(requested_ell)
            if resolved_ell <= 0:
                raise ValueError("ell must be strictly positive.")
            ell_method = "user supplied"
        parameters["ell"] = resolved_ell
        print(
            f"{kernel}: {len(characteristics)} characteristics, "
            f"{parameters['n_random_features']} features, ell={resolved_ell:.4g}"
        )
    else:
        ell_method = None
        print(f"Linear ridge: {len(characteristics)} characteristics")

    rff_kernel = PortfolioKernel(kernel=kernel, **parameters)
    monthly_statistics = {
        year: _prepare_rff_year(rff_kernel, panels_by_year[year])
        for year in first_training_years
    }
    # Only compressed monthly loadings are needed for fitting. Releasing the
    # stock-level panels here keeps long-history experiments memory bounded.
    panels_by_year.clear()
    diagnostics = []
    monthly_oos = []
    selected_weight_records = []
    model_coefficients = []
    train_eigenvalues = []
    refit_eigenvalues = []
    train_target_alignment = []
    refit_target_alignment = []
    model_test_years = []
    selected_indices = []
    model_volatility_scales = []
    refit_dual_coefficients = []
    refit_month_dates = []

    for test_year, train_year_list, validation_year_list, test_year_list in candidate_windows:
        required_years = set(train_year_list + validation_year_list + test_year_list)
        for stale_year in set(panels_by_year) - required_years:
            del panels_by_year[stale_year]
            monthly_statistics.pop(stale_year, None)

        for year in sorted(required_years):
            if year not in monthly_statistics:
                panels = load_year(year)
                monthly_statistics[year] = _prepare_rff_year(rff_kernel, panels)
                if native_oos_only or year not in test_year_list:
                    del panels_by_year[year]
            elif year in test_year_list and year not in panels_by_year:
                load_year(year)

        train_months = [
            month for year in train_year_list for month in monthly_statistics[year]
        ]
        validation_months = [
            month for year in validation_year_list for month in monthly_statistics[year]
        ]
        refit_months = train_months + validation_months
        train_G = _managed_payoff_matrix(train_months)
        validation_G = _managed_payoff_matrix(validation_months)

        train_path = _ridge_path_from_managed_payoffs(
            train_G,
            lambda_grid=lambda_grid,
        )
        train_returns = train_G @ train_path["coefficients"]
        validation_returns = validation_G @ train_path["coefficients"]
        train_sharpes = np.array(
            [_annualized_sharpe(train_returns[:, index]) for index in range(train_returns.shape[1])]
        )
        validation_sharpes = np.array(
            [
                _annualized_sharpe(validation_returns[:, index])
                for index in range(validation_returns.shape[1])
            ]
        )
        train_losses = np.array(
            [_response_one_loss(train_returns[:, index]) for index in range(train_returns.shape[1])]
        )
        validation_losses = np.array(
            [
                _response_one_loss(validation_returns[:, index])
                for index in range(validation_returns.shape[1])
            ]
        )
        selection_scores = np.where(
            np.isfinite(validation_losses), validation_losses, np.inf
        )
        if np.all(np.isposinf(selection_scores)):
            raise ValueError(f"All validation losses are undefined for {test_year}.")
        selected_index = int(np.argmin(selection_scores))
        validation_volatilities = np.array(
            [
                _annualized_volatility(validation_returns[:, index])
                for index in range(validation_returns.shape[1])
            ]
        )
        if target_volatility is None:
            volatility_scales = np.ones_like(validation_volatilities)
        else:
            if target_volatility <= 0:
                raise ValueError("target_volatility must be positive or None.")
            volatility_scales = np.divide(
                float(target_volatility),
                validation_volatilities,
                out=np.ones_like(validation_volatilities),
                where=np.isfinite(validation_volatilities)
                & (validation_volatilities > 1e-12),
            )

        refit_path = _ridge_path_from_managed_payoffs(
            _managed_payoff_matrix(refit_months),
            lambda_grid=train_path["lambda_grid"],
        )
        test_months = [
            month for year in test_year_list for month in monthly_statistics[year]
        ]
        if native_oos_only:
            window_oos = _evaluate_native_oos_months(
                test_months,
                refit_path["coefficients"],
                refit_path["lambda_grid"],
                test_year,
                selected_index,
            )
            window_weights = []
        else:
            test_panels = [
                panel for year in test_year_list for panel in panels_by_year[year]
            ]
            window_oos, window_weights = _evaluate_oos_panels(
                test_panels,
                test_months,
                rff_kernel,
                refit_path["coefficients"],
                refit_path["lambda_grid"],
                test_year,
                selected_index,
                volatility_scales=volatility_scales,
                max_gross_exposure=max_gross_exposure,
                save_weights=save_weights,
            )
        monthly_oos.extend(window_oos)
        selected_weight_records.extend(window_weights)
        window_oos_frame = pd.DataFrame(window_oos)

        for index, lambda_ in enumerate(refit_path["lambda_grid"]):
            lambda_oos = window_oos_frame.loc[
                window_oos_frame["penalty_index"].eq(index),
                "portfolio_return",
            ]
            lambda_oos_raw = window_oos_frame.loc[
                window_oos_frame["penalty_index"].eq(index),
                "portfolio_return_raw",
            ]
            lambda_oos_native = window_oos_frame.loc[
                window_oos_frame["penalty_index"].eq(index),
                "portfolio_return_native",
            ]
            diagnostics.append(
                {
                    "test_year": test_year,
                    "penalty_index": index,
                    "lambda": float(lambda_),
                    "in_sample_sharpe": float(train_sharpes[index]),
                    "validation_sharpe": float(validation_sharpes[index]),
                    "in_sample_loss": float(train_losses[index]),
                    "validation_loss": float(validation_losses[index]),
                    "oos_sharpe": float(_annualized_sharpe(lambda_oos)),
                    "oos_sharpe_raw": float(_annualized_sharpe(lambda_oos_raw)),
                    "oos_loss_raw": float(_response_one_loss(lambda_oos_raw)),
                    "oos_loss_native": float(_response_one_loss(lambda_oos_native)),
                    "validation_annualized_volatility": float(
                        validation_volatilities[index]
                    ),
                    "volatility_scale": float(volatility_scales[index]),
                    "train_effective_dimension": float(
                        train_path["effective_dimensions"][index]
                    ),
                    "effective_dimension": float(
                        refit_path["effective_dimensions"][index]
                    ),
                    "selected_by_validation": index == selected_index,
                    "n_train_months": len(train_months),
                    "n_refit_months": len(refit_months),
                }
            )

        model_coefficients.append(refit_path["coefficients"])
        train_eigenvalues.append(train_path["eigenvalues"])
        refit_eigenvalues.append(refit_path["eigenvalues"])
        train_target_alignment.append(train_path["target_alignment"])
        refit_target_alignment.append(refit_path["target_alignment"])
        model_volatility_scales.append(volatility_scales)
        refit_dual_coefficients.append(refit_path["dual_coefficients"])
        refit_month_dates.append(
            np.asarray([month["date"].to_datetime64() for month in refit_months])
        )
        model_test_years.append(test_year)
        selected_indices.append(selected_index)
        for year in test_year_list:
            panels_by_year.pop(year, None)
        print(
            f"Window {test_year}: selected lambda={refit_path['lambda_grid'][selected_index]:.3g}, "
            f"validation loss={validation_losses[selected_index]:.4g}"
        )

    diagnostics_frame = pd.DataFrame(diagnostics).sort_values(
        ["test_year", "penalty_index"]
    )
    monthly_oos_frame = pd.DataFrame(monthly_oos).sort_values(
        ["penalty_index", "date"]
    )
    monthly_oos_frame["equity"] = monthly_oos_frame.groupby("penalty_index")[
        "portfolio_return"
    ].transform(lambda values: (1.0 + values).cumprod())

    selected_oos = monthly_oos_frame.loc[
        monthly_oos_frame["selected_by_validation"]
    ].sort_values("date").copy()
    selected_oos["equity"] = (1.0 + selected_oos["portfolio_return"]).cumprod()
    selected_oos["cumulative_pnl"] = 1.0 + selected_oos["portfolio_return"].cumsum()

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "lambda_diagnostics.parquet"
    monthly_path = output_dir / "monthly_oos_portfolios.parquet"
    selected_path = output_dir / "selected_oos_portfolio.parquet"
    models_path = output_dir / "models.npz"
    metadata_path = output_dir / "metadata.json"

    diagnostics_frame.to_parquet(diagnostics_path, index=False)
    monthly_oos_frame.to_parquet(monthly_path, index=False)
    selected_oos.to_parquet(selected_path, index=False)
    paths = {
        "diagnostics": diagnostics_path,
        "monthly_oos": monthly_path,
        "selected_oos": selected_path,
        "models": models_path,
        "metadata": metadata_path,
    }
    if save_weights:
        weights_path = output_dir / "selected_oos_weights.parquet"
        pd.DataFrame(selected_weight_records).to_parquet(weights_path, index=False)
        paths["selected_weights"] = weights_path

    model_payload = {
        "test_years": np.asarray(model_test_years),
        "lambda_grid": np.asarray(refit_path["lambda_grid"]),
        "coefficients": np.stack(model_coefficients),
        "train_eigenvalues": _pad_vectors(train_eigenvalues),
        "refit_eigenvalues": _pad_vectors(refit_eigenvalues),
        "train_target_alignment": _pad_vectors(train_target_alignment),
        "refit_target_alignment": _pad_vectors(refit_target_alignment),
        "selected_indices": np.asarray(selected_indices),
        "volatility_scales": np.stack(model_volatility_scales),
        "refit_dual_coefficients": np.stack(refit_dual_coefficients),
        "refit_month_dates": np.stack(refit_month_dates),
        "characteristics": np.asarray(characteristics),
    }
    if hasattr(rff_kernel, "rff_weights_"):
        model_payload["rff_weights"] = rff_kernel.rff_weights_
        model_payload["rff_phases"] = rff_kernel.rff_phases_
    np.savez_compressed(
        models_path,
        **model_payload,
    )
    elapsed_seconds = time.perf_counter() - started_at
    metadata = {
        "estimator": "monthly response-one kernel ridge portfolio",
        "kernel": kernel,
        "kernel_parameters": parameters,
        "ell_method": ell_method,
        "characteristics": characteristics,
        "n_characteristics": len(characteristics),
        "lambda_grid": np.asarray(refit_path["lambda_grid"]).tolist(),
        "lambda_units": "direct ridge penalty in the response-one objective",
        "train_years": train_years,
        "validation_years": validation_years,
        "test_years": test_years,
        "test_year_start": test_year_start,
        "test_year_end": test_year_end,
        "weight_definition": "w_it = phi(z_it)' beta (free weights; no division by N_t)",
        "max_gross_exposure": max_gross_exposure,
        "target_annualized_volatility": target_volatility,
        "volatility_scaling": "validation-period realized volatility, fixed before each test window",
        "leverage_rule": "monthly proportional rescaling if sum(abs(raw weights)) exceeds the cap",
        "training_objective": "mean_t (1 - portfolio_return_t)^2 + lambda * ||beta||^2",
        "lambda_selection": "minimum response-one validation loss of the unconstrained paper estimator",
        "refit": "train plus validation after lambda selection",
        "effective_dimension": "sum_j mu_j / (mu_j + lambda), mu_j eigenvalues of G'G/T",
        "completed_windows": len(model_test_years),
        "save_weights": bool(save_weights),
        "native_oos_only": bool(native_oos_only),
        "elapsed_seconds": elapsed_seconds,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "diagnostics": diagnostics_frame,
        "monthly_oos": monthly_oos_frame,
        "selected_oos": selected_oos,
        "paths": paths,
    }
