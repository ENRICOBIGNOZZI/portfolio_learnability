"""Train one portfolio model and save its results."""

from pathlib import Path

import numpy as np
import pandas as pd

from download_JKP.read_dataset import (
    DEFAULT_DATA_DIR,
    available_jkp_characteristics,
    available_jkp_years,
    load_dataset,
)
from Kernels.kernel_function import (
    PortfolioKernel,
    effective_dimension,
    fit_lambda_grid,
    kernel_eigenvalues,
    make_complexity_lambda_grid,
    make_return_dictionary,
    make_return_matrix,
)
from median_lengthscale import median_lengthscale
from Utils.utils import (
    build_expanding_windows,
    build_monthly_panels,
    compute_sharpe_ratio,
    limit_gross_exposure,
)


panel_cache = {}


def evaluate_lambdas(
    months,
    kernel,
    betas,
    max_gross_exposure,
):
    """Compute portfolio returns for every lambda, optionally with a gross cap."""
    lambda_values = list(betas)
    beta_matrix = np.column_stack(
        [betas[value] for value in lambda_values]
    )

    returns = {value: [] for value in lambda_values}
    raw_gross = {value: [] for value in lambda_values}
    scales = {value: [] for value in lambda_values}

    for month in months:
        features = kernel.features(month["x"])
        raw_weights = (
            features @ beta_matrix
        ) / month["n_assets"]

        month_raw_gross = np.abs(raw_weights).sum(axis=0)
        month_scales = np.ones(len(lambda_values))
        if max_gross_exposure is not None:
            too_large = month_raw_gross > max_gross_exposure
            month_scales[too_large] = (
                max_gross_exposure
                / month_raw_gross[too_large]
            )

        weights = raw_weights * month_scales
        month_returns = month["r"] @ weights

        for column, lambda_value in enumerate(lambda_values):
            returns[lambda_value].append(month_returns[column])
            raw_gross[lambda_value].append(
                month_raw_gross[column]
            )
            scales[lambda_value].append(month_scales[column])

    results = {}

    for lambda_value in lambda_values:
        results[lambda_value] = {
            "returns": np.asarray(returns[lambda_value]),
            "raw_gross_exposure": np.asarray(
                raw_gross[lambda_value]
            ),
            "scale_factor": np.asarray(scales[lambda_value]),
        }

    return results


def load_panels(characteristics):
    """Load the panels once and reuse them for every kernel."""
    cache_name = tuple(characteristics)

    if cache_name in panel_cache:
        return panel_cache[cache_name]

    years = available_jkp_years(DEFAULT_DATA_DIR)
    panels = []

    for year in years:
        data = load_dataset(
            characteristics=characteristics,
            years=[year],
        )
        panels.extend(
            build_monthly_panels(
                data,
                characteristics=characteristics,
            )
        )

    panel_cache[cache_name] = panels
    return panels


def train_model(
    lambda_grid,
    kernel_name,
    characteristics,
    n_random_features=2000,
    max_gross_exposure=None,
    number_of_lambdas=120,
    lengthscale_multipliers=(0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
):
    """Train one kernel with expanding windows."""
    use_all_characteristics = (
        isinstance(characteristics, str)
        and characteristics.lower() == "all"
    )

    if use_all_characteristics:
        characteristics = available_jkp_characteristics(
            DEFAULT_DATA_DIR
        )
    elif isinstance(characteristics, str):
        characteristics = [characteristics]

    if kernel_name in {
        "gaussian",
        "matern12",
        "matern32",
        "matern52",
        "ntk",
    }:
        if n_random_features < 1000:
            raise ValueError(
                "Use at least 1000 random features."
            )

    if max_gross_exposure is not None and max_gross_exposure <= 0:
        raise ValueError(
            "max_gross_exposure must be positive when supplied."
        )
    cap_value = (
        np.nan
        if max_gross_exposure is None
        else float(max_gross_exposure)
    )

    uses_lengthscale = kernel_name in {
        "gaussian",
        "matern12",
        "matern32",
        "matern52",
    }

    project_folder = Path(__file__).resolve().parent
    if use_all_characteristics:
        characteristic_name = "all"
    else:
        characteristic_name = "_".join(characteristics)
    results_folder = (
        project_folder
        / "results"
        / kernel_name
        / characteristic_name
    )
    results_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    panels = load_panels(characteristics)

    windows = build_expanding_windows(
        panels,
        initial_train_years=10,
        validation_years=5,
        test_years=1,
    )

    if lambda_grid is not None:
        lambda_grid = np.sort(
            np.asarray(lambda_grid, dtype=float)
        )

    lengthscale = 1.0
    if uses_lengthscale:
        median = median_lengthscale(
            windows[0]["train"]
        )
        print("Median lengthscale:", median)

        lengthscale_results = []
        best_lengthscale_loss = np.inf
        best_lengthscale_grid = None

        first_train = windows[0]["train"]
        first_validation = windows[0]["validation"]

        for multiplier in lengthscale_multipliers:
            trial_lengthscale = median * multiplier
            trial_kernel = PortfolioKernel(
                kernel=kernel_name,
                ell=trial_lengthscale,
                n_random_features=n_random_features,
            )
            trial_months = first_train + first_validation
            trial_returns = make_return_dictionary(
                trial_months,
                trial_kernel,
            )
            trial_train_matrix = make_return_matrix(
                first_train,
                trial_kernel,
                trial_returns,
            )
            trial_validation_matrix = make_return_matrix(
                first_validation,
                trial_kernel,
                trial_returns,
            )

            trial_grid = lambda_grid
            if trial_grid is None:
                trial_eigenvalues = kernel_eigenvalues(
                    trial_train_matrix
                )
                trial_grid = make_complexity_lambda_grid(
                    trial_eigenvalues,
                    number_of_lambdas=number_of_lambdas,
                )

            trial_betas, _ = fit_lambda_grid(
                trial_train_matrix,
                trial_grid,
            )

            trial_best_lambda = None
            trial_best_loss = np.inf
            trial_best_sharpe = np.nan

            for lambda_value in trial_grid:
                trial_validation_returns = (
                    trial_validation_matrix
                    @ trial_betas[lambda_value]
                )
                trial_loss = np.mean(
                    (1.0 - trial_validation_returns) ** 2
                )

                if trial_loss < trial_best_loss:
                    trial_best_lambda = lambda_value
                    trial_best_loss = trial_loss
                    trial_best_sharpe = compute_sharpe_ratio(
                        trial_validation_returns
                    )

            lengthscale_results.append(
                {
                    "kernel": kernel_name,
                    "median_lengthscale": median,
                    "multiplier": multiplier,
                    "lengthscale": trial_lengthscale,
                    "best_lambda": trial_best_lambda,
                    "validation_loss": trial_best_loss,
                    "validation_sharpe": trial_best_sharpe,
                }
            )

            if trial_best_loss < best_lengthscale_loss:
                lengthscale = trial_lengthscale
                best_lengthscale_loss = trial_best_loss
                best_lengthscale_grid = trial_grid

        if lambda_grid is None:
            lambda_grid = best_lengthscale_grid

        lengthscale_file = (
            results_folder
            / "lengthscale_diagnostics.parquet"
        )
        pd.DataFrame(lengthscale_results).to_parquet(
            lengthscale_file,
            index=False,
        )
        print("Selected lengthscale:", lengthscale)
        print("Lengthscale diagnostics:", lengthscale_file)

    kernel = PortfolioKernel(
        kernel=kernel_name,
        ell=lengthscale,
        n_random_features=n_random_features,
    )
    return_dictionary = make_return_dictionary(
        panels,
        kernel,
    )

    test_results = []
    eigenvalue_results = []
    lambda_results = []
    lambda_return_results = []

    for window_number, window in enumerate(
        windows,
        start=1,
    ):
        train_matrix = make_return_matrix(
            window["train"],
            kernel,
            return_dictionary,
        )
        validation_matrix = make_return_matrix(
            window["validation"],
            kernel,
            return_dictionary,
        )
        betas, _ = fit_lambda_grid(
            train_matrix,
            lambda_grid,
        )
        validation_results = evaluate_lambdas(
            window["validation"],
            kernel,
            betas,
            max_gross_exposure,
        )

        test_year = window["test_years"][0]
        best_lambda = None
        best_validation_loss = np.inf
        window_lambda_results = []

        for lambda_value in lambda_grid:
            beta = betas[lambda_value]
            raw_validation_returns = validation_matrix @ beta
            validation_returns = validation_results[
                lambda_value
            ]["returns"]
            validation_loss = np.mean(
                (1.0 - validation_returns) ** 2
            )

            window_lambda_results.append(
                {
                    "kernel": kernel_name,
                    "test_year": test_year,
                    "train_start": window["train_years"][0],
                    "train_end": window["train_years"][-1],
                    "validation_start": window["validation_years"][0],
                    "validation_end": window["validation_years"][-1],
                    "lengthscale": lengthscale,
                    "lambda": lambda_value,
                    "max_gross_exposure": cap_value,
                    "validation_sharpe": compute_sharpe_ratio(
                        validation_returns
                    ),
                    "validation_loss": validation_loss,
                    "validation_sharpe_raw": compute_sharpe_ratio(
                        raw_validation_returns
                    ),
                    "validation_loss_raw": np.mean(
                        (1.0 - raw_validation_returns) ** 2
                    ),
                    "validation_average_raw_gross_exposure": np.mean(
                        validation_results[lambda_value][
                            "raw_gross_exposure"
                        ]
                    ),
                    "validation_average_scale_factor": np.mean(
                        validation_results[lambda_value][
                            "scale_factor"
                        ]
                    ),
                }
            )

            raw_validation_loss = np.mean(
                (1.0 - raw_validation_returns) ** 2
            )

            if raw_validation_loss < best_validation_loss:
                best_lambda = lambda_value
                best_validation_loss = raw_validation_loss

        train_and_validation = np.vstack(
            (
                train_matrix,
                validation_matrix,
            )
        )
        test_matrix = make_return_matrix(
            window["test"],
            kernel,
            return_dictionary,
        )
        refit_betas, eigenvalues = fit_lambda_grid(
            train_and_validation,
            lambda_grid,
        )
        test_lambda_results = evaluate_lambdas(
            window["test"],
            kernel,
            refit_betas,
            max_gross_exposure,
        )

        for number, eigenvalue in enumerate(
            eigenvalues,
            start=1,
        ):
            eigenvalue_results.append(
                {
                    "kernel": kernel_name,
                    "test_year": test_year,
                    "lengthscale": lengthscale,
                    "eigenvalue_number": number,
                    "eigenvalue": eigenvalue,
                }
            )

        for result in window_lambda_results:
            lambda_value = result["lambda"]
            lambda_beta = refit_betas[lambda_value]
            raw_test_returns = test_matrix @ lambda_beta
            limited_test_returns = test_lambda_results[
                lambda_value
            ]["returns"]
            raw_test_gross = test_lambda_results[
                lambda_value
            ]["raw_gross_exposure"]
            test_scale = test_lambda_results[
                lambda_value
            ]["scale_factor"]

            result["effective_dimension"] = effective_dimension(
                eigenvalues,
                lambda_value,
            )
            result["in_sample_sharpe"] = compute_sharpe_ratio(
                train_and_validation @ lambda_beta
            )
            result["out_of_sample_sharpe"] = compute_sharpe_ratio(
                limited_test_returns
            )
            result["out_of_sample_sharpe_raw"] = compute_sharpe_ratio(
                raw_test_returns
            )
            result["test_average_raw_gross_exposure"] = np.mean(
                test_lambda_results[lambda_value][
                    "raw_gross_exposure"
                ]
            )
            result["test_average_scale_factor"] = np.mean(
                test_lambda_results[lambda_value][
                    "scale_factor"
                ]
            )
            result["selected"] = (
                lambda_value == best_lambda
            )

            for month_number, month in enumerate(window["test"]):
                lambda_return_results.append(
                    {
                        "formation_date": month["formation_date"],
                        "return_date": month["return_date"],
                        "kernel": kernel_name,
                        "lengthscale": lengthscale,
                        "lambda": lambda_value,
                        "max_gross_exposure": cap_value,
                        "raw_portfolio_return": raw_test_returns[
                            month_number
                        ],
                        "portfolio_return": limited_test_returns[
                            month_number
                        ],
                        "raw_gross_exposure": raw_test_gross[
                            month_number
                        ],
                        "gross_exposure": (
                            raw_test_gross[month_number]
                            * test_scale[month_number]
                        ),
                        "scale_factor": test_scale[month_number],
                    }
                )

        lambda_results.extend(
            window_lambda_results
        )

        beta = refit_betas[best_lambda]
        window_test_returns = []
        window_weights = []

        for month in window["test"]:
            features = kernel.features(
                month["x"]
            )
            raw_weights = (
                features @ beta
            ) / month["n_assets"]
            (
                weights,
                raw_gross_exposure,
                scale_factor,
            ) = limit_gross_exposure(
                raw_weights,
                max_gross_exposure,
            )
            raw_portfolio_return = raw_weights @ month["r"]
            portfolio_return = weights @ month["r"]
            absolute_weights = np.abs(weights)
            gross_exposure = absolute_weights.sum()
            net_exposure = weights.sum()
            max_abs_weight = absolute_weights.max()
            p99_abs_weight = np.quantile(absolute_weights, 0.99)

            window_test_returns.append(
                portfolio_return
            )
            test_results.append(
                {
                    "formation_date": month["formation_date"],
                    "return_date": month["return_date"],
                    "kernel": kernel_name,
                    "lengthscale": lengthscale,
                    "lambda": best_lambda,
                    "max_gross_exposure": cap_value,
                    "raw_portfolio_return": raw_portfolio_return,
                    "portfolio_return": portfolio_return,
                    "raw_gross_exposure": raw_gross_exposure,
                    "gross_exposure": gross_exposure,
                    "net_exposure": net_exposure,
                    "max_abs_weight": max_abs_weight,
                    "p99_abs_weight": p99_abs_weight,
                    "scale_factor": scale_factor,
                }
            )
            window_weights.append(
                pd.DataFrame(
                    {
                        "formation_date": month["formation_date"],
                        "return_date": month["return_date"],
                        "kernel": kernel_name,
                        "lengthscale": lengthscale,
                        "lambda": best_lambda,
                        "id": month["id"],
                        "permno": month["permno"],
                        "raw_weight": raw_weights,
                        "weight": weights,
                        "scale_factor": scale_factor,
                    }
                )
            )

        lengthscale_name = (
            f"lengthscale_{lengthscale:.3g}"
        )
        lambda_name = f"lambda_{best_lambda:.0e}"
        weights_folder = (
            results_folder
            / "weights"
            / lengthscale_name
            / lambda_name
        )
        weights_folder.mkdir(
            parents=True,
            exist_ok=True,
        )
        weights_file = (
            weights_folder
            / f"weights_{test_year}.parquet"
        )
        pd.concat(
            window_weights,
            ignore_index=True,
        ).to_parquet(
            weights_file,
            index=False,
        )

        print()
        print("Kernel:", kernel_name)
        print("Window:", window_number)
        print("Test year:", test_year)
        if uses_lengthscale:
            print("Lengthscale:", lengthscale)
        print("Best lambda:", best_lambda)
        print(
            "Average test return:",
            np.mean(window_test_returns),
        )

    returns_file = (
        results_folder
        / "portfolio_returns.parquet"
    )
    eigenvalues_file = (
        results_folder
        / "kernel_eigenvalues.parquet"
    )
    lambda_file = (
        results_folder
        / "lambda_diagnostics.parquet"
    )
    lambda_returns_file = (
        results_folder
        / "lambda_portfolio_returns.parquet"
    )

    pd.DataFrame(test_results).to_parquet(
        returns_file,
        index=False,
    )
    pd.DataFrame(eigenvalue_results).to_parquet(
        eigenvalues_file,
        index=False,
    )
    pd.DataFrame(lambda_results).to_parquet(
        lambda_file,
        index=False,
    )
    pd.DataFrame(lambda_return_results).to_parquet(
        lambda_returns_file,
        index=False,
    )

    print()
    print("Results saved in:", results_folder)

    return results_folder
