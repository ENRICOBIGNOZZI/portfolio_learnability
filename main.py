"""Run all portfolio kernels."""

from train_model import train_model


lambda_grid = None
number_of_lambdas = 60
lengthscale_multipliers = [
    0.5,
    1.0,
    2.0,
]
characteristics = "all"
max_gross_exposure = 2.0
kernel_names = [
    "linear",
    "gaussian",
    "ntk",
    "matern12",
    "matern32",
    "matern52",
]

for kernel_name in kernel_names:
    train_model(
        lambda_grid=lambda_grid,
        kernel_name=kernel_name,
        characteristics=characteristics,
        n_random_features=1000,
        max_gross_exposure=max_gross_exposure,
        number_of_lambdas=number_of_lambdas,
        lengthscale_multipliers=lengthscale_multipliers,
    )
