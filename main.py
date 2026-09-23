"""Run all portfolio kernels."""

import argparse

from train_model import train_model


lambda_grid = None
number_of_lambdas = 120
lengthscale_multipliers = [
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    4.0,
]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--characteristics", nargs="+", default=["all"])
parser.add_argument("--kernels", nargs="+", choices=["linear", "gaussian", "ntk", "matern12", "matern32", "matern52"])
args = parser.parse_args()
characteristics = "all" if args.characteristics == ["all"] else args.characteristics
max_gross_exposure = None
kernel_names = [
    "linear",
    "gaussian",
    "ntk",
    "matern12",
    "matern32",
    "matern52",
]
if args.kernels:
    kernel_names = args.kernels

for kernel_name in kernel_names:
    print(f"Training {kernel_name}: {characteristics}", flush=True)
    train_model(
        lambda_grid=lambda_grid,
        kernel_name=kernel_name,
        characteristics=characteristics,
        n_random_features=2000,
        max_gross_exposure=max_gross_exposure,
        number_of_lambdas=number_of_lambdas,
        lengthscale_multipliers=lengthscale_multipliers,
    )
