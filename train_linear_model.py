"""Train the all-characteristic direct linear-ridge benchmark."""

from pathlib import Path

from Utils.utils import train_model


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "linear_all_characteristics"


def main():
    results = train_model(
        data_dir=PROJECT_ROOT / "data" / "JKP_USA",
        characteristics="all",
        kernel="linear",
        kernel_parameters={"batch_size": 4096},
        rho_grid=None,
        train_years=10,
        validation_years=5,
        test_years=1,
        target_volatility=0.10,
        max_gross_exposure=10.0,
        output_dir=RESULTS_DIR,
        save_weights=False,
    )
    for name, path in results["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
