"""Run the paper-consistent Gaussian RFF portfolio experiment."""

from pathlib import Path

from Utils.utils import train_model


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    results = train_model(
        data_dir=PROJECT_ROOT / "data" / "JKP_USA",
        characteristics="all",
        kernel="gaussian",
        kernel_parameters={
            "ell": "median",
            "n_random_features": 1024,
            "random_state": 0,
            "batch_size": 4096,
        },
        # Default: 29 relative penalties from 1e-10 to 1e4.
        rho_grid=None,
        train_years=10,
        validation_years=5,
        test_years=1,
        output_dir=(
            PROJECT_ROOT / "results" / "gaussian_rff_all_characteristics"
        ),
        target_volatility=0.10,
        max_gross_exposure=10.0,
        save_weights=False,
    )

    print("\nSaved training outputs:")
    for name, path in results["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
