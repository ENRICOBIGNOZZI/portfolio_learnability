"""Regenerate all figures sequentially from existing results; no model training."""
import importlib
import argparse
from pathlib import Path
import time
from Utils.paper_figures import ROOT, FIGURES
from Utils.reproducibility import write_manifest

MODULES = [
    "figure_1_complexity_performance", "figure_2_in_sample_oos",
    "figure_3_history_opens_directions", "figure_4_economic_spectrum",
    "figure_5_learnable_frontier", "figure_6_economic_outcomes",
    "figure_7_tuning_stability", "figure_8_exposure_and_risk",
    "figure_9_benchmark_uncertainty", "figure_10_complexity_by_period",
    "appendix_loss_calibration", "appendix_historical_kernel_weights",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=MODULES, help="Generate only these scripts (omit .py).")
    parser.add_argument("--output-dir", type=Path, default=FIGURES,
                        help="Destination for plots, tables and reproduction manifest.")
    args = parser.parse_args(argv)
    modules = args.only or MODULES
    if "figure_3_history_opens_directions" in modules:
        for months in (60, 120, 180, 240, 360, 480):
            path = ROOT / f"results/history_length/matern32_T{months}/metadata.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing history experiment: {path}. Train it explicitly first.")
    started = time.perf_counter()
    args.output_dir = args.output_dir.resolve()
    for name in modules:
        print(f"Generating {name}...", flush=True)
        importlib.import_module(name).main(figure_dir=args.output_dir)
    write_manifest(args.output_dir, modules, time.perf_counter()-started)


if __name__ == "__main__":
    main()
