"""Train a characteristic subset and regenerate every figure and both reports.

Example: python3 run_experiment.py --characteristics be_me
Use --skip-training after a completed baseline run, or --report-only for plots.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from lambda_scaling import KERNELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--characteristics", nargs="+", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--skip-training", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    selection = ["--characteristics", *args.characteristics]
    commands = []
    if not (args.skip_training or args.report_only):
        commands.append(["main.py", *selection])
    commands.extend([
        ["eigenvalues.py", *selection, *(["--plots-only"] if args.report_only else [])],
        ["complexity.py", *selection],
        ["equities.py", *selection],
    ])
    commands.extend(["plot_equity.py", "--kernel", kernel, *selection] for kernel in KERNELS)
    commands.extend([
        ["complexity_analysis.py", *selection],
        ["lambda_scaling.py", *selection, *(["--report-only"] if args.report_only else [])],
    ])
    env = os.environ.copy()
    for name in ["OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"]:
        env[name] = "2"
    env["MPLBACKEND"] = "Agg"
    started = datetime.now(timezone.utc).isoformat()
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run([sys.executable, "-u", *command], cwd=root, env=env, check=True)
    name = "_".join(args.characteristics)
    output = root / "results" / ("lambda_scaling" if name == "all" else f"lambda_scaling_{name}")
    manifest = {"characteristics": args.characteristics, "kernels": list(KERNELS),
                "commands": commands, "started_utc": started,
                "completed_utc": datetime.now(timezone.utc).isoformat()}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Complete:", output / "report.html", flush=True)


if __name__ == "__main__":
    main()
