"""Record local inputs, source code, packages and outputs using bounded reads."""
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import sys

from Utils.paper_figures import ROOT


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_manifest(figure_dir, modules, elapsed_seconds):
    figure_dir = Path(figure_dir)
    # Include the saved result bundles on which this figure suite depends.
    result_dirs = [ROOT / "results/matern32_rff_all_characteristics",
                   ROOT / "results/linear_all_characteristics"]
    if "figure_3_history_opens_directions" in modules:
        result_dirs += [ROOT / f"results/history_length/matern32_T{t}"
                        for t in (60, 120, 180, 240, 360, 480)]
    inputs = sorted({p for directory in result_dirs for pattern in
                     ("metadata.json", "lambda_diagnostics.parquet", "monthly_oos_portfolios.parquet",
                      "selected_oos_portfolio.parquet", "models.npz")
                     for p in directory.glob(pattern)})
    sources = sorted({p for directory in (ROOT, ROOT/"Utils", ROOT/"Kernels", ROOT/"download_JKP")
                      for p in directory.glob("*.py")} | {ROOT/"requirements-reproduce.txt"})
    outputs = sorted({p for module in modules
                      for p in figure_dir.glob(("appendix_historical_*" if module ==
                                                "appendix_historical_kernel_weights" else module+"*"))
                      if p.suffix in {".png", ".svg", ".parquet", ".csv"}})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "matplotlib", "pyarrow", "scipy")},
        "command_arguments": sys.argv, "modules": modules,
        "elapsed_seconds": elapsed_seconds,
        "execution": "sequential figure generation from saved fits; no training",
        "bootstrap": {"random_seed": 0, "curve_and_history_draws": 500,
                      "paired_draws": 1000, "default_block_months": 12,
                      "paired_block_sensitivity_months": [6, 12, 24],
                      "refit_models_in_bootstrap": False},
        "inputs": [file_record(p) for p in inputs],
        "source_files": [file_record(p) for p in sources],
        "outputs": [file_record(p) for p in outputs],
    }
    path = figure_dir / "reproduction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Reproduction manifest: {path}")
    return path
