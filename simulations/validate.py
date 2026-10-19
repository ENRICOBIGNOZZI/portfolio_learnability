"""Independent accounting checks on the delivered Monte Carlo CSV files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .run import source_hash


def validate(out):
    out = Path(out)
    config = json.loads((out / "config.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    frame = pd.read_csv(out / "trials.csv")
    assert source_hash(config) == manifest["source_hash"]
    assert len(frame) == len(config["cells"]) * config["trials"] * len(config["T_train"]) * 2
    assert set(frame.rule) == {"pilot_frozen", "literal_C1"}
    assert set(frame.T_train) == set(config["T_train"])
    assert set(frame.trial) == set(range(config["trials"]))
    assert not frame.duplicated(["N", "rho", "trial", "T_train", "rule"]).any()
    for (N, rho, rule), group in frame.groupby(["N", "rho", "rule"]):
        assert [int(N), float(rho)] in config["cells"]
        assert len(group) == config["trials"] * len(config["T_train"])
        assert group.C_lambda.nunique() == 1
    np.testing.assert_allclose(frame.lambda_reg, frame.C_lambda * frame.T_train**(-2/3), rtol=1e-12)
    np.testing.assert_allclose(frame.SR_gap, config["SR_ann"] - frame.SR_hat, atol=1e-13)
    pstar = config["SR_ann"]**2 / (12 + config["SR_ann"]**2)
    np.testing.assert_allclose(frame.Q_star, 1-pstar, atol=1e-14)
    assert np.isfinite(frame.select_dtypes("number")).all().all()
    assert frame.e4_relative_error.max() < 1e-10
    assert frame.regret.min() >= -1e-10
    assert (frame.complexity > 0).all()
    assert (frame.complexity <= np.minimum(frame.T_train, config["features"]) + 1e-9).all()
    # Qhat-Qstar and exact conditional regret differ only by the common finite
    # evaluation oracle moment, independent of the learned policy and of rho.
    accounting = {}
    for N, group in frame.groupby("N"):
        residual = group.Q_hat - group.Q_star - group.regret
        assert np.ptp(residual) < 1e-10
        accounting[int(N)] = float(residual.mean())
    spectrum = pd.read_csv(out / "complexity_lambda_trials.csv")
    for key, group in spectrum.groupby(["N", "rho", "trial"]):
        assert len(group) == 65
        ordered = group.sort_values("lambda_reg")
        assert np.all(np.diff(ordered.complexity) < 0)
        assert (ordered.complexity > 0).all()
        assert (ordered.complexity <= config["features"] + 1e-9).all()
    fits = json.loads((out / "rate_fits.json").read_text())
    percentiles=pd.read_csv(out/'percentiles.csv')
    keys=['N','rho','T_train','rule']
    actual=frame.groupby(keys)[['SR_hat','SR_gap','complexity']].quantile(.95,interpolation='linear')
    exported=percentiles.set_index(keys).sort_index()
    for original,column in [('SR_hat','SR_p95'),('SR_gap','gap_p95'),('complexity','complexity_p95')]:
        np.testing.assert_allclose(exported[column],actual[original].sort_index(),rtol=1e-11,atol=1e-13)
    for key in ["SR_T_N", "SR_T_rho", "SR_complexity_N", "C_T"]:
        assert fits[key]["fit_points"] == len(config["T_train"])
        assert fits[key]["all_points_used"]
    files = [p for p in out.rglob("*") if p.is_file() and "cache" not in p.parts
             and p.name not in ["validation.json", "visual_review.json"]]
    report = dict(passed=True, observations=len(frame),
                  completed_replications=len(frame[["N", "rho", "trial"]].drop_duplicates()),
                  expected_replications=config["trials"] * len(config["cells"]),
                  max_e4_relative_error=float(frame.e4_relative_error.max()),
                  evaluation_oracle_moment_offsets=accounting,
                  source_hash_matches=True, monotone_complexity_lambda=True,
                  percentile_plotted=95,percentile_interpolation='linear',
                  scientific_fit_results="Descriptive fits retained, including benchmark mismatches.",
                  files_sha256={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(files)})
    (out / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=2))
    return report


if __name__ == "__main__":
    validate(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "outputs/new_montecarlo")
