"""Run the frozen new DGP experiment; workers only control execution concurrency."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .model import (calibrate, characteristics, evaluate_policies, fit_path,
                    managed_returns, matern_rff, returns)

ROOT = Path(__file__).resolve().parent


def setup(config, N):
    cal = calibrate(N, config["SR_ann"], config["sigma_eps"], terms=config["fourier_terms"])
    omega, phase = matern_rff(config["D"], config["s"], config["features"],
                             config["ell"], config["feature_seed"])
    return cal, omega, phase


def simulate(config, N, rho, seed, size):
    cal, omega, phase = setup(config, N)
    rng = np.random.default_rng(seed)
    Z = characteristics(size, N, rho, rng)
    R, mu, h = returns(Z, cal, rng, config["fourier_terms"])
    G = managed_returns(Z, R, omega, phase)
    # E4 check, without an N x N matrix.
    from .model import moments
    _, B, u, hh = moments(Z[:1], cal, config["fourier_terms"])
    u, B, hh = u[0], B[0], hh[0]
    residual = ((cal["sigma_eps"]**2 + u @ u) * u
                + B @ (cal["Sigma_factor"] @ (B.T @ u))
                + hh * (hh @ u) - cal["kappa_0"] * u)
    e4 = np.max(np.abs(residual)) / max(np.max(np.abs(cal["kappa_0"] * u)), 1e-30)
    return G, e4


def pilot_job(args):
    config, N, trial = args
    with threadpool_limits(limits=1):
        n = config["pilot_train"]
        G, _ = simulate(config, N, config["fixed_rho"],
                        np.random.SeedSequence([config["pilot_seed"], N, trial]),
                        n + config["pilot_validation"])
        penalties = np.asarray(config["pilot_C_grid"]) * n**(-2 / 3)
        theta, _, _ = fit_path(G[:n], penalties)
        payoff = G[n:] @ theta
        return N, trial, np.mean((1 - payoff)**2, axis=0)


def trial_job(args):
    config, N, rho, trial, constant, cache = args
    dest = Path(cache) / f"N{N}_rho{rho:g}_trial{trial:03d}.npz"
    if dest.exists():
        return str(dest)
    with threadpool_limits(limits=1):
        seed = np.random.SeedSequence([config["training_seed"], N, trial])
        # Same primitive streams across rho; independent across trials and N.
        G, e4 = simulate(config, N, rho, seed, max(config["T_train"]))
        coefficients, complexities = [], []
        for n in config["T_train"]:
            penalties = np.array([constant, 1.0]) * n**(-2 / 3)
            theta, complexity, eig = fit_path(G[:n], penalties)
            coefficients.append(theta)
            complexities.append(complexity)
        lambda_grid = (constant * max(config["T_train"])**(-2 / 3)
                       * np.geomspace(0.01, 100, 65))
        c_grid = np.sum(eig[:, None] / (eig[:, None] + lambda_grid), axis=0)
        temporary = dest.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, theta=np.column_stack(coefficients),
                            complexity=complexities, lambda_grid=lambda_grid,
                            C_grid=c_grid, eigenvalues=eig, e4=e4)
        temporary.replace(dest)
    return str(dest)


def source_hash(config):
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in [ROOT / "model.py", ROOT / "run.py", ROOT / "source/user_simulation.py"]:
        h.update(path.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/paper.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/new_montecarlo")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        from .report import build
        build(args.output)
        return
    import pandas as pd
    config = json.loads(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    signature = source_hash(config)
    marker = cache / "source_hash.txt"
    if marker.exists() and marker.read_text() != signature:
        raise RuntimeError("Configuration/source changed: use a fresh output directory")
    marker.write_text(signature)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = time.time()
    rows = []
    if not (out / "pilot.csv").exists():
        jobs = [(config, N, j) for N in config["N_values"] for j in range(config["pilot_trials"])]
        with ProcessPoolExecutor(args.workers) as pool:
            for N, j, losses in pool.map(pilot_job, jobs):
                rows.extend(dict(N=N, trial=j, C_lambda=C, validation_loss=loss)
                            for C, loss in zip(config["pilot_C_grid"], losses))
        pd.DataFrame(rows).to_csv(out / "pilot.csv", index=False)
    pilot = pd.read_csv(out / "pilot.csv")
    averages = pilot.groupby(["N", "C_lambda"]).validation_loss.mean().reset_index()
    constants = {int(N): float(frame.loc[frame.validation_loss.idxmin(), "C_lambda"])
                 for N, frame in averages.groupby("N")}
    (out / "frozen_constants.json").write_text(json.dumps(constants, indent=2) + "\n")
    print("Frozen independent-pilot constants:", constants, flush=True)
    jobs = [(config, N, rho, j, constants[N], str(cache)) for N, rho in config["cells"]
            for j in range(config["trials"])]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(trial_job, job) for job in jobs]
        for k, future in enumerate(as_completed(futures), 1):
            future.result()
            if k % 10 == 0 or k == len(jobs):
                print(f"Training {k}/{len(jobs)}; elapsed {(time.time()-start)/60:.1f} min", flush=True)
    records, spectra, evaluation_checks = [], [], []
    with threadpool_limits(limits=max(1, min(4, args.workers))):
        for N in config["N_values"]:
            cal, omega, phase = setup(config, N)
            # Independent iid states: rho does not change the stationary marginal.
            rng = np.random.default_rng(np.random.SeedSequence([config["evaluation_seed"], N]))
            Z_eval = rng.uniform(-1, 1, size=(config["evaluation_states"], N, config["D"]))
            cells = [(n, rho) for n, rho in config["cells"] if n == N]
            for _, rho in cells:
                coefficients, identifiers, complexity_all = [], [], []
                for j in range(config["trials"]):
                    with np.load(cache / f"N{N}_rho{rho:g}_trial{j:03d}.npz") as trial:
                        coefficients.append(trial["theta"])
                        complexity_all.append(trial["complexity"])
                        identifiers.append(float(trial["e4"]))
                        spectra.extend(dict(N=N, rho=rho, trial=j, lambda_reg=lam, complexity=C)
                                       for lam, C in zip(trial["lambda_grid"], trial["C_grid"]))
                theta = np.column_stack(coefficients)
                # Split state halves for a numerical integration sensitivity check.
                halves = np.array_split(Z_eval, 2)
                evaluations = [evaluate_policies(theta, z, cal, omega, phase,
                                                 config["fourier_terms"], block=1) for z in halves]
                # Pool raw first and second moments before computing Sharpe.
                mean = (evaluations[0]["mean"] + evaluations[1]["mean"]) / 2
                second = (evaluations[0]["second"] + evaluations[1]["second"]) / 2
                sharpe = np.sqrt(12) * mean / np.sqrt(second - mean**2)
                regret = (evaluations[0]["regret"] + evaluations[1]["regret"]) / 2
                oracle_eval = float(np.mean([e["SR_oracle_eval"][0] for e in evaluations]))
                for j in range(config["trials"]):
                    for k, n in enumerate(config["T_train"]):
                        for r, rule in enumerate(["pilot_frozen", "literal_C1"]):
                            idx = j * len(config["T_train"]) * 2 + k * 2 + r
                            records.append(dict(N=N, rho=rho, trial=j, T_train=n,
                                rule=rule, C_lambda=constants[N] if r == 0 else 1.0,
                                lambda_reg=(constants[N] if r == 0 else 1.0) * n**(-2/3),
                                complexity=complexity_all[j][k, r], SR_hat=sharpe[idx],
                                SR_gap=config["SR_ann"]-sharpe[idx],
                                SR_half1=evaluations[0]["SR_hat"][idx],
                                SR_half2=evaluations[1]["SR_hat"][idx],
                                SR_oracle_eval=oracle_eval, regret=regret[idx],
                                Q_hat=1-2*mean[idx]+second[idx], Q_star=1-cal["p_star"],
                                e4_relative_error=identifiers[j]))
                print(f"Evaluated N={N}, rho={rho:g}", flush=True)
    pd.DataFrame(records).to_csv(out / "trials.csv", index=False)
    pd.DataFrame(spectra).to_csv(out / "complexity_lambda_trials.csv", index=False)
    manifest = dict(source_hash=signature, config=config, constants=constants,
                    elapsed_seconds=time.time()-start, workers=args.workers,
                    python=platform.python_version(), numpy=np.__version__,
                    platform=platform.platform(), completed_trials=len(jobs),
                    evaluation="2048 independent uniform characteristic states; exact conditional innovation moments",
                    original_worktree="Preserved, including untracked simulazioni.py and simulazioni_2.py")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from .report import build
    build(out)


if __name__ == "__main__":
    main()
