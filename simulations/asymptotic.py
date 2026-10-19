"""Long-horizon, prefix-stable simulation of the unchanged user economy.

Nyström approximates the same nonperiodic Matérn-3/2 kernel as the supplied RFF.
Only sufficient statistics and fitted policies are cached, never stock panels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from .model import calibrate, matern_rff, phi, moments, returns, fit_path

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def exp_accelerator():
    import sys
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        function = lib.vvexp
        function.argtypes = [ctypes.c_void_p] * 3
        return lib, function
    return None


def matern_kernel(z, centers, ell=1.):
    r = cdist(z, centers)
    r *= -np.sqrt(3) / ell
    acceleration = exp_accelerator()
    if acceleration:
        e = np.empty_like(r)
        size = ctypes.c_int(r.size)
        acceleration[1](e.ctypes.data, r.ctypes.data, ctypes.byref(size))
    else:
        e = np.exp(r)
    r *= -1
    r += 1
    r *= e
    return r


class Basis:
    def __init__(self, kind, rank, config):
        self.kind, self.rank, self.ell = kind, rank, config["ell"]
        if kind == "nystrom":
            points = qmc.Sobol(3, scramble=True, seed=config["center_seed"]).random_base2(
                int(np.ceil(np.log2(rank))))[:rank]
            self.centers = 2 * points - 1
            gram = matern_kernel(self.centers, self.centers, self.ell)
            gram.flat[::rank+1] += config["kernel_jitter"]
            self.L = linalg.cholesky(gram, lower=True, check_finite=False)
        elif kind == "rff":
            self.omega, self.phase = matern_rff(M=rank, ell=self.ell,
                                               seed=config["feature_seed"])
            self.L = None
        else:
            raise ValueError(kind)

    def raw(self, z):
        return matern_kernel(z, self.centers, self.ell) if self.kind == "nystrom" else phi(z, self.omega, self.phase)

    def whiten_moments(self, mean, second):
        if self.L is None:
            return mean, second
        mean = linalg.solve_triangular(self.L, mean, lower=True, check_finite=False)
        a = linalg.solve_triangular(self.L, second, lower=True, check_finite=False)
        second = linalg.solve_triangular(self.L, a.T, lower=True, check_finite=False).T
        return mean, (second+second.T)/2

    def raw_coefficients(self, theta):
        if self.L is None:
            return theta
        return linalg.solve_triangular(self.L.T, theta, lower=False, check_finite=False)

    def whiten_rows(self, rows):
        return rows if self.L is None else linalg.solve_triangular(
            self.L, rows.T, lower=True, check_finite=False).T


def draw_blocks(N, rho, trial, seed, total, block=128):
    """Each complete date block has its own streams; larger horizons share prefixes."""
    previous = None
    scale = np.sqrt(1-rho*rho)
    cal = calibrate(N)
    for index, start in enumerate(range(0, total, block)):
        rng_z = np.random.default_rng(np.random.SeedSequence([seed, N, trial, index, 0]))
        rng_r = np.random.default_rng(np.random.SeedSequence([seed, N, trial, index, 1]))
        state = rng_z.normal(size=(block, N, 3))
        for j in range(block):
            if previous is not None:
                state[j] = rho*previous + scale*state[j]
            previous = state[j]
        z = 2*ndtr(state)-1
        r, _, _ = returns(z, cal, rng_r)
        take = min(block, total-start)
        yield start, z[:take], r[:take]


def managed_block(basis, z, r):
    g = np.empty((len(z), basis.rank))
    for j in range(len(z)):
        g[j] = r[j] @ basis.raw(z[j]) / z.shape[1]
    return g


def sufficient_path(config, N, rho, trial, seed, times, basis):
    times = sorted(times)
    total = times[-1]
    sum_g = np.zeros(basis.rank)
    sum_gg = np.zeros((basis.rank, basis.rank))
    for start, z, r in draw_blocks(N, rho, trial, seed, total, config["stream_block"]):
        g = managed_block(basis, z, r)
        cursor = 0
        boundaries = [t for t in times if start < t <= start+len(g)]
        for boundary in sorted(set(boundaries+[start+len(g)])):
            end = boundary-start
            piece = g[cursor:end]
            sum_g += piece.sum(axis=0)
            sum_gg += piece.T @ piece
            cursor = end
            if boundary in times:
                yield boundary, sum_g/boundary, sum_gg/boundary


def fit_statistics(basis, mean, second, lambdas):
    m, s = basis.whiten_moments(mean, second)
    eig, u = linalg.eigh(s, check_finite=False)
    tolerance = max(abs(eig[-1]), 1e-30)*1e-9
    if eig[0] < -tolerance:
        raise ArithmeticError(f"Negative training spectrum: {eig[0]}")
    eig = np.maximum(eig, 0)
    theta = u @ ((u.T @ m)[:, None] / (eig[:, None]+np.asarray(lambdas)))
    c = np.sum(eig[:, None]/(eig[:, None]+np.asarray(lambdas)), axis=0)
    return basis.raw_coefficients(theta), c, eig


def scientific_hash(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name in ["model.py", "asymptotic.py", "source/user_simulation.py"]:
        digest.update((ROOT/name).read_bytes())
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def pilot_job(args):
    config, N, trial, cache = args
    path = Path(cache)/f"pilot_N{N}_trial{trial:03d}.npz"
    if path.exists():
        return str(path)
    with threadpool_limits(limits=1):
        basis = Basis("nystrom", config["pilot_rank"], config)
        n, validation = config["pilot_train"], config["pilot_validation"]
        pieces = [managed_block(basis,z,r) for _,z,r in draw_blocks(N,.5,trial,config["pilot_seed"],n+validation,config["stream_block"])]
        g = np.concatenate(pieces)
        theta, _, _ = fit_path(basis.whiten_rows(g[:n]),
                               np.asarray(config["pilot_C_grid"])*n**(-2/3))
        coeff = basis.raw_coefficients(theta)
        loss = np.mean((1-g[n:]@coeff)**2, axis=0)
        atomic_npz(path, loss=loss)
    return str(path)


def training_job(args):
    config, N, rho, trial, seed, times, kind, rank, constant, cache, tag = args
    path = Path(cache)/f"{tag}_{kind}{rank}_N{N}_rho{rho:g}_trial{trial:03d}.npz"
    if path.exists():
        return str(path)
    with threadpool_limits(limits=1):
        basis = Basis(kind, rank, config)
        policies, complexities = [], []
        for n, m, s in sufficient_path(config,N,rho,trial,seed,times,basis):
            coeff, c, eig = fit_statistics(basis,m,s,constant*n**(-2/3)*np.array(config["lambda_multipliers"]))
            policies.append(coeff)
            complexities.append(c)
            if n == config["spectrum_T_train"]:
                spectrum = eig.copy()
        if config["spectrum_T_train"] not in times:
            spectrum = eig.copy()
        atomic_npz(path, theta=np.column_stack(policies), complexity=complexities,
                   eigenvalues=spectrum, T_train=times)
    return str(path)


def evaluation_job(args):
    config, N, kind, rank, cache = args
    path = Path(cache)/f"evaluation_{kind}{rank}_N{N}.npz"
    if path.exists():
        return str(path)
    with threadpool_limits(limits=1):
        basis = Basis(kind,rank,config)
        cal = calibrate(N)
        rng = np.random.default_rng(np.random.SeedSequence([config["evaluation_seed"],N]))
        means, seconds, oracles = [], [], []
        count = config["evaluation_states"]//2
        for half in range(2):
            m_sum = np.zeros(rank)
            s_sum = np.zeros((rank,rank))
            p_sum = 0.
            for start in range(0,count,config["evaluation_block"]):
                b = min(count-start,config["evaluation_block"])
                z = rng.uniform(-1,1,(b,N,3))
                _, B, mu, h = moments(z,cal)
                kernel = basis.raw(z.reshape(-1,3))
                k = kernel.reshape(b,N,rank)
                m = np.einsum("bnr,bn->br",k,mu)/N
                hh = np.einsum("bnr,bn->br",k,h)/N
                exposures = np.einsum("bnr,bnk->brk",k,B)/N
                exposures = exposures @ linalg.cholesky(cal["Sigma_factor"],lower=True)
                ef = exposures.transpose(0,2,1).reshape(-1,rank)
                m_sum += m.sum(axis=0)
                s_sum += m.T@m + hh.T@hh + ef.T@ef + (cal["sigma_eps"]/N)**2 * (kernel.T@kernel)
                p_sum += np.sum(mu**2)/cal["kappa_0"]
            means.append(m_sum/count)
            seconds.append(s_sum/count)
            oracles.append(p_sum/count)
        atomic_npz(path, means=means, seconds=seconds, oracles=oracles,
                   evaluation_states=config["evaluation_states"])
    return str(path)


def evaluate_coefficients(coeff, evaluation, pstar=3/7):
    """Raw moments and an unbiased two-fold oracle control variate.

    For each fold, the control coefficient is estimated on the other independent
    fold. E[p(Z)]=pstar is known analytically; training never sees these moments.
    """
    ms = evaluation["means"]
    ss = evaluation["seconds"]
    ps = evaluation["oracles"]
    raw_m = ms.mean(axis=0)
    raw_s = ss.mean(axis=0)
    cv_m, cv_s = [], []
    for j in range(2):
        control = ms[1-j]/ps[1-j]
        cv_m.append(ms[j]-(ps[j]-pstar)*control)
        cv_s.append(ss[j]-(ps[j]-pstar)*np.outer(control,control))
    mean = np.mean(cv_m,axis=0)@coeff
    second = np.einsum("il,ij,jl->l",coeff,np.mean(cv_s,axis=0),coeff,optimize=True)
    raw_mean = raw_m@coeff
    raw_second = np.einsum("il,ij,jl->l",coeff,raw_s,coeff,optimize=True)
    variance = second-mean**2
    if np.any(variance<=0):
        raise ArithmeticError("Nonpositive evaluation variance")
    sharpe = np.sqrt(12)*mean/np.sqrt(variance)
    raw_sharpe = np.sqrt(12)*raw_mean/np.sqrt(raw_second-raw_mean**2)
    return dict(SR_hat=sharpe,SR_raw=raw_sharpe,SR_gap=3-sharpe,
                Q_hat=1-2*mean+second,regret=second-2*mean+pstar,
                payoff_mean=mean,payoff_second=second)


def execute(jobs, function, workers, label):
    start=time.time()
    with ProcessPoolExecutor(workers) as pool:
        futures=[pool.submit(function,job) for job in jobs]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%5==0 or i==len(jobs):
                print(f"{label}: {i}/{len(jobs)}; elapsed {(time.time()-start)/60:.1f} min",flush=True)


def prepare(config,out):
    out.mkdir(parents=True,exist_ok=True)
    cache=out/"cache"
    cache.mkdir(exist_ok=True)
    marker=cache/"source_hash.txt"
    signature=scientific_hash(config)
    if marker.exists() and marker.read_text()!=signature:
        raise RuntimeError("Scientific source or design changed; use a new output directory")
    marker.write_text(signature)
    (out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    return cache


def pilot(config,out,workers):
    cache=prepare(config,out)
    jobs=[(config,N,j,str(cache)) for N in config["N_values"] for j in range(config["pilot_trials"])]
    execute(jobs,pilot_job,workers,"Independent calibration pilot")
    rows=[]
    for _,N,j,_ in jobs:
        with np.load(cache/f"pilot_N{N}_trial{j:03d}.npz") as a:
            rows.extend(dict(N=N,trial=j,C_lambda=C,validation_loss=loss)
                        for C,loss in zip(config["pilot_C_grid"],a["loss"]))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"pilot.csv",index=False)
    avg=frame.groupby(["N","C_lambda"]).validation_loss.mean().reset_index()
    constants={str(N):float(f.loc[f.validation_loss.idxmin(),"C_lambda"]) for N,f in avg.groupby("N")}
    (out/"frozen_constants.json").write_text(json.dumps(constants,indent=2)+"\n")
    print("FROZEN CONSTANTS",constants,flush=True)


def collect(config,out,tag,settings,times_by_cell,trials,seed):
    cache=out/"cache"
    constants=json.loads((out/"frozen_constants.json").read_text())
    rows=[]
    spectral=[]
    for N,rho,kind,rank in settings:
        times=times_by_cell[(N,rho)]
        with np.load(cache/f"evaluation_{kind}{rank}_N{N}.npz") as e:
            evaluation={k:e[k] for k in e.files}
        for trial in range(trials):
            with np.load(cache/f"{tag}_{kind}{rank}_N{N}_rho{rho:g}_trial{trial:03d}.npz") as result:
                metrics=evaluate_coefficients(result["theta"],evaluation)
                for i,n in enumerate(times):
                    for j,mult in enumerate(config["lambda_multipliers"]):
                        index=i*len(config["lambda_multipliers"])+j
                        row=dict(N=N,rho=rho,trial=trial,representation=kind,rank=rank,
                                 T_train=n,T_pilot_train=config["pilot_train"],
                                 T_pilot_validation=config["pilot_validation"],
                                 T_evaluation=config["evaluation_states"],lambda_multiplier=mult,
                                 C_lambda=constants[str(N)],lambda_reg=constants[str(N)]*mult*n**(-2/3),
                                 complexity=float(result["complexity"][i,j]))
                        row.update({key:float(value[index]) for key,value in metrics.items()})
                        rows.append(row)
                if tag=="confirmation":
                    grid=constants[str(N)]*config["spectrum_T_train"]**(-2/3)*np.geomspace(.01,100,65)
                    eig=result["eigenvalues"]
                    cs=np.sum(eig[:,None]/(eig[:,None]+grid),axis=0)
                    spectral.extend(dict(N=N,rho=rho,trial=trial,T_train=config["spectrum_T_train"],lambda_reg=lam,complexity=c)
                                    for lam,c in zip(grid,cs))
    pd.DataFrame(rows).to_csv(out/(tag+"_trials.csv"),index=False)
    if spectral:
        pd.DataFrame(spectral).to_csv(out/"complexity_lambda_trials.csv",index=False)
    return pd.DataFrame(rows)


def fidelity(config,out,workers):
    cache=prepare(config,out)
    constants=json.loads((out/"frozen_constants.json").read_text())
    N=config["fixed_N"]
    rho=config["fixed_rho"]
    settings=[(N,rho,"nystrom",r) for r in config["candidate_ranks"]]+[(N,rho,"rff",1000)]
    times=config["fidelity_T_train"]
    jobs=[(config,N,rho,j,config["fidelity_seed"],times,kind,rank,constants[str(N)],str(cache),"fidelity")
          for N,rho,kind,rank in settings for j in range(config["fidelity_trials"])]
    execute(jobs,training_job,workers,"Paired numerical-fidelity pilot")
    execute([(config,N,kind,rank,str(cache)) for N,rho,kind,rank in settings],evaluation_job,min(2,workers),"Independent evaluation operators")
    frame=collect(config,out,"fidelity",settings,{(N,rho):times},config["fidelity_trials"],config["fidelity_seed"])
    avg=frame[frame.lambda_multiplier==1].groupby(["representation","rank","T_train"])[["SR_gap","complexity"]].mean()
    reference=avg.loc[("nystrom",max(config["candidate_ranks"]))]
    checks=[]
    for rank in config["candidate_ranks"]:
        candidate=avg.loc[("nystrom",rank)]
        late=reference.index>=config["asymptotic_T_min"]
        c_error=float(np.max(np.abs(candidate.complexity[late]/reference.complexity[late]-1)))
        sr_error=float(np.max(np.abs(candidate.SR_gap[late]-reference.SR_gap[late])))
        checks.append(dict(rank=rank,max_relative_C_difference=c_error,max_absolute_SR_difference=sr_error,
                           passes=bool(c_error<=config["rank_C_tolerance"] and sr_error<=config["rank_SR_tolerance"])))
    selected=next(v["rank"] for v in checks if v["passes"])
    selection=dict(selected_rank=selected,reference_rank=max(config["candidate_ranks"]),checks=checks,
                   rule="Smallest rank meeting prespecified numerical-error tolerances; slopes never enter selection.")
    (out/"rank_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    print("NUMERICAL RANK SELECTION",json.dumps(selection),flush=True)


def confirmation(config,out,workers):
    cache=prepare(config,out)
    selection=json.loads((out/"rank_selection.json").read_text())
    rank=selection["selected_rank"]
    constants=json.loads((out/"frozen_constants.json").read_text())
    settings=[(N,rho,"nystrom",rank) for N,rho in config["cells"]]
    times_by_cell={(N,rho):(config["T_train_reference"] if N==config["fixed_N"] and rho==config["fixed_rho"] else config["T_train"]) for N,rho,_,_ in settings}
    execute([(config,N,"nystrom",rank,str(cache)) for N in config["N_values"]],evaluation_job,min(2,workers),"Confirmation evaluation operators")
    jobs=[(config,N,rho,j,config["confirmation_seed"],times_by_cell[N,rho],kind,rank,constants[str(N)],str(cache),"confirmation")
          for N,rho,kind,rank in settings for j in range(config["trials"])]
    execute(jobs,training_job,workers,"Long-horizon confirmation")
    collect(config,out,"confirmation",settings,times_by_cell,config["trials"],config["confirmation_seed"])
    (out/"completed.json").write_text(json.dumps(dict(completed_replications=len(jobs),rank=rank,source_hash=scientific_hash(config)),indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"config/asymptotic.json")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/asymptotic_v2")
    parser.add_argument("--stage",choices=["pilot","fidelity","confirmation","all"],default="all")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    for stage,fn in [("pilot",pilot),("fidelity",fidelity),("confirmation",confirmation)]:
        if args.stage in [stage,"all"]:
            fn(config,args.output,args.workers)


if __name__=="__main__":
    main()
