"""Persistent characteristic-factor economy; exact population risk, no test simulation.

Compute and render are separate so public figures can be rebuilt from aggregate outputs.
Private replication checkpoints make the full Monte Carlo resumable.
"""
from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import lfilter
from scipy.stats import linregress
from data_pipeline import write_json, digest

T_GRID = (60, 90, 120, 180, 240, 360, 540, 720, 1080, 1440)
SEED = 20261003
RHO_F = .30
RHO_Z = .90
C_MU = .0004
SIGNAL = .16
SIGMA_E = .15
LAMBDA_COUNT = 360
BOOTSTRAPS = 500
CONFIGS = [('NL', 1.5, 2000, 500), ('NL', 1.25, 2000, 200),
           ('NL', 2., 2000, 200), ('L', 1.5, 2000, 200),
           ('NL', 1.5, 1000, 200), ('NL', 1.5, 4000, 200),
           ('R1', 1.5, 2000, 500), ('R1', 1.5, 1000, 200),
           ('R1', 1.5, 4000, 200)]
ECONOMY_CODES = {'NL': 0, 'L': 1, 'R1': 2}


def population(j, b, economy='NL'):
    if economy not in ECONOMY_CODES:
        raise ValueError(f'Unknown economy: {economy}')
    index = np.arange(1, j+1, dtype=float)
    mu = C_MU * index**(-b)
    shape = 1 / index
    if economy == 'R1':
        shape = 1 / (np.sqrt(index) * np.log1p(index))
    if economy == 'L':
        shape[1:] = 0
    scale = np.sqrt(SIGNAL / np.dot(mu, shape**2))
    theta = scale * shape
    return mu, theta, mu * theta, float(scale)


def factor_path(rng, length, mu, theta, rho=RHO_F):
    """Stationary Gaussian AR(1); diagonal minus rank-one covariance square root."""
    h = np.sqrt(mu)*theta
    q = np.dot(h, h)
    if not 0 < q < 1:
        raise ValueError('Factor covariance is not positive definite.')
    z = rng.standard_normal((length, len(mu)))
    z -= np.outer(z@h, h)/(1+np.sqrt(1-q))
    z *= np.sqrt(mu)
    # First innovation is the stationary initial draw, subsequent ones are AR shocks.
    z[1:] *= np.sqrt(1-rho*rho)
    return lfilter([1.], [1., -rho], z, axis=0) + mu*theta


def path_estimates(f, penalties, mu, theta, validation=True):
    """Dual eigendecomposition, then exact primal regret along the entire ridge path."""
    t = len(f)
    if t > f.shape[1]:
        # Primal decomposition avoids numerical null-space activation when T > J.
        eig,u=eigh(f.T@f,check_finite=False,driver='evd')
        eig=np.maximum(eig,0)
        estimate=u@((u.T@f.sum(axis=0))[:,None]/(eig[:,None]+t*penalties))
    else:
        eig, u = eigh(f@f.T, check_finite=False, driver='evd')
        if eig.min() < -1e-10*eig.max():
            raise ValueError('Indefinite second-moment Gram matrix.')
        eig = np.maximum(eig, 0)
        coef = (u.T@np.ones(t))[:, None]/(eig[:, None]+t*penalties)
        estimate = (f.T@u)@coef
    regret = np.sum(mu[:, None]*(estimate-theta[:, None])**2, axis=0)
    complexity = np.sum(eig[:, None]/(eig[:, None]+t*penalties), axis=0)
    if not validation:
        return regret, complexity, estimate
    v = min(60, t//3)
    inner = f[:-v]
    if len(inner)>f.shape[1]:
        d,w=eigh(inner.T@inner,check_finite=False,driver='evd')
        d=np.maximum(d,0)
        a=(w.T@inner.sum(axis=0))[:,None]/(d[:,None]+len(inner)*penalties)
        predicted=(f[-v:]@w)@a
    else:
        d, w = eigh(inner@inner.T, check_finite=False, driver='evd')
        d = np.maximum(d, 0)
        a = (w.T@np.ones(t-v))[:, None]/(d[:, None]+(t-v)*penalties)
        predicted = ((f[-v:]@inner.T)@w)@a
    losses = np.mean((1-predicted)**2, axis=0)
    chosen = np.flatnonzero(losses == losses.min())[-1]
    return regret, complexity, int(chosen)


def stock_audit(j, b, economy, months=3, n=5000):
    if not j < n:
        raise ValueError('DCT audit requires J < N.')
    rng = np.random.default_rng(SEED+round(b*100)+j+ECONOMY_CODES[economy])
    mu, theta, mean, scale = population(j, b, economy)
    factors = factor_path(rng, months, mu, theta)
    latent = rng.normal(size=n)
    records = []
    for month in range(months):
        if month:
            latent = RHO_Z*latent+np.sqrt(1-RHO_Z**2)*rng.normal(size=n)
        rank = np.argsort(np.argsort(latent, kind='stable'), kind='stable')+1
        characteristic = (rank-.5)/n
        phi = np.sqrt(2)*np.cos(np.pi*np.outer(characteristic, np.arange(1,j+1)))
        gram = phi.T@phi/n
        orth = float(np.max(np.abs(gram-np.eye(j))))
        noise = rng.normal(size=n)
        residual = SIGMA_E*(noise-phi@(phi.T@noise/n))
        residual_error = float(np.linalg.norm(phi.T@residual)/n)
        returns = phi@factors[month]+residual
        errors = []
        for _ in range(5):
            policy = rng.normal(size=j)
            errors.append(abs((phi@policy)@returns/n-policy@factors[month]))
        records.append(dict(month=month, orthogonality_error=orth,
            residual_projection_error=residual_error, payoff_error=float(max(errors)),
            stock_residual_std=float(residual.std()), stock_return_std=float(returns.std())))
    smallest = eigh(np.diag(mu)-np.outer(mean,mean), eigvals_only=True,
                     subset_by_index=[0,0], check_finite=False)[0]
    assert all(max(r['orthogonality_error'],r['residual_projection_error'],r['payoff_error'])<1e-10 for r in records)
    assert smallest>0
    return dict(N=n,J=j,b=b,a=.5 if economy=='R1' else 1.,
        log_power=1 if economy=='R1' else 0,
        target_shape='1/(sqrt(j)*log(j+1))' if economy=='R1' else ('1/j' if economy=='NL' else 'first coordinate'),
        economy=economy,rho_Z=RHO_Z,rho_F=RHO_F,
        sigma_e=SIGMA_E,c_mu=C_MU,c_theta=scale,min_eigenvalue_V_F=float(smallest),
        theta_S_theta=float(np.dot(mu,theta**2)),months=records,passed=True)


def long_path_audit(j,b,economy,length=200000):
    """Stream a single stationary path; all coordinate means/diagonals, 16x16 block."""
    rng=np.random.default_rng(SEED+991+j+round(100*b)+ECONOMY_CODES[economy])
    mu,theta,mean,_=population(j,b,economy)
    h=np.sqrt(mu)*theta; q=h@h
    prev=None; sums=np.zeros(j); squares=np.zeros(j); cross=np.zeros((16,16))
    lag=0.; lagden=0.; count=0
    for start in range(0,length,2048):
        m=min(2048,length-start)
        z=rng.normal(size=(m,j));z-=np.outer(z@h,h)/(1+np.sqrt(1-q))
        if prev is None:
            z[1:]*=np.sqrt(1-RHO_F**2)
        else:
            z*=np.sqrt(1-RHO_F**2);z[0]+=RHO_F*prev
        x=lfilter([1.],[1.,-RHO_F],z,axis=0)
        if prev is not None:
            lag+=float(prev@x[0]);lagden+=float(prev@prev)
        lag+=float(np.sum(x[:-1]*x[1:]));lagden+=float(np.sum(x[:-1]**2))
        prev=x[-1].copy()
        y=x+h
        sums+=y.sum(axis=0);squares+=np.sum(y*y,axis=0);cross+=y[:,:16].T@y[:,:16]
        count+=m
    mean_scaled=sums/count; diag_scaled=squares/count
    # Marginal mean MCSE adjusted for exactly known scalar AR persistence.
    mcse=np.sqrt((1-h*h)/count*(1+RHO_F)/(1-RHO_F))
    mean_z=float(np.max(np.abs(mean_scaled-h)/mcse))
    diagerr=float(np.max(np.abs(diag_scaled-1)))
    blockerr=float(np.max(np.abs(cross/count-np.eye(16))))
    ar=lag/lagden
    passed=mean_z<6.5 and diagerr<.035 and blockerr<.025 and abs(ar-RHO_F)<.005
    assert passed, (mean_z,diagerr,blockerr,ar)
    return dict(months=count,max_mean_abs_z=mean_z,max_second_moment_relative_error=diagerr,
        whitened_second_moment_block_max_error=blockerr,estimated_ar=ar,
        means_first_8=(mean_scaled[:8]*np.sqrt(mu[:8])).tolist(),
        target_means_first_8=mean[:8].tolist(),second_moments_first_8=(diag_scaled[:8]*mu[:8]).tolist(),
        target_second_moments_first_8=mu[:8].tolist(),passed=passed)


def slopes(oracle, bootstrap=None):
    output=[]
    t=oracle['T'].to_numpy()
    subsets={'full':np.arange(len(t)), 'upper_half':np.arange(len(t)//2,len(t)),
             'largest_four':np.arange(len(t)-4,len(t))}
    b=float(oracle.b.iloc[0])
    for subset,ix in subsets.items():
        for metric,theory in [('lambda_oracle',-b/(b+1)),('C_oracle',1/(b+1)),('regret_oracle',-b/(b+1))]:
            fit=linregress(np.log(t[ix]),np.log(oracle[metric].to_numpy()[ix]))
            row=dict(subset=subset,metric=metric,T_min=int(t[ix[0]]),T_max=int(t[ix[-1]]),
                slope=fit.slope,ols_se=fit.stderr,r_squared=fit.rvalue**2,theory_r1=theory)
            if bootstrap is not None:
                x=np.log(t[ix]);weights=(x-x.mean())/np.sum((x-x.mean())**2)
                boot=np.log(bootstrap[metric][:,ix])@weights
                row.update(bootstrap_se=float(boot.std(ddof=1)),bootstrap_low=float(np.quantile(boot,.025)),bootstrap_high=float(np.quantile(boot,.975)))
            output.append(row)
    return pd.DataFrame(output)


def replication(job):
    economy,b,j,r,penalties=job
    mu,theta,_,_=population(j,b,economy)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,round(b*100),j,ECONOMY_CODES[economy],r]))
    f=factor_path(rng,max(T_GRID),mu,theta)
    risk=[];empirical=[];chosen=[]
    for t in T_GRID:
        a,c,k=path_estimates(f[:t],penalties,mu,theta)
        risk.append(a);empirical.append(c);chosen.append(k)
    return r,np.array(risk),np.array(empirical),np.array(chosen)


def run_config(economy,b,j,reps,cache,workers=2):
    name=f'{economy}_b{round(b*100):03d}_J{j}_R{reps}'
    cache.mkdir(parents=True,exist_ok=True)
    checkpoint=cache/(name+'.npz')
    mu,theta,mean,_=population(j,b,economy)
    penalties=np.geomspace(mu[-1]*1e-5,mu[0]*1e3,LAMBDA_COUNT)
    for extension in range(5):
        token=dict(economy=economy,b=b,J=j,R=reps,seed=SEED,T=list(T_GRID),lambda_values=penalties.tolist())
        meta=cache/(name+'.json')
        if checkpoint.exists() and meta.exists() and json.loads(meta.read_text())['design']==token:
            loaded=np.load(checkpoint);risks=loaded['risks'];empirical=loaded['empirical'];choices=loaded['choices'];done=int(loaded['done'])
            elapsed=float(json.loads(meta.read_text())['elapsed_seconds'])
        else:
            risks=np.full((reps,len(T_GRID),len(penalties)),np.nan);empirical=np.full_like(risks,np.nan)
            choices=np.full((reps,len(T_GRID)),-1,dtype=int);done=0;elapsed=0.
        start=time.perf_counter()
        # Independent replicate seeds; shared nested T prefixes preserve paired bootstrap.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=((economy,b,j,r,penalties) for r in range(done,reps))
            for r,risk,emp,chosen in pool.map(replication,jobs):
                risks[r]=risk;empirical[r]=emp;choices[r]=chosen
                if (r+1)%25==0 or r+1==reps:
                    duration=elapsed+time.perf_counter()-start
                    np.savez_compressed(checkpoint,risks=risks,empirical=empirical,choices=choices,done=r+1)
                    write_json(meta,dict(design=token,elapsed_seconds=duration))
                    print(name,f'{r+1}/{reps}',f'{duration:.1f}s',flush=True)
        meanrisk=risks.mean(axis=0);oi=meanrisk.argmin(axis=1)
        if np.all((oi>0)&(oi<len(penalties)-1)):
            break
        lo=penalties[0]/100 if (oi==0).any() else penalties[0]
        hi=penalties[-1]*100 if (oi==len(penalties)-1).any() else penalties[-1]
        penalties=np.geomspace(lo,hi,LAMBDA_COUNT)
        print('Extending oracle grid',name,lo,hi,flush=True)
    else:
        raise RuntimeError('Oracle remains on grid boundary.')
    popc=np.sum(mu[:,None]/(mu[:,None]+penalties),axis=0)
    ridge_target=mean[:,None]/(mu[:,None]+penalties)
    bias=np.sum(mu[:,None]*(ridge_target-theta[:,None])**2,axis=0)
    rng=np.random.default_rng(SEED+337)
    bootmean=np.empty((BOOTSTRAPS,len(T_GRID),len(penalties)))
    # Resample whole independent paths, keeping all T and lambdas paired.
    weights=rng.multinomial(reps,np.full(reps,1/reps),size=BOOTSTRAPS)/reps
    bootmean[:]=np.einsum('br,rtl->btl',weights,risks,optimize=True)
    bi=bootmean.argmin(axis=2);bt=np.arange(len(T_GRID))[None,:]
    bootstrap={'lambda_oracle':penalties[bi],'C_oracle':popc[bi], 'regret_oracle':bootmean[np.arange(BOOTSTRAPS)[:,None],bt,bi]}
    surface=[];oracle=[];validation=[]
    for k,t in enumerate(T_GRID):
        choice=choices[:,k];valrisk=risks[np.arange(reps),k,choice];idx=oi[k]
        common=dict(economy=economy,b=b,J=j,R=reps,T=t)
        surface.append(pd.DataFrame({**common,'lambda':penalties,'population_C':popc,'population_C_over_T':popc/t,
            'mean_regret':meanrisk[k],'regret_mcse':risks[:,k].std(axis=0,ddof=1)/np.sqrt(reps),
            'mean_empirical_C':empirical[:,k].mean(axis=0),'population_ridge_regret':bias}))
        row={**common,'lambda_oracle':penalties[idx],'C_oracle':popc[idx], 'C_oracle_over_T':popc[idx]/t,
            'regret_oracle':meanrisk[k,idx],'regret_mcse':risks[:,k,idx].std(ddof=1)/np.sqrt(reps),'oracle_grid_index':int(idx)}
        for metric,arr in bootstrap.items():
            row[metric+'_ci_low'],row[metric+'_ci_high']=np.quantile(arr[:,k],[.025,.975])
        oracle.append(row)
        validation.append({**common,'median_lambda_validation':float(np.median(penalties[choice])),
            'median_population_C_over_T':float(np.median(popc[choice]/t)),
            'mean_validation_regret':float(valrisk.mean()),'validation_regret_mcse':float(valrisk.std(ddof=1)/np.sqrt(reps)),
            'oracle_regret':meanrisk[k,idx],'validation_oracle_ratio':float(valrisk.mean()/meanrisk[k,idx]),
            'fraction_validation_grid_boundary':float(np.mean((choice==0)|(choice==len(penalties)-1)))})
    oracle=pd.DataFrame(oracle)
    rate=slopes(oracle,bootstrap).assign(economy=economy,b=b,J=j,R=reps)
    return pd.concat(surface),oracle,pd.DataFrame(validation),rate


def compute(args):
    started=time.perf_counter();out=Path(args.output)/'simulations';out.mkdir(parents=True,exist_ok=True)
    audits=[]
    for economy,b,j,reps in CONFIGS:
        name=f'{economy}_b{round(b*100):03d}_J{j}'
        auditfile=Path(args.cache)/(name+'_audit.json');auditfile.parent.mkdir(parents=True,exist_ok=True)
        if auditfile.exists():audit=json.loads(auditfile.read_text())
        else:
            print('Stock and long-path audit',name,flush=True)
            audit=stock_audit(j,b,economy);audit['long_path']=long_path_audit(j,b,economy)
            write_json(auditfile,audit)
        assert audit['passed'] and audit['long_path']['passed']
        audits.append(audit)
    write_json(out/'characteristic_factor_dgp_audit.json',audits)
    collected=[[],[],[],[]]
    for config in CONFIGS:
        for dest,frame in zip(collected,run_config(*config,Path(args.cache),args.workers)):dest.append(frame)
    surface,oracle,validation,rate=[pd.concat(x,ignore_index=True) for x in collected]
    surface.to_parquet(out/'characteristic_factor_full_surface.parquet',index=False)
    oracle.to_csv(out/'characteristic_factor_oracle_by_T.csv',index=False)
    validation.to_csv(out/'characteristic_factor_validation_by_T.csv',index=False)
    rate.to_csv(out/'characteristic_factor_rate_slopes.csv',index=False)
    robust=rate[rate.economy.isin(['NL','R1'])&(rate.b==1.5)].copy()
    robust.to_csv(out/'characteristic_factor_finite_J_robustness.csv',index=False)
    write_json(out/'characteristic_factor_run.json',dict(seed=SEED,configs=CONFIGS,T_grid=T_GRID,
        positive_lambda_count=LAMBDA_COUNT,bootstrap_replicates=BOOTSTRAPS,headline_subset='upper_half',headline_economy='R1',
        target_shapes={'R1':'1/(sqrt(j)*log(j+1))','NL':'1/j','L':'first coordinate'},
        elapsed_this_invocation_seconds=time.perf_counter()-started,source_sha256=digest(__file__),
        workers=args.workers,checkpoint_compute_seconds=sum(json.loads(p.read_text())['elapsed_seconds'] for p in Path(args.cache).glob('*_R*.json'))))
    print('Monte Carlo complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--cache',default='results/characteristic_factor')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();compute(args)
