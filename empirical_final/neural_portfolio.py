"""Chronological learned portfolio features with an exact ridge readout.

No individual-stock prediction loss is used. Each gradient is the derivative
of a squared *complete monthly aggregate*, at the current full-history head.
Only private checkpoints contain extractors, stock inputs, or stock weights.
"""
import argparse
import copy
import json
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch import nn
from portfolio import annual_splits, ridge_path, sharpe
from data_pipeline import digest
from empirical_final.run import clean_manifest
from empirical_final.compact_common import (ROOT, OUT, PRIVATE, PROTOCOL, SCALE,
    setup, save, audit, source, account_month, summarize_accounts)

CFG = PROTOCOL['neural']
torch.set_num_threads(1)


class Extractor(nn.Module):
    def __init__(self, dimension=130, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.Sequential(nn.Linear(dimension,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh())

    def forward(self, x):
        h = self.layers(x)
        return torch.cat([torch.ones((len(x),1), dtype=x.dtype, device=x.device),h], dim=1)


class Panels:
    """One immutable mmap per year; bounded resident tensors and no stock subsampling."""
    def __init__(self):
        self.clean = clean_manifest()
        self.names = self.clean['characteristics']
        self.dates = pd.date_range('1963-01-31','2024-12-31',freq='ME')
        self.cache = OrderedDict()
        self.root = PRIVATE/'neural_inputs'
        self.root.mkdir(exist_ok=True)
        self.records = []
        for item in self.clean['files']:
            year = int(item['name'].split('_')[-1].split('.')[0])
            folder = self.root/str(year)
            folder.mkdir(exist_ok=True)
            meta_path = folder/'manifest.json'
            raw = source(ROOT/'data/clean'/item['name'])
            if digest(raw) != item['sha256']:
                raise ValueError('Changed private input')
            identity = dict(clean_sha256=item['sha256'], names=self.names, dtype='float32', order='eom,id')
            if not meta_path.exists():
                frame = pd.read_parquet(raw).sort_values(['eom','id'], kind='stable')
                np.save(folder/'x.npy', frame[self.names].to_numpy(np.float32))
                np.save(folder/'r.npy', frame.r.to_numpy(np.float32))
                np.save(folder/'ids.npy', frame.id.to_numpy(np.int64))
                groups = frame.groupby('eom', sort=True).size()
                offsets = np.r_[0, groups.cumsum().to_numpy()]
                meta = dict(identity=identity, offsets=offsets.tolist(), dates=[str(d.date()) for d in groups.index],
                            files={p.name:digest(p) for p in folder.glob('*.npy')})
                meta_path.write_text(json.dumps(meta, indent=2))
            meta = json.loads(meta_path.read_text())
            if meta['identity'] != identity:
                raise ValueError('Neural panel provenance mismatch')
            for name, expected in meta['files'].items():
                if digest(folder/name) != expected:
                    raise ValueError('Neural mmap changed')
            self.records += [(year, meta['offsets'][j], meta['offsets'][j+1]) for j in range(12)]
        if len(self.records) != 744:
            raise ValueError('Incomplete neural panel')

    def __getitem__(self, index):
        year, start, stop = self.records[index]
        if year not in self.cache:
            folder = self.root/str(year)
            # Copy-on-write mapping: torch gets a writable view without modifying files.
            self.cache[year] = tuple(np.load(folder/f'{n}.npy', mmap_mode='c') for n in ['x','r','ids'])
            if len(self.cache) > 3:
                self.cache.popitem(last=False)
        self.cache.move_to_end(year)
        x,r,ids = self.cache[year]
        return torch.from_numpy(x[start:stop]), torch.from_numpy(r[start:stop]), ids[start:stop]


def monthly_features(model, x, r):
    return torch.mean(model(x)*r[:,None], dim=0)


def managed(model, panels, indices):
    with torch.no_grad():
        return np.array([monthly_features(model, *panels[int(i)][:2]).numpy().astype(float) for i in indices])


def head_scale(g):
    return float(np.sum(g*g)/len(g)/g.shape[1])


def solve_head(g, penalty):
    sigma = g.T@g/len(g)
    mu = g.mean(axis=0)
    a = np.linalg.solve(sigma+penalty*np.eye(g.shape[1]), mu)
    return a, sigma


def extractor_training(panels, indices, seed, budgets):
    """Block-coordinate optimization of the direct maximum-Sharpe criterion.

    At fixed head, uniform complete-month draws have exactly the full-history
    objective gradient in expectation. Full-history ridge refreshes minimize
    the Euclidean head objective. Validation/test observations never enter.
    """
    model = Extractor(len(panels.names), seed)
    initial = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(seed+20261010)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['learning_rate'])
    g = managed(model, panels, indices)
    penalty = head_scale(g)*CFG['extractor_head_penalty_ratio']
    a,_ = solve_head(g, penalty)
    logs, snapshots = [], {}
    nparams = sum(p.numel() for p in model.parameters())
    def record(step):
        objective = float(np.mean((1-g@a)**2)+penalty*(a@a))
        objective += CFG['hidden_penalty']*sum(float((p*p).sum().detach()) for p in model.parameters())/nparams
        logs.append(dict(step=step, objective=objective, raw_Q=float(np.mean((1-g@a)**2)),
                         head_norm=float(np.linalg.norm(a)), penalty=penalty))
    record(0)
    for step in range(1,max(budgets)+1):
        optimizer.zero_grad()
        at = torch.tensor(a, dtype=torch.float32)
        # Backpropagate each complete month separately to bound graph memory.
        chosen = rng.choice(indices, CFG['complete_months_per_step'], replace=True)
        for i in chosen:
            x,r,_ = panels[int(i)]
            monthly = monthly_features(model,x,r)@at
            ((1-monthly)**2/len(chosen)).backward()
        hidden_penalty = CFG['hidden_penalty']*sum((p*p).sum() for p in model.parameters())/nparams
        hidden_penalty.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step % CFG['head_refresh_steps'] == 0 or step in budgets:
            g = managed(model, panels, indices)
            a,_ = solve_head(g, penalty)
            record(step)
        if step in budgets:
            change = np.sqrt(sum(float(((model.state_dict()[k]-initial[k])**2).sum()) for k in initial))
            snapshots[step] = dict(state=copy.deepcopy(model.state_dict()), g=g.copy(),
                                  extractor_penalty=penalty, parameter_change=change, logs=copy.deepcopy(logs))
    return snapshots


def pilot():
    setup()
    started = time.monotonic()
    panels = Panels()
    train = np.arange(120)
    snapshots = extractor_training(panels,train,0,CFG['budgets'])
    model = Extractor()
    model.load_state_dict(snapshots[max(CFG['budgets'])]['state'])
    g = snapshots[max(CFG['budgets'])]['g']
    a,sigma = solve_head(g,head_scale(g)*.01)
    beta,_,c = ridge_path(g,[head_scale(g)*.01])
    np.testing.assert_allclose(a,beta[:,0],rtol=1e-8,atol=1e-8)
    # Finite difference checks the squared full-month aggregation gradient.
    model = model.double()
    x,r,_ = panels[0]; x,r = x.double(),r.double()
    loss = (1-monthly_features(model,x,r)@torch.tensor(a))**2
    loss.backward()
    parameter = next(model.parameters()); analytic = float(parameter.grad[0,0])
    original = float(parameter[0,0].detach()); eps=1e-5
    with torch.no_grad():
        parameter[0,0]=original+eps
        plus=float((1-monthly_features(model,x,r)@torch.tensor(a))**2)
        parameter[0,0]=original-eps
        minus=float((1-monthly_features(model,x,r)@torch.tensor(a))**2)
        parameter[0,0]=original
    error = abs((plus-minus)/(2*eps)-analytic)
    if error > 1e-6:
        raise ValueError('Neural monthly gradient failed')
    elapsed=time.monotonic()-started
    audit('pilot',dict(elapsed_seconds=elapsed, torch=torch.__version__,
        monthly_gradient_error=error, exact_head_max_error=float(np.max(np.abs(a-beta[:,0]))),
        logs=snapshots[max(CFG['budgets'])]['logs'],
        initial_objective=snapshots[32]['logs'][0]['objective'],
        final_objective=snapshots[96]['logs'][-1]['objective'],
        parameter_change=snapshots[96]['parameter_change'],
        runtime_plan='CPU, one numerical thread, three mmap years resident; 47 chronological refits per seed; checkpoints resumable. Pilot timing is conservative only for initial history, later histories are longer.'))
    print(json.dumps(dict(pilot_seconds=elapsed, gradient_error=error,
                         logs=snapshots[96]['logs']),indent=2),flush=True)


def run(seed):
    setup(); started=time.monotonic(); panels=Panels()
    rf=pd.read_csv(source(ROOT/'data/risk_free.csv'),parse_dates=['return_date']).set_index('return_date').rf
    states={s[0]:pd.Series(dtype=float) for s in PROTOCOL['cost_scenarios']}
    previous=pd.Series(dtype=float)
    monthly=[]; annual=[]; spectra=[]; candidates=[]; logs=[]
    root=PRIVATE/f'neural_seed_{seed}';root.mkdir(exist_ok=True)
    identity=dict(protocol_sha256=digest(ROOT/'empirical_final/compact_protocol.json'),
                  code_sha256=digest(__file__), clean_sha256=digest(ROOT/'data/clean/manifest.json'))
    for year,train,val,test in annual_splits(panels.dates):
        checkpoint=root/f'{year}.pt'
        if checkpoint.exists():
            cp=torch.load(checkpoint,weights_only=False)
            if cp['identity']!=identity:
                raise ValueError('Neural checkpoint changed')
        else:
            snapshots=extractor_training(panels,train,seed,CFG['budgets'])
            choices=[]
            for budget,snap in snapshots.items():
                model=Extractor(len(panels.names),seed);model.load_state_dict(snap['state'])
                vg=managed(model,panels,val)
                grid=head_scale(snap['g'])*np.array(CFG['readout_penalty_ratios'])
                beta,_,_=ridge_path(snap['g'],grid)
                loss=np.mean((1-vg@beta)**2,axis=0)
                for j in range(len(grid)):
                    choices.append(dict(budget=budget,candidate=j,ratio=CFG['readout_penalty_ratios'][j],
                                        penalty=float(grid[j]),validation_Q=float(loss[j])))
            best=min(choices,key=lambda r:(r['validation_Q'],r['budget'],r['candidate']))
            h=np.r_[train,val]
            refit=extractor_training(panels,h,seed,[best['budget']])[best['budget']]
            model=Extractor(len(panels.names),seed);model.load_state_dict(refit['state'])
            a,sigma=solve_head(refit['g'],head_scale(refit['g'])*best['ratio'])
            tg=managed(model,panels,test)
            cp=dict(identity=identity,best=best,choices=choices,inner_logs={k:v['logs'] for k,v in snapshots.items()},
                    refit=refit,a=a,sigma=sigma,tg=tg)
            torch.save(cp,checkpoint)
        best=cp['best'];refit=cp['refit'];a=cp['a'];g=refit['g']
        model=Extractor(len(panels.names),seed);model.load_state_dict(refit['state'])
        penalty=head_scale(g)*best['ratio']
        beta,mu,c=ridge_path(g,[penalty])
        np.testing.assert_allclose(a,beta[:,0],rtol=1e-7,atol=1e-7)
        values,vectors=np.linalg.eigh(cp['sigma'])
        values=np.maximum(values,0)[::-1];vectors=vectors[:,::-1]
        coefficients=vectors*((vectors.T@g.mean(axis=0))/(values+penalty))[None,:]
        np.testing.assert_allclose(coefficients.sum(axis=1),a,rtol=1e-7,atol=1e-7)
        residual=float(np.linalg.norm((cp['sigma']+penalty*np.eye(65))@a-g.mean(axis=0))/np.linalg.norm(g.mean(axis=0)))
        if c[0]>min(g.shape)+1e-6:
            raise ValueError('Head complexity exceeds rank')
        annual.append(dict(seed=seed,decision_year=year,budget=best['budget'],ratio=best['ratio'],
            selected_penalty=penalty,validation_Q=best['validation_Q'],C=c[0],rank=int((values>values[0]*1e-12).sum()),
            parameter_change=refit['parameter_change'],head_residual=residual,
            refit_initial_objective=refit['logs'][0]['objective'],refit_final_objective=refit['logs'][-1]['objective'],
            last_known_payoff=str((panels.dates[val[-1]]+pd.offsets.MonthEnd(1)).date()),
            first_test_payoff=str((panels.dates[test[0]]+pd.offsets.MonthEnd(1)).date()),
            lambda_boundary=best['candidate'] in [0,len(CFG['readout_penalty_ratios'])-1]))
        for r in cp['choices']:
            candidates.append(dict(seed=seed,decision_year=year,selected=r==best,**r))
        for stage,ll in list(cp['inner_logs'].items())+[('refit',refit['logs'])]:
            for row in ll:logs.append(dict(seed=seed,decision_year=year,stage=str(stage),**row))
        for j,value in enumerate(values):
            spectra.append(dict(seed=seed,decision_year=year,rank=j+1,eigenvalue=value,
                                normalized_eigenvalue=value/values.sum(),selected_penalty=penalty))
        weight_rows=[]
        for ii,index in enumerate(test):
            x,r,ids=panels[int(index)]
            with torch.no_grad():features=model(x).numpy().astype(float)
            w=SCALE*features@a/len(r)
            date=panels.dates[index];retdate=date+pd.offsets.MonthEnd(1)
            expected=SCALE*float(cp['tg'][ii]@a)
            error=abs(float(w@r.numpy())-expected)
            if error>2e-7:
                raise ValueError('Neural stock weights fail monthly reconstruction')
            target=pd.Series(w,index=ids);returns=pd.Series(r.numpy().astype(float),index=ids)
            for scenario,tb,bb in PROTOCOL['cost_scenarios']:
                step,states[scenario]=account_month(target,returns,float(rf.loc[retdate]),states[scenario],previous,tb,bb)
                monthly.append(dict(policy='neural',seed=seed,scenario=scenario,decision_year=year,
                                    formation_date=date,return_date=retdate,raw_return=float(cp['tg'][ii]@a),
                                    reconstruction_error=error,**step))
            previous=target
            weight_rows.append(pd.DataFrame(dict(id=ids,formation_date=date,weight=w)))
        pd.concat(weight_rows).to_parquet(root/f'weights_{year}.parquet',index=False)
        print(f'Neural seed {seed} {year}: budget={best["budget"]}, C={c[0]:.2f}, elapsed={(time.monotonic()-started)/60:.1f} min',flush=True)
    save(f'neural_monthly_seed_{seed}',monthly);save(f'neural_annual_seed_{seed}',annual)
    save(f'neural_spectra_seed_{seed}',spectra);save(f'neural_validation_seed_{seed}',candidates)
    save(f'neural_optimization_seed_{seed}',logs)
    save(f'neural_performance_seed_{seed}',summarize_accounts(pd.DataFrame(monthly)))
    audit(f'neural_seed_{seed}',dict(elapsed_seconds=time.monotonic()-started,identity=identity,
        torch=torch.__version__,seed=seed,annual_refits=47,monthly_payoffs=564,
        private_checkpoint_hashes={p.name:digest(p) for p in root.glob('*.pt')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    pilot() if args.pilot else run(args.seed)
