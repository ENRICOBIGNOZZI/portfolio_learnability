"""Small independent stationary/break portfolio controls, separate from market evidence."""
import numpy as np
import pandas as pd
from .calendar import next_month, ages
from .config import write_json, provenance
from .data_adapter import Panel
from .training import fit
from .portfolio import holdings, payoff, ensemble
from .memory import Memory
from .guard import select_guard
from .metrics import summary


def run_synthetic(directory, monitor=None):
    if (directory/'source.json').exists():
        raise FileExistsError('Use a new synthetic run directory.')
    rows, outcomes, fit_count = [], [], 0
    optimizer = dict(steps=5,learning_rate=.01,weight_decay=0.,stock_block=64,batch_dates=8,threads=1)
    gate_config = dict(window=60,block=12,replicates=2000,seed=20261003)
    rules = [Memory('uniform'),Memory('theory',10,1),Memory('taper',120,1)]
    write_json(directory/'source.json',provenance(dict(
        experiment='SYNTHETIC_MECHANISM_CONTROL',optimizer=optimizer,guard=gate_config,
        seeds=[8342,8343],neural_seeds=[0,1],architecture=dict(depth=2,width=8),
        memories=[r.__dict__ for r in rules],months=240,stocks=24,features=3,
        first_decision=60,refit_interval=12,break_time=150,
        scenario_pairing='Same innovations within each stationary/break seed pair',
        note='Mechanism stress test: synthetic memories are not subject to the equity n_eff>=36 filter')))
    for scenario in ['stationary','break']:
        for replication in [0,1]:
            rng = np.random.default_rng(8342+replication)
            data = []
            dates = pd.date_range('2000-01-31',periods=240,freq='ME')
            for t,date in enumerate(dates):
                z = rng.uniform(-.5,.5,(24,3))
                mu = -.04 if scenario == 'break' and t>=150 else .04
                r = z[:,0]*(mu+rng.normal(0,.05))+rng.normal(0,.03,24)
                data.append(Panel(date,np.arange(24),z,next_month(date),r,next_month(date)))
            history = []
            for t in range(60,240):
                if monitor:
                    monitor.check()
                if t%12==0:
                    age,span = ages(dates[t],[p.return_realization_date for p in data[:t]],next_month(dates[0]))
                    bank=[]
                    for rule in rules:
                        w=rule.weights(age,span)
                        pair=[fit(data[:t],w,dict(depth=2,width=8),s,optimizer,dates[t],monitor)[0] for s in [0,1]]
                        fit_count += len(pair)
                        bank.append(pair)
                all_weights=[ensemble([holdings(m,data[t].Z)[1] for m in pair]) for pair in bank]
                old=np.array(history)
                gate=select_guard(old[:,0] if len(old) else np.array([]),
                    old[:,1:3] if len(old) else np.empty((0,2)),gate_config)
                chosen=gate['index']+1 if gate['active'] else 0
                p=[payoff(w,data[t].forward_excess_returns) for w in all_weights]
                history.append(p)
                rows.append(dict(scenario=scenario,replication=replication,t=t,full=p[0],theory=p[1],taper=p[2],
                                 guarded=p[chosen],active=gate['active'],choice=chosen))
            part=pd.DataFrame(rows)
            part=part[(part.scenario==scenario)&(part.replication==replication)]
            activation_after=part[(part.t>=150)&part.active]
            new_activation = part.active & ~part.active.shift(fill_value=False)
            new_after = part[(part.t>=150)&new_activation]
            pre = part.loc[part.t<150]
            post = part.loc[part.t>=150]
            outcomes.append(dict(scenario=scenario,replication=replication,months=len(part),
                guard_activation=float(part.active.mean()),switches=int(part.choice.ne(part.choice.shift()).iloc[1:].sum()),
                active_just_before_break=bool(pre.active.iloc[-1]) if scenario=='break' else None,
                first_active_month_after_break=int(activation_after.t.iloc[0]-150) if len(activation_after) and scenario=='break' else None,
                first_new_activation_after_break=int(new_after.t.iloc[0]-150) if len(new_after) and scenario=='break' else None,
                pre_break_activation=float(pre.active.mean()) if scenario=='break' else None,
                post_break_activation=float(post.active.mean()) if scenario=='break' else None,
                post_break_loss_penalty=float(np.mean((1-post.guarded)**2-(1-post.full)**2)) if scenario=='break' else None,
                stationary_or_break_loss_penalty=float(np.mean((1-part.guarded)**2-(1-part.full)**2)),
                full=summary(part.full),guarded=summary(part.guarded)))
            print('synthetic',scenario,replication,'completed',flush=True)
    pd.DataFrame(rows).to_csv(directory/'synthetic_monthly.csv',index=False)
    result=dict(status='SYNTHETIC_RUN_COMPLETED',independent_seed_blocks=2,scenario_streams=4,
                pairing='Stationary and break share innovations within a replication',fits=fit_count,
                interpretation='Small mechanism controls, not JKP evidence or a calibrated coverage guarantee',results=outcomes)
    write_json(directory/'synthetic.json',result)
    return result
