"""Development-only runtime calibration; no evaluation outcomes select budgets."""
import pandas as pd
from .calendar import ages, next_month
from .config import write_json
from .data_adapter import PanelStore
from .memory import uniform
from .training import fit


def benchmark(c,directory,monitor):
    store = PanelStore('retrospective_complete_payoff')
    origin = pd.Timestamp('1973-01-31')
    history = store.history(origin)
    rows = []
    for a in [{'depth':2,'width':32},{'depth':4,'width':64}]:
        optimizer = {**c['optimizer'],'steps':2,'batch_dates':8}
        _,info = fit(history,uniform(len(history)),a,0,optimizer,origin,monitor)
        rows.append(dict(architecture=a,**info))
        print('development benchmark',a,info['seconds'],flush=True)
    result = dict(status='REAL_DEVELOPMENT_BENCHMARK',origin=str(origin.date()),fits=rows,
                  interpretation='Runtime only; no final-period performance observed')
    write_json(directory/'benchmark.json',result)
    return result
