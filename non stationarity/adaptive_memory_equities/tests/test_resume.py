import json
import numpy as np
import pandas as pd
from adaptive_memory_equities.config import PACKAGE, load_config
from adaptive_memory_equities.calendar import next_month
from adaptive_memory_equities.data_adapter import Panel, History
from adaptive_memory_equities.prequential import run


class FixtureStore:
    mode='retrospective_complete_payoff'
    fingerprint='synthetic-resume-fixture'
    feature_hash='fixture'
    names=['a','b','c']

    def __init__(self):
        self.dates=pd.date_range('2000-01-31','2005-12-31',freq='ME')
        rng=np.random.default_rng(12)
        self.panels={d:Panel(d,np.arange(4),rng.normal(size=(4,3)),next_month(d),
            rng.normal(.02,.1,4),next_month(d)) for d in self.dates}

    def panel(self,date):
        return self.panels[pd.Timestamp(date)]

    def history(self,origin,start):
        return History(self,[d for d in self.dates if next_month(d)<=origin],origin)


def test_resume_matches_uninterrupted_and_decisions_are_immutable(tmp_path):
    c=load_config(PACKAGE/'configs/smoke.yaml')
    c.update(history_start='2000-01-31',formation_start='2004-01-31',formation_end='2005-01-31')
    c['architectures']=[dict(depth=2,width=8)]
    c['baseline_architecture']='L2W8'
    c['optimizer']={**c['optimizer'],'steps':1,'batch_dates':2}
    c['memories']=dict(theory=[[.1,1]],taper=[[240,1]])
    store=FixtureStore()
    uninterrupted=tmp_path/'whole'; resumed=tmp_path/'resumed'
    run(c,uninterrupted,store=store)
    run(c,resumed,stop_after=5,store=store)
    run(c,resumed,resume=True,store=store)
    for p in (uninterrupted/'events').glob('*.json'):
        # Identical deterministic events include hashes of actual saved holdings.
        a=json.loads(p.read_text()); b=json.loads((resumed/'events'/p.name).read_text())
        assert a==b
    before={p.name:p.read_bytes() for p in (resumed/'decisions').glob('*.json')}
    run(c,resumed,resume=True,store=store)
    assert before=={p.name:p.read_bytes() for p in (resumed/'decisions').glob('*.json')}
