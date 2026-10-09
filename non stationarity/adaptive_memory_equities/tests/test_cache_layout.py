"""Optional local-source parity checks for the resource-only cache layout change."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from adaptive_memory_equities.config import PACKAGE, REPO
from adaptive_memory_equities.data_adapter import PanelStore
from adaptive_memory_equities.training import fit
from adaptive_memory_equities.memory import uniform


@pytest.mark.skipif(not (REPO/'data/clean/manifest.json').exists(),reason='Licensed local snapshot absent')
def test_annual_cache_is_bitwise_equal_to_original_prepared_source():
    store=PanelStore('retrospective_complete_payoff')
    source=pd.read_parquet(REPO/'data/clean/jkp_1963.parquet')
    for date in ['1963-01-31','1963-06-30','1963-12-31']:
        p=store.panel(date)
        reference=source.loc[source.eom.eq(pd.Timestamp(date))].sort_values('id')
        np.testing.assert_array_equal(p.Z,reference[store.names].to_numpy())
        np.testing.assert_array_equal(p.forward_excess_returns,reference.r.to_numpy())
        np.testing.assert_array_equal(p.asset_ids,reference.id.to_numpy())
        assert p.Z.flags.c_contiguous


@pytest.mark.skipif(not (REPO/'data/clean/manifest.json').exists(),reason='Licensed local snapshot absent')
def test_storage_order_does_not_change_actual_fit_or_calibration():
    store=PanelStore('retrospective_complete_payoff')
    row=[store.panel(d) for d in pd.date_range('1963-01-31',periods=12,freq='ME')]
    column=[]
    for p in row:
        q=copy.copy(p);q.Z=np.array(p.Z,order='F',copy=True);column.append(q)
    optimizer=dict(steps=2,learning_rate=.01,weight_decay=0.,stock_block=512,batch_dates=0,threads=1)
    a,info_a=fit(row,uniform(12),dict(depth=2,width=32),0,optimizer)
    b,info_b=fit(column,uniform(12),dict(depth=2,width=32),0,optimizer)
    for x,y in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)
    assert info_a['loss_after']==pytest.approx(info_b['loss_after'],abs=1e-12)
