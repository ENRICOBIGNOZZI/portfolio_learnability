import numpy as np
import pandas as pd
from adaptive_memory_equities.prequential import choose_methods
from adaptive_memory_equities.metrics import paired_comparisons


def test_new_adaptive_candidate_does_not_erase_baseline_architecture_history():
    keys = ['L1W8|uniform:0:1', 'L2W8|uniform:0:1', 'L1W8|taper:120:1']
    metadata = {k: dict(architecture=k.split('|')[0], family=k.split('|')[1].split(':')[0]) for k in keys}
    config = dict(architectures=[dict(depth=1,width=8),dict(depth=2,width=8)],
                  baseline_architecture='L2W8', memories=dict(taper=[[120,1]]),
                  guard=dict(window=60,replicates=20,block=12,seed=0),soft_temperature=8.)
    # The adaptive candidate has just become admissible, whereas both uniform
    # architectures have 60 genuine, already realized prequential observations.
    past = [dict(experts={keys[0]: .1+j*.0001, keys[1]: -.1},
                 methods={scope+'/full': dict(raw_excess=-.1) for scope in ['L1W8','L2W8','joint']})
            for j in range(60)]
    selections, gates = choose_methods(keys,metadata,past,config)
    assert selections['joint/full'] == {keys[0]:1.}
    assert selections['joint/taper/guarded'] == selections['joint/full']
    assert gates['joint/taper']['reason'] == 'INSUFFICIENT_PREQUENTIAL_HISTORY'


def test_degenerate_sharpe_comparison_is_explicitly_unidentified():
    rng = np.random.default_rng(10)
    wide = pd.DataFrame(dict(base=rng.normal(0,.02,36), constant=np.zeros(36)))
    rows = paired_comparisons(wide,'base',replicates=40)
    loss = next(r for r in rows if r['metric']=='loss')
    sharpe = next(r for r in rows if r['metric']=='sharpe')
    assert np.isfinite(loss['difference'])
    assert sharpe['status'] == 'NOT_IDENTIFIED'
    assert sharpe['reason'] == 'DEGENERATE_VARIANCE'
    assert sharpe['difference'] is None
