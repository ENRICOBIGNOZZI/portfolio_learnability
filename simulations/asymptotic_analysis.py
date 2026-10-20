"""Analysis-only additions to a frozen simulation design."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def analysis_design(config, output):
    """Add declared fit windows without changing data, seeds or primary cutoff."""
    effective=dict(config)
    path=Path(output)/'analysis_spec.json'
    if not path.exists():
        return effective,dict(source='frozen simulation config',
                              sensitivity_T_min=config['sensitivity_T_min'])
    raw=path.read_bytes()
    spec=json.loads(raw)
    if spec['primary_T_train_min']!=config['asymptotic_T_min']:
        raise ValueError('An analysis extension cannot change the primary cutoff')
    cuts=spec['sensitivity_T_min']
    if cuts!=sorted(set(cuts)) or any(type(t) is not int or t<=0 for t in cuts):
        raise ValueError('Analysis cutoffs must be distinct positive ordered integers')
    if not set(config['sensitivity_T_min'])<=set(cuts):
        raise ValueError('All original sensitivity cutoffs must be retained')
    if not set(cuts)<=set(config['T_train_reference']):
        raise ValueError('Analysis cutoffs must be actual reference training dates')
    effective['sensitivity_T_min']=cuts
    return effective,dict(source=path.name,sha256=hashlib.sha256(raw).hexdigest(),
                          primary_T_train_min=spec['primary_T_train_min'],
                          sensitivity_T_min=cuts,reason=spec['reason'])
