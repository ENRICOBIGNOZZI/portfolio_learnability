"""Shared paths, public aggregate export, and provenance for compact Section VI."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import digest
from empirical_final.run import ROOT, bind, read_json

OUT = ROOT / 'outputs/empirical_compact_20261010'
PRIVATE = ROOT / 'results/empirical_compact_20261010'
PROTOCOL_PATH = ROOT / 'empirical_final/compact_protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
SCALE = float(PROTOCOL['scale'].split(', ')[-1].split(';')[0])


def setup():
    for p in [OUT/'tables', OUT/'audit', OUT/'figures', OUT/'publication', PRIVATE]:
        p.mkdir(parents=True, exist_ok=True)
    frozen = OUT/'audit/protocol.json'
    if frozen.exists() and frozen.read_bytes() != PROTOCOL_PATH.read_bytes():
        raise ValueError('Frozen protocol changed: use a new version')
    frozen.write_bytes(PROTOCOL_PATH.read_bytes())


def save(name, frame):
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    frame.to_csv(OUT/'tables'/f'{name}.csv', index=False, float_format='%.13g')
    return frame


def audit(name, value):
    (OUT/'audit'/f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def source(path):
    return bind(path)


def verify_audit_inputs(path):
    meta = read_json(path)
    for name, expected in meta.get('inputs', {}).items():
        bind(ROOT/name, expected)
    for name, expected in meta.get('shared_sources', {}).items():
        bind(ROOT/name, expected)
    return meta


def generated_weight_cache(tag, path):
    """Reuse only this revision's hash-frozen interrupted stock computations.

    Callers additionally verify current bank scores and all realized managed
    payoffs. These private position files are never included in public output.
    """
    manifest=OUT/'audit'/f'{tag}_generated_weight_cache.json'
    if not manifest.exists():return None
    meta=json.loads(manifest.read_text())
    if meta['protocol_sha256']!=digest(PROTOCOL_PATH) or meta['clean_manifest_sha256']!=digest(ROOT/'data/clean/manifest.json'):
        raise ValueError('Generated weight cache protocol/data changed')
    expected=meta['files'].get(path.name)
    if expected is None:return None
    bind(path,expected)
    return pd.read_parquet(path)


def account_month(target, returns, rf, state, previous, trade_bps=0, borrow_bps=0):
    from empirical_final.core import accounting_step
    ids = target.index.union(state.index)
    aligned = target.reindex(ids, fill_value=0).to_numpy()
    step = accounting_step(aligned, returns.reindex(ids).to_numpy(), rf,
                           state.reindex(ids, fill_value=0).to_numpy(), trade_bps/1e4, borrow_bps/1e4)
    drift = pd.Series(step.pop('next_drift'), index=ids)
    drift = drift[aligned != 0]
    step.update(gross_exposure=float(target.abs().sum()), net_exposure=float(target.sum()),
                target_turnover=float(target.subtract(previous, fill_value=0).abs().sum()))
    return step, drift


def summarize_accounts(monthly, keys=('policy','scenario')):
    from empirical_final.core import performance
    rows = []
    for key, f in monthly.groupby(list(keys), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(keys, key)) | dict(months=len(f), first_return=f.return_date.min(),
            last_return=f.return_date.max(), mean_gross=float(f.gross_exposure.mean()),
            annual_turnover=float(12*f.turnover.mean()), annual_target_turnover=float(12*f.target_turnover.mean()),
            annual_trading_fees=float(12*f.trading_fee.mean()), annual_borrow_fees=float(12*f.borrowing_fee.mean()),
            **performance(f.excess_return, f.total_return)))
    return pd.DataFrame(rows)
