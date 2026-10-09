"""Read-only reuse of frozen JKP source and original safe formation transforms."""
from dataclasses import dataclass
import gc
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .config import REPO, PACKAGE, digest, canonical_hash, write_json
from .calendar import next_month, eligible

# Import the existing loader without modifying it or depending on the caller's cwd.
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))
from data_pipeline import formation_mask, rank_months, META


@dataclass
class Panel:
    formation_date: pd.Timestamp
    asset_ids: np.ndarray
    Z: np.ndarray
    return_realization_date: pd.Timestamp
    forward_excess_returns: np.ndarray
    available_at: pd.Timestamp
    feature_manifest_hash: str = ''
    sample_mode: str = 'formation_audit'

    @property
    def availability_metadata(self):
        return dict(available_at=str(self.available_at), convention='month close, snapshot label',
                    historical_vintage_certified=False)

    @property
    def universe_metadata(self):
        return dict(N_t=len(self.asset_ids), sample_mode=self.sample_mode,
                    formation_only=self.sample_mode == 'formation_audit')


def transform(frame, names, mode):
    frame = frame.loc[formation_mask(frame)].copy()
    if mode == 'retrospective_complete_payoff':
        frame = frame.loc[np.isfinite(frame.ret_exc_lead1m)]
    elif mode != 'formation_audit':
        raise ValueError(mode)
    return rank_months(frame, names).sort_values(['eom', 'id'], kind='stable')


def selection():
    chosen = json.loads((REPO/'characteristic_selection.json').read_text())
    names = chosen['selected']
    dictionary = json.loads((REPO/'characteristics.json').read_text())
    forbidden = {'id', 'permno', 'eom', 'ret_exc_lead1m', 'return_known', 'return_date'}
    if len(set(names)) != len(names) or not set(names) <= set(dictionary) or set(names) & forbidden:
        raise ValueError('Frozen feature allowlist mismatch.')
    if chosen['reference_end'] != '1972-12-31':
        raise ValueError('Unexpected characteristic freeze.')
    return names, canonical_hash(names)


def audit(destination, monitor=None):
    """Recount raw source year by year; exact next-calendar check with a small index."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    raw = REPO/'data/raw'
    manifest = json.loads((raw/'manifest.json').read_text())
    if manifest['return_units'] != 'decimal' or manifest['status'] != 'complete':
        raise ValueError('Raw snapshot not complete/decimal.')
    names, fh = selection()
    rows = []
    matched = discrepancies = recoverable = 0
    rank_error = 0.
    for item in manifest['files']:
        if monitor:
            monitor.check()
        path = raw/item['name']
        if digest(path) != item['sha256']:
            raise ValueError('Raw checksum mismatch: '+path.name)
        f = pd.read_parquet(path, columns=list(dict.fromkeys(META+names+['current_excess_return'])))
        f['eom'] = pd.to_datetime(f.eom)
        if f.duplicated(['id','eom']).any() or f[['id','eom']].isna().any().any():
            raise ValueError('Duplicate/missing raw key.')
        formed = transform(f, names, 'formation_audit')
        if formed.empty:
            continue
        # Current year and next January suffice; calendar merge never uses next available row.
        following = [f[['id','eom','current_excess_return']]]
        year = int(formed.eom.dt.year.iloc[0])
        candidates = [x for x in manifest['files'] if x['name'].startswith(f'jkp_{year+1}')]
        if candidates:
            nxt = pd.read_parquet(raw/candidates[0]['name'], columns=['id','eom','current_excess_return'])
            following.append(nxt.loc[pd.to_datetime(nxt.eom).dt.month.eq(1)])
        current = pd.concat(following).rename(columns={'eom':'return_date'})
        current['return_date'] = pd.to_datetime(current.return_date)
        formed['return_date'] = formed.eom + pd.offsets.MonthEnd(1)
        joined = formed[['id','eom','return_date','ret_exc_lead1m']].merge(
            current, on=['id','return_date'], how='left', validate='one_to_one')
        valid = np.isfinite(joined.ret_exc_lead1m)
        known = np.isfinite(joined.current_excess_return)
        matched += int((valid & known).sum())
        discrepancies += int((valid & known & (abs(joined.ret_exc_lead1m-joined.current_excess_return)>1e-8)).sum())
        recoverable += int((~valid & known).sum())
        # Formation-time invariance checked on every year's complete cross-sections.
        altered = f.copy()
        altered['ret_exc_lead1m'] = np.nan
        alternate = transform(altered, names, 'formation_audit')
        if not np.array_equal(formed.id, alternate.id):
            raise AssertionError('Future-dependent universe.')
        rank_error = max(rank_error, float(np.max(abs(formed[names].to_numpy()-alternate[names].to_numpy()))))
        for date, m in joined.groupby('eom'):
            missing = ~np.isfinite(m.ret_exc_lead1m)
            rows.append(dict(formation_date=str(date.date()), stocks=len(m),
                             missing_payoffs=int(missing.sum()),
                             unresolved=int((missing & ~np.isfinite(m.current_excess_return)).sum())))
        print(f'audit {year}: stocks={len(formed)} missing={int((~valid).sum())}', flush=True)
        del f, formed, alternate, altered, current, following, joined
        gc.collect()
    counts = pd.DataFrame(rows)
    counts.to_csv(destination/'counts.csv', index=False)
    result = dict(status='PAYOFF_COMPLETENESS_FAILED' if counts.unresolved.sum() else 'COMPLETE',
        formation_stock_months=int(counts.stocks.sum()), months=len(counts),
        missing_payoffs=int(counts.missing_payoffs.sum()), unresolved=int(counts.unresolved.sum()),
        months_with_missing=int((counts.unresolved>0).sum()), recoverable= recoverable,
        adjacent_calendar_pairs=matched, adjacent_discrepancies=discrepancies,
        future_perturbation_rank_error=rank_error, feature_count=len(names), feature_manifest_hash=fh,
        raw_manifest_hash=digest(raw/'manifest.json'), verified_raw_files=len(manifest['files']),
        vintage='current retrospective snapshot; historical availability not certified')
    if rank_error or discrepancies:
        raise ValueError('Formation or lead-label audit failed.')
    write_json(destination/'audit.json', result)
    return result


class PanelStore:
    """Annual row-contiguous memory maps; monthly views, never whole-history RAM."""
    def __init__(self, mode, cache_root=None):
        import pyarrow as pa
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        self.mode = mode
        self.names, self.feature_hash = selection()
        self.root = REPO/'data'/('clean' if mode == 'retrospective_complete_payoff' else 'formation_only')
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        source_hash = (self.manifest.get('source_raw_manifest_sha256') if mode == 'retrospective_complete_payoff'
                       else self.manifest['audit']['source_manifest_sha256'])
        if source_hash != digest(REPO/'data/raw/manifest.json'):
            raise ValueError('Prepared panels belong to a different raw snapshot.')
        if mode == 'retrospective_complete_payoff' and self.manifest['characteristics'] != self.names:
            raise ValueError('Clean dictionary differs from frozen allowlist.')
        self.fingerprint = canonical_hash(dict(manifest=digest(self.root/'manifest.json'),
            raw=digest(REPO/'data/raw/manifest.json'), features=self.feature_hash,
            transform=digest(REPO/'data_pipeline.py'), mode=mode, cache_schema=2))
        self.cache = Path(cache_root or PACKAGE/'private_runs/cache')/self.fingerprint
        self.cache.mkdir(parents=True, exist_ok=True)
        self.dates = pd.date_range('1963-01-31','2024-12-31',freq='ME')
        self.verified = set()
        self.verified_cache = set()
        self._open_year = None
        self._arrays = None
        self._offsets = None

    def _year(self, year):
        item = next(x for x in self.manifest['files'] if x['name'] == f'jkp_{year}.parquet')
        source = self.root/item['name']
        if year not in self.verified:
            if digest(source) != item['sha256']:
                raise ValueError('Prepared source checksum mismatch.')
            self.verified.add(year)
        done = self.cache/f'{year}.json'
        if done.exists():
            if year not in self.verified_cache:
                record = json.loads(done.read_text())
                if record['source_hash'] != item['sha256'] or any(
                        digest(self.cache/name) != checksum for name,checksum in record['files'].items()):
                    raise ValueError('Local panel cache checksum mismatch.')
                self.verified_cache.add(year)
            return
        f = pd.read_parquet(source, columns=['id','eom','return_date','r',*self.names])
        if f.duplicated(['id','eom']).any():
            raise ValueError('Duplicate panel keys.')
        if not (pd.to_datetime(f.return_date) == pd.to_datetime(f.eom)+pd.offsets.MonthEnd(1)).all():
            raise ValueError('Lead label shifted twice or calendar misaligned.')
        f = f.sort_values(['eom','id'],kind='stable').reset_index(drop=True)
        z = np.ascontiguousarray(f[self.names].to_numpy(float))
        if not np.isfinite(z).all() or (abs(z) > .50000001).any():
            raise ValueError('Invalid frozen ranks.')
        saved, offsets = {}, {}
        for date, indices in f.groupby('eom',sort=True).indices.items():
            offsets[f'{pd.Timestamp(date):%Y-%m}'] = [int(indices[0]),int(indices[-1])+1]
        arrays = dict(z=z,ids=f.id.to_numpy(np.int64),r=f.r.to_numpy(float))
        for suffix,array in arrays.items():
            p = self.cache/f'{year}-{suffix}.npy'
            with p.with_suffix('.tmp').open('wb') as stream:
                np.save(stream,array,allow_pickle=False)
            p.with_suffix('.tmp').replace(p)
            saved[p.name] = digest(p)
        write_json(done,dict(source_hash=item['sha256'],files=saved,offsets=offsets,layout='C',cache_schema=2))
        self.verified_cache.add(year)

    def panel(self, date):
        date = pd.Timestamp(date)
        self._year(date.year)
        if self._open_year != date.year:
            self._arrays = {s:np.load(self.cache/f'{date.year}-{s}.npy',mmap_mode='r',allow_pickle=False)
                            for s in ['z','ids','r']}
            self._offsets = json.loads((self.cache/f'{date.year}.json').read_text())['offsets']
            self._open_year = date.year
        start,stop = self._offsets[f'{date:%Y-%m}']
        arrays = {k:v[start:stop] for k,v in self._arrays.items()}
        return Panel(date, arrays['ids'], arrays['z'], next_month(date), arrays['r'],
                     next_month(date), self.feature_hash, self.mode)

    def history(self, origin, start='1963-01-31'):
        # This object exposes historical dates only; training cannot ask it for test data.
        dates = [d for d in self.dates if d >= pd.Timestamp(start) and next_month(d) <= pd.Timestamp(origin)]
        return History(self, dates, origin)


class History:
    def __init__(self, store, dates, origin):
        self.store, self.dates, self.origin = store, dates, pd.Timestamp(origin)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        p = self.store.panel(self.dates[index])
        if not eligible(p, self.origin):
            raise ValueError('Unavailable payoff requested by training.')
        return p
