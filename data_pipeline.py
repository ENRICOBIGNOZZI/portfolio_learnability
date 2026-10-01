"""Frozen characteristic cleaning, with no future-return filter on holdings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

START = pd.Timestamp('1963-01-01')
END = pd.Timestamp('2024-12-31')
REFERENCE_SELECTION_END = pd.Timestamp('2023-12-31')
INITIAL_TRAIN_END = pd.Timestamp('1972-12-31')
FLAGS = ['common', 'primary_sec', 'obs_main', 'exch_main']
META = ['permno', 'id', 'eom', 'excntry', 'size_grp', 'me', 'ret_exc_lead1m',
        'crsp_shrcd', 'crsp_exchcd', *FLAGS]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str))


def formation_mask(frame):
    dates = pd.to_datetime(frame['eom'])
    mask = dates.between(START, END)
    mask &= frame['excntry'].eq('USA') & frame['id'].notna() & frame['id'].between(1, 99999)
    mask &= frame['crsp_shrcd'].isin([10, 11, 12])
    mask &= frame['crsp_exchcd'].isin([1, 2, 3])
    mask &= frame['size_grp'].notna() & frame['size_grp'].ne('nano')
    for flag in FLAGS:
        mask &= frame[flag].eq(1)
    return mask.fillna(False)


def characteristic_selection(frame, candidates, count=130):
    """Small-fixture API; production accumulates the same counts by year."""
    known = frame.loc[formation_mask(frame) &
                      pd.to_datetime(frame['eom']).le(REFERENCE_SELECTION_END)]
    if known.empty:
        raise ValueError('No reference-sample observations.')
    clean = known[candidates].apply(pd.to_numeric, errors='coerce')
    clean = clean.replace([np.inf, -np.inf], np.nan)
    coverage = clean.notna().sum()
    return choose_names(coverage, len(known), count)


def choose_names(coverage, observations, count=130):
    if observations <= 0 or len(coverage) < count:
        raise ValueError('Insufficient candidate characteristics or reference observations.')
    table = pd.DataFrame({'characteristic':coverage.index,
                          'missing_share':1-coverage.to_numpy()/observations})
    table = table.sort_values(['missing_share', 'characteristic'], kind='stable')
    chosen = table.head(count)['characteristic'].tolist()
    if table.head(count)['missing_share'].ge(1).any():
        raise ValueError('Fewer than 130 characteristics are observed in the reference sample.')
    # Fixed alphabetical coordinate order is independent of later coverage.
    return sorted(chosen), table


def rank_months(frame, names):
    """Average ranks on observed values; zero is neutral, not always a median."""
    result = frame.copy()
    result[names] = result[names].apply(pd.to_numeric, errors='coerce')
    result[names] = result[names].replace([np.inf, -np.inf], np.nan)
    result = result.loc[result[names].isna().mean(axis=1).le(0.30)].copy()
    grouped = result.groupby('eom', sort=False)[names]
    ranks = grouped.rank(method='average', na_option='keep')
    counts = grouped.transform('count')
    result[names] = ((ranks-1)/(counts-1)-0.5).mask(counts <= 1).fillna(0.0)
    return result


def sample_initial(panels, max_points=1000, seed=0):
    if not panels:
        raise ValueError('Empty initial sample.')
    rng = np.random.default_rng(seed)
    per_month = int(np.ceil(max_points/len(panels)))
    sample = np.vstack([x[rng.choice(len(x), min(per_month, len(x)), replace=False)]
                        for x in panels])
    if len(sample) > max_points:
        sample = sample[rng.choice(len(sample), max_points, replace=False)]
    return sample


def prepare(raw_dir, output_dir):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    raw_manifest = json.loads((raw_dir/'manifest.json').read_text())
    protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())
    if raw_manifest.get('return_units') != 'decimal':
        raise ValueError('Raw snapshot must document decimal return units.')
    if raw_manifest['status'] != 'complete':
        raise ValueError('Raw download is not complete.')
    candidates = raw_manifest['characteristics']
    pinned_candidates = json.loads(Path(__file__).with_name('characteristics.json').read_text())
    if set(candidates) != set(pinned_candidates):
        raise ValueError('Raw candidate names differ from the pinned JKP dictionary.')
    if len(candidates) != 153 or len(set(candidates)) != 153:
        raise ValueError('Require the pinned 153-variable JKP dictionary.')
    raw_files = [raw_dir/item['name'] for item in raw_manifest['files']]
    if len(raw_files) != len(set(raw_files)):
        raise ValueError('Duplicate raw files.')
    seen_keys = set()
    import pyarrow.parquet as pq
    for item, file in zip(raw_manifest['files'], raw_files, strict=True):
        schema = set(pq.ParquetFile(file).schema_arrow.names)
        required = set(META + candidates + ['current_excess_return', 'current_total_return'])
        if required - schema:
            raise ValueError(f'Incomplete raw schema {file.name}: {sorted(required-schema)}')
        keys = pd.read_parquet(file, columns=['id','eom'])
        if keys.isna().any().any() or keys.duplicated().any():
            raise ValueError('Missing or duplicated raw id/eom.')
        pairs = set(zip(keys.id, pd.to_datetime(keys.eom)))
        if seen_keys.intersection(pairs):
            raise ValueError('Duplicate security-month across raw files.')
        seen_keys.update(pairs)
        if digest(file) != item['sha256']:
            raise ValueError('Raw snapshot checksum mismatch: '+file.name)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory; no old clean cache is reused.')
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = pd.Series(0, index=candidates, dtype='int64')
    n_calibration = 0
    for path in raw_files:
        period = path.stem.rsplit('_', 1)[1]
        year = int(period.split('-')[0])
        if year > REFERENCE_SELECTION_END.year:
            continue
        frame = pd.read_parquet(path, columns=[*META, *candidates])
        frame['eom'] = pd.to_datetime(frame['eom'])
        frame = frame.loc[formation_mask(frame) & frame.eom.le(REFERENCE_SELECTION_END)]
        values = frame[candidates].replace([np.inf, -np.inf], np.nan)
        coverage += values.notna().sum()
        n_calibration += len(frame)
    names, coverage_table = choose_names(coverage, n_calibration)
    coverage_table.to_csv(output_dir/'characteristic_coverage.csv', index=False)
    write_json(output_dir/'characteristic_provenance.json', {
        **protocol['characteristic_selection'], 'candidates':candidates, 'selected':names,
        'coverage_observations':n_calibration,
        'coverage':coverage_table.to_dict(orient='records')})
    write_json(output_dir/'characteristics.json', names)

    # Recovery uses observed next-calendar-month returns, never eligibility.
    return_history = pd.concat([pd.read_parquet(p, columns=[
        'id', 'eom', 'current_excess_return', 'current_total_return']) for p in raw_files], ignore_index=True)
    return_history['eom'] = pd.to_datetime(return_history['eom'])
    if return_history.duplicated(['id', 'eom']).any():
        raise ValueError('Duplicate raw security-month returns; do not silently aggregate.')
    return_history = return_history.rename(columns={'eom':'return_date',
        'current_excess_return':'next_current_excess_return',
        'current_total_return':'next_current_total_return'})
    return_history['next_observation_present'] = True
    counts, missing, files, initial = [], [], [], []
    n_recovered = 0
    for path in raw_files:
        frame = pd.read_parquet(path, columns=[*META, *names,
            'current_excess_return', 'current_total_return']).rename(columns={
                'current_excess_return':'formation_current_excess_return',
                'current_total_return':'formation_current_total_return'})
        frame['eom'] = pd.to_datetime(frame['eom'])
        frame = frame.loc[formation_mask(frame)].copy()
        n_universe = len(frame)
        frame = rank_months(frame, names)
        if frame.empty:
            continue
        if frame.duplicated(['id', 'eom']).any():
            raise ValueError('Duplicate formation security-month.')
        frame = frame.sort_values(['eom', 'id'], kind='stable')
        frame['return_date'] = frame.eom + pd.offsets.MonthEnd(1)
        frame = frame.merge(return_history, on=['id', 'return_date'], how='left',
                            validate='many_to_one', sort=False)
        frame['r'] = frame.ret_exc_lead1m.where(np.isfinite(frame.ret_exc_lead1m))
        recover = frame.r.isna() & np.isfinite(frame.next_current_excess_return)
        frame.loc[recover, 'r'] = frame.loc[recover, 'next_current_excess_return']
        n_recovered += int(recover.sum())
        frame['return_known'] = np.isfinite(frame.r)
        unresolved = frame.loc[~frame.return_known, ['id','permno','eom','return_date',
            'ret_exc_lead1m','formation_current_excess_return','formation_current_total_return',
            'next_current_excess_return','next_current_total_return',
            'next_observation_present']].rename(columns={'eom':'formation_date'})
        unresolved['reason_unresolved'] = np.where(
            unresolved.next_observation_present.eq(True),
            'Next calendar month exists but its JKP excess return is missing',
            'No next-calendar-month security observation in the same raw snapshot')
        missing.append(unresolved)
        for date, month in frame.groupby('eom', sort=True):
            counts.append({'formation_date':date, 'stocks':len(month),
                           'unknown_returns':int((~month.return_known).sum())})
            if date <= INITIAL_TRAIN_END:
                initial.append(month[names].to_numpy(float))
        result_path = output_dir/path.name
        frame[['id','eom','return_date','me','r','return_known',*names]].to_parquet(
            result_path, index=False, compression='zstd')
        files.append({'name':result_path.name, 'rows':len(frame), 'sha256':digest(result_path)})
        print(f'Prepared {path.stem}: universe={n_universe:,}, retained={len(frame):,}', flush=True)
    if len(initial) != 120:
        raise ValueError('Initial training must contain 120 formation months.')
    sample = sample_initial(initial)
    np.save(output_dir/'initial_sample.npy', sample)
    missing_table = pd.concat(missing, ignore_index=True)
    missing_table.to_csv(output_dir/'unresolved_returns.csv', index=False)
    pd.DataFrame(counts).to_csv(output_dir/'universe_counts.csv', index=False)
    manifest = {'status':'complete' if missing_table.empty else 'unresolved_returns',
        'cleaning':'AIPT Section 2.5 stock-characteristic cleaning; 2024 is an extension outside feature selection',
        'source_raw_manifest_sha256':digest(raw_dir/'manifest.json'),
        'raw_manifest':raw_manifest, 'protocol':protocol,
        'protocol_sha256':digest(Path(__file__).with_name('protocol.json')),
        'characteristic_provenance':json.loads((output_dir/'characteristic_provenance.json').read_text()),
        'characteristics':names, 'feature_count':len(names),
        'coverage_start':'1963-01-01','coverage_end':'2023-12-31',
        'coverage_observations':n_calibration, 'row_missing_limit':0.30,
        'rank':'(average_rank-1)/(observed_count-1)-0.5',
        'imputation':'zero after ranking; neutral value, not guaranteed median with ties',
        'recovered_returns':n_recovered, 'unresolved_returns':len(missing_table),
        'unresolved_policy':'retain formation stocks, prohibit training/report until reconciled',
        'files':files, 'sample_sha256':digest(output_dir/'initial_sample.npy'),
        'formation_start':'1963-01-31','formation_end':'2024-12-31',
        'average_stocks_per_month':float(pd.DataFrame(counts).stocks.mean()),
        'raw_rows':sum(f['rows'] for f in raw_manifest['files'])}
    write_json(output_dir/'manifest.json', manifest)
    print('Unresolved forward returns:', len(missing_table), flush=True)
    if not missing_table.empty:
        raise ValueError('Unresolved payoffs: inspect unresolved_returns.csv. No fit is permitted.')
    return manifest


def load_panels(clean_dir):
    root = Path(clean_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['unresolved_returns']:
        raise ValueError('Unresolved returns remain. Reconcile the raw source and rebuild; '
                         'do not drop securities or replace unknown returns by zero.')
    names = manifest['characteristics']
    if len(names) != 130 or len(set(names)) != 130:
        raise ValueError('Exactly 130 distinct predictors required.')
    panels = []
    for item in manifest['files']:
        if digest(root/item['name']) != item['sha256']:
            raise ValueError('Clean panel checksum mismatch.')
        frame = pd.read_parquet(root/item['name'])
        x = frame[names].to_numpy(float)
        if not np.isfinite(x).all() or np.max(np.abs(x)) > .5:
            raise ValueError('Invalid ranked characteristics.')
        if not np.isfinite(frame.r).all() or not frame.return_known.all():
            raise ValueError('Unresolved payoff cannot enter a fit.')
        if frame.duplicated(['id','eom']).any():
            raise ValueError('Duplicate clean security-month.')
        for date, g in frame.groupby('eom', sort=True):
            g = g.sort_values('id', kind='stable')
            panels.append({'date':pd.Timestamp(date), 'return_date':g.return_date.iloc[0],
                           'ids':g.id.to_numpy(), 'x':g[names].to_numpy(float),
                           'r':g.r.to_numpy(float)})
    panels.sort(key=lambda p:p['date'])
    actual = pd.DatetimeIndex([p['date'] for p in panels])
    expected = pd.date_range('1963-01-31', '2024-12-31', freq='ME')
    if not actual.equals(expected):
        raise ValueError('The formation panel must contain every month, 1963--2024.')
    return panels, manifest
