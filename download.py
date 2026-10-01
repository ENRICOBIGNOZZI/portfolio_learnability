"""One WRDS connection and one SELECT; no authentication retries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(out, names_file):
    import psycopg2
    names = json.loads(names_file.read_text())
    if len(names) != 153 or len(set(names)) != 153:
        raise ValueError('Expected the 153 published JKP characteristic names.')
    if any(not n.replace('_', '').isalnum() for n in names):
        raise ValueError('Invalid column identifier.')
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out/'manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        for f in manifest['files']:
            if sha256(out/f['name']) != f['sha256']:
                raise ValueError('Cached raw data checksum mismatch.')
        print('Existing complete raw cache reused. WRDS connections: 0.', flush=True)
        return
    username = os.environ.get('WRDS_USERNAME')
    password = os.environ.get('WRDS_PASSWORD')
    if not username or not password:
        raise RuntimeError('WRDS_USERNAME and WRDS_PASSWORD secrets are required.')
    attempt = {'connection_attempts': 1, 'select_calls': 0, 'status': 'connecting'}
    attempt_file = out/'connection_attempt.json'
    with attempt_file.open('x') as stream:
        json.dump(attempt, stream)
    columns = ['id','eom','excntry','gvkey','permno','size_grp','me',
               'ret_exc_lead1m','common','exch_main','primary_sec','obs_main',
               'crsp_shrcd','crsp_exchcd'] + names
    sql = ('SELECT '+', '.join('g."'+n+'"' for n in columns)+
           ', (to_jsonb(g)->>\'ret_exc\')::double precision AS current_excess_return '+
           'FROM contrib.global_factor g '+
           "WHERE excntry = 'USA' AND id <= 99999 "+
           "AND eom >= '1963-01-01' AND eom < '2025-02-01' ORDER BY eom, id")
    (out/'query.sql').write_text(sql)
    conn = None
    files, parts = [], []
    year_now, total = None, 0
    started = time.monotonic()
    def flush(year, buffers):
        data = pd.concat(buffers, ignore_index=True)
        p = out/f'jkp_{year}.parquet'
        data.to_parquet(p, index=False, compression='zstd')
        files.append({'name':p.name, 'rows':len(data), 'sha256':sha256(p)})
        print(f'Raw {year}: {len(data):,} rows saved.', flush=True)
    try:
        print('WRDS: single authorized connection attempt; no retries.', flush=True)
        conn = psycopg2.connect(user=username, password=password,
            host='wrds-pgdata.wharton.upenn.edu', port=9737, dbname='wrds',
            sslmode='require', connect_timeout=90,
            application_name='portfolio_paper_single_request')
        with conn.cursor(name='paper_stream') as cursor:
            cursor.itersize = 10000
            attempt['select_calls'] = 1
            cursor.execute(sql)
            output_columns = columns + ['current_excess_return']
            while True:
                records = cursor.fetchmany(10000)
                if not records:
                    break
                frame = pd.DataFrame.from_records(records, columns=output_columns)
                frame['eom'] = pd.to_datetime(frame['eom'])
                for col in names + ['me', 'ret_exc_lead1m', 'current_excess_return']:
                    frame[col] = pd.to_numeric(frame[col], errors='coerce').astype(float)
                for col in ['id','permno','crsp_shrcd','crsp_exchcd',
                            'common','exch_main','primary_sec','obs_main']:
                    frame[col] = pd.to_numeric(frame[col], errors='raise').astype('Int64')
                for col in ['excntry','gvkey','size_grp']:
                    frame[col] = frame[col].astype('string')
                for year, block in frame.groupby(frame.eom.dt.year, sort=True):
                    year = int(year)
                    if year_now is not None and year != year_now:
                        flush(year_now, parts)
                        parts = []
                    year_now = year
                    parts.append(block.copy())
                total += len(frame)
        if parts:
            flush(year_now, parts)
        if total == 0:
            raise RuntimeError('WRDS returned no rows.')
        manifest = {'status':'complete','source':'WRDS contrib.global_factor',
                    'characteristics':names, 'files':files, 'rows':total,
                    'git_sha':os.environ.get('GITHUB_SHA'),
                    'elapsed_seconds':time.monotonic()-started,
                    'connection_attempts':1,'select_calls':1,
                    'query_sha256':sha256(out/'query.sql')}
        manifest_file.write_text(json.dumps(manifest, indent=2))
        attempt['status'] = 'complete'
    except Exception as error:
        attempt['status'] = 'failed'
        attempt['error_type'] = type(error).__name__
        raise RuntimeError('Single WRDS attempt failed: '+type(error).__name__+
                           '. No retry was attempted.') from None
    finally:
        attempt_file.write_text(json.dumps(attempt, indent=2))
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/raw'))
    parser.add_argument('--names', type=Path, required=True)
    args = parser.parse_args()
    download(args.out, args.names)
