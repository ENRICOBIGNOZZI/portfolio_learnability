"""Reproduce the paper from raw data; never calls WRDS implicitly."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from data import digest, load_panels, prepare, write_json
from model import (FEATURE_COUNTS, KERNELS, FeatureBank, array_hash, exact_kernel,
                   fit_windows, l1_target_turnover, managed_matrix, median_distance, sharpe)


def check_approximation(sample, ell, output):
    sample = np.asarray(sample, float)
    x = sample[:min(128, len(sample))]
    rows = []
    for kernel in ('gaussian', 'matern32'):
        exact = exact_kernel(x, x, kernel, ell)
        for seed in (0, 1, 2):
            bank = FeatureBank(kernel, x.shape[1], max(FEATURE_COUNTS), ell, seed)
            for p in FEATURE_COUNTS:
                phi = bank.features(x, p)
                error = phi @ phi.T-exact
                rows.append({'kernel':kernel, 'seed':seed, 'features':p,
                    'relative_frobenius_error':np.linalg.norm(error)/np.linalg.norm(exact),
                    'rmse':np.sqrt(np.mean(error**2)), 'max_error':np.max(np.abs(error)),
                    'sample_points':len(x), 'sample_sha256':array_hash(x)})
    pd.DataFrame(rows).to_csv(output/'rff_approximation.csv', index=False)


def fit(clean_dir, output_dir, kernel, seed=0):
    panels, clean_manifest = load_panels(clean_dir)
    output = Path(output_dir)
    case_root = output/'public'/kernel/f'seed_{seed}'
    if case_root.exists():
        raise FileExistsError('Never mix or overwrite runs: '+str(case_root))
    case_root.mkdir(parents=True)
    sample = np.load(Path(clean_dir)/'initial_sample.npy', allow_pickle=False)
    if array_hash(sample) is None:
        raise AssertionError('Invalid sample.')
    ell = median_distance(sample)
    dates = pd.DatetimeIndex([p['date'] for p in panels])
    bank = FeatureBank(kernel, sample.shape[1], max(FEATURE_COUNTS), ell, seed)
    print('Computing managed payoffs:', kernel, seed, flush=True)
    g_max = managed_matrix(panels, bank)
    check_approximation(sample, ell, case_root)
    write_json(case_root/'source.json', {
        'git_sha':os.environ.get('GITHUB_SHA'), 'clean_manifest_sha256':digest(Path(clean_dir)/'manifest.json'),
        'feature_bank':bank.metadata(), 'initial_sample_sha256':array_hash(sample),
        'managed_matrix_sha256':array_hash(g_max),
        'formation_start':str(dates.min().date()), 'formation_end':str(dates.max().date()),
        'refit_timing':'after first formation month close; labels known then',
        'no_test_selection_of_feature_count':True})
    counts = (g_max.shape[1],) if kernel == 'linear' else FEATURE_COUNTS
    for p in counts:
        folder = case_root/f'p_{p}'
        folder.mkdir()
        g = g_max if kernel == 'linear' else g_max[:, :p]*np.sqrt(bank.maximum/p)
        monthly, diagnostics, paths, coefficients = [], [], [], []
        previous = None
        save_weights = seed == 0 and (kernel == 'linear' or p == max(FEATURE_COUNTS))
        private = output/'private'/kernel/f'seed_{seed}'/f'p_{p}'
        if save_weights:
            private.mkdir(parents=True, exist_ok=True)
        for result in fit_windows(g, dates):
            year, choice = result['year'], result['choice']
            test = result['test']
            test_returns = result['test_returns']
            test_sharpe = sharpe(test_returns, axis=0)
            for j, penalty in enumerate(result['penalties']):
                diagnostics.append({'test_year':year, 'lambda':penalty,
                    'effective_complexity':result['complexity'][j],
                    'validation_loss':result['validation_loss'][j],
                    'historical_sharpe':result['historical_sharpe'][j],
                    'oos_sharpe':test_sharpe[j], 'selected':j == choice,
                    'n_train':len(result['train']), 'n_validation':len(result['validation'])})
                for h, index in enumerate(test):
                    paths.append({'test_year':year, 'formation_date':dates[index],
                        'return_date':panels[index]['return_date'], 'lambda':penalty,
                        'excess_return':test_returns[h, j]})
            weight_rows = []
            for h, index in enumerate(test):
                panel = panels[index]
                row = {'test_year':year, 'formation_date':panel['date'],
                    'return_date':panel['return_date'], 'kernel':kernel, 'seed':seed,
                    'features':p, 'lambda':result['penalties'][choice],
                    'effective_complexity':result['complexity'][choice],
                    'raw_excess_return':test_returns[h, choice]}
                if save_weights:
                    weights = bank.features(panel['x'], p) @ result['beta']/len(panel['ids'])
                    if not np.isclose(weights @ panel['r'], row['raw_excess_return'],
                                      rtol=1e-8, atol=1e-10):
                        raise AssertionError('Stock-weight and managed-payoff returns disagree.')
                    turnover, previous = l1_target_turnover(panel['ids'], weights, previous)
                    row.update(raw_gross=np.abs(weights).sum(), raw_net=weights.sum(),
                               raw_target_turnover=turnover)
                    weight_rows.append(pd.DataFrame({'formation_date':panel['date'],
                        'return_date':panel['return_date'], 'id':panel['ids'],
                        'raw_weight':weights}))
                monthly.append(row)
            if save_weights:
                pd.concat(weight_rows, ignore_index=True).to_parquet(
                    private/f'weights_{year}.parquet', index=False, compression='zstd')
                coefficients.append(result['beta'])
            if kernel == 'linear' and year == 1978:
                initial_gross = []
                for index in result['validation']:
                    panel = panels[index]
                    weights = bank.features(panel['x']) @ result['selected_train_beta']/len(panel['ids'])
                    initial_gross.append(float(np.abs(weights).sum()))
                median_gross = float(np.median(initial_gross))
                if not np.isfinite(median_gross) or median_gross <= 0:
                    raise ValueError('Cannot fix a positive portfolio scale.')
                write_json(case_root/'calibration.json', {
                    'scale':1.8/median_gross, 'target_initial_median_gross':1.8,
                    'raw_initial_validation_median_gross':median_gross,
                    'available_at':'1978-01-31', 'common_to_every_strategy':True,
                    'time_varying_exposure_normalization':False})
            if year == 2024:
                pd.DataFrame({'rank':np.arange(1,len(result['mu'])+1),
                              'eigenvalue':result['mu']}).to_csv(folder/'spectrum_2024.csv',index=False)
            print(f'{kernel} seed={seed} P={p} test={year}: '
                  f'C={result["complexity"][choice]:.2f}', flush=True)
        frame = pd.DataFrame(monthly)
        if len(frame) != 564 or frame.return_date.nunique() != 564:
            raise ValueError('Incomplete out-of-sample return history.')
        frame.to_csv(folder/'monthly.csv', index=False)
        pd.DataFrame(diagnostics).to_csv(folder/'diagnostics.csv', index=False)
        pd.DataFrame(paths).to_parquet(folder/'lambda_returns.parquet', index=False, compression='zstd')
        if save_weights:
            np.savez_compressed(private/'selected_coefficients.npz', beta=np.stack(coefficients))
        write_json(folder/'manifest.json', {
            'status':'complete','kernel':kernel, 'seed':seed, 'features':p,
            'oos_sharpe':float(sharpe(frame.raw_excess_return)),
            'raw_response_one_loss':float(np.mean((1-frame.raw_excess_return)**2)),
            'source_sha256':digest(case_root/'source.json'),
            'files':{name:digest(folder/name) for name in (
                'monthly.csv','diagnostics.csv','lambda_returns.parquet','spectrum_2024.csv')}})
    return case_root


def get_risk_free(destination):
    import io
    import re
    import zipfile
    import requests
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Do not overwrite a frozen risk-free file.')
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    text = archive.read(archive.namelist()[0]).decode('utf-8-sig')
    rows = []
    for line in text.splitlines():
        cells = [v.strip() for v in line.split(',')]
        if len(cells) >= 5 and re.fullmatch(r'\d{6}', cells[0]):
            date = pd.to_datetime(cells[0], format='%Y%m')+pd.offsets.MonthEnd(0)
            rows.append({'return_date':date,'rf':float(cells[4])/100.0})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.return_date.duplicated().any():
        raise ValueError('Unexpected French risk-free format.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination,index=False)
    write_json(destination.with_suffix('.json'), {'url':url,'sha256':digest(destination),
        'units':'monthly decimal','download_date':str(pd.Timestamp.now('UTC'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--clean', type=Path, required=True)
    p.add_argument('--return-corrections', type=Path)
    p = commands.add_parser('fit')
    p.add_argument('--clean', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kernel', choices=KERNELS, required=True)
    p.add_argument('--seed', type=int, default=0)
    p = commands.add_parser('risk-free')
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('report')
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--risk-free', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.raw, args.clean, args.return_corrections)
    elif args.command == 'fit':
        fit(args.clean, args.out, args.kernel, args.seed)
    elif args.command == 'risk-free':
        get_risk_free(args.out)
    elif args.command == 'report':
        from figures import report
        report(args.results, args.risk_free, args.out)


if __name__ == '__main__':
    main()
