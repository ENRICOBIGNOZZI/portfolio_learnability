"""Reproduce the paper from raw data; never calls WRDS implicitly."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import digest, load_panels, prepare, write_json
from kernels import FEATURE_COUNT, KERNELS, FeatureBank, array_hash, median_distance
from portfolio import fit_windows, managed_matrix, sharpe
import subprocess


def fit(clean_dir, output_dir, kernel, seed=0):
    panels, clean_manifest = load_panels(clean_dir)
    output = Path(output_dir)
    case_root = output/'public'/kernel/f'seed_{seed}'
    if case_root.exists():
        raise FileExistsError('Never mix or overwrite runs: '+str(case_root))
    case_root.mkdir(parents=True)
    sample = np.load(Path(clean_dir)/'initial_sample.npy', allow_pickle=False)
    if digest(Path(clean_dir)/'initial_sample.npy') != clean_manifest['sample_sha256']:
        raise ValueError('Initial sample checksum mismatch.')
    if sample.shape != (1000, 130):
        raise ValueError('Require exactly 1,000 initial vectors with 130 predictors.')
    if seed != 0:
        raise ValueError('The experiment fixes seed=0.')
    ell = median_distance(sample)
    dates = pd.DatetimeIndex([p['date'] for p in panels])
    bank = FeatureBank(kernel, sample.shape[1], FEATURE_COUNT, ell, seed)
    print('Computing managed payoffs:', kernel, seed, flush=True)
    g_max = managed_matrix(panels, bank)
    write_json(case_root/'source.json', {
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'code_checksums':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        'clean_manifest':clean_manifest, 'clean_manifest_sha256':digest(Path(clean_dir)/'manifest.json'),
        'feature_bank':bank.metadata(), 'initial_sample_sha256':array_hash(sample),
        'managed_matrix_sha256':array_hash(g_max),
        'formation_start':str(dates.min().date()), 'formation_end':str(dates.max().date()),
        'refit_timing':'after first formation month close; labels known then',
        'no_test_selection_of_feature_count':True})
    counts = (g_max.shape[1],) if kernel == 'linear' else (FEATURE_COUNT,)
    for p in counts:
        folder = case_root/f'p_{p}'
        folder.mkdir()
        g = g_max if kernel == 'linear' else g_max[:, :p]*np.sqrt(bank.maximum/p)
        monthly, diagnostics, paths, coefficients = [], [], [], []
        save_weights = seed == 0 and (kernel == 'linear' or p == FEATURE_COUNT)
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
                    row.update(raw_gross=np.abs(weights).sum(), raw_net=weights.sum())
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
                    'raw_validation_gross':initial_gross,
                    'selected_lambda':float(result['penalties'][choice]),
                    'selected_train_beta_sha256':array_hash(result['selected_train_beta']),
                    'available_at':'1978-01-31', 'formation_start':'1973-01-31',
                    'formation_end':'1977-12-31', 'return_end':'1978-01-31',
                    'timing_caveat':'Full 1977 formation validation requires January 1978 payoff.', 'common_to_every_strategy':True,
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
            'quadratic_portfolio_loss':float(np.mean((1-frame.raw_excess_return)**2)),
            'calibration_sha256':digest(case_root/'calibration.json') if kernel=='linear' else None,
            'lambda_grid_sha256':array_hash(result['penalties']),
            'lambda_grid':result['penalties'].tolist(),
            'source_sha256':digest(case_root/'source.json'),
            'files':{name:digest(folder/name) for name in (
                'monthly.csv','diagnostics.csv','lambda_returns.parquet','spectrum_2024.csv')}})
    return case_root


def get_risk_free(raw_dir, destination):
    """Use the same cash return JKP subtracted, from the identical raw snapshot."""
    raw_dir, destination = Path(raw_dir), Path(destination)
    manifest = json.loads((raw_dir/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest.get('return_units') != 'decimal':
        raise ValueError('Require a complete decimal-return raw snapshot.')
    if destination.exists():
        raise FileExistsError('Do not overwrite a frozen cash-return series.')
    frames = []
    for item in manifest['files']:
        path = raw_dir/item['name']
        if digest(path) != item['sha256']:
            raise ValueError('Raw checksum mismatch.')
        f = pd.read_parquet(path, columns=['id','eom','current_total_return','current_excess_return'])
        f = f.loc[f.id.between(1,99999)].copy()
        f['rf'] = f.current_total_return-f.current_excess_return
        frames.append(f[['eom','rf']].loc[np.isfinite(f.rf)])
    frame = pd.concat(frames)
    rows = []
    for date, g in frame.groupby('eom'):
        if g.rf.max()-g.rf.min() > 1e-8:
            raise ValueError('Inconsistent cash rates within a CRSP month.')
        rows.append({'return_date':date,'rf':float(g.rf.median())})
    destination.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(destination,index=False)
    write_json(destination.with_suffix('.json'), {
        'source':'Same raw snapshot: JKP current total minus current excess return',
        'raw_manifest_sha256':digest(raw_dir/'manifest.json'),
        'units':'monthly decimal','sha256':digest(destination)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--clean', type=Path, required=True)
    p = commands.add_parser('fit')
    p.add_argument('--clean', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kernel', choices=KERNELS, required=True)
    p = commands.add_parser('risk-free')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('report')
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--risk-free', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.raw, args.clean)
    elif args.command == 'fit':
        fit(args.clean, args.out, args.kernel)
    elif args.command == 'risk-free':
        get_risk_free(args.raw, args.out)
    elif args.command == 'report':
        from figures import report
        report(args.results, args.risk_free, args.out)


if __name__ == '__main__':
    main()
