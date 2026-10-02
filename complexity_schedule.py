"""Initial C0 validation and annual spectral plug-in regularization; four figures."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline import digest, load_panels, write_json
from kernels import FeatureBank, median_distance, array_hash
from portfolio import annual_splits, complexity_grid, dual_path, managed_matrix, sharpe


def spectral_fit(eigenvalues):
    """Fixed central rank band avoids dominant first modes and numerical tail."""
    mu = np.sort(np.asarray(eigenvalues, float))[::-1]
    if not np.isfinite(mu).all() or len(mu) == 0 or mu[0] <= 0:
        raise ValueError('Invalid managed spectrum.')
    mu = mu[mu > mu[0]*1e-12]
    lo, hi = max(2, int(np.ceil(.1*len(mu)))), int(np.floor(.6*len(mu)))
    rank = np.arange(lo, hi+1)
    if len(rank) < 5:
        raise ValueError('Insufficient spectral ranks for a slope estimate.')
    x, y = np.log(rank), np.log(mu[rank-1])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y-(intercept+slope*x)
    b = float(-slope)
    if not np.isfinite(b) or b <= 1:
        raise ValueError(f'Estimated b={b} outside the manuscript domain b>1; do not clip.')
    return {'b':b, 'r_squared':float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
            'rank_low':lo, 'rank_high':hi, 'positive_rank':len(mu)}


def nested_grid(g):
    """Preserve the original 120 penalties exactly and add 80 complexity midpoints."""
    base = complexity_grid(g, 120)
    s = np.linalg.svd(g, compute_uv=False)
    mu = s*s/len(g)
    mu = mu[mu > mu.max()*1e-12]
    c = np.sum(mu[:, None]/(mu[:, None]+base[None, :]), axis=0)
    added = []
    for j in np.linspace(0, 118, 80, dtype=int):
        target = (c[j]+c[j+1])/2
        low, high = base[j], base[j+1]
        for _ in range(80):
            middle = np.sqrt(low*high)
            if np.sum(mu/(mu+middle)) > target:
                low = middle
            else:
                high = middle
        added.append(np.sqrt(low*high))
    result = np.sort(np.r_[base, added])
    if len(np.unique(result)) != 200:
        raise ValueError('The nested grid must contain 200 distinct values.')
    return result


def calibrate(g, dates):
    _, train, validation, _ = next(annual_splits(dates))
    history = np.asarray(g)[train]
    penalties = nested_grid(history)
    gram = history @ history.T
    alpha, mu, _ = dual_path(gram, penalties)
    spectrum = spectral_fit(mu)
    exponent = spectrum['b']/(spectrum['b']+1)
    c0 = penalties*len(train)**exponent
    validation_returns = (np.asarray(g)[validation] @ history.T) @ alpha
    losses = np.mean((1-validation_returns)**2, axis=0)
    choice = int(np.argmin(losses))
    return {'c0':c0, 'choice':choice, 'loss':losses, 'lambda':penalties,
            'spectrum':spectrum, 'train_months':len(train), 'validation_months':len(validation)}


def window(g, dates, split, calibration):
    year, train, validation, test = split
    history_indices = np.r_[train, validation]
    history = np.asarray(g)[history_indices]
    gram = history @ history.T
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    if values.min() < -max(np.abs(values).max(), 1e-30)*1e-10:
        raise ValueError('Non-positive-semidefinite managed Gram matrix.')
    values = np.maximum(values, 0.)
    mu = values[::-1]/len(history)
    spectrum = spectral_fit(mu)
    exponent = spectrum['b']/(spectrum['b']+1)
    penalties = calibration['c0']*len(history)**(-exponent)
    alpha = vectors @ ((vectors.T@np.ones(len(history)))[:, None]/
                       (values[:, None]+len(history)*penalties[None, :]))
    # This evaluation happens after fitting. Test returns cannot enter b or lambda.
    returns = (np.asarray(g)[test]@history.T)@alpha
    complexity = np.sum(mu[:, None]/(mu[:, None]+penalties[None, :]), axis=0)
    choice = calibration['choice']
    return {'year':year, 'lambda':penalties, 'complexity':complexity,
            'loss':np.mean((1-returns)**2, axis=0), 'sharpe':sharpe(returns, axis=0),
            'spectrum':spectrum, 'returns':returns, 'T':len(history),
            'selected_beta':history.T@alpha[:, choice], 'test':test,
            'last_known_payoff':str((dates[history_indices[-1]]+pd.offsets.MonthEnd(1)).date())}


def cache_managed(clean, cache):
    cache = Path(cache)
    path, manifest_path = cache/'matern32_managed.npy', cache/'manifest.json'
    sample = np.load(Path(clean)/'initial_sample.npy', allow_pickle=False)
    clean_meta = json.loads((Path(clean)/'manifest.json').read_text())
    if digest(Path(clean)/'initial_sample.npy') != clean_meta['sample_sha256']:
        raise ValueError('Initial sample checksum mismatch.')
    bank = FeatureBank('matern32', 130, 10000, median_distance(sample), 0)
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text())
        if (meta['clean_manifest_sha256'] != digest(Path(clean)/'manifest.json') or
                meta['feature_bank'] != bank.metadata() or digest(path) != meta['managed_file_sha256']):
            raise ValueError('Managed cache provenance mismatch.')
        g = np.load(path, allow_pickle=False)
        if array_hash(g) != meta['managed_matrix_sha256']:
            raise ValueError('Managed array checksum mismatch.')
    else:
        panels, _ = load_panels(clean)
        rows = []
        for i in range(len(panels)):
            rows.append(managed_matrix([panels[i]], bank)[0])
            if (i+1) % 12 == 0:
                print('Managed payoffs:', panels.dates[i].date(), flush=True)
        g = np.vstack(rows)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(path, g)
        meta = {'kernel':'matern32', 'feature_bank':bank.metadata(),
                'clean_manifest_sha256':digest(Path(clean)/'manifest.json'),
                'managed_matrix_sha256':array_hash(g), 'managed_file_sha256':digest(path),
                'dates':[str(d.date()) for d in panels.dates]}
        write_json(manifest_path, meta)
    dates = pd.DatetimeIndex(meta['dates'])
    if g.shape != (744,10000) or not np.isfinite(g).all():
        raise ValueError('Require 744 months of 10,000 finite managed payoffs.')
    list(annual_splits(dates))
    return g, dates, meta


def heatmap_values(paths, count=260):
    """Interpolate within observed complexity support; never extrapolate a year."""
    centers = np.geomspace(paths.complexity.min(), paths.complexity.max(), count)
    years = sorted(paths.year.unique())
    z = np.full((len(years), count), np.nan)
    for i, year in enumerate(years):
        d = paths[paths.year == year].sort_values('complexity')
        inside = (centers >= d.complexity.min()) & (centers <= d.complexity.max())
        z[i, inside] = np.interp(np.log(centers[inside]), np.log(d.complexity), d.oos_loss)
    return centers, np.asarray(years), z


def normalized_inputs(paths, selected):
    if selected.year.duplicated().any() or selected['T'].isna().any():
        raise ValueError('Require one historical sample size per year.')
    sizes = selected.set_index('year')['T']
    if not np.isfinite(sizes).all() or (sizes <= 0).any() or (sizes % 1 != 0).any():
        raise ValueError('T must be a positive integer number of estimation months.')
    paths, selected = paths.copy(), selected.copy()
    paths['T'] = paths.year.map(sizes)
    if paths['T'].isna().any():
        raise ValueError('Missing historical sample size for a plotted year.')
    paths['complexity_over_T'] = paths.complexity/paths['T']
    selected['complexity_over_T'] = selected.complexity/selected['T']
    return paths, selected


def plot_results(paths, selected, out, normalized=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix', 'font.size':12, 'axes.titlesize':18,
        'axes.labelsize':14, 'axes.edgecolor':'#9b9ea6', 'axes.linewidth':.7,
        'xtick.labelsize':11, 'ytick.labelsize':11, 'pdf.fonttype':42,
        'savefig.facecolor':'white'})
    if normalized:
        paths, selected = normalized_inputs(paths, selected)
        paths['complexity'] = paths.complexity_over_T
        selected['complexity'] = selected.complexity_over_T
    cmap, norm = plt.get_cmap('coolwarm'), Normalize(1978,2024)
    xlabel = r'Effective portfolio complexity $\widehat{\mathcal{C}}_T(\lambda)$'
    if normalized:
        xlabel = r'Complexity per observation $\widehat{\mathcal{C}}_T(\lambda)/T$'
    def base(title, subtitle):
        fig, ax = plt.subplots(figsize=(9.1,6.6))
        fig.subplots_adjust(left=.115, right=.86, bottom=.205, top=.81)
        fig.text(.115,.945,title,fontsize=19,weight='bold',ha='left')
        fig.text(.115,.895,subtitle,fontsize=11.5,color='#565b65',ha='left')
        ax.grid(which='major',color='#dce0e5',lw=.65)
        ax.grid(which='minor',color='#edf0f3',lw=.4)
        ax.set_axisbelow(True)
        return fig, ax
    def save(fig, stem, note):
        if normalized:
            note += '\nT counts historical monthly managed payoffs used for fitting (180-732), excluding the test year.'
        fig.text(.115,.055,note,fontsize=10,color='#555b65',ha='left',va='bottom',linespacing=1.45)
        fig.savefig(out/(stem+'.pdf'))
        fig.savefig(out/(stem+'.png'),dpi=220)
        plt.close(fig)
    common = 'Matérn-3/2 · JKP U.S. stocks · 200 initial $C_0$ candidates'
    for metric, title, stem in [
        ('oos_loss','Out-of-sample loss and portfolio complexity','fig01_oos_loss'),
        ('oos_sharpe','Out-of-sample Sharpe and portfolio complexity','fig02_oos_sharpe')]:
        fig, ax = base(title,common)
        for year, d in paths.groupby('year'):
            d = d.sort_values('complexity')
            ax.plot(d.complexity,d[metric],color=cmap(norm(year)),lw=.9,alpha=.72)
        ax.set_xscale('log'); ax.set_xlabel(xlabel)
        if metric == 'oos_loss':
            ax.set_yscale('log');ax.set_ylabel(r'OOS loss $\widehat Q_{T+1}(\lambda)$')
        else:
            ax.set_ylabel('OOS Sharpe ratio (annualized)');ax.axhline(0,color='#5f646f',lw=.8)
        cax = fig.add_axes([.89,.205,.019,.605])
        cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax)
        cb.set_label('Formation year',fontsize=11)
        cb.set_ticks([1978,1990,2000,2010,2024])
        save(fig,stem,'Each line: one annual refit, evaluated on its 12 subsequent monthly payoffs.\n'
             '1978-2024 formations; February-January payoffs. No smoothing or imposed curve shape.')
    fig, ax = base('The out-of-sample loss landscape',common)
    centers, years, z = heatmap_values(paths)
    edges = np.exp(np.r_[np.log(centers[0])-(np.log(centers[1])-np.log(centers[0]))/2,
                          (np.log(centers[:-1])+np.log(centers[1:]))/2,
                          np.log(centers[-1])+(np.log(centers[-1])-np.log(centers[-2]))/2])
    mesh = ax.pcolormesh(edges,np.r_[years-.5,years[-1]+.5],np.ma.masked_invalid(z),
        cmap='coolwarm',norm=LogNorm(vmin=paths.oos_loss.min(),vmax=paths.oos_loss.max()),rasterized=True)
    ax.set_xscale('log');ax.set_xlabel(xlabel);ax.set_ylabel('Formation year')
    ax.set_ylim(years[-1]+.5,years[0]-.5);ax.grid(False)
    ax.plot(selected.complexity,selected.year,color='#171b24',lw=1.35,label='Fixed $C_0$ path')
    ax.legend(loc='lower left',fontsize=10,framealpha=.94)
    cax=fig.add_axes([.89,.205,.019,.605]);fig.colorbar(mesh,cax=cax,label='OOS loss')
    save(fig,'fig03_oos_loss_heatmap','Loss interpolated in log complexity within each year; blank cells lie outside observed support.\n'
         'Black path uses $C_0$ selected once in the initial validation; OOS losses never select it.')
    fig, ax = base('Complexity per observation and regularization' if normalized else 'Complexity and regularization through time',
                   'Matérn-3/2 · $C_0$ fixed initially · spectral b and portfolio coefficients refitted annually')
    fig.subplots_adjust(right=.855)
    right = ax.twinx()
    a, = ax.plot(selected.year,selected.complexity,'o-',color='#155ce3',ms=3.2,lw=1.65,
                  label='Effective complexity')
    b, = right.plot(selected.year,selected['lambda'],'s-',color='#cf342e',ms=3,lw=1.6,
                     label=r'Regularization $\lambda_T$')
    ax.set_yscale('log');right.set_yscale('log');ax.set_xlabel('Formation year')
    ax.set_ylabel(r'Selected complexity $\widehat{\mathcal{C}}_T(\lambda_T)$',color='#155ce3')
    right.set_ylabel(r'Regularization $\lambda_T$',color='#cf342e',labelpad=12)
    if normalized:
        ax.set_ylabel(r'Selected complexity per observation $\widehat{\mathcal{C}}_T(\lambda_T)/T$',color='#155ce3')
        a.set_label('Complexity / T')
    ax.tick_params(axis='y',which='both',colors='#155ce3');right.tick_params(axis='y',which='both',colors='#cf342e')
    fig.legend(handles=[a,b],loc='upper left',bbox_to_anchor=(.105,.865),
               ncol=2,fontsize=11,frameon=False)
    save(fig,'fig04_complexity_regularization',
         r'$\lambda_T=C_0 T^{-\hat b_T/(\hat b_T+1)}$; $T$ counts historical monthly observations.'+'\n'
         '$C_0$ chosen on 1973-1977 validation. Annual changes in b can produce local increases in λ.')


def render_normalized(source_dir):
    source = Path(source_dir)
    manifest = json.loads((source/'manifest.json').read_text())
    for name in ['paths.csv', 'annual_schedule.csv']:
        if digest(source/name) != manifest['outputs'][name]:
            raise ValueError('Published path checksum mismatch.')
    out = source/'normalized'
    if out.exists():
        raise FileExistsError('Never overwrite published normalized figures.')
    out.mkdir()
    paths = pd.read_csv(source/'paths.csv')
    selected = pd.read_csv(source/'annual_schedule.csv')
    p, s = normalized_inputs(paths, selected)
    p.to_csv(out/'paths.csv',index=False)
    s.to_csv(out/'annual_schedule.csv',index=False)
    plot_results(paths,selected,out,normalized=True)
    write_json(out/'manifest.json',{
        'status':'complete', 'normalization':'effective complexity / T',
        'T_definition':'number of historical monthly managed payoffs in each annual refit, excluding its OOS window',
        'T_first':int(s['T'].iloc[0]),'T_last':int(s['T'].iloc[-1]),
        'first_complexity_over_T':float(s.complexity_over_T.iloc[0]),
        'last_complexity_over_T':float(s.complexity_over_T.iloc[-1]),
        'source_manifest_sha256':digest(source/'manifest.json'),
        'source_inputs':{n:manifest['outputs'][n] for n in ['paths.csv','annual_schedule.csv']},
        'code_sha256':digest(Path(__file__)),
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'no_refit':True,'no_return_or_loss_changes':True,
        'outputs':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})


def run(clean, cache, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Use a fresh output directory; never mix empirical runs.')
    out.mkdir(parents=True)
    g, dates, source = cache_managed(clean,cache)
    calibration = calibrate(g,dates)
    choice = calibration['choice']
    pd.DataFrame({'candidate':np.arange(200),'C0':calibration['c0'],
        'initial_lambda':calibration['lambda'],'validation_loss':calibration['loss'],
        'selected':np.arange(200)==choice}).to_csv(out/'initial_validation.csv',index=False)
    records, annual, monthly, private_returns = [], [], [], []
    for split in annual_splits(dates):
        r = window(g,dates,split,calibration)
        for j in range(200):
            records.append({'year':r['year'],'candidate':j,'C0':calibration['c0'][j],
                'lambda':r['lambda'][j],'complexity':r['complexity'][j],
                'oos_loss':r['loss'][j],'oos_sharpe':r['sharpe'][j],'selected':j==choice})
        annual.append({'year':r['year'],'T':r['T'],'C0':calibration['c0'][choice],
            'lambda':r['lambda'][choice],'complexity':r['complexity'][choice],
            'oos_loss':r['loss'][choice],'oos_sharpe':r['sharpe'][choice],
            'last_known_payoff':r['last_known_payoff'],**r['spectrum']})
        for h,i in enumerate(r['test']):
            monthly.append({'formation_date':dates[i],'return_date':dates[i]+pd.offsets.MonthEnd(1),
                'raw_excess_return':r['returns'][h,choice],'lambda':r['lambda'][choice]})
        private_returns.append(r['returns'])
        print(f"Year {r['year']}: b={r['spectrum']['b']:.4f}, lambda={r['lambda'][choice]:.6g}, C={r['complexity'][choice]:.3f}",flush=True)
    paths, selected, month = pd.DataFrame(records),pd.DataFrame(annual),pd.DataFrame(monthly)
    paths.to_csv(out/'paths.csv',index=False)
    selected.to_csv(out/'annual_schedule.csv',index=False)
    month.to_csv(out/'selected_monthly.csv',index=False)
    np.savez_compressed(Path(cache)/'schedule_payoffs.npz',returns=np.concatenate(private_returns))
    # Adversarial check on actual data, covering both first and final refits.
    changed = g.copy();changed[180:] *= -7
    perturbed = calibrate(changed,dates)
    for key in ['c0','lambda','loss']:
        np.testing.assert_allclose(calibration[key],perturbed[key],rtol=1e-12,atol=1e-12)
    assert calibration['choice']==perturbed['choice']
    for split in [next(annual_splits(dates)),list(annual_splits(dates))[-1]]:
        original = window(g,dates,split,calibration)
        changed = g.copy();changed[split[3]] *= -11
        perturbed = window(changed,dates,split,calibration)
        for key in ['lambda','complexity','selected_beta']:
            np.testing.assert_allclose(original[key],perturbed[key],rtol=1e-12,atol=1e-12)
        assert original['spectrum']==perturbed['spectrum']
    plot_results(paths,selected,out)
    meta = {'status':'complete','protocol':json.loads(Path('schedule_protocol.json').read_text()),
        'protocol_sha256':digest(Path('schedule_protocol.json')),'source':source,
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_checksums':{name:digest(Path(name)) for name in ['complexity_schedule.py','portfolio.py','kernels.py','data_pipeline.py']},
        'C0':float(calibration['c0'][choice]),'initial_choice_index':choice,
        'initial_spectrum':calibration['spectrum'],'initial_lambda':float(calibration['lambda'][choice]),
        'initial_choice_at_boundary':choice in (0,199),'annual_refits':47,'oos_months':564,
        'full_oos_sharpe':float(sharpe(month.raw_excess_return)),
        'b_min':float(selected.b.min()),'b_max':float(selected.b.max()),
        'lambda_increase_years':selected.loc[selected['lambda'].diff()>0,'year'].tolist(),
        'audit':'Actual future payoff perturbations leave initial C0 and first/final historical b, lambda, complexity and coefficients unchanged.',
        'not_claimed':'A finite empirical slope does not establish the population spectral assumption or exact theorem constants.',
        'outputs':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/'manifest.json',meta)
    print('COMPLETE',out,'Sharpe',meta['full_oos_sharpe'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean',default='data/clean')
    parser.add_argument('--cache',default='results/schedule_cache')
    parser.add_argument('--out',default='paper/complexity_schedule')
    parser.add_argument('--normalized-only',action='store_true',help='Render C/T versions from an existing completed output, without refitting.')
    args=parser.parse_args()
    if args.normalized_only:
        render_normalized(args.out)
    else:
        run(args.clean,args.cache,args.out)
