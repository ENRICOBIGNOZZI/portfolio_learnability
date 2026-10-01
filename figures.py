"""Exactly five figures, one performance table, and a source-bound empirical section."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_pipeline import digest, write_json
from kernels import array_hash
from portfolio import sharpe

LABELS = {'linear':'Linear', 'gaussian':'Gaussian', 'matern32':'Matérn-3/2'}
COLORS = {'linear':'#2B2B2B', 'gaussian':'#355C7D', 'matern32':'#8A5A64'}
LINES = {'linear':'-', 'gaussian':'--', 'matern32':'-.'}
STEMS = ('fig01_wealth_drawdown', 'fig02_exposure', 'fig03_complexity_gaussian',
         'fig04_complexity_matern32', 'fig05_managed_spectrum')


def wealth_drawdown(total_returns):
    r = np.asarray(total_returns, float)
    if not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Total returns must be finite decimal values strictly above -100%.')
    wealth = np.cumprod(1+r)
    drawdown = wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1
    # Independent scalar recursion, including the unit-value starting point.
    capital, peak = 1., 1.
    independent = []
    for value in r:
        capital *= 1+value
        peak = max(peak, capital)
        independent.append(capital/peak-1)
    np.testing.assert_allclose(drawdown, independent, rtol=1e-12, atol=1e-12)
    return wealth, drawdown


def scaled_history(frame, rf, kappa):
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError('One positive fixed kappa is required.')
    expected = pd.date_range('1978-01-31','2024-12-31',freq='ME')
    frame = frame.sort_values('formation_date').copy()
    if not pd.DatetimeIndex(frame.formation_date).equals(expected):
        raise ValueError('Require exactly 564 unique OOS formation months.')
    if not (frame.return_date == frame.formation_date+pd.offsets.MonthEnd(1)).all():
        raise ValueError('Return dates must be the next calendar month.')
    if not np.isfinite(frame[['raw_excess_return','raw_gross','raw_net']]).all().all():
        raise ValueError('Nonfinite portfolio input.')
    if (frame.raw_gross < np.abs(frame.raw_net)-1e-8).any():
        raise ValueError('Gross exposure cannot be below absolute net exposure.')
    frame = frame.merge(rf,on='return_date',how='left',validate='one_to_one')
    if not np.isfinite(frame.rf).all():
        raise ValueError('Missing cash return.')
    frame['excess_return'] = kappa*frame.raw_excess_return
    frame['total_return'] = frame.rf+frame.excess_return
    frame['gross'] = kappa*frame.raw_gross
    frame['net'] = kappa*frame.raw_net
    frame['kappa'] = kappa
    frame['wealth'], frame['drawdown'] = wealth_drawdown(frame.total_return)
    frame['excess_wealth'] = np.cumprod(1+frame.excess_return)
    np.testing.assert_allclose(sharpe(frame.excess_return),sharpe(frame.raw_excess_return))
    return frame


def drawdown_episodes(frame):
    episodes, current = [], None
    last_peak = frame.return_date.iloc[0]-pd.offsets.MonthEnd(1)
    for row in frame.itertuples():
        if row.drawdown < -1e-12:
            if current is None:
                current = {'peak_date':str(last_peak.date()), 'trough_date':str(row.return_date.date()),
                           'drawdown':float(row.drawdown), 'recovery_date':None}
            if row.drawdown < current['drawdown']:
                current.update(trough_date=str(row.return_date.date()),drawdown=float(row.drawdown))
        else:
            if current is not None:
                current['recovery_date'] = str(row.return_date.date())
                episodes.append(current)
                current = None
            last_peak = row.return_date
    if current is not None:
        episodes.append(current)
    return sorted(episodes,key=lambda x:x['drawdown'])[:5]


def get_case(root, kernel):
    count = 131 if kernel == 'linear' else 10000
    folder = root/kernel/'seed_0'/f'p_{count}'
    meta = json.loads((folder/'manifest.json').read_text())
    if (meta['status'],meta['kernel'],meta['features'],meta['seed']) != ('complete',kernel,count,0):
        raise ValueError('Invalid model metadata.')
    for name, checksum in meta['files'].items():
        if digest(folder/name) != checksum:
            raise ValueError('Result checksum mismatch: '+name)
    source_path = folder.parent/'source.json'
    if digest(source_path) != meta['source_sha256']:
        raise ValueError('Source checksum mismatch.')
    source = json.loads(source_path.read_text())
    for name, checksum in source['code_checksums'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Fit/report code differs: '+name)
    if source['clean_manifest']['protocol']['protocol_version'] != 'final-empirical-rebuild-v1':
        raise ValueError('Old results cannot enter this report.')
    penalties = np.asarray(meta['lambda_grid'])
    if len(penalties) != 120 or array_hash(penalties) != meta['lambda_grid_sha256']:
        raise ValueError('Frozen lambda grid mismatch.')
    return folder, meta, source


def style():
    plt.rcParams.update({'font.family':'serif','font.size':10,'mathtext.fontset':'stix',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42})


def canvas(panels=2):
    fig, axes = plt.subplots(panels,1,figsize=(7.2,6.2 if panels==2 else 4),
                             layout='constrained', squeeze=False)
    for ax in axes[:,0]:
        ax.grid(axis='y',color='#E2E2E2',linewidth=.45)
    return fig, axes[:,0]


def line(ax, frame, x, y, kernel):
    ax.plot(frame[x],frame[y],color=COLORS[kernel],linestyle=LINES[kernel],
            linewidth=1.15,label=LABELS[kernel])


def save(fig, directory, stem):
    fig.savefig(directory/(stem+'.pdf'))
    fig.savefig(directory/(stem+'.png'),dpi=300)
    plt.close(fig)


def final_path(diagnostics, path):
    curve = diagnostics.loc[diagnostics.test_year.eq(2024)].copy()
    final = path.loc[path.test_year.eq(2024)]
    if len(curve) != 120 or curve['lambda'].nunique()!=120 or curve.selected.sum()!=1:
        raise ValueError('Require the full final-window path and one validation selection.')
    dates = pd.date_range('2024-01-31','2024-12-31',freq='ME')
    if len(final)!=1440 or final.duplicated(['formation_date','lambda']).any():
        raise ValueError('The final path requires twelve actual returns per lambda.')
    for penalty, g in final.groupby('lambda'):
        g = g.sort_values('formation_date')
        if not pd.DatetimeIndex(g.formation_date).equals(dates):
            raise ValueError('Pooled OOS returns are not the 2024 test window.')
        if not (g.return_date==g.formation_date+pd.offsets.MonthEnd(1)).all():
            raise ValueError('Incorrect final test payoff dates.')
        match = np.flatnonzero(np.isclose(curve['lambda'],penalty,rtol=1e-12,atol=0))
        if len(match)!=1:
            raise ValueError('Lambda path mismatch.')
        np.testing.assert_allclose(curve.oos_sharpe.iloc[match[0]],sharpe(g.excess_return),rtol=1e-8)
    if not np.isfinite(curve[['historical_sharpe','oos_sharpe','effective_complexity']]).all().all():
        raise ValueError('Undefined final-window Sharpe or complexity.')
    return curve.sort_values('effective_complexity')


def complexity_figure(directory, kernel, frame):
    peak = frame.loc[frame.oos_sharpe.idxmax()]
    selected = frame.loc[frame.selected].iloc[0]
    fig, axes = canvas()
    for ax, column, label in zip(axes,['historical_sharpe','oos_sharpe'],
            ['Historical Sharpe','2024 test Sharpe']):
        line(ax,frame,'effective_complexity',column,kernel)
        ax.axvline(selected.effective_complexity,color='#777777',linestyle=':',linewidth=.8)
        ax.set_ylabel(label)
    axes[1].scatter([selected.effective_complexity],[selected.oos_sharpe],s=25,
        facecolors='white',edgecolors=COLORS[kernel],label='Validation selected',zorder=4)
    axes[1].scatter([peak.effective_complexity],[peak.oos_sharpe],s=22,
        marker='x',color='#444444',label='Ex-post maximum',zorder=5)
    axes[1].set_xlabel(r'Effective complexity $C(\lambda)$')
    axes[1].legend(fontsize=8)
    save(fig,directory,STEMS[2 if kernel=='gaussian' else 3])
    return {'kernel':kernel,'ex_post_complexity':float(peak.effective_complexity),
            'ex_post_sharpe':float(peak.oos_sharpe),
            'selected_complexity':float(selected.effective_complexity)}


def report(results, risk_free_file, output):
    root, output = Path(results)/'public', Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Report output must be empty.')
    rf = pd.read_csv(risk_free_file,parse_dates=['return_date'])
    rf_source = json.loads(Path(risk_free_file).with_suffix('.json').read_text())
    if rf_source['sha256'] != digest(risk_free_file) or rf_source['units']!='monthly decimal':
        raise ValueError('Cash-return provenance mismatch.')
    calibration_path = root/'linear/seed_0/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    if pd.Timestamp(calibration['available_at']) > pd.Timestamp('1978-01-31'):
        raise ValueError('Scale was unavailable at the first formation trade.')
    scale = float(calibration['scale'])
    histories, sources, cases, curves, spectra, stats, audit = {}, [], {}, {}, {}, [], {}
    for kernel in LABELS:
        folder, meta, source = get_case(root,kernel)
        sources.append(source); cases[kernel]=meta
        if kernel=='linear' and digest(calibration_path)!=meta['calibration_sha256']:
            raise ValueError('Calibration checksum mismatch.')
        clean = source['clean_manifest']
        if clean['protocol']['timing']['status']!='user_confirmed':
            raise ValueError('Timing convention must be resolved before publication.')
        if clean['unresolved_returns'] or clean['feature_count']!=130 or clean['status']!='complete':
            raise ValueError('Invalid cleaned panel.')
        if clean['source_raw_manifest_sha256'] != rf_source['raw_manifest_sha256']:
            raise ValueError('Cash returns must use the same raw snapshot.')
        frame = scaled_history(pd.read_csv(folder/'monthly.csv',
            parse_dates=['formation_date','return_date']),rf,scale)
        histories[kernel]=frame
        sr = float(sharpe(frame.excess_return))
        stats.append({'Policy':LABELS[kernel],'Annual mean excess return':12*frame.excess_return.mean(),
            'Annual volatility':np.sqrt(12)*frame.excess_return.std(ddof=1),'Sharpe ratio':sr,
            'Maximum drawdown':frame.drawdown.min(),'Mean gross exposure':frame.gross.mean(),
            'Mean net exposure':frame.net.mean()})
        pre = frame.loc[frame.return_date < '2020-01-01']
        audit[kernel]={'maximum_drawdown_date':str(frame.loc[frame.drawdown.idxmin(),'return_date'].date()),
            'five_largest_episodes':drawdown_episodes(frame),
            'requires_review':bool(sr>=2.5 and pre.drawdown.min()>-.05),
            'stress_periods':{label:{'min_drawdown':float(g.drawdown.min()),
                'min_monthly_return':float(g.total_return.min())} for label, start, end in [
                ('1987','1987','1987-12-31'),('2000-2002','2000','2002-12-31'),
                ('2008-2009','2008','2009-12-31'),('2020','2020','2020-12-31'),
                ('2022','2022','2022-12-31')]
                for g in [frame.loc[frame.return_date.between(start,end)]]}}
        if kernel!='linear':
            spectra[kernel]=pd.read_csv(folder/'spectrum_2024.csv')
            curves[kernel]=final_path(pd.read_csv(folder/'diagnostics.csv',float_precision='round_trip'),
                                    pd.read_parquet(folder/'lambda_returns.parquet'))
    for key in ['clean_manifest_sha256','initial_sample_sha256','git_sha','code_checksums']:
        if any(s[key]!=sources[0][key] for s in sources):
            raise ValueError('Mixed representation provenance: '+key)
    if any(s['feature_bank']['ell']!=sources[0]['feature_bank']['ell'] for s in sources):
        raise ValueError('Representations must share ell.')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'audit.json',audit)
    if any(a['requires_review'] for a in audit.values()):
        raise ValueError('Strong Sharpe with tiny pre-2020 drawdowns: audit required before publication.')
    figdir=output/'figures';figdir.mkdir()
    tables=output/'tables';tables.mkdir()
    inputs=output/'inputs';inputs.mkdir()
    input_hashes={}
    for kernel, frame in histories.items():
        path=inputs/(kernel+'_monthly.csv');frame.to_csv(path,index=False)
        input_hashes[str(path.relative_to(output))]=digest(path)
    style()
    fig,axes=canvas()
    for kernel, frame in histories.items():
        baseline=pd.DataFrame({'return_date':[pd.Timestamp('1978-01-31')], 'wealth':[1.], 'drawdown':[0.]})
        plot=pd.concat([baseline,frame],ignore_index=True)
        line(axes[0],plot,'return_date','wealth',kernel)
        plot=plot.assign(drawdown_percent=100*plot.drawdown)
        line(axes[1],plot,'return_date','drawdown_percent',kernel)
    axes[0].set_yscale('log');axes[0].set_ylabel('Cumulative total wealth')
    axes[1].set_ylabel('Drawdown (%)');axes[0].legend(ncol=3,fontsize=9)
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(10));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    save(fig,figdir,STEMS[0])
    fig,axes=canvas()
    for kernel, frame in histories.items():
        line(axes[0],frame,'formation_date','gross',kernel)
        line(axes[1],frame,'formation_date','net',kernel)
    axes[0].set_ylabel('Gross exposure');axes[1].set_ylabel('Net exposure')
    axes[0].legend(ncol=3,fontsize=9);axes[1].axhline(0,color='#999999',linewidth=.6)
    save(fig,figdir,STEMS[1])
    peaks=[]
    for kernel, curve in curves.items():
        path=inputs/(kernel+'_complexity_2024.csv');curve.to_csv(path,index=False)
        input_hashes[str(path.relative_to(output))]=digest(path)
        peaks.append(complexity_figure(figdir,kernel,curve))
    fig,axes=canvas(1)
    for kernel, spectrum in spectra.items():
        if not np.isfinite(spectrum.eigenvalue).all() or (np.diff(spectrum.eigenvalue)>1e-12).any():
            raise ValueError('Invalid ordered managed spectrum.')
        path=inputs/(kernel+'_spectrum_2024.csv');spectrum.to_csv(path,index=False)
        input_hashes[str(path.relative_to(output))]=digest(path)
        line(axes[0],spectrum.loc[spectrum.eigenvalue>0],'rank','eigenvalue',kernel)
    axes[0].set(xscale='log',yscale='log',xlabel='Eigenvalue rank',ylabel='Managed-payoff eigenvalue')
    axes[0].legend();save(fig,figdir,STEMS[4])
    performance=pd.DataFrame(stats)
    performance.to_csv(tables/'performance.csv',index=False)
    compact = performance.rename(columns={
        'Annual mean excess return':'Annual mean excess', 'Annual volatility':'Annual vol.',
        'Sharpe ratio':'Sharpe', 'Maximum drawdown':'Max. drawdown',
        'Mean gross exposure':'Mean gross', 'Mean net exposure':'Mean net'})
    compact.to_latex(tables/'performance.tex',index=False,float_format='%.3f',escape=True)
    make_text(output,performance,peaks,sources[0]['feature_bank']['ell'])
    write_json(output/'characteristics.json',clean['characteristic_provenance'])
    figure_inputs={STEMS[0]:[k for k in input_hashes if 'monthly' in k],
                   STEMS[1]:[k for k in input_hashes if 'monthly' in k],
                   STEMS[2]:['inputs/gaussian_complexity_2024.csv'],
                   STEMS[3]:['inputs/matern32_complexity_2024.csv'],
                   STEMS[4]:[k for k in input_hashes if 'spectrum' in k]}
    manifest={'status':'complete','git_sha':sources[0]['git_sha'],
        'code_checksums':sources[0]['code_checksums'],'raw_manifest':clean['raw_manifest'],
        'characteristics':clean['characteristics'],'characteristic_provenance':clean['characteristic_provenance'],
        'clean_manifest':clean,'ell':sources[0]['feature_bank']['ell'],
        'sampled_vector_sha256':sources[0]['initial_sample_sha256'],'sample_dates':['1963-01-31','1972-12-31'],
        'seed':0,'P':10000,'lambda_grid':{k:v['lambda_grid'] for k,v in cases.items()},
        'lambda_grid_sha256':{k:v['lambda_grid_sha256'] for k,v in cases.items()},
        'kappa':scale,'calibration':calibration,'calibration_sha256':digest(calibration_path),
        'risk_free_sha256':digest(risk_free_file),'oos_formation':['1978-01-31','2024-12-31'],
        'oos_returns':['1978-02-28','2025-01-31'],'oos_months':564,
        'wealth_definition':'product(1 + same-snapshot cash return + kappa * raw portfolio excess return)',
        'drawdown_definition':'wealth / running maximum including unit starting wealth - 1',
        'complexity_2024_peaks':peaks,
        'figures':{stem:{'inputs':{p:input_hashes[p] for p in figure_inputs[stem]},
             'outputs':{ext:digest(figdir/(stem+'.'+ext)) for ext in ('pdf','png')}} for stem in STEMS}}
    manifest['outputs']={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'reproduction_manifest.json',manifest)


def make_text(output, performance, peaks, ell):
    summaries=[]
    for row in performance.to_dict(orient='records'):
        policy=row['Policy'].replace('Matérn',r'Mat\'ern')
        summaries.append(f"{policy}: Sharpe {row['Sharpe ratio']:.2f}, maximum drawdown {100*row['Maximum drawdown']:.1f}\\%")
    text=r'''\section{Empirical analysis}
We use the U.S. stock-level data of Jensen, Kelly, and Pedersen (2023).
Inspired by Section 2.5 of Didisheim, Ke, Kelly, and Malamud (2024), we retain
130 of the 153 characteristics by coverage using only the initial training
sample, 1963--1972. Validation and test observations cannot affect selection.
The published candidate dictionary is a fixed retrospective research specification;
we do not claim a historical data vintage or contemporaneous signal discovery.
We retain common stocks on the main U.S. exchanges, excluding nano stocks,
apply the 30\% row-missingness threshold, and exclude stock-months with missing
JKP next-month excess returns before monthly ranking and the calculation of $N_t$.
This complete-case convention is our sample choice, not a rule attributed to
Didisheim et al.; the sample is conditional on future payoff availability.
We rank observed characteristics into $[-0.5,0.5]$ and assign residual missing
characteristics neutral zero. We do not impute missing payoffs.

Training expands from 1963, with the preceding five formation years reserved
for validation and annual refits at the January formation close. The experiment
contains 47 test years and 564 formation months, January 1978--December 2024;
realized payoffs run from February 1978 to January 2025. Ridge penalties are
selected using the portfolio criterion introduced earlier, along 120-point grids
fixed using only the initial 1963--1972 managed-payoff spectra.

Linear is the finite-dimensional benchmark. Gaussian represents smooth nonlinear
policies; Mat\'ern-3/2 permits a broader nonlinear class. Each nonlinear policy
uses one fixed map of 10,000 random Fourier features, seed zero. Both share
a lengthscale of ELL, the median distance among 1,000 initial-training vectors.
A common fixed positive scale targets median gross exposure 1.8 for the initial
Linear validation policy. The last validation payoff becomes known at the
January 1978 formation close, before the first OOS payoff.

Figure~\ref{fig:wealth} and Table~\ref{tab:performance} summarize the new run.
SUMMARY.
Total wealth compounds the cash return plus the scaled portfolio excess payoff,
adding cash exactly once. Drawdown uses exactly that wealth path, including
its starting value.
\begin{figure}[htbp]\centering
\includegraphics[width=.9\linewidth]{figures/fig01_wealth_drawdown.pdf}
\caption{Cumulative total wealth (log scale) and drawdown from the same wealth series.
All policies use the same fixed pre-payoff scale.}\label{fig:wealth}
\end{figure}
\begin{table}[htbp]\centering\small
\resizebox{\linewidth}{!}{\input{tables/performance.tex}}
\caption{Performance from the 564 monthly OOS observations. Returns and volatility
are annualized from decimal monthly excess returns; maximum drawdown is signed.}
\label{tab:performance}\end{table}

Figure~\ref{fig:exposure} reports gross and net stock exposure under the same scale.
\begin{figure}[htbp]\centering
\includegraphics[width=.9\linewidth]{figures/fig02_exposure.pdf}
\caption{Gross and net exposures for the three policies.}\label{fig:exposure}
\end{figure}

Representation determines which investment opportunities can potentially be
expressed. The managed-portfolio spectrum describes their payoff structure.
Effective complexity determines how much of that spectrum is actually used
after regularization. Figure~\ref{fig:spectrum} shows the ordered positive
second-moment eigenvalues on the common history preceding the 2024 test window.
\begin{figure}[htbp]\centering
\includegraphics[width=.9\linewidth]{figures/fig05_managed_spectrum.pdf}
\caption{Managed-portfolio spectra on log-log axes.}\label{fig:spectrum}
\end{figure}

Figures~\ref{fig:gaussian} and~\ref{fig:matern} plot historical and subsequent
2024 test Sharpe against $C(\lambda)=\sum_j\mu_j/(\mu_j+\lambda)$ along the
same frozen penalty paths. The test panels use only the twelve payoffs from
February 2024 through January 2025. Validation selections are distinguished
from descriptive ex-post maxima, which never determine the policy.
\begin{figure}[htbp]\centering
\includegraphics[width=.9\linewidth]{figures/fig03_complexity_gaussian.pdf}
\caption{Gaussian effective complexity: historical and true 2024 test Sharpe.}
\label{fig:gaussian}\end{figure}
\begin{figure}[htbp]\centering
\includegraphics[width=.9\linewidth]{figures/fig04_complexity_matern32.pdf}
\caption{Mat\'ern-3/2 effective complexity: historical and true 2024 test Sharpe.}
\label{fig:matern}\end{figure}
'''
    (Path(output)/'empirics.tex').write_text(text.replace('ELL',f'{ell:.4f}').replace('SUMMARY','; '.join(summaries)))
