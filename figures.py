"""Paper figures: no legacy inputs, no smoothing, no result-dependent model choice."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import fitz
from data import digest, write_json
from model import FEATURE_COUNTS, sharpe

LABELS = {'linear':'Linear', 'gaussian':'Gaussian', 'matern32':'Matérn-3/2'}
COLORS = {'linear':'#243B53', 'gaussian':'#A45F35', 'matern32':'#77364F'}
LINES = {'linear':'--', 'gaussian':'-', 'matern32':'-.'}


def style():
    plt.rcParams.update({'font.family':'serif', 'font.size':10,
        'mathtext.fontset':'stix', 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':0.7, 'legend.frameon':False, 'figure.facecolor':'white',
        'axes.facecolor':'white', 'pdf.fonttype':42, 'ps.fonttype':42})


def canvas(ylabel, xlabel=''):
    fig = plt.figure(figsize=(7.4, 4.1))
    ax = fig.add_axes([0.13,0.15,0.83,0.79])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', linewidth=0.45, color='#E1E4E8')
    return fig, ax


def save(fig, output, stem):
    fig.savefig(output/(stem+'.pdf'))
    fig.savefig(output/(stem+'.png'), dpi=250)
    plt.close(fig)


def stack(output, top, bottom, stem):
    doc = fitz.open()
    a, b = fitz.open(output/(top+'.pdf')), fitz.open(output/(bottom+'.pdf'))
    width = max(a[0].rect.width, b[0].rect.width)
    ha, hb = a[0].rect.height, b[0].rect.height
    page = doc.new_page(width=width, height=ha+hb)
    page.show_pdf_page(fitz.Rect(0,0,width,ha), a, 0)
    page.show_pdf_page(fitz.Rect(0,ha,width,ha+hb), b, 0)
    doc.save(output/(stem+'.pdf'))
    page.get_pixmap(matrix=fitz.Matrix(2,2)).save(output/(stem+'.png'))
    doc.close(); a.close(); b.close()


def time_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('')


def get_case(root, kernel, count, seed=0):
    folder = root/kernel/f'seed_{seed}'/f'p_{count}'
    meta = json.loads((folder/'manifest.json').read_text())
    if meta['status'] != 'complete' or meta['kernel'] != kernel or meta['features'] != count:
        raise ValueError('Invalid case metadata.')
    for name, expected in meta['files'].items():
        if digest(folder/name) != expected:
            raise ValueError('Result checksum mismatch: '+str(folder/name))
    source_file = folder.parent/'source.json'
    if digest(source_file) != meta['source_sha256']:
        raise ValueError('Result provenance mismatch.')
    return folder, meta, json.loads(source_file.read_text())


def complexity_figure(output, kernel, diagnostic):
    frame = diagnostic.loc[diagnostic.test_year.eq(2024)].sort_values('effective_complexity')
    if len(frame) != 120 or frame.selected.sum() != 1:
        raise ValueError('The final-window curve must contain 120 penalties and one validation choice.')
    peak = frame.loc[frame.oos_sharpe.idxmax()]
    selected = frame.loc[frame.selected].iloc[0]
    upper = kernel+'_historical_2024'
    lower = kernel+'_test_2024'
    for column, title, stem in [
        ('historical_sharpe','Historical fit (train + validation)',upper),
        ('oos_sharpe','Test window 2024 (February 2024–January 2025)',lower)]:
        fig, ax = canvas('Annualized Sharpe ratio', r'Effective complexity $\widehat{\mathcal{C}}(\lambda)$')
        ax.set_title(LABELS[kernel]+' — '+title, loc='left', fontsize=10, pad=10)
        color = '#66788A' if column == 'historical_sharpe' else COLORS[kernel]
        ax.plot(frame.effective_complexity, frame[column], color=color,
                linewidth=1.45, marker='o', markersize=2.2, markeredgewidth=0)
        ax.set_xlim(-5, max(frame.effective_complexity)*1.04)
        if column == 'historical_sharpe':
            ax.set_xlabel('')
        else:
            ax.scatter([peak.effective_complexity],[peak.oos_sharpe],s=110,marker='*',
                facecolor='#D1AD54',edgecolor='#343434',linewidth=0.8,zorder=5,
                label=f'Ex-post peak: C = {peak.effective_complexity:.1f}')
            ax.scatter([selected.effective_complexity],[selected.oos_sharpe],s=36,
                facecolor='white',edgecolor=color,linewidth=1.1,zorder=6,
                label='Validation-selected policy')
            ax.legend(loc='best',fontsize=8)
        save(fig, output, stem)
    stack(output, upper, lower, 'complexity_'+kernel+'_2024')
    for stem in (upper, lower):
        for suffix in ('.pdf','.png'):
            (output/(stem+suffix)).unlink()
    interior = peak.effective_complexity not in (
        frame.effective_complexity.iloc[0], frame.effective_complexity.iloc[-1])
    return {'kernel':kernel, 'ex_post_complexity':float(peak.effective_complexity),
        'ex_post_sharpe':float(peak.oos_sharpe), 'interior':bool(interior),
        'selected_complexity':float(selected.effective_complexity),
        'selected_sharpe':float(selected.oos_sharpe)}


def report(results, risk_free_file, output):
    root, output = Path(results)/'public', Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Report output must be empty. Old figures cannot enter this build.')
    output.mkdir(parents=True, exist_ok=True)
    figdir = output/'figures'; figdir.mkdir()
    style()
    rf = pd.read_csv(risk_free_file, parse_dates=['return_date'])
    if rf.return_date.duplicated().any() or not np.isfinite(rf.rf).all():
        raise ValueError('Invalid risk-free series.')
    calibration = json.loads((root/'linear/seed_0/calibration.json').read_text())
    scale = float(calibration['scale'])
    histories, spectra, peaks, stats, sources = {}, {}, [], [], []
    for kernel, count in [('linear',131),('gaussian',10000),('matern32',10000)]:
        folder, meta, source = get_case(root,kernel,count)
        sources.append(source)
        frame = pd.read_csv(folder/'monthly.csv',parse_dates=['formation_date','return_date'])
        frame = frame.sort_values('return_date').merge(rf, on='return_date',how='left',validate='one_to_one')
        if len(frame) != 564 or frame.rf.isna().any():
            raise ValueError('Every OOS month needs its correctly dated risk-free return.')
        frame['excess_return'] = frame.raw_excess_return*scale
        frame['total_return'] = frame.excess_return+frame.rf
        frame['gross'] = frame.raw_gross*scale
        frame['net'] = frame.raw_net*scale
        frame['target_turnover'] = frame.raw_target_turnover*scale
        if np.any(1+frame.total_return <= 0):
            raise ValueError('Wealth cannot be compounded: nonpositive monthly wealth factor.')
        frame['wealth'] = np.cumprod(1+frame.total_return)
        wealth = np.r_[1.0, frame.wealth.to_numpy()]
        drawdown = wealth/np.maximum.accumulate(wealth)-1
        net_proxy = frame.excess_return-0.0025*frame.target_turnover
        stats.append({'kernel':kernel, 'annual_mean_excess':12*frame.excess_return.mean(),
            'annual_volatility':np.sqrt(12)*frame.excess_return.std(ddof=1),
            'sharpe':float(sharpe(frame.excess_return)), 'max_drawdown':-float(drawdown.min()),
            'mean_gross':frame.gross.mean(), 'mean_net':frame.net.mean(),
            'gross_p99':frame.gross.quantile(.99),
            'annual_target_turnover':12*frame.target_turnover.mean(),
            'net_sharpe_25bps_target_turnover_proxy':float(sharpe(net_proxy)),
            'native_response_one_loss':meta['raw_response_one_loss'],
            'terminal_wealth':frame.wealth.iloc[-1]})
        histories[kernel] = frame
        if kernel != 'linear':
            spectra[kernel] = pd.read_csv(folder/'spectrum_2024.csv')
            diagnostic = pd.read_csv(folder/'diagnostics.csv')
            peaks.append(complexity_figure(figdir,kernel,diagnostic))
    if len({s['clean_manifest_sha256'] for s in sources}) != 1:
        raise ValueError('Representations were not estimated on the same cleaned panel.')
    if len({s['initial_sample_sha256'] for s in sources}) != 1:
        raise ValueError('Representations use different calibration samples.')
    for frame in histories.values():
        if not frame.return_date.equals(histories['linear'].return_date):
            raise ValueError('OOS calendars do not agree.')
    performance = pd.DataFrame(stats)
    performance.to_csv(output/'performance.csv',index=False)
    pd.DataFrame(peaks).to_csv(output/'complexity_2024.csv',index=False)

    fig, ax = canvas('Cumulative wealth')
    for kernel, frame in histories.items():
        date0 = frame.return_date.iloc[0]-pd.offsets.MonthEnd(1)
        ax.plot(pd.DatetimeIndex([date0,*frame.return_date]), np.r_[1,frame.wealth],
                color=COLORS[kernel], linestyle=LINES[kernel],linewidth=1.5,label=LABELS[kernel])
    ax.set_yscale('log'); time_axis(ax); ax.legend(loc='upper left',ncol=3,fontsize=9)
    save(fig,figdir,'cumulative_wealth')
    for name, ylabel in [('gross','Gross exposure'),('net','Net exposure')]:
        fig, ax = canvas(ylabel)
        for kernel, frame in histories.items():
            ax.plot(frame.formation_date,frame[name],color=COLORS[kernel],
                    linestyle=LINES[kernel],linewidth=1.15,label=LABELS[kernel])
        if name == 'gross':
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(0,color='#989898',linewidth=.6,zorder=0)
        time_axis(ax); ax.legend(loc='best',ncol=3,fontsize=9)
        save(fig,figdir,name+'_exposure')
    stack(figdir,'gross_exposure','net_exposure','portfolio_exposure')
    for name in ('gross_exposure','net_exposure'):
        for suffix in ('.png','.pdf'):
            (figdir/(name+suffix)).unlink()

    fig, ax = canvas('Managed-payoff eigenvalue','Eigenvalue rank')
    for kernel, frame in spectra.items():
        frame = frame.loc[frame.eigenvalue > 0]
        ax.plot(frame['rank'],frame.eigenvalue,color=COLORS[kernel],
                linestyle=LINES[kernel],linewidth=1.4,label=LABELS[kernel])
    ax.set_xscale('log'); ax.set_yscale('log'); ax.legend(loc='best')
    save(fig,figdir,'managed_spectrum')

    numerical = pd.read_csv(root/'gaussian/seed_0/rff_approximation.csv')
    numerical.to_csv(output/'rff_approximation.csv',index=False)
    fig, ax = canvas('Relative kernel approximation error','Number of Random Fourier Features')
    for kernel in ('gaussian','matern32'):
        frame = numerical.loc[numerical.kernel.eq(kernel)]
        center = frame.loc[frame.seed.eq(0)].sort_values('features')
        bands = frame.groupby('features').relative_frobenius_error.agg(['min','max']).sort_index()
        ax.plot(center.features,center.relative_frobenius_error,
                color=COLORS[kernel],marker='o',markersize=3,linewidth=1.4,label=LABELS[kernel])
        ax.fill_between(bands.index.to_numpy(),bands['min'].to_numpy(),bands['max'].to_numpy(),
                        color=COLORS[kernel],alpha=.12,linewidth=0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(FEATURE_COUNTS)
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.legend()
    save(fig,figdir,'rff_approximation')

    rff_performance = []
    for kernel in ('gaussian','matern32'):
        for p in FEATURE_COUNTS:
            folder, meta, source = get_case(root,kernel,p)
            if source['clean_manifest_sha256'] != sources[0]['clean_manifest_sha256']:
                raise ValueError('RFF comparison changed the information set.')
            rff_performance.append({'kernel':kernel,'features':p,'seed':0,
                                    'sharpe':meta['oos_sharpe'],'native_loss':meta['raw_response_one_loss']})
    rff = pd.DataFrame(rff_performance)
    rff.to_csv(output/'rff_oos_performance.csv',index=False)
    fig, ax = canvas('Out-of-sample Sharpe ratio','Number of Random Fourier Features')
    for kernel in ('gaussian','matern32'):
        frame = rff.loc[rff.kernel.eq(kernel)].sort_values('features')
        ax.plot(frame.features,frame.sharpe,color=COLORS[kernel],
                linewidth=1.4,marker='o',markersize=3,label=LABELS[kernel])
    ax.set_xscale('log'); ax.set_xticks(FEATURE_COUNTS)
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.legend()
    save(fig,figdir,'rff_oos_performance')
    make_text(output, performance, peaks, sources[0]['feature_bank']['ell'])
    write_json(output/'manifest.json',{'status':'complete',
        'clean_manifest_sha256':sources[0]['clean_manifest_sha256'],
        'risk_free_sha256':digest(risk_free_file), 'calibration':calibration,
        'primary_source_pdf_sha256':'5ba2ce32de3ac9a21e5dd6396dc82d85378e48814910baf536a1af013372f424',
        'files':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}})


def make_text(output, performance, peaks, ell):
    p = performance.set_index('kernel')
    table = [r'\begin{table}[!htbp]\centering',r'\caption{Out-of-sample portfolio performance.}',
             r'\label{tab:performance}',r'\small',r'\begin{tabular}{lrrrrrrr}',r'\toprule',
             r'Policy & Mean (\%) & Vol. (\%) & SR & DD (\%) & Gross & TO/yr & Net SR$^*$\\',r'\midrule']
    for kernel in LABELS:
        row = p.loc[kernel]
        label = r'Mat\'ern-$3/2$' if kernel == 'matern32' else LABELS[kernel]
        table.append(f'{label} & {100*row.annual_mean_excess:.2f} & {100*row.annual_volatility:.2f} & '
            f'{row.sharpe:.2f} & {100*row.max_drawdown:.2f} & {row.mean_gross:.2f} & '
            f'{row.annual_target_turnover:.2f} & {row.net_sharpe_25bps_target_turnover_proxy:.2f}'+r'\\')
    table += [r'\bottomrule\end{tabular}',r'\par\smallskip\begin{minipage}{0.96\textwidth}\footnotesize',
        r'Mean and volatility are annualized excess-return statistics. DD is the maximum total-wealth drawdown. '
        r'Gross is the time average of $\sum_i|w_{i,t}|$. TO/yr is annualized L1 target-weight turnover, '
        r'not drift-adjusted trading volume. $^*$Net SR subtracts 25 basis points per unit of this turnover proxy; '
        r'it is not a full implementation-cost estimate. The first rebalance starts from cash.',
        r'\end{minipage}\end{table}']
    (output/'performance.tex').write_text('\n'.join(table))
    interior = all(v['interior'] for v in peaks)
    finding = ('Both representations display an interior maximum over the evaluated test-window grid. '
               'This is descriptive evidence of a finite-sample tradeoff, not a universal optimal-complexity law.'
               if interior else
               'At least one representation attains its test-window maximum at a boundary of the evaluated grid. '
               'These results therefore do not establish an interior optimum for both representations.')
    text = r'''\section{Empirical Portfolio Learnability}
\label{sec:empirics}
We study how much characteristic-based portfolio demand can be learned from a finite return history.
The data are the U.S. stock-level panel of \citet{jensenkellypedersen2023}.
We retain the JKP main observations of primary common securities on NYSE, AMEX, and NASDAQ
with CRSP share codes 10, 11, or 12, and exclude the nano size group.
Following the characteristic-panel design of \citet{didisheim2024aipt}, we retain 130 of the
153 published characteristics, remove stock-months with more than 30\% missing inputs,
and rank observed characteristics cross-sectionally each month to $[-0.5,0.5]$.
Our explicit timing modification is to determine the 130 best-covered characteristics using
only 1963--1972 and then freeze their identities. Thus we do not reproduce their full-sample
coverage selection. Remaining missing ranks are set to the neutral value zero.
The formation universe never depends on whether a future return happens to be observed;
unresolved payoffs must be reconciled before an empirical report is produced.

The first training period is 1963--1972, followed by validation in 1973--1977.
The training history expands annually while the preceding five formation years remain the
validation sample. A grid of 120 penalties is constructed from the initial training spectrum
and fixed thereafter. The validation response-one loss selects the penalty; coefficients are
then refitted using training and validation observations available at the first trade date.
Policies are re-estimated annually and weights are rebalanced monthly.
Formation dates run from January 1978 through December 2024; the corresponding realized
returns run from February 1978 through January 2025.

We compare Linear, Gaussian, and Mat\'ern-$3/2$ policies using the same information and calendar.
The nonlinear specifications use 10,000 Random Fourier Features. Their common bandwidth is
fixed at the median pairwise Euclidean distance among 1,000 characteristic vectors sampled
from the initial training period, $\ell=ELL$.
The Gaussian and Mat\'ern models impose different smoothness penalties on portfolio demand;
the independently sampled finite-feature spaces should not be described as a nested sequence.
A single positive portfolio scale, calibrated from the first Linear validation portfolio's
gross exposure, is fixed before its first out-of-sample return and used by all specifications.
There is no monthly gross cap or ex-post volatility normalization.

Figure~\ref{fig:wealth} reports cumulative total wealth, compounded as
$\prod_t(1+R_{f,t}+R^p_{t})$, before implementation costs.
Table~\ref{tab:performance} reports excess-return performance together with exposure and trading intensity.
The full-period annualized Sharpe estimates are SRL for Linear, SRG for Gaussian, and SRM for Mat\'ern-$3/2$.
These are point estimates from the fitted policies, not estimates of population representation gaps.
\begin{figure}[!htbp]\centering
\includegraphics[width=0.95\textwidth]{figures/cumulative_wealth.pdf}
\caption{Cumulative out-of-sample total wealth. The vertical axis is logarithmic.}
\label{fig:wealth}\end{figure}
\input{performance.tex}
\begin{figure}[!htbp]\centering
\includegraphics[width=0.90\textwidth]{figures/portfolio_exposure.pdf}
\caption{Gross exposure $\sum_i|w_{i,t}|$ and net dollar exposure $\sum_iw_{i,t}$ at each formation date.
Net dollar exposure is not an estimate of market beta.}
\label{fig:exposure}\end{figure}

We next hold each representation fixed and vary regularization. Let $\widehat\mu_{j,T}$
be the eigenvalues of the uncentered second-moment matrix of monthly managed payoffs.
The effective portfolio complexity is
\[
\widehat{\mathcal C}_T(\lambda)=
\sum_j\frac{\widehat\mu_{j,T}}{\widehat\mu_{j,T}+\lambda}.
\]
This measures the degrees of freedom retained by the regularization filter, not the number
of stocks, profitable factors, or nonzero portfolio holdings. The 10,000-coordinate maps can
have at most 732 nonzero sample eigenvalues before the final test window.
Figure~\ref{fig:spectrum} describes this finite managed-payoff spectrum. Eigenvalues measure
second-moment exposure in a specified feature geometry; profitability additionally depends
on mean-payoff alignment, and sampling uncertainty is not determined by eigenvalues alone.
\begin{figure}[!htbp]\centering
\includegraphics[width=0.85\textwidth]{figures/managed_spectrum.pdf}
\caption{Unnormalized managed-payoff eigenvalues before the final test window. Both axes are logarithmic;
the feature maps and the 732-observation refit are held fixed.}
\label{fig:spectrum}\end{figure}

Figures~\ref{fig:gaussiancomplexity} and~\ref{fig:materncomplexity} contrast historical fit
with subsequent performance in the final test window. Each point is a fitted penalty value;
lines simply connect evaluated points. FINDING
The stars identify test-grid maxima after observing twelve returns, while circles identify
the choices made using validation data. The stars do not enter the implemented backtest.
A one-year maximum is noisy and does not identify the population-optimal penalty or a convergence rate.
\begin{figure}[!htbp]\centering
\includegraphics[width=0.90\textwidth]{figures/complexity_gaussian_2024.pdf}
\caption{Gaussian: historical and subsequent Sharpe along the final regularization path.
The test returns are February 2024--January 2025.}
\label{fig:gaussiancomplexity}\end{figure}
\begin{figure}[!htbp]\centering
\includegraphics[width=0.90\textwidth]{figures/complexity_matern32_2024.pdf}
\caption{Mat\'ern-$3/2$: the same estimation calendar and interpretation as in the Gaussian comparison.}
\label{fig:materncomplexity}\end{figure}

Finally, we distinguish effective complexity from the numerical size of the kernel approximation.
We repeat the experiment at $P\in\{250,500,1000,2000,4000,10000\}$ using nested frequency banks
and the same bandwidth and characteristic panel. Increasing $P$ retains all previous frequencies
and phases, with the $\sqrt{2/P}$ normalization adjusted accordingly.
Figure~\ref{fig:rffperformance} reports the seed-zero out-of-sample performance at each $P$;
10,000 remains the prespecified main specification and is not selected using these test statistics.
Figure~\ref{fig:rffapproximation} separately checks approximation of the exact kernels on initial-training
characteristic vectors. Its central lines use seed zero and its shading spans seeds zero, one, and two;
it is not a confidence interval or an ensemble portfolio.
\begin{figure}[!htbp]\centering
\includegraphics[width=0.85\textwidth]{figures/rff_oos_performance.pdf}
\caption{Out-of-sample portfolio Sharpe versus the number of Random Fourier Features.
Each policy uses chronological validation; finite-$P$ comparisons use a shared, nested random bank.}
\label{fig:rffperformance}\end{figure}
\begin{figure}[!htbp]\centering
\includegraphics[width=0.85\textwidth]{figures/rff_approximation.pdf}
\caption{Relative Frobenius error between the exact characteristic kernel matrix and its RFF approximation.
Only characteristic vectors from the initial training sample enter this diagnostic.}
\label{fig:rffapproximation}\end{figure}

The empirical design separates three objects: the characteristic representation, its numerical
approximation, and the effective degrees of freedom retained after regularization.
Their effects should not be conflated. The regularization-path figures describe how a fixed
investment representation translates into historical fit and subsequent portfolio performance.
'''
    text = text.replace('ELL',f'{ell:.6f}').replace('SRL',f'${p.loc["linear","sharpe"]:.2f}$')
    text = text.replace('SRG',f'${p.loc["gaussian","sharpe"]:.2f}$').replace('SRM',f'${p.loc["matern32","sharpe"]:.2f}$')
    text = text.replace('FINDING',finding)
    (output/'empirics.tex').write_text(text)
    bibliography = r'''@article{jensenkellypedersen2023,
 author={Jensen, Theis Ingerslev and Kelly, Bryan and Pedersen, Lasse Heje},
 title={Is There a Replication Crisis in Finance?}, journal={The Journal of Finance},
 year={2023}, volume={78}, number={5}, pages={2465--2518}, doi={10.1111/jofi.13249}}
@techreport{didisheim2024aipt,
 author={Didisheim, Antoine and Ke, Shikun and Kelly, Bryan T. and Malamud, Semyon},
 title={APT or ``AIPT''? The Surprising Dominance of Large Factor Models},
 institution={National Bureau of Economic Research}, type={Working Paper},
 number={33012}, year={2024}, note={September 2024 version; Section 2.5}}
'''
    (output/'references.bib').write_text(bibliography)
    (output/'main.tex').write_text(r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}\usepackage{amsmath,amssymb,graphicx,booktabs}
\usepackage[round,authoryear]{natbib}\usepackage{microtype}\usepackage[hidelinks]{hyperref}
\begin{document}\input{empirics.tex}\clearpage
\bibliographystyle{plainnat}\bibliography{references}\end{document}
''')
