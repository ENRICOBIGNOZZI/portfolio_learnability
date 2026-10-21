# Nuove simulazioni D=3

Questa e l'unica simulazione attiva della branch `empirics-montecarlo-20261019`.
Il codice originale ricevuto e conservato senza modifiche in
`source/user_simulation.py`; non viene eseguito all'importazione.
Gli empirici preesistenti restano disponibili. `branch_scope.json` elenca le
rimozioni delle vecchie simulazioni; la cartella di lavoro originale e intatta.

## Conferma completata: 500 repliche e figure da paper

`outputs/asymptotic_dense_v3` ha completato **500 repliche**, 100 per
ciascuna delle cinque celle: N=300,600,1200 a rho=0.5 e N=600 a rho=0,0.95.
Sono pubblicati 71.400 portafogli stimati e 32.500 punti dello spettro.
I confronti usano 46 T_train effettivi da 100 a 12800; il riferimento
N=600, rho=0.5 usa 54 tempi fino a 25600. Ogni punto e realmente ristimato.

- [Report completo con tutte le cinque figure](outputs/asymptotic_dense_v3/report.html)
- [PDF completo: cinque figure e tre pagine diagnostiche](outputs/asymptotic_dense_v3/requested_plots.pdf)
- [PDF singoli e immagini da paper](outputs/asymptotic_dense_v3/figures/)
- [Fit principali](outputs/asymptotic_dense_v3/main_rate_fits.csv) e [tutti i tagli](outputs/asymptotic_dense_v3/window_sensitivity.csv)
- [Repliche complete](outputs/asymptotic_dense_v3/confirmation_trials.csv) e [audit indipendente](outputs/asymptotic_dense_v3/validation.json)

Le figure mostrano media, STD campionaria fra repliche (ddof=1) e
95esimo percentile empirico. L'asse temporale e T_train; Sharpe-complexity
segue il percorso bilanciato. Le finestre 100,200,400,800,1600,3200,6400
sono tutte riportate; il fit principale resta da 1600.
La tabella con gap Sharpe-T -0.5264 appartiene al **primo run completo**,
`outputs/new_montecarlo`, con cinque T da 100 a 1600, distinto da questa conferma.
Il [PDF del riferimento](outputs/asymptotic_dense_v3/cells/N600_rho0.5/reference_plots.pdf)
mostra media, STD, 95esimo percentile e tutti i tagli piccoli e grandi.

| Rate, riferimento a 100 repliche | Benchmark | Primario da 1600 | Secondario da 200 |
|---|---:|---:|---:|
| Gap Sharpe vs T_train | -0.6667 | -0.6894 | -0.6487 |
| C vs T_train | +0.3333 | +0.3107 | +0.3263 |
| C vs lambda | -0.5000 | -0.4621 | -0.4757 |
| Gap Sharpe vs C | -2.0000 | -2.2179 | -1.9839 |

Il fit da 100, che usa tutta la griglia, da C(T)=+0.3331. Tutti i tagli
sono in `cells/N600_rho0.5/window_sensitivity.csv`. C(lambda) usa sempre
lo spettro a T_train=12800 e cambia soltanto la finestra di penalita.
I quattro R2 con esponente teorico imposto del fit primario superano 0.99.
Le pendenze libere restano diverse dai benchmark: i quattro fit primari
non superano il criterio aggiuntivo dell'intervallo bootstrap prefissato.
Il pilot a ranghi superiori studia separatamente l'errore numerico residuo.
La conferma a rango 1024 e completa; il controllo aggiuntivo ai ranghi
4096 e 8192 e separato e descritto sotto.

Anche la cella **N=300, rho=0.5 ha 100 repliche complete**, su 46 T_train
da 100 a 12800.
I [28 fit di N=300](outputs/asymptotic_dense_v3/cells/N300_rho0.5/window_sensitivity.csv)
comprendono tutti i tagli, anche 100,200,400,800. Il fit principale da 1600
usa 20 tempi effettivi e restituisce:

| Rate, N=300 | Benchmark | Esponente libero | R2, esponente teorico imposto |
|---|---:|---:|---:|
| Gap Sharpe vs T_train | -0.6667 | -0.6398 | 0.9966 |
| C vs T_train | +0.3333 | +0.3197 | 0.9978 |
| C vs lambda | -0.5000 | -0.4703 | 0.9958 |
| Gap Sharpe vs C | -2.0000 | -2.0006 | 0.9986 |

Per gap(C), il bootstrap 95% dell'esponente e [-2.0514,-1.9505]: passa
anche il criterio prefissato dell'intervallo e R2. Gli altri tre rate
mostrano scarti finiti e restano riportati. C(lambda) usa lo stesso spettro
a T_train=12800; nessuna scelta del taglio usa la qualita del fit.
La descrizione di questa cella mantiene N=600 come riferimento primario
del disegno generale.

La cella **N=600, rho=0 ha 100 repliche complete**, anch'essa su 46 T_train
da 100 a 12800.
I [28 fit di rho=0](outputs/asymptotic_dense_v3/cells/N600_rho0/window_sensitivity.csv)
mantengono ogni taglio dichiarato. Nel fit primario da 1600:

| Rate, N=600 e rho=0 | Benchmark | Esponente libero | R2, esponente teorico imposto |
|---|---:|---:|---:|
| Gap Sharpe vs T_train | -0.6667 | -0.6596 | 0.9989 |
| C vs T_train | +0.3333 | +0.3152 | 0.9962 |
| C vs lambda | -0.5000 | -0.4621 | 0.9931 |
| Gap Sharpe vs C | -2.0000 | -2.0920 | 0.9987 |

Il bootstrap 95% dell'esponente del gap(T) e [-0.6798,-0.6413], quindi
questo fit passa anche il criterio prefissato dell'intervallo e R2.
Gli altri tre fit primari non lo superano. Il taglio secondario da 800 da
gap(C)=-2.0105; la tabella pubblica anche tutti gli altri esiti, senza
scegliere la finestra dopo aver visto la pendenza. Le ampiezze fra rho
sono confrontate sotto sulla griglia comune 1600..12800.
Il riferimento rho=0.5 conserva il suo fit primario fino a 25600.

Anche **N=600, rho=0.95 ha 100 repliche complete**, su tutti i 46 T_train:
questa cella fa parte delle **500 repliche complete** del disegno. I
[28 fit di rho=0.95](outputs/asymptotic_dense_v3/cells/N600_rho0.95/window_sensitivity.csv)
restituiscono nel primario da 1600 gap(T)=-0.6918, C(T)=+0.3152,
C(lambda)=-0.4621 e gap(C)=-2.1945. I quattro R2 a esponente teorico imposto
superano 0.993, ma nessuno dei quattro intervalli delle pendenze libere
include il benchmark. Restano riportati tutti i tagli, anche piccoli.

Il [confronto delle ampiezze fra rho](outputs/asymptotic_dense_v3/rho_rate_constants.csv)
e completo: N=600, 100 repliche per rho e gli stessi 20 T_train da 1600 a
12800, con fit gap(T)=a_rho*T^(-2/3) senza intercetta.

| rho | Media a | STD di a | 95esimo percentile di a | Rapporto a/a_0.5 | CI 95% del rapporto |
|---|---:|---:|---:|---:|---:|
| 0 | 12.4648 | 1.7235 | 15.1504 | 0.9848 | [0.9556,1.0120] |
| 0.5 | 12.6567 | 1.5147 | 15.2322 | 1.0000 | [1.0000,1.0000] |
| 0.95 | 12.4584 | 1.5937 | 15.0597 | 0.9843 | [0.9547,1.0147] |

Gli intervalli appaiati dei due confronti includono 1: le ampiezze osservate
sono molto simili, coerentemente con l'ortogonalita condizionale del DGP.
Sono coefficienti descrittivi a T finito, distinti dalle costanti dei bound.
Sulla griglia comune le pendenze libere del gap(T) sono -0.6596,-0.6794,
-0.6918. Il riferimento mantiene separatamente il suo fit primario fino
a 25600; nessuna finestra e scelta in base ai risultati.

La cella **N=1200, rho=0.5 e completa su 100 repliche e 46 T_train**.
I [28 fit di N=1200](outputs/asymptotic_dense_v3/cells/N1200_rho0.5/window_sensitivity.csv)
restituiscono nel primario da 1600:

| Rate, N=1200 | Benchmark | Esponente libero | R2, esponente teorico imposto |
|---|---:|---:|---:|
| Gap Sharpe vs T_train | -0.6667 | -0.6618 | 0.9997 |
| C vs T_train | +0.3333 | +0.3164 | 0.9967 |
| C vs lambda | -0.5000 | -0.4645 | 0.9940 |
| Gap Sharpe vs C | -2.0000 | -2.0912 | 0.9988 |

Il bootstrap 95% del gap(T) e [-0.6807,-0.6434], quindi questo fit passa
il criterio prefissato di intervallo e R2. Gli altri tre primari non lo
superano. Il taglio secondario da 400 da gap(C)=-1.9849; quello da 200
da C(T)=+0.3328. Questi confronti restano etichettati come secondari,
senza modificare il riferimento N=600 o il taglio principale da 1600.

Il pilot numerico indipendente gia completato ha otto repliche per
rappresentazione; i suoi esiti non sono risultati della conferma a 100.

| Quantita | Benchmark | Pilot Nyström1024, 8 repliche |
|---|---:|---:|
| Gap Sharpe vs T_train | -0.6667 | -0.6736 |
| C vs T_train | +0.3333 | +0.3151 |
| C vs lambda | -0.5000 | -0.4621 |
| Gap Sharpe vs C | -2.0000 | -2.1365 |

I fit temporali del pilot usano T_train=1600,3200,6400,12800; C(lambda)
usa lo spettro a T_train=12800 e la finestra di penalita prefissata.
Fonti: `outputs/asymptotic_v2/fidelity_trials.csv` e
`fidelity_rate_diagnostics.json`. Pendenze e R2 non hanno selezionato il rango.

Il [confronto secondario dei tagli nel pilot](outputs/asymptotic_v2/fidelity_windows.html)
pubblica tutti i tagli osservabili 800,1600,3200, per tutte e quattro le
rappresentazioni e tutti e quattro i rate: 48 fit. Da 800 si usano tutti i
cinque punti disponibili. Al riferimento Nyström2048 si ottengono
gap(T)=-0.6522, C(T)=+0.3285, C(lambda)=-0.4790, gap(C)=-1.9840.
Al rango selezionato 1024 gli stessi esponenti sono -0.6523,+0.3198,
-0.4661,-2.0375. Questi esiti restano preliminari a otto repliche e non
cambiano il fit principale da 1600 o la selezione numerica del rango.
I tagli 100,200,400 richiedono i nuovi dati: non vengono attribuiti al pilot.
Per riprodurre l'analisi dai CSV e cache originali:
`python3 -m simulations.fidelity_windows simulations/outputs/asymptotic_v2`.
La stessa invocazione con `--validate` verifica i 48 fit dai soli CSV
pubblicati e riproduce le pendenze della tabella principale precedente.
Per ricostruire il report del riferimento:
`python3 -m simulations.reference_report simulations/outputs/asymptotic_dense_v3`;
la stessa invocazione con `--validate` verifica dati, STD, percentili e 28 fit.
Per le sole tabelle di una cella esportata si aggiungono
`--tables-only --N 300 --rho 0.5`; per verificarle si usano
`--validate --N 300 --rho 0.5`. La griglia di training e i semi del disegno
restano quelli originali; il riferimento locale serve a fittare C(lambda)
per la cella richiesta.

Il [protocollo asintotico](ASYMPTOTIC_PROTOCOL.md) definisce l'estensione di
T_train fino a 12800 e 25600, il controllo del rango e le finestre finali di fit.
Il [disegno fitto](config/asymptotic_dense.json) ha 46 punti effettivi per
confronto e 54 nel riferimento, con 28 punti nella coda T_train>=1600.
Ogni punto e ristimato. Le figure mostrano media +/- deviazione standard
campionaria tra repliche (ddof=1) e 95esimo percentile empirico, con asse
temporale T_train esplicito. Non si usano SE per le bande.
I PDF vettoriali hanno font serif incorporati e colori Okabe-Ito;
media, dispersione, percentile e benchmark hanno segni distinti.
I CSV e i valori delle simulazioni originali restano invariati; i loro
percentili sono aggiunti in `outputs/new_montecarlo/percentiles.csv`.

La sensibilita ai tagli include anche quelli piccoli: 100,200,400,800,
oltre a 1600,3200,6400. Il fit principale resta da 1600, i dati restano
tutti visibili e si pubblicano le quattro pendenze per ogni finestra,
compresa C(lambda) sullo stesso spettro di training. L'aggiunta secondaria
di 200 e registrata in `analysis_spec.json`, separata dal disegno congelato.

Il [pilot di precisione aggiuntivo](PRECISION_PROTOCOL.md) confronta ranghi
4096 e 8192, senza cambiare kernel, otto semi appaiati o C0 del pilot.
Parte da T_train=100 e arriva a 25600, con tutti i tagli piccoli e grandi.
Il rango viene valutato con un certificato di errore della complessita
empirica; le pendenze non determinano la selezione. Questo calcolo resta
distinto dalla conferma a 100 repliche. Il rango **4096 ha completato tutte
le otto repliche e la valutazione**, su nove T_train da 100 a 25600.
Il primario da 1600 restituisce gap(T)=-0.6908, C(T)=+0.3259,
C(lambda)=-0.4802 e gap(C)=-2.1184, con R2 a potenza teorica >0.997.
Gli intervalli bootstrap includono il benchmark per i due rate del gap,
non per i due rate di C. Il certificato massimo e 2.4938%, sopra la soglia
numerica prefissata del 2%; non viene quindi dichiarato selezionato.
I [28 fit del rango 4096](outputs/asymptotic_precision_v4/precision_nystrom4096_fits.json)
e i dati grezzi sono pubblicati con audit indipendente, inclusi gli
intervalli bootstrap. Il confronto con 8192 e le nuove figure restano da
completare. La finestra e lo spettro del nuovo pilot sono diversi dalla
precedente tabella Nyström1024: vedere il protocollo per il confronto corretto.

## Output del primo run completo: T_train da 100 a 1600

- `outputs/new_montecarlo/report.html`: figure, metodo, risultati e limiti dei fit.
- `outputs/new_montecarlo/requested_plots.pdf`: cinque figure richieste e audit della costante originale.
- `outputs/new_montecarlo/trials.csv`: tutte le 100 repliche per ciascuna delle cinque celle, per entrambe le regole di lambda.
- `outputs/new_montecarlo/summary.csv`: medie, deviazioni standard ed errori Monte Carlo.
- `outputs/new_montecarlo/rate_fits.csv` e `.json`: esponenti liberi, intervalli bootstrap, fit a esponente teorico fissato e diagnostica del floor.
- `outputs/new_montecarlo/complexity_lambda_trials.csv`: intera superficie della complexity a T=1600.
- `outputs/new_montecarlo/pilot.csv`: tutti i punteggi della selezione preliminare.
- `outputs/new_montecarlo/manifest.json`, `checks.json`: configurazione, provenienza e controlli.

## Risultati del primo run completo

500 repliche completate, 100 per cella. Riferimento dei fit: N=600, rho=0.5;
lo Sharpe viene fittato come gap rispetto a 3. Nessun T e escluso.

| Grafico | Esponente teorico | Esponente libero | R2 con esponente teorico |
|---|---:|---:|---:|
| Sharpe vs T, N diversi | -0.6667 | -0.5264 | 0.9741 |
| Sharpe vs T, rho diversi | -0.6667 | -0.5264 | 0.9741 |
| Sharpe vs C | -2.0000 | -1.5719 | 0.9580 |
| C vs T | 0.3333 | 0.3352 | 0.9965 |
| C vs lambda | -0.5000 | -0.4579 | 0.9879 |

I cinque fit a esponente teorico hanno R2>0.90, ma le pendenze finite non
coincidono tutte con i benchmark asintotici. Per il gap di Sharpe vs T il fit
aggiungendo un floor ha R2=0.9984; e una diagnostica della rappresentazione
finita e non dimostra la convergenza verso SR*=3. Le curve per rho sono quasi
sovrapposte, coerentemente con la forte ortogonalita condizionale del DGP.
A T=1600 lo Sharpe medio delle policy calibrate e circa 2.89 in tutte le celle.
L'errore massimo relativo E4 e 4.12e-16; tutti i 5000 record delle due regole
superano l'audit indipendente. PDF e singole figure sono renderizzati e verificati.

## Riproduzione

```sh
python3 -m pip install -r requirements.txt
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.run --workers 5
python3 -m simulations.run --render-only
python3 -m simulations.validate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

Le versioni effettivamente usate sono fissate anche in `simulations/requirements.txt`
(per una riproduzione in un ambiente dedicato; i requisiti empirici restano separati).

I checkpoint delle policy sono locali e ignorati da Git, mentre dati aggregati,
repliche, codice, PDF e PNG sono versionati. Un hash di codice e configurazione
impedisce il riuso di checkpoint incompatibili. Una modifica scientifica richiede
una nuova directory `--output`. `--workers` cambia soltanto la concorrenza.

## Disegno fissato prima dell'esecuzione

`config/paper.json` fissa 100 repliche e T_train=100,200,400,800,1600.
Le cinque celle coprono esattamente i confronti richiesti: N=300,600,1200 a
rho=0.5 e rho=0,0.5,0.95 a N=600. Non si eseguono le quattro combinazioni
incrociate non usate nei grafici. Non sono scartati trial o estremi di T.
La banca RFF e quella dell'utente: D=s=3, M=1000, ell=1, seed=42.
SR ottimo marginale annualizzato=3 per tutti gli N, per calibrazione.
Il confronto tra N include questa ricalibrazione delle ampiezze e della scala
dello spettro; non tiene fissa l'intera economia aumentando soltanto gli asset.

La regola originale `lambda=T^(-2/3)` viene mantenuta come confronto.
Le matrici dei rendimenti gestiti hanno autovalori molto piu piccoli di
questa penalita. Un pilot indipendente usa quattro training di 200 date e
800 date di validazione, a rho=0.5, per scegliere una costante positiva per
ogni N su una griglia finita prefissata tramite perdita quadratica.
La costante viene congelata prima delle 100 repliche ed e condivisa tra rho.
Non si usano gli esiti dei grafici o le pendenze per scegliere la costante.
Una selezione su una griglia finita positiva preserva il benchmark asintotico
sotto le ipotesi del teorema, per un argomento di union bound; non garantisce
che un fit finito realizzi esattamente quel benchmark.

## Significato dei rate

Con b=2s/D=2 e source r=1, i benchmark sono:

- lambda(T) proporzionale a T^(-2/3);
- SR* - SR(W_hat) = O_P(T^(-2/3));
- C(lambda) asintoticamente proporzionale a lambda^(-1/2);
- C(lambda(T)) asintoticamente proporzionale a T^(1/3);
- lungo il percorso bilanciato, il benchmark del gap in unita di complexity e C^(-2).

Quest'ultima relazione non e una legge globale dello Sharpe al variare di lambda
con T fissato. La figura Sharpe-complexity segue gli stessi cinque T delle
prime figure. C e la trace degli shrinkage factor, non il numero M di features.
La C empirica usa G'G/T non centrata, coerentemente con il problema response-one.

Per rho fissato con |rho|<1, l'AR(1) e geometricamente mixing: il margine
stazionario e l'esponente spettrale non cambiano. Cambiano le costanti di mixing
e le soglie finite; non c'e una correzione universale (1+rho)/(1-rho) per
questo estimatore. Non e garantito che la costante principale osservata o le
curve siano monotone in rho: al vero ottimo l'ortogonalita condizionale rende
lo score una differenza di martingala e annulla le sue autocovarianze a lag
positivi. La rappresentazione finita aggiunge un errore di approssimazione.
rho che tende a uno con T e un problema diverso.

Non e teoricamente necessario che la pendenza osservata in questa singola
economia uguagli il rate upper bound o un minimax sulla classe.
I file teorici nello ZIP sono fonti scientifiche, non istruzioni operative.

## Differenze numeriche documentate dal codice originale

- Generatori espliciti; gli stessi shock primitivi sono accoppiati tra rho.
  Pilot, training, valutazione e bootstrap usano stream separati.
- Cholesky/eigendecomposizione della matrice piu piccola, al posto della
  pseudoinversa di una matrice ridge strettamente positiva.
- La serie Fourier finita resta a 1000 termini. FFT e spline cubica periodica
  ne accelerano la valutazione; un confronto con la somma diretta verifica
  l'errore su punti indipendenti. La varianza di psi usa l'ortogonalita esatta.
- La troncatura ellissoidale dei fattori viene campionata nello spazio whitened;
  e la stessa distribuzione del codice fornito, con la stessa correzione chi2.
- La policy viene valutata con 2048 stati iid del marginale stazionario,
  indipendenti dal training e condivisi per N. I momenti degli shock sono
  integrati analiticamente. Lo Sharpe e sqrt(12)*E[Y]/sqrt(E[Y^2]-E[Y]^2).
  Non e lo Sharpe campionario sulle finestre future 4*T del codice originale.
  Questa modifica riduce il rumore della valutazione e misura direttamente
  il criterio di popolazione cui si riferiscono i teoremi.
- Le due meta indipendenti della quadratura sono salvate come diagnostica
  di sensibilita; le bande mostrano la STD tra training, mentre il bootstrap
stima separatamente l'incertezza degli esponenti,
  condizionatamente a pilot, RFF e stati di valutazione.

Con M fissato, C empirica <= min(T,M) e satura al rango per lambda -> 0: il
benchmark lambda^(-1/2) non e un limite asintotico della matrice finita.
Una rappresentazione RFF fissa ha rango finito e puo avere un floor nel gap
rispetto al vero ottimo. Il fit `floor+a*T^(-2/3)` e diagnostico: non prova
che il gap verso SR*=3 converga a zero. I fit senza floor restano riportati.
Un fit e classificato buono soltanto con R2>=0.90 e benchmark entro il 95%
bootstrap dell'esponente libero. Le figure non sono modificate per farlo passare.

## Risorse empiriche locali

La worktree locale collega `data/` e `results/` ai dati privati della cartella
originale, senza copiarli o versionarli. Per gli audit storici che verificano
vecchi hash, `paper/theory/` e un collegamento locale ignorato ai vecchi sorgenti.
Questi sorgenti non appartengono al tree della nuova branch e non sono inclusi
in `paper/main.tex`. Un log storico ignorato e collegato nello stesso modo.
Una clonazione senza queste risorse puo eseguire i test numerici, mentre i test
empirici su dati reali richiedono il relativo archivio locale.
