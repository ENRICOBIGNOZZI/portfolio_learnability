# Nuove simulazioni D=3

Questa e l'unica simulazione attiva della branch `empirics-montecarlo-20261019`.
Il codice originale ricevuto e conservato senza modifiche in
`source/user_simulation.py`; non viene eseguito all'importazione.
Gli empirici preesistenti restano disponibili. `branch_scope.json` elenca le
rimozioni delle vecchie simulazioni; la cartella di lavoro originale e intatta.

## Output

- `outputs/new_montecarlo/report.html`: figure, metodo, risultati e limiti dei fit.
- `outputs/new_montecarlo/requested_plots.pdf`: cinque figure richieste e audit della costante originale.
- `outputs/new_montecarlo/trials.csv`: tutte le 100 repliche per ciascuna delle cinque celle, per entrambe le regole di lambda.
- `outputs/new_montecarlo/summary.csv`: medie, deviazioni standard ed errori Monte Carlo.
- `outputs/new_montecarlo/rate_fits.csv` e `.json`: esponenti liberi, intervalli bootstrap, fit a esponente teorico fissato e diagnostica del floor.
- `outputs/new_montecarlo/complexity_lambda_trials.csv`: intera superficie della complexity a T=1600.
- `outputs/new_montecarlo/pilot.csv`: tutti i punteggi della selezione preliminare.
- `outputs/new_montecarlo/manifest.json`, `checks.json`: configurazione, provenienza e controlli.

## Riproduzione

```sh
python3 -m pip install -r requirements.txt
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.run --workers 5
python3 -m simulations.run --render-only
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
  di sensibilita; le bande e il bootstrap includono l'incertezza tra training,
  condizionatamente a pilot, RFF e stati di valutazione.

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
