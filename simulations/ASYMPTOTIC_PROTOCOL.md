# Estensione asintotica, griglia fitta e dispersione tra repliche

Pilot e controllo di precisione sono fissati in `config/asymptotic.json`.
La successiva richiesta di molti piu punti temporali viene registrata in
`config/asymptotic_dense.json` prima di avviare la conferma. Cambiano solo
il nome del disegno e le due griglie di training della conferma; nessuna
replica di conferma era stata eseguita sul vecchio disegno.
I vecchi risultati restano in `outputs/new_montecarlo`; le figure mostrano
media +/- una STD e 95esimo percentile, mantenendo invariati i CSV originali.
I percentili del vecchio studio sono aggiunti nel separato `percentiles.csv`.

## Requisiti e disegno

- Stessa economia del codice fornito, D=s=3, b=2, Sharpe ottimo annualizzato 3.
- 100 repliche indipendenti per ciascuna delle cinque celle originali.
- 46 valori effettivi di T_train da 100 a 12800 per tutti i confronti:
  40 valori logaritmici arrotondati, uniti alle otto ancore originarie
  100,200,400,800,1600,3200,6400,12800.
- Riferimento N=600, rho=0.5: 54 valori effettivi fino a T_train=25600,
  unendo altri nove valori logaritmici da 12800 a 25600.
  Ogni punto comporta un nuovo fit delle stesse statistiche di training
  accumulate fino a quel T: non si interpolano risultati di simulazione.
  Il fit principale del riferimento contiene 28 punti con T_train>=1600.
- Asse temporale: soltanto T_train. Il pilot ha 800 date di training e 1600
  di validazione su uno stream separato; nessuna di queste date e aggiunta
  all'ascissa della simulazione di conferma.
- Fit principale su T_train>=1600, scelto prima dei nuovi risultati.
  Si pubblicano anche tutti i tagli prefissati 100,400,800,1600,3200,6400
  quando rimangono almeno tre punti. La finestra non e scelta dal miglior R2.
- Per C(lambda), spettro di training a T_train=12800 e fit sul tratto di
  penalita corrispondente a T_train da 1600 a 12800.
- Le curve Sharpe-complexity restano sul percorso bilanciato confermato:
  lambda(T_train)=C0*T_train^(-2/3).

## Precisione del kernel

Aumentare le repliche riduce l'incertezza della media ma non l'errore di
approssimazione. Per lo stesso kernel non periodico

    K(z,z') = (1+sqrt(3)*||z-z'||/ell)*exp(-sqrt(3)*||z-z'||/ell)

si confronta la banca originale RFF1000 con Nyström a ranghi 512,1024,2048.
I centri Sobol annidati sono fissati indipendentemente dai dati.
Se LL'=K(C,C)+1e-12*I, le features sono K(z,C)*L^(-T). Il training usa
la norma euclidea in questa base, coerente con la proiezione RKHS.

La costante positiva C0 viene selezionata su sei pilot indipendenti per N,
a T_train=800, minimizzando la perdita quadratica di validazione su una
griglia finita prefissata. Resta congelata per tutti i T e rho.
Un secondo pilot indipendente di otto repliche appaiate confronta i ranghi.
Si sceglie il piu piccolo rango che, sui T>=1600, differisce dal riferimento
2048 al massimo del 5% nella complexity media e di 0.002 nello Sharpe medio.
Pendenze e R2 non entrano nella scelta. Tutti gli esiti vengono pubblicati,
compresi i ranghi che falliscono. Il riferimento e anch'esso finito:
questo confronto e evidenza numerica di convergenza, non prova del limite infinito.

Riferimento metodologico primario:
[Rudi, Camoriano e Rosasco (2015), Less is More: Nystrom Computational Regularization](https://arxiv.org/abs/1507.04717).
Il lavoro riguarda regressione kernel; non sostituisce la verifica delle
ipotesi finanziarie e di dipendenza del manoscritto.

## Simulazione, valutazione e STD

I dati si generano in blocchi con stream separati per caratteristiche e
rendimenti. Aumentare T_train conserva esattamente il prefisso precedente;
aumentare il rango non cambia i dati. Si conservano solo statistiche
sufficienti e coefficienti, evitando cache di pannelli di grandi dimensioni.
La forma distributiva degli shock troncati e quella del codice originale.

La valutazione integra gli shock analiticamente su 8192 stati indipendenti.
Primo e secondo momento vengono stimati anche con una control variate
incrociata che usa il momento noto dell'oracolo p*=3/7. Per ciascuna meta f:

    c_other = m_other / p_other
    m_cv_f = m_f - c_other*(p_f-p*)
    s_cv_f = s_f - c_other^2*(p_f-p*)

L'altra meta e indipendente da f, quindi i momenti corretti restano non
distorti. Il rapporto di Sharpe rimane una stima non lineare. Il training
non usa l'oracolo. SR_raw, i momenti corretti e il confronto tra valutazioni
restano nei CSV: nessuna stima viene troncata o spostata per adattare un rate.

Le bande delle performance e le barre di errore mostrano media +/-
una deviazione standard campionaria tra le repliche (ddof=1), senza dividere
per sqrt(trials). I tratteggi colorati aggiungono il 95esimo percentile
empirico di ciascuna metrica, a ogni punto, con interpolazione lineare
tra le statistiche d'ordine delle 100 repliche. Questa interpolazione
definisce il quantile empirico, non inserisce punti temporali simulati.
Non sono SE o intervalli di confidenza della media. Gli intervalli bootstrap
degli esponenti sono distinti e vengono salvati nelle tabelle di fit.
Nel grafico Sharpe-complexity si mostra la STD su entrambe le coordinate;
il percentile dello Sharpe e posto alla complexity media del medesimo T,
senza attribuirgli il significato di quantile congiunto delle coordinate.
Nella griglia fitta, la STD verticale e una banda su ogni punto; le barre
orizzontali della complexity sono alle ancore temporali originarie per
evitare sovrapposizioni. Tutte le STD di ogni T restano nei CSV.
Le figure per il paper sono PDF vettoriali con font serif incorporati,
palette Okabe-Ito e simboli differenti per distinguere le serie anche in
scala di grigi. Le copie PNG sono a 300 dpi. I cinque grafici principali
sono separati dalle tre pagine di diagnostica nel PDF completo.

I benchmark rimangono gap(T_train)=O_P(T_train^(-2/3)), C(T_train)~T_train^(1/3),
C(lambda)~lambda^(-1/2) e gap(C)~C^(-2) lungo il percorso bilanciato.
Un upper bound non impone un'esatta pendenza finita per ogni singola economia.
Tutti i fit, anche quelli lontani dal benchmark, vengono conservati.

## Riproduzione

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.asymptotic --stage pilot --workers 4
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.asymptotic --stage fidelity --workers 4
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.asymptotic_dense --workers 4
python3 -m simulations.asymptotic_report simulations/outputs/asymptotic_dense_v3
python3 -m simulations.asymptotic_validate simulations/outputs/asymptotic_dense_v3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_new_montecarlo.py tests/test_asymptotic.py
```

Il controllo di hash rifiuta checkpoint provenienti da codice o disegni
scientifici diversi. `--workers` modifica soltanto la concorrenza.
`asymptotic_dense` riusa soltanto i pilot completati: verifica l'hash
scientifico del genitore, ammette esclusivamente le due nuove griglie di
conferma e il nome del disegno, e registra hash dei file e disegno originale
in `reused_stages.json`. Non vengono riusate policy di conferma di altre
griglie. Training, valutazione e raccolta usano le stesse funzioni scientifiche.
Per limitare l'uso del disco, ogni cella completa viene esportata e verificata
in `cells/` prima di eliminare soltanto i suoi coefficienti intermedi.
Un riavvio verifica gli hash delle celle esportate e riprende il resto.
Il riferimento viene eseguito per primo, senza cambiare dati o semi.

Gli output finali sono in `outputs/asymptotic_dense_v3/`; l'eventuale assenza di
`completed.json` indica che la conferma non e ancora stata raccolta interamente.
Per stabilire se un calcolo e ancora attivo si verifica anche il processo,
non la sola presenza di file di stato.
