# Pilot appaiato con maggiore precisione numerica

Disegno aggiuntivo richiesto dopo la tabella preliminare a otto repliche.
Non sostituisce la conferma a 100 repliche per ciascuna delle cinque celle.
Il kernel resta il Matérn 3/2 non periodico sul cubo originale, con la
stessa economia, calibrazione e penalizzazione del pilot completato.

`config/asymptotic_precision.json` congela prima del calcolo:

- N=600, rho=0.5; le stesse otto repliche appaiate del pilot di precisione;
- ranghi Nyström 4096 e 8192, centri Sobol annidati indipendenti dai dati;
- T_train = 100,200,400,800,1600,3200,6400,12800,25600;
- C0=5.62341325190349e-7, gia selezionato dal pilot separato;
- percorso bilanciato lambda=C0*T_train^(-2/3), moltiplicatori 0.3,1,3;
- 8192 stati di valutazione indipendenti, stessi stati dei confronti precedenti;
- fit principale da 1600 e tutti i tagli secondari 100,200,400,800,1600,3200,6400;
- C(lambda) sullo spettro a T_train=25600, con ogni finestra di lambda
  identificata dal corrispondente tratto del percorso bilanciato.

L'obiettivo e misurare l'errore numerico residuo, non scegliere il rango
dalla pendenza o selezionare i risultati che coincidono con gli esponenti.
La legge spettrale asintotica non impone pendenze esattamente uguali su
intervalli finiti. Tutti gli intervalli e gli esiti, anche sfavorevoli,
rimangono pubblicati.

## Certificato per la complessita empirica

Sia A la matrice Gram T x T dei rendimenti gestiti, normalizzata per T,
calcolata con il kernel completo, e B la sua proiezione Nyström con jitter
positivo. Si ha A >= B >= 0 nell'ordine semidefinito positivo. Per
f_lambda(u)=u/(u+lambda), la monotonia degli autovalori e
0 <= f'_lambda(u) <= 1/lambda danno

    0 <= tr f_lambda(A) - tr f_lambda(B) <= tr(A-B)/lambda.

La traccia di A richiede soltanto le diagonali:
R_t' K(Z_t,Z_t) R_t / (N^2*T). Non occorre formare l'intera Gram T x T.
Il bound relativo pubblicato divide il limite superiore per C(B): e
conservativo anche rispetto all'errore relativo con denominatore C(A).
Si controllano tutte le otto repliche, tutti i T del percorso bilanciato
e tutte le lambda della finestra principale sullo spettro fissato.
Si sceglie il rango piu piccolo che certifica al massimo il 2% di errore.
Se nessuno dei due ranghi passa, si pubblica l'esito senza dichiarare
risolta la precisione. Questo e un certificato della complessita sul
campione, non della complessita di popolazione o del rate dello Sharpe.

Riferimento metodologico generale per le approssimazioni kernel a rango
ridotto: [Bach, 2013](https://proceedings.mlr.press/v30/Bach13.pdf).
Il certificato specifico sopra e verificato anche costruendo direttamente
la Gram completa in un test piccolo, indipendente dal solver Nyström.

## Intervalli numerici per le pendenze di C

Il certificato puntuale racchiude anche la pendenza che si otterrebbe con
il kernel empirico completo, sugli stessi dati e sulla stessa finestra.
Per replica e punto vale `C_B <= C_A <= min(C_B + Delta_C, T_train)`.
Si mediano prima questi limiti tra repliche, ottenendo `L_i <= mean(C_A)_i <= U_i`:
il fit pubblicato usa il logaritmo della media, non la media dei logaritmi.

Posto `x_i=log(T_train)` oppure `log(lambda)`, la pendenza OLS con intercetta
e `beta = sum(w_i*log(mean(C_A)_i))`, dove
`w_i=(x_i-mean(x))/sum((x_i-mean(x))^2)`. Il minimo numerico usa `log(L_i)`
per i pesi positivi e `log(U_i)` per quelli negativi; il massimo inverte
le scelte. Si ottiene un intervallo conservativo condizionato ai campioni
osservati. Non serve conoscere gli autovalori dell'intera Gram T x T.

Questi intervalli numerici sono distinti dagli intervalli bootstrap della
pendenza tra repliche. Riguardano la complessita empirica; non racchiudono
la complessita di popolazione o il gap Sharpe del portafoglio con kernel
completo. Se un benchmark asintotico e esterno all'intervallo, la pendenza
del kernel empirico completo su quella finestra e diversa dal benchmark:
questo non contraddice un rate asintotico.

Il report finale produce `precision_complexity_slope_enclosures.csv` per
C(T) e C(lambda), su tutti i tagli e per entrambi i ranghi. Il validatore
ricostruisce i limiti dalle tracce grezze e verifica gli estremi con una
regressione indipendente. Due test controllano gli estremi di tutte le
combinazioni puntuali e l'inclusione delle pendenze calcolate direttamente
con una Gram Matérn completa. La selezione del rango resta determinata
esclusivamente dalla soglia del 2% sul livello di C.

## Memoria, ripresa e output

Le righe dei rendimenti gestiti risiedono in un memmap temporaneo;
non si conservano pannelli di titoli. Ridge usa la forma duale o primale
secondo la dimensione minore. Si usa un solo worker per limitare la RAM.
La valutazione contrae direttamente le policy con i momenti condizionati,
senza un operatore di valutazione rango x rango. Un test confronta questa
valutazione con l'operatore indipendente del codice precedente.

Ogni replica salva atomicamente coefficienti, spettro, complessita e tracce.
Soltanto dopo elimina il proprio scratch. Il riavvio riusa le repliche
complete; un hash del codice scientifico e del disegno impedisce di mescolare
versioni. I file finali includono CSV per tutte le repliche, certificati,
spettro, fit per tutti i tagli e manifest SHA256. Otto repliche restano
un pilot: STD e 95esimo percentile empirico hanno precisione limitata.

Esecuzione, un rango alla volta:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -u -m simulations.asymptotic_precision --rank 4096 --workers 1
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -u -m simulations.asymptotic_precision --rank 8192 --workers 1
python3 -m simulations.precision_report simulations/outputs/asymptotic_precision_v4
python3 -m simulations.precision_validate simulations/outputs/asymptotic_precision_v4
```

Il report richiede il completamento di entrambi i ranghi e pubblica un PDF
vettoriale di tre pagine: livelli Sharpe, quattro rate e sensibilita di
tutte le pendenze ai tagli. Non dichiara superata la soglia di precisione
se il certificato conservativo non la dimostra. La validazione ricalcola
dal CSV le statistiche, la regolarizzazione, i momenti, le pendenze,
le griglie di fit e la selezione del rango, e verifica gli SHA dei dati.
Il caso di prova completo ha inoltre verificato che sostituire una STD
con il suo SE venga rilevato. Il caso di prova e una verifica del codice,
distinta dai risultati scientifici sotto riportati.

## Primo rango completato: 4096

Otto repliche complete e valutate su tutti i nove T_train da 100 a 25600:
216 portafogli, 520 punti dello spettro e tutti i 28 fit dichiarati.
Il fit principale usa 1600,3200,6400,12800,25600; C(lambda) usa lo spettro
a 25600 nella finestra prefissata. Gli esponenti liberi sono:

| Rate | Benchmark | Rango 4096 | R2 con esponente teorico imposto |
|---|---:|---:|---:|
| Gap Sharpe vs T_train | -0.6667 | -0.6908 | 0.9984 |
| C vs T_train | +0.3333 | +0.3259 | 0.9992 |
| C vs lambda | -0.5000 | -0.4802 | 0.9982 |
| Gap Sharpe vs C | -2.0000 | -2.1184 | 0.9978 |

Gli intervalli bootstrap del gap(T) e del gap(C) includono i benchmark;
quelli dei due rate di C non li includono. Il massimo certificato relativo
e **2.4938%**, sopra la soglia prefissata del 2%: il rango 4096 non passa
questa soglia e non viene dichiarato selezionato. Il rango 8192 resta
da completare, come pure il confronto e le figure finali. La finestra
temporale e lo spettro differiscono da quelli della precedente tabella
Nyström1024: le pendenze non vanno confrontate come se il solo rango cambiasse.

I dati e i fit sono in `outputs/asymptotic_precision_v4/precision_nystrom4096_*`.
L'audit del singolo rango si riproduce senza richiedere il report finale:

```sh
python3 -m simulations.precision_validate simulations/outputs/asymptotic_precision_v4 --rank 4096
```

L'audit verifica manifest, momenti, griglie, tracce e certificati, e
ricalcola da percorsi grezzi le pendenze, i fit a potenza fissata e gli
intervalli bootstrap con regressione centrata indipendente da `polyfit`.
Per gap(C) ricampiona insieme entrambe le coordinate. Il risultato distingue
esplicitamente dati corretti, soglia numerica superata e studio completo;
nessun report grafico viene dichiarato verificato da questo controllo parziale.

Anche gli intervalli numerici del rango 4096 sono calcolati su tutte le
otto repliche, mantenendo tutti i tagli. Nel primario da 1600 a 25600:

| Pendenza | Fit Nystrom4096 | Minimo kernel empirico completo | Massimo kernel empirico completo | Benchmark compreso |
|---|---:|---:|---:|---|
| C(T) | +0.325929 | +0.321395 | +0.335833 | Si: +1/3 |
| C(lambda) | -0.480170 | -0.497703 | -0.470953 | No: -1/2 |

La seconda riga esclude l'uguaglianza esatta del fit finito del kernel
empirico completo con -1/2 su quei campioni e quella finestra; il benchmark
asintotico resta tale. La prima riga e compatibile con +1/3 ma non determina
la pendenza effettiva del kernel completo. Sono limiti numerici condizionati,
distinti dai bootstrap e dalla complessita di popolazione.
I 14 intervalli sono in `precision_nystrom4096_complexity_slope_enclosures.csv`,
con manifest e controllo indipendente in
`precision_nystrom4096_slope_enclosures_validation.json`. La selezione del
rango e il completamento del confronto con 8192 restano separati.

## Ripresa quando varia lo spazio libero

Il controllo preventivo del disco ha interrotto un tentativo a rango 4096
dopo quattro repliche salvate. I quattro checkpoint sono stati verificati
e riusati, con codice scientifico, configurazione, semi e penalita invariati.
Anche i 20 checkpoint gia salvati per N=1200 hanno mantenuto i loro hash;
il solo anticipo di quella cella e stato ridotto da due worker a uno.

`asymptotic_precision_pipeline` coordina la ripresa dello stesso modulo
scientifico. Prima di nuovi training richiede il margine del solver originale
(scratch piu 2 GiB), piu 256 MiB, in due controlli consecutivi. Se tutte le
policy del rango sono gia presenti, servono soltanto valutazione ed esportazione.
Si riprovano esclusivamente le interruzioni esplicite per margine di disco
o ENOSPC; gli altri errori fermano il tentativo. Le esecuzioni precedenti
restano in `cache/execution_history/`, senza modificare dati o selezionare semi.

Anche il rango 8192 ha incontrato il controllo di spazio prima della quarta
replica, dopo tre checkpoint completi e verificati. Il coordinatore ha
atteso il margine richiesto e ha ripreso automaticamente dai tre checkpoint,
con gli stessi semi e codice scientifico. Non e stato necessario cancellare
dati o cambiare la dimensione dello studio.

Per contenere il picco di memoria, il rango maggiore attende il completamento
di tutte le celle della conferma. Poi il report e il validatore vengono eseguiti
automaticamente; restano la revisione visiva e la pubblicazione.

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -u -m simulations.asymptotic_precision_pipeline \
  simulations/outputs/asymptotic_precision_v4 \
  --wait-main-output simulations/outputs/asymptotic_dense_v3
```
