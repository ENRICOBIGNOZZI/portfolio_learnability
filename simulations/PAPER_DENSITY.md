# Densificazione della figura per il paper

Calcolo avviato dopo la richiesta di maggiore densita. Si mantengono
**N=600, rho=0.5, rango 8192, otto repliche e C0 invariato**, aggiungendo
training intermedi reali nella finestra gia scelta. I risultati della
griglia piu densa non sono ancora disponibili.

La finestra del paper passa da sei a **26 valori reali di T_train**:

```
800, 919, 1056, 1213, 1393, 1600, 1838, 2111, 2425,
2786, 3200, 3676, 4222, 4850, 5572, 6400, 7352, 8445,
9701, 11143, 12800, 14703, 16890, 19401, 22286, 25600
```

Si conservano inoltre 100,200,400: 29 training per replica e 232 policy
complessive, comprese le 72 policy dei nove ancoraggi precedenti.
Il nuovo calcolo riguarda soltanto il percorso bilanciato,
lambda=C0*T_train^(-2/3). I confronti con gli altri moltiplicatori rimangono
nelle analisi originali. C(lambda) mantiene il medesimo spettro a 25600;
la maggiore densita riguarda le curve in T_train e il percorso gap(C).
I marcatori aggiunti saranno risultati simulati, non interpolazioni.

Si usano le **stesse otto traiettorie**, non nuove repliche indipendenti.
La finestra da 800 e stata dichiarata prima della nuova simulazione e
non viene riselezionata dai nuovi risultati. Il primario da 1600 e tutti
i tagli di sensibilita restano disponibili. Le pendenze potranno cambiare
per effetto dei nuovi punti inseriti nella regressione.

## Calcolo esatto piu veloce

Con una sola penalita serve il ridge a quella penalita e la sua complessita,
senza diagonalizzare nuovamente ogni matrice. Per S=G'G/T, oppure GG'/T
nella forma duale, e d=dim(S), si usano le identita esatte

```
theta = soluzione ridge mediante Cholesky
C = d - lambda * tr((S + lambda I)^(-1))
tr((S + lambda I)^(-1)) = ||L^(-1)||_F^2,  L L' = S + lambda I
```

La traccia viene calcolata per intero, senza stimatori stocastici o
troncamenti spettrali. Sette test confrontano policy e C con il solver
spettrale nelle forme primale e duale, anche in presenza di collinearita.
Un benchmark indipendente a dimensione 1600 misura 2.002 s per il fit
spettrale e 0.149 s per questo passaggio, circa 13.5 volte piu veloce,
con errore relativo sulla policy 9.5e-12 ed errore assoluto su C 2.4e-10.
Questo rapporto riguarda il passaggio di algebra lineare, non l'intero run.

Lo spettro completo a T_train=25600 proviene dallo **stesso identico
campione finale** del run v4: modello, semi, centri e normalizzazione
coincidono. `prior_eigenvalues.csv` e `prior_trace_anchors.csv` ne conservano
i valori con precisione sufficiente al round-trip. La provenienza include
hash del disegno, del codice e dei checkpoint originali. Il nuovo training
ricostruisce tutti i dati e controlla tracce e C degli ancoraggi precedenti;
il report verifica anche lo Sharpe e i momenti delle 72 policy appaiate.
Il riuso dello spettro serve alla curva C(lambda); le policy dei nuovi T
sono stimate dai relativi prefissi di training.

Il primo tentativo con il solver spettrale completo, `paper_density_v1`,
e stato fermato intenzionalmente **prima di produrre checkpoint completi**
per usare il metodo equivalente piu veloce. Griglia, finestra e parametri
restano quelli gia dichiarati. La sostituzione e registrata in
`paper_density_v1/superseded_attempt.json`; il disegno iniziale rimane
pubblicato. Il calcolo effettivo e in **paper_density_v2**.

## Protocollo e riproduzione

Il disegno e in `config/paper_density_fast.json` e
`outputs/paper_density_v2/plan.json`. `training_backend.json` lega codice,
input spettrali e risultati agli hash. Durante il run non modificare il
modulo `paper_density_compute.py`, gli input congelati o i moduli scientifici
precedenti. Si usa un solo worker con i thread numerici limitati a uno.

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m simulations.paper_density_compute \
  --config simulations/config/paper_density_fast.json \
  --output simulations/outputs/paper_density_v2
python3 -m simulations.paper_density_report --source simulations/outputs/paper_density_v2
```

Il report verifica dati, ancoraggi, quattro rate e intervalli numerici di
C, poi genera la figura, la tabella e la didascalia LaTeX con la finestra fissata.
La produzione del PDF richiede successivamente rendering e revisione
visiva; lo stato distingue questa revisione dai controlli numerici.
Checkpoint, log e scratch rimangono locali. Si pubblicheranno dati, fit,
certificati, figura e didascalia completati.
