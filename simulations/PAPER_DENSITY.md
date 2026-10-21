# Densificazione della figura per il paper

Calcolo avviato dopo la richiesta di maggiore densita. La scelta e
mantenere **N=600, rho=0.5, rango 8192, otto repliche e C0 invariato**,
aggiungendo training intermedi reali nella finestra gia scelta.
I risultati della griglia piu densa non sono ancora disponibili.

La griglia nella finestra del paper passa da sei a **26 valori di T_train**:

```
800, 919, 1056, 1213, 1393, 1600, 1838, 2111, 2425,
2786, 3200, 3676, 4222, 4850, 5572, 6400, 7352, 8445,
9701, 11143, 12800, 14703, 16890, 19401, 22286, 25600
```

Si conservano inoltre 100,200,400 per il confronto con tutti gli ancoraggi
originali: 29 training per replica, 232 policy complessive. Il nuovo calcolo
riguarda il percorso bilanciato, lambda=C0*T_train^(-2/3), senza ricalcolare
gli altri moltiplicatori di lambda. I confronti originali rimangono disponibili.
C(lambda) mantiene il medesimo spettro a T_train=25600 e la medesima griglia
spettrale: i nuovi punti aumentano la densita delle curve in T_train e del
percorso gap(C). Non si aggiungono marcatori interpolati alle simulazioni.

Semi, economia, kernel, rango e penalita coincidono con il controllo
precedente; si usano le **stesse otto traiettorie**, non otto nuove repliche
indipendenti. La finestra del paper da 800 e dichiarata prima di calcolare
la griglia piu densa e non viene riselezionata in base ai nuovi risultati.
Il primario da 1600 e tutti i tagli di sensibilita restano disponibili.
Le pendenze potranno cambiare per effetto dei nuovi punti nella regressione.

Il disegno congelato e in `config/paper_density.json` e
`outputs/paper_density_v1/plan.json`, con hash della scelta effettuata sul
pilot precedente. Il codice scientifico di `asymptotic_precision` rimane
invariato; si usa un worker con i thread numerici limitati a uno.

Dopo il training, `paper_density_report` verifica tutti i dati con il
validatore indipendente del rango, confronta le 72 policy dei nove
ancoraggi originali, ricontrolla i quattro fit dai dati grezzi e genera la
figura con la finestra fissata. Verifica anche gli intervalli numerici di
pendenza di C su tutti i tagli. La didascalia LaTeX distingue la precedente
scelta descrittiva dalla griglia piu densa fissata prima di questo calcolo.

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m simulations.asymptotic_precision_pipeline \
  simulations/outputs/paper_density_v1 \
  --config simulations/config/paper_density.json --compute-only
python3 -m simulations.paper_density_report --source simulations/outputs/paper_density_v1
```

La generazione del PDF richiede successivamente rendering e revisione
visiva. Lo stato del report distingue questa revisione dai controlli
numerici. Durante il calcolo non modificare configurazione o moduli
scientifici congelati. Checkpoint e scratch rimangono locali; si
pubblicheranno dati, fit, certificati, figura e didascalia completati.
