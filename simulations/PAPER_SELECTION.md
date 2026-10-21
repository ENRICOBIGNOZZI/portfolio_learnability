# Figure illustrative per il paper: scelta a posteriori

La figura piu vicina congiuntamente ai quattro benchmark usa **N=600,
rho=0.5, rango 8192 e T_train da 800 a 25600**, sulle otto repliche del
controllo di precisione. Le 100 repliche del confronto separato usano
**N=1200, rho=0.5, rango 1024 e T_train da 200 a 12800**.

| Quantita | Benchmark | Illustrazione, 8 repliche | Confronto, 100 repliche |
|---|---:|---:|---:|
| Gap Sharpe vs T_train | -0.6667 | -0.6706 | -0.6431 |
| C vs T_train | +0.3333 | +0.3316 | +0.3328 |
| C vs lambda | -0.5000 | -0.4867 | -0.4782 |
| Gap Sharpe vs C | -2.0000 | -2.0209 | -1.9304 |

Il minimo R2 dei quattro fit a esponente teorico fissato e 0.9953 nella
prima figura e 0.9932 nella seconda. La massima discrepanza relativa fra
pendenza libera e benchmark e rispettivamente 2.67% e 4.36%.
La figura a otto repliche e un'illustrazione numerica; il confronto a 100
repliche permette di valutare separatamente il comportamento su piu repliche.
Le bande sono una STD campionaria (ddof=1), accompagnata dal percentile
empirico 95 della quantita sull'asse verticale. In gap(C), il percentile
del gap e posto alla complessita media dello stesso T e le barre
orizzontali rappresentano la STD di C.

## Regola di scelta e portata

La scelta e **esplorativa e a posteriori**, richiesta per la presentazione
nel paper. Non modifica il primario gia pubblicato da T_train=1600.
Gli intervalli delle vecchie analisi non sono corretti per questa selezione;
le figure selezionate illustrano la vicinanza delle curve ai benchmark,
senza fornire un test confermativo indipendente dei rate asintotici.

Ogni candidato deve contenere tutti e quattro i rate, con lo stesso taglio
iniziale e lo stesso estremo temporale che definisce la finestra. Si
richiede R2 >= 0.99 per tutti i fit teorici e si minimizza il massimo di
`abs(p_hat-p_theory)/abs(p_theory)` sui quattro rate. I pareggi si risolvono
con l'errore relativo medio, il taglio e lo scenario. Non si assemblano
quattro finestre diverse, una per ciascuna pendenza.

Il rango ad alta precisione e quello gia scelto dal certificato numerico
su C, senza riaprirne la selezione in base alle pendenze. Il confronto a
100 repliche sceglie fra le cinque celle del disegno completato. I due
studi hanno ranghi, N e numerosita diversi e rimangono separati.

`all_candidates.csv` conserva 49 specifiche: cinque celle per sette tagli
nella conferma e due ranghi per sette tagli nel controllo di precisione.
Sono 42 quelle con estremi temporali comuni a tutti i rate. Le sette del
riferimento N=600,rho=0.5 della conferma hanno gap/C(T) fino a 25600 ma
una finestra C(lambda) che termina al T equivalente di 12800: sono quindi
escluse dalla nuova scelta di una singola finestra comune. Rimangono valide
e disponibili nelle analisi originali. I pilot precedenti sono esclusi
anche dal confronto fra candidati. Il rango 4096 resta nella tabella di
sensibilita, mentre il rango certificato per l'illustrazione e 8192.

## File pronti per il paper

Tutti i file sono in [outputs/paper_selection_v1](outputs/paper_selection_v1/):

- `high_precision_rates.pdf`: quattro pannelli, otto repliche, sei T_train reali e 17 lambda.
- `confirmation_rates.pdf`: quattro pannelli, 100 repliche, 40 T_train reali e 20 lambda.
- `paper_figures.tex`: ambienti figure e didascalie in inglese, inclusa la selezione a posteriori.
- `paper_table.tex`: tabella LaTeX dei quattro esponenti per entrambi gli studi.
- `selection.json`, `all_candidates.csv`, `rates.csv`: criteri, graduatoria e valori completi.
- `*_curve.csv`: medie, STD, percentile, complessita orizzontale e benchmark di ciascun pannello.
- `validation.json`: verifica delle curve rispetto ai dati originali, hash e revisione visiva.

I percorsi LaTeX sono relativi alla cartella `paper/`, come gli altri input
del manoscritto del repository. Non e modificato il manoscritto empirico:
le due figure e le didascalie sono pronte per l'inserimento nel paper scelto.
Le figure PDF hanno font TrueType incorporati; sono disponibili anche PNG
300 dpi. Gli unici tratti densi generati dal codice sono i benchmark
analitici; i marcatori delle simulazioni corrispondono ai punti reali.

Rigenerazione, senza nuovi training o modifiche ai dati originali:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m simulations.paper_selection
```

La rigenerazione imposta la revisione visiva come pendente: i PDF ricreati
vanno renderizzati e ricontrollati prima di sostituire quelli revisionati.
