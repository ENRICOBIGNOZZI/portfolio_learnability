# Esperimento con lambda scalata

Esecuzione completa (usa i dati locali e la baseline già salvata):

```sh
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python3 lambda_scaling.py
```

Solo rigenerazione di grafici e report:

```sh
python3 lambda_scaling.py --report-only
```

Il risultato è `results/lambda_scaling/report.html`. La galleria comprende
tutti i grafici originali e le nuove curve per i sei kernel. PNG e HTML 3D
si aprono anche individualmente. I CSV riassuntivi sono `performance.csv`,
`parameters.csv` e `learning_rates.csv`. I parquet per kernel permettono di
ricostruire le figure senza addestrare nuovamente i modelli.

La baseline non viene modificata: viene letta dai suoi parquet e verificata
con SHA-256 insieme a tutti gli altri risultati precedenti. La ricostruzione
dei payoff gestiti deve inoltre riprodurre i rendimenti raw originali entro
1e-7 in ciascuna finestra annuale.

## Regole confrontate

- Baseline: selezione annuale della lambda sulla validation, come esistente.
- Controllo: lambda scelta una sola volta e poi costante.
- Paper: `lambda(T) = lambda0 * (T/120)**(-a)`, `a = b/(b*r+1)`.
  Si sceglie solo lambda0 sui primi 60 mesi di validation dopo 120 mesi di
  train. Il refit iniziale ha 180 mesi e applica già lo scaling. La lengthscale
  e la griglia sono quelle della baseline, fissate nella prima validation.
- Sensibilità: r assunto pari a 1.05, 1.5 (principale), 2; nessuna selezione
  fra r guardando il test. Per NTK e Matérn, b viene descritto dallo spettro del
  primo train, sui ranghi 5-72; sono salvati anche due fit alternativi.
  Lineare e gaussiano usano la convenzione b infinito, quindi a=1/r.
  Il caso gaussiano rappresenta solo la potenza, senza pretendere di stimare
  eventuali correzioni logaritmiche o una regola lambda univoca.
- Empirica: potenza stimata separatamente dai minimi di validation per
  training annidati di 36-120 mesi, tutti precedenti al test; potenza e
  lambda0 sono poi congelate. Le potenze negative non vengono corrette.

Ogni strategia ricalcola i coefficienti del portafoglio annualmente. Il cap
lordo è lo stesso della baseline e viene applicato ai nuovi pesi esatti.
I grafici di complexity variano la costante iniziale su tutta la griglia:
non trattano le lambda effettive di anni diversi come se fossero uguali.

## Rate empirico

Il learning exponent del paper è `p = b*r/(b*r+1)`, diverso dalla potenza
`a` di lambda. Il tentativo empirico usa train trailing annidati di
36, 60, 90, 120, 180, 240, 360 mesi sui medesimi test 1993-2024, rendimenti
raw e regola principale. Il fit profila `SR(T) = S_inf - A*(T/36)**(-p)`
con `A >= 0`, `p` tra 0.02 e 2, salvando il profilo completo e 400 bootstrap
annuali appaiati. Non osserva SR* e non dimostra il bound probabilistico
del paper. Non monotonicità, optima ai bordi e ampi intervalli devono essere
letti come problemi di identificazione, non come potenze precise.

Il dataset, le feature casuali e la preparazione sono quelli della baseline.
Il filtro delle caratteristiche sull'intero periodo e la ricerca pregressa
sugli stessi dati rimangono limiti del confronto. Le equity capitalizzano
meccanicamente rendimenti excess e non riconciliano finanziamento e costi.

Verifica numerica mirata:

```sh
python3 -m pytest -q test_lambda_scaling.py
```
