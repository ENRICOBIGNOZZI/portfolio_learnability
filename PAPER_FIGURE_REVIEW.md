# Diagnostiche empiriche per il paper

Revisione degli script dell'8 settembre 2026. Il PDF originale è stato usato
esclusivamente come riferimento ed è rimasto intatto. Le figure teoriche alle
pagine 11 e 13 sono schemi: non vanno interpretate come risultati empirici.

## Figure da privilegiare

Per il corpo del paper userei le Figure 1 (complessità e performance),
3 (lunghezza della storia, con loss e incertezza), 4 (spettro e filtro),
6 (risultati economici e drawdown) e 9 (confronto con linear ridge).
La Figura 2 documenta l'interpolazione del target response-one. Le Figure 7,
8, 10 e le appendici spiegano stabilità, implementazione e sottoperiodi.
La Figura 5 è un'illustrazione empirica di media e rischio, non una stima
della frontiera efficiente di popolazione.

| Script | Domanda e modifica |
| --- | --- |
| `figure_1_complexity_performance.py` | L'OOS cambia lungo la griglia? Distingue strategie a rho fisso, selezione annuale e minimo ex post; intervalli pointwise condizionati ai fit. |
| `figure_2_in_sample_oos.py` | Quanto il fit in sample si separa dall'OOS? Corretto l'asse: train/validation hanno C su 120 mesi, OOS C su 180 mesi di refit. |
| `figure_3_history_opens_directions.py` | Più storia migliora davvero l'apprendimento? Aggiunti loss OOS, riferimento zero esposizione e dispersione bootstrap dei minimi. |
| `figure_4_economic_spectrum.py` | Quali direzioni vengono attivate? Aggiunto il filtro mu/(mu+lambda) scelto dalla validazione. |
| `figure_5_learnable_frontier.py` | Come cambia il rapporto media-rischio? Mostra rendimenti nativi e implementati, punti realizzati e limiti di riscalamento costante compatibili con il cap nel campione. |
| `figure_6_economic_outcomes.py` | Quando vengono guadagnati o persi i rendimenti? Aggiunto drawdown; l'indice di rendimenti in eccesso non viene più chiamato ricchezza totale. |
| `figure_7_tuning_stability.py` | La scelta di rho è stabile? Superficie di validation loss, rho annuale e C train/refit. |
| `figure_8_exposure_and_risk.py` | Che esposizioni sostengono i risultati? Pesi netti/lordi, volatilità realizzata e variazioni dei pesi. |
| `figure_9_benchmark_uncertainty.py` | Il Matérn migliora sul linear ridge? Confronto paired, 1.000 bootstrap a blocchi di 6/12/24 mesi e sottoperiodi di calendario. |
| `figure_10_complexity_by_period.py` | L'ottimo di complessità è stabile nel tempo? Curve per cinque sottoperiodi fissi e scelta annuale di validazione. |
| `appendix_loss_calibration.py` | La loss alta riflette direzione o scala? Decomposizione empirica esatta, senza modificare i pesi o il tuning. |

## Risultati da interpretare apertamente

- Nei 564 mesi di formazione 1978-2024, lo Sharpe implementato è 2,765 per
  Matérn e 2,870 per linear ridge. La differenza è -0,104; l'intervallo
  bootstrap paired al 95% con blocchi di 12 mesi è [-0,373; 0,250]. Questi
  risultati non sostengono una superiorità complessiva del Matérn.
- Il cap lordo 10 non si attiva in nessuno dei 564 mesi delle due strategie
  selezionate. I massimi osservati sono 9,660 e 8,170. Le differenze fra
  rendimenti nativi e implementati dipendono quindi dalla scala di volatilità
  che cambia fra finestre, non dal cap nei portafogli selezionati.
- Il target ex ante del 10% non è la volatilità OOS realizzata: quest'ultima
  è 13,512% e 12,731%. La Figura 8 mostra anche gli episodi di superamento.
- Nel 2020-2024 il Matérn selezionato ha Sharpe implementato -0,075 e loss
  nativa 2,352. I risultati aggregati di lungo periodo nascondono questa
  debolezza. Il sottoperiodo contiene soltanto 60 osservazioni mensili.
- La selezione raggiunge rho=1e-10, limite inferiore della griglia, nel 2009
  e nel 2015. In quegli anni non abbiamo identificato un minimo interno.
- Nell'esercizio sulle lunghezze la loss della strategia selezionata migliora
  fino a T=240, poi peggiora; gli intervalli sull'ottimo ex post sono ampi.
  Non è una verifica della legge asintotica o del suo esponente.
- Nella Figura 5 i due estremi sono scelti usando solo la complessità,
  senza cercare il punto che produce la narrazione desiderata. Nella scala
  nativa il caso meno regolarizzato può avere Sharpe superiore alla scelta
  di validazione. Ciò non va nascosto: loss, Sharpe e scala non coincidono
  per una regola stimata arbitraria.

## Significato della decomposizione della loss

Per una serie nativa R e a*=mean(R)/mean(R²), vale esattamente

`mean((1-R)^2) = min_a mean((1-aR)^2) + mean(R²)*(1-a*)²`.

Il primo termine è calcolato ex post ed è soltanto diagnostico: usa anche
i rendimenti OOS, ammette scale negative e ignora il cap. Nessuno dei
nuovi script lo usa per scegliere lambda o modificare i portafogli. Il
collegamento con lo Sharpe quadrato perde il segno del rendimento medio.

## Limiti da mantenere visibili

1. Il bootstrap ricampiona le serie OOS già stimate, con blocchi comuni
   nei confronti paired. Non ripete il training, la pulizia, la selezione
   delle caratteristiche o il tuning. Gli intervalli sono descrittivi,
   pointwise e senza correzione per confronti multipli. Il blocco 24 mesi
   nel sottoperiodo da 60 mesi è un controllo molto poco informativo.
2. L'IS Sharpe è una media di finestre sovrapposte; quello OOS è calcolato
   su una serie mensile concatenata. Il grafico esplicita questa differenza.
3. La complessità esposta è una media fra finestre. Un punto della strategia
   a selezione annuale non deve necessariamente trovarsi sulla curva a rho
   fisso. Lambda assoluto cambia con lo spettro: il parametro confrontabile
   lungo il percorso è rho=lambda/mu_max.
4. L'esercizio storico usa T mesi di train più 60 di validazione nel refit,
   quindi il sistema più lungo ha 540 mesi. Non si deve dire che tutte le
   regressioni sono al massimo 480 x 480.
5. La proiezione del target sullo spettro empirico non identifica il
   parametro di regolarità r del paper. Il nome Matérn non basta a stabilire
   l'esponente b dello spettro dei payoff gestiti.
6. I dati salvati hanno data di formazione e rendimenti `ret_exc_lead1m`.
   Gli assi temporali economici mostrano il mese successivo. I sottoperiodi
   restano dichiaratamente definiti per anno di formazione. Il prodotto
   di (1 + rendimento in eccesso) è un indice, non ricchezza comprensiva
   del rendimento monetario.
7. Il turnover salvato è la distanza L1 fra pesi-obiettivo consecutivi:
   non considera il drift delle posizioni prima del ribilanciamento.
   Nessun costo è stato aggiunto.

## Verifiche ulteriori che richiedono nuove stime

Prima di interpretare i risultati come una validazione definitiva del paper,
sono utili questi esperimenti separati. Non sono stati eseguiti da questa
revisione, che riutilizza i fit esistenti:

- Robustezza RFF: semi diversi e 512/1024/2048 feature a protocollo identico;
  distinguere approssimazione numerica e complessità economica. I risultati
  attuali sono per una rappresentazione finita e un solo seme.
- Protocollo strettamente point-in-time: la pulizia attuale sceglie le
  caratteristiche mediante copertura sull'intero 1963-2024
  (`build_clean_jkp_cache`), non solo sul primo train. Inoltre la disponibilità
  effettiva delle caratteristiche alla formazione e il confine dell'ultimo
  rendimento forward meritano un audit dedicato. Non è un errore risolto da
  un nuovo grafico e non è incluso negli intervalli bootstrap.
- Specificazione della policy: il codice condivide beta fra titoli,
  `w_it=phi(z_it)'beta`. Il paper esemplifica anche kernel matriciali
  `k(z,z') I_N` con coefficienti distinti per asset. La policy condivisa
  corrisponde a `K(Z,Z')=Phi(Z)Phi(Z')'`; questa specializzazione va dichiarata
  nell'eventuale sezione empirica, soprattutto con universo variabile.
- Una simulazione con popolazione nota può misurare realmente il gap di
  frontiera e studiare le velocità di apprendimento; il minimo OOS osservato
  non sostituisce la frontiera di popolazione.

## Riproduzione

`python3 run_paper_figures.py` rigenera sequenzialmente i grafici dai risultati
esistenti, senza training. Gli script restano indipendenti e condividono
`Utils/empirical_analysis.py` e `Utils/paper_figures.py`. Gli output nuovi e
rivisti sono in `results/paper_figures/`, in PNG, SVG e tabelle CSV/Parquet.
