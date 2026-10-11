# Chiusura Rich6D — 12 ottobre 2026

La CV iniziale riduce la loss soprattutto correggendo la scala del portafoglio. Il miglioramento non si traduce in un vantaggio finale dello Sharpe medio. La specifica resta invariata: una selezione sui primi 60 dati, poi a congelata e lambda=a T^-0.6. Nessuna nuova campagna completa.

| Baseline T=4860, tutte le 300 storie | CV iniziale | Vecchia costante |
|---|---:|---:|
| Regret | 0.011296 | 0.022205 |
| Sharpe annualizzato | 2.932497 | 2.934774 |
| Errore lungo la direzione (ray gap) | 0.011119 | 0.010733 |
| Errore di scala | 0.000177 | 0.011472 |
| Bias ridge entro il sottospazio | 0.003179 | 0.019338 |
| Spostamento di stima + cross firmato | 0.006339 | 0.001089 |

Il floor di rappresentazione è 0.001777, pari al 15.73% del regret medio CV: il criterio del 5% fallisce. Lo spostamento e il cross sono separati solo dove abbiamo recuperato i coefficienti. La loro somma sulle 300 storie non è chiamata varianza. Ray gap, scale gap ed errore direzionale normalizzato sono identificati univocamente da loss e Sharpe, senza scegliere radici per u/v; le identità sono verificate anche sui momenti ricalcolati.

Le pendenze su tutte le 13 medie sono -0.4378196 per il regret e -0.3986480 per lo Sharpe gap. Sottrarre il floor porta la prima soltanto a -0.470399. Le quattro finestre già definite e le distribuzioni normalizzate restano complete: non abbiamo scelto un cutoff per ottenere -0.6.

**resolved:** provenance e aritmetica delle 1500 storie; due oracle distinti sulla stessa griglia; componenti identificate e bias ridge a ogni lambda effettiva; 79 test superati; quattro figure principali PDF/PNG e appendice; nota teorica verificata nel suo ambito; sezione compilata e collegamento minimo del paper. Empirica e teoremi conservati.

**failed:** floor/regret oltre il 5% nelle celle esportate. Nessuna soglia è stata allentata.

**unresolved:** certificazione numerica globale. L'audit copre i dodici indici rappresentativi e quattro stress aggiuntivi a T=60, ma una sola storia agli orizzonti intermedi/massimi. Gli stress non entrano nella MCSE dei rappresentativi. Il rango modifica 8/80 scelte iniziali; i confronti sui dodici rappresentativi a T=60 passano, mentre ai grandi T manca la MCSE tra repliche. Nel singolo baseline finale, a fissata, la variazione di regret P8192-P4096 è +0.00000247; è un confronto puntuale, non un certificato. Il rango 8192 non ha un controllo a rango superiore.

**not_run:** gli altri audit completi pianificati, la ridge di popolazione a P8192 alle lambda CV, tutti i momenti grezzi mancanti e la sensibilità Fourier ai grandi T. Il pilota M128/512/2048 a T=60 lascia invariate le scelte e mostra variazioni relative della policy al più 1.50e-10; il bound conservativo a tutte le lambda CV resta non informativo in alcune celle estreme.

Per chiudere le parti irrisolte occorre completare i controlli appaiati dichiarati e valutare il costo di un eventuale rango superiore prima di una produzione. Un sottospazio fisso con floor positivo non eredita convergenza a zero verso l'ottimo completo dal teorema kernel. Il risultato r=1 riguarda l'economia ideale infinita; non certifica M128 o Nyström.

[Rapporto tecnico completo](closeout_report.md) · [Comandi di riproduzione](../../closeout/README.md) · PDF compilati nella cartella paper/.
