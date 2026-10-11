# CODEX — Chiusura scientifica e numerica delle simulazioni Rich6D

## Obiettivo e fonti

Repository: `ENRICOBIGNOZZI/portfolio_learnability`.
Stato verificato in questa revisione: `05461a909a10ccbcfbbfe99dcf98c2a516c49248`.
Prima di lavorare, registra HEAD, branch, stato del working tree e modifiche successive. Non sovrascrivere lavoro concorrente. Questo incarico riguarda SOLO simulazioni e relativo pacchetto editoriale; non gli esperimenti empirici JKP, la nonstationarity o nuovi algoritmi.

Leggi:
- `simulations/initial_cv/{selector,run,summarize,audit,report}.py`;
- `simulations/estimator/ridge.py`, `simulations/extended/` e i test pertinenti;
- `simulations/outputs/rich6d_initial_cv60_v1/`;
- `simulations/outputs/rich6d_rough_sr3_extended_v2/`;
- `simulations/outputs/rich6d_sr3_numerical_followup/`;
- le ipotesi e le prove correnti in `paper/theory/`;
- l'allegato `Portfolio_Source_Regularity_Theory.tex`, con il PDF corrispondente. Tratta questa nota come una derivazione da verificare, non come una certificazione già acquisita. Se manca l'allegato, segnala esattamente quali verifiche teoriche non puoi completare.

Il risultato desiderato è una sezione di simulazioni riproducibile, numericamente controllata e fedele alla teoria. NON cercare una configurazione o una finestra temporale che produca una pendenza desiderata. Non garantire un voto editoriale o l'accettazione da parte di un journal.

## 1. Congela la specifica economica e statistica

La specifica principale resta:

`lambda_(k,T) = a_hat_(k,60) * T^(-0.6)`.

Per ciascuna replica e ciascuna economia, scegli `a` UNA SOLA VOLTA sui primi 60 dati. Mantieni la griglia originaria di 71 costanti fra 1e-8 e 1e-1, i quattro fold forward con training 20/30/40/50 e successivi dieci dati di validation, la loss response-one, la ponderazione per osservazione e il tie-break originario. Ogni fold usa il proprio numero di osservazioni di training nell'esponente. Ristima la policy su tutti i dati disponibili, senza ulteriori scelte di `a`.

Mantieni DGP, Matérn, lengthscale, parametri economici, Fourier M=128, campioni e pairing originali, 300 repliche per economia, le cinque economie e tutti gli orizzonti. Il baseline arriva a T=4860; il confronto fra economie usa la griglia comune fino a T=3240. Conserva gli ottimi analitici specifici di N e la convenzione Sharpe annualizzato = sqrt(12) per Sharpe mensile marginale. Loss e regret non si annualizzano.

Non cambiare T0, loss di selezione, esponente, griglia, distribuzione dei coefficienti o target. Non eliminare repliche estreme, scelte ai bordi o orizzonti sfavorevoli. Non reinterpretare le nuove diagnostiche come preregistrate prima di vedere i risultati: registra che sono analisi successive. Conserva la vecchia regola come confronto, non come vincitore scelto sulla base della pendenza.

## 2. Completa le diagnostiche della quantità realmente coperta dal teorema

La pipeline corrente salva il regret, ma non lo include fra le quantità per cui `initial_cv/summarize.py` calcola le pendenze. Aggiungi una diagnostica del regret senza sostituire o nascondere quella dello Sharpe.

Per ogni economia, replica, T e specifica, esporta:
- regret response-one rispetto all'ottimo economico analitico;
- Sharpe e Sharpe gap, mantenendo anche le unità mensili;
- complessità empirica, complessità di popolazione del modello numerico e loro rapporti con T;
- T^(0.6) per regret e per Sharpe gap; C/T^(0.4);
- media, mediana, deviazione standard, percentili 2.5/97.5 e MCSE della media.

Calcola le pendenze descrittive sulle quattro finestre GIA' definite: completa, T>=360, T>=720, T>=1440. Le incertezze devono preservare l'intera replica e la dipendenza fra i suoi orizzonti; riusa il metodo a covarianza completa e verifica con jackknife. Non confondere pendenza del log della media con media delle pendenze individuali.

Controllo aritmetico da riprodurre, non obiettivo da raggiungere: sulle 13 medie baseline iniziale-CV archiviate la pendenza del regret è circa -0.4378196, e quella dello Sharpe gap -0.3986480. Quindi la distinzione loss/Sharpe da sola non risolve l'apparente lentezza finita. Quantifica anche i quantili delle quantità normalizzate: il teorema è in probabilità, non un'identità sulla media Monte Carlo. Un grafico finito non prova né confuta da solo il rate asintotico.

Non scegliere un cutoff per vicinanza a -0.6. Un eventuale sottoinsieme numericamente affidabile deve derivare da criteri numerici espliciti, avere una maschera esportata e accompagnarsi sempre alla griglia completa. Distingui la scelta di scala grafica dalla selezione dei dati usati per stimare una pendenza.

## 3. Misura l'errore di scala e l'errore direzionale

`moment_metrics` calcola già media e secondo momento, ma restituisce soltanto loss e Sharpe; il runner iniziale-CV non salva i coefficienti. Verifica prima eventuali cache disponibili. Non ricostruire momenti ambigui scegliendo arbitrariamente una radice delle equazioni loss/Sharpe.

Aggiungi un helper compatibile con l'API esistente. Per ciascuna policy salva i momenti MENSILI:

`u = E[Y]`, `v = E[Y^2]`, `q = E[(Y_star)^2] = 1 - Q_star`.

Non confondere questo `q` con una notazione del manoscritto che eventualmente indica Q_star.

Implementa e verifica, policy per policy:

`c = u/q`
`d = v - u^2/q`
`regret = q*(c-1)^2 + d`
`Sharpe_monthly = u / sqrt(v-u^2)`.

Per c>0, verifica anche:

`Sharpe_monthly = Sharpe_star_monthly / sqrt(1 + d/[c^2*q*(1-q)])`.

La grandezza d, presa da sola, non è invariante alla scala. Riporta quindi anche il termine direzionale normalizzato e, per u>0 e v>0, la decomposizione lungo la direzione:

`ray_gap = q - u^2/v`
`scale_gap = (v-u)^2/v`
`regret = ray_gap + scale_gap`
`ray_gap = 1/(1+Sharpe_monthly^2) - Q_star`.

Tratta separatamente media negativa/nulla e varianza nulla; non cambiare il segno della policy tramite informazione di popolazione. Non cancellare valori negativi oltre l'errore di arrotondamento dichiarato. Le identità sono per policy: non applicarle alle sole medie di gruppo.

Distingui questa decomposizione dalla decomposizione rispetto alla population ridge. Per ciascuna lambda realmente selezionata, valuta anche:

`regret = projection_floor + regularization_bias_within_P + estimation_displacement + signed_cross_term`.

Deriva i termini rispetto all'ottimo nel sottospazio e alla population ridge dello STESSO sottospazio, usando gli stessi momenti. Il termine di spostamento dalla population ridge non è automaticamente una varianza, e il cross term non si annulla automaticamente, soprattutto con a dipendente dai dati. Verifica la ricostruzione del regret numericamente. Riporta queste diagnostiche in tabelle e appendice, senza aggiungere una quinta figura principale di bias-decomposition.

Riusa coefficienti o storie disponibili. Se i momenti mancanti richiedono nuove stime, rigenera soltanto le storie necessarie con i semi originali e verifica che loss/Sharpe precedenti siano riprodotti. Salva momenti e sidecar sufficienti per evitare un'altra rigenerazione. Non promettere che tutto sia postprocessing se gli input non esistono.

## 4. Confronta le due diverse definizioni di oracle

Mantieni distinti:
1. initial-CV, specifica principale implementabile;
2. old-fixed, confronto storico;
3. oracle-Q: un'unica lambda per economia/T che minimizza il regret medio sulla griglia diagnostica originaria;
4. oracle-SR: un'unica lambda che massimizza lo Sharpe medio sulla stessa griglia.

Non usare la media dei massimi scelti separatamente per replica. Ricava entrambi gli oracle dagli output originali completi. Non usarli mai per calibrare la strategia. Riusa, dove disponibile, la verifica even/odd cross-fitted del follow-up; resta comunque un diagnostico con informazione di popolazione. Non presentare le bande dell'oracle come intervalli corretti per la selezione.

Riporta la distribuzione delle costanti iniziali e i casi di bordo senza rimuoverli. Eventuali classi di a sono diagnostiche descrittive, non nuove regole di trading o prove causali. Confronta initial-CV con oracle-Q sulla loss e con oracle-SR sullo Sharpe: non mescolare i due obiettivi.

## 5. Fai un audit numerico delle NUOVE policy CV

Il confronto P4096/P8192 esistente usa la vecchia costante, sul solo baseline e su dodici storie pilota. Non certifica le policy selezionate dalla CV, che utilizzano penalizzazioni diverse.

Prepara un protocollo e una stima di risorse PRIMA di nuovi calcoli. Primo stadio: confronto abbinato P4096/P8192 su un sottoinsieme dichiarato di storie iniziale-CV, includendo orizzonti brevi, intermedi e massimi, tutte le economie e casi di lambda piccola. Distingui le dodici repliche rappresentative con indici fissati dagli eventuali stress test scelti tramite la costante iniziale, non tramite Sharpe o pendenza. Non confondere gli indici dei vecchi rank-pilot con i semi della produzione.

Separa due verifiche:
A. stessa a selezionata a P4096 mantenuta nei due ranghi: isola l'approssimazione della policy;
B. CV rifatta soltanto all'inizio, sui medesimi 60 dati, a ciascun rango: misura la sensibilità dell'intera procedura. Questa è un'audit separato, non una modifica delle costanti archiviate.

Conserva il criterio storico del 5%, con gli stessi denominatori esplicitati: projection floor/regret, differenza media abbinata più 1.96 MCSE e quadratura indipendente. Aggiungi sensibilità di media, secondo momento, Sharpe gap e componente direzionale con soglie dichiarate prima del confronto. Usa una seconda quadratura indipendente per valutare policy fissate. Un optimum stimato sulla stessa quadratura non è un certificato dell'ottimo esatto.

Non inferire una certificazione da P8192 contro P4096 per il rango massimo, né da pochi orizzonti per tutti quelli non verificati. L'ottimo economico resta il benchmark principale. Gli ottimi nei sottospazi servono soltanto a quantificare l'errore numerico.

Se l'audit richiede un rango maggiore, valuta costi e benefici prima di una nuova produzione. Non mescolare ranghi diversi lungo una curva senza un protocollo dichiarato. Non avviare automaticamente una nuova campagna completa di 1.500 storie. Se la risoluzione resta insufficiente, consegna una limitazione quantitativa e stato `unresolved`, senza allentare le soglie. Un audit ristretto autorizza conclusioni ristrette.

Verifica inoltre residuali normalizzati delle equazioni normali, stabilità alle lambda minime, equivalenza primal/dual su problemi piccoli e al cambio di regime, equivalenza della base Cholesky/simmetrica, assenza di jitter non dichiarato e conservazione delle normalizzazioni 1/N e 1/T.

## 6. Collega correttamente economia ideale, Fourier e Nyström

Controlla la nota teorica allegata contro il DGP effettivo. Mantieni separati:
- economia ideale a serie Fourier infinita;
- economia numericamente simulata con M=128;
- policy stimata nel sottospazio Nyström di dimensione P.

Verifica il risultato r=1 per l'economia ideale e la proposizione sulla selezione iniziale in una griglia finita positiva. Non dedurre la massima source regularity del modello M=128 dalla sola differenziabilità, né dedurre un esponente osservato esatto dal risultato r=1. Registra qualunque lacuna teorica invece di correggere silenziosamente i teoremi protetti.

Esplicita che la proposizione sulla CV vale per lo stimatore kernel del teorema: un sottospazio fisso con errore di rappresentazione positivo non ha regret verso l'ottimo completo che tende a zero. Per le simulazioni serve quantificare l'errore computazionale oltre a quello statistico. Non applicare automaticamente teoremi Nyström esterni, con ipotesi diverse, al nostro problema dipendente e con feature non limitate.

Riusa il controllo Fourier M=128/512/2048 e il bound deterministico della coda. Verifica la sensibilità di momenti, policy stimate e scelte iniziali alle effettive lambda CV, non soltanto la vicinanza puntuale di W_star. Se basta un bound perturbativo rigoroso e informativo, preferiscilo ad altri fit; altrimenti usa storie e innovazioni abbinate. Non cambiare M della produzione per migliorare una pendenza.

Le economie rho condividono il medesimo operatore di popolazione e la medesima funzione C(lambda), ma dopo CV possono avere lambda selezionate diverse: le curve C(lambda_hat) non devono essere artificialmente rese uguali. Per N usa ciascun ottimo analitico e chiarisci la dipendenza da N delle costanti teoriche.

## 7. Test, figure e collegamento al manoscritto

Aggiungi test delle identità finanziarie con policy ottima, multipli positivi dell'ottimo, direzioni perturbate, casi nulli/negativi e unità mensili/annuali. Controlla assenza di accesso ai dati futuri e agli operatori di popolazione durante CV, costante iniziale immutabile, pairing, riproduzione dei vecchi output e rifiuto di checkpoint/manifests alterati. Testa entrambe le definizioni di oracle e l'aritmetica delle pendenze contro calcoli indipendenti.

Mantieni quattro figure principali, senza introdurre altri modelli:
0. spettro economico e filtri, con diagnostica dell'energia del target nelle stesse direzioni;
1. recupero dello Sharpe/frontiera con il vero ottimo economico;
2. complessità empirica e di popolazione lungo il percorso;
3. Sharpe contro complessità, con le scelte implementabili chiaramente distinte dagli oracle.

Conserva gli scenari N/rho, i risultati completi e le diagnostiche aggiuntive in appendice. La decomposizione finanziaria deve spiegare i risultati attraverso valori calcolati, non essere soltanto una didascalia. Per la complessità di popolazione valuta ogni lambda selezionata e poi aggrega: C(media lambda) non sostituisce la media di C(lambda). Non sommare autovalori ordinati di operatori diversi.

Bande delle repliche e intervalli della media devono avere etichette diverse. Riporta sempre condizioni numeriche non superate. Nessuno smoothing o ritaglio nasconde anomalie. Genera PDF vettoriali e PNG; verifica visivamente ogni figura e il documento compilato.

Prepara `simulation_section.tex`, `simulation_appendix.tex`, tabelle generate dai CSV e una patch minima del collegamento in `paper/main.tex`: nello stato revisionato `simoutput` punta ancora a `confirmation_v2`. Non alterare la sezione empirica o i teoremi. Compila una copia di staging del paper e verifica tutti gli input e riferimenti prima di proporre la patch.

## 8. Consegna e criterio di chiusura

Lavora in una directory nuova, per esempio `simulations/outputs/rich6d_initial_cv60_closeout_v1/`, con pochi moduli aggiuntivi riutilizzabili, non una nuova duplicazione del framework. Mantieni gli esperimenti storici immutabili. Nessun push o spesa cloud aggiuntiva senza autorizzazione; non riavviare job già attivi.

Consegna codice e test, dati per replica, tabelle, figure, sezione/appendice compilabili, comando di riproduzione da input verificati, semi/hash/versioni e rapporto breve con stati separati: `resolved`, `failed`, `unresolved`, `not_run`.

Il rapporto deve rispondere quantitativamente:
- Che cosa migliora la CV iniziale, e dove?
- Quanto del regret è scala, direzione, shrinkage, spostamento di stima e approssimazione?
- Quanto restano stabili le NUOVE policy aumentando rango e quadratura?
- Quali conclusioni finite sono sostenute e quali affermazioni asintotiche non sono certificate?

Chiudi il lavoro quando i risultati principali hanno evidenza riproducibile e limitazioni esplicite. Se una verifica necessaria fallisce, descrivi il blocco preciso e il minimo lavoro ulteriore richiesto. La chiusura non dipende dal raggiungimento della pendenza -0.6 o dal fatto che una strategia vinca ovunque.
