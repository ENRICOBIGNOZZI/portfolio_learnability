# Audit del codice e dei risultati — portfolio_learnability

**Snapshot controllato:** `cdfbd11a5390c15c54cace9b95e7d5e40c2ec35e`  
**Repository:** `ENRICOBIGNOZZI/portfolio_learnability`  
**Data della verifica:** 8 ottobre 2026.  
**Intervento eseguito:** sola lettura del repository; nessuna modifica o push.

## 1. Ambito e limiti

Sono stati letti tutti i 21 moduli Python non vuoti del nuovo sottosistema `simulations/`, i due moduli di test specifici delle simulazioni, `kernels.py`, `verify_final_outputs.py`, il workflow, i manifest di provenienza e i risultati rilevanti. Sono stati inoltre eseguiti controlli numerici locali mirati e indipendenti, descritti in `evidence/independent_checks.json`.

Questo **non** equivale a una nuova esecuzione delle 3.500 valutazioni Monte Carlo, né a una nuova esecuzione locale dei 123 test o a un nuovo audit completo del codice empirico JKP. I checkpoint `.npz` delle repliche non sono pubblicati nell'albero Git controllato. Il computer collegato risultava offline. La CI del commit risulta riuscita: esegue test e rigenera lo smoke, non riproduce l'intera produzione storica.

Distinguere quattro livelli: codice letto; identità verificate indipendentemente; risultati aggregati pubblicati; risultati di produzione dichiarati nei report. Non confonderli.

## 2. Correzione importante: la sorgente teorica coincide

Il dubbio precedente sulla cartella locale `Portfolio 8` è risolto per il contenuto teorico importato. I Git blob dei nove file di `paper/theory/` coincidono con quelli calcolati sui byte di `Portfolio_Paper_25_FINAL_MINIMAL.zip`. `representation_tables.tex` coincide con i primi 3.648 byte di `appendice_2.tex`. Le due figure teoriche coincidono con gli hash SHA-256 nel manifest di preservazione.

SHA-256 dello ZIP di riferimento:

```text
a6419bc9cca08d70f67dbbf8cfc044cb8a4049e276b3b7a6586eb436770be86b
```

Loss, estimator, population problem, Sharpe e prove importate **non richiedono un ripristino**. Aggiornare la documentazione di provenienza, non sostituire la teoria. Il confronto non costituisce una nuova certificazione matematica integrale di quelle prove.

Evidenza: `evidence/source_comparison.json`, `evidence/verified_theory_blobs.json`; repository `simulations/outputs/audit/manuscript_preservation.json`.

### Osservazione preesistente nel testo protetto, da non correggere di nascosto

Nell'equazione `eq:estimator` di `paper/theory/direct_port_learning.tex`, la catena di uguaglianze identifica `\widehat W=argmin(...)`, una policy, con `min_U{1/(1+SR_lambda(U)^2)}`, un valore scalare. Sono oggetti diversi. La prima definizione mediante argmin è quella implementata dal codice; la successiva uguaglianza deve essere segnalata separatamente all'autore e non riscritta nell'incarico sulle simulazioni. È presente anche il segnaposto testuale `table ....`.

Queste osservazioni non implicano che Codex abbia alterato la teoria: sono già nei byte originali preservati. La correzione richiede autorizzazione distinta. Non costituiscono motivo per cambiare loss o scala del fitting Python.

## 3. Nucleo matematico e stimatore: nessun errore individuato nei passaggi controllati

Il baseline costruisce triplette di caratteristiche con shift comune `r/3`, usa tre fattori economici e shock idiosincratici gaussiani non proiettati. Per i loadings specificati vale esattamente

```text
||beta(z)||² = c_beta
B_t' B_t / N = (c_beta/3) I_3
E[F F' | information_t] = I_3
Cov(F | information_t) = I_3 - mu_F mu_F'
```

La policy `W*(z)=beta(z)'mu_F/(c_beta/3+sigma_eps²/N)` soddisfa la normal equation condizionale. Il calcolo dei momenti di popolazione in `experiments/population.py` conserva correttamente la dipendenza interna alle triplette. Non tratta erroneamente le azioni come indipendenti.

La rappresentazione Nyström usa una base ortonormale RKHS; `ridge_path` impiega il secondo momento **non centrato**, la media del managed payoff e la penalizzazione corretta. La decomposizione del regret include il termine incrociato con segno, non una falsa somma di due contributi necessariamente positivi.

Controlli locali: errori della normal equation dello stimatore sotto `3.1e-16`; controlli indipendenti di E4 e bilanciamento sotto `1.5e-17` e `2.5e-18`, rispettivamente, negli stati verificati. Questi numeri sono verifiche di identità, non una prova tramite simulazione di tutte le ipotesi.

Sorgenti: `dgp/balanced.py`, `estimator/kernel.py`, `estimator/ridge.py`, `experiments/population.py`, `experiments/monte_carlo.py`.

## 4. Gap di integrità verificati nel codice — priorità P0

### 4.1 Invalidazione incompleta delle cache

In `run.py`, `SCIENCE_SOURCES` non comprende lo stesso `run.py`. Eppure `context_for` definisce rappresentazione, riferimento di popolazione e contesto scientifico. Una modifica di questa logica può non cambiare il `run_hash` dei checkpoint. Il fatto che `run.py` compaia fra gli hash dei risultati derivati non invalida automaticamente la cache scientifica: il manifest dei risultati derivati viene riscritto durante il run.

Inoltre `environment_audit` riutilizza un audit in base a fingerprint della configurazione, flag `passed` e numero di stati; non ne verifica tutte le dipendenze di codice e delle soglie. Il manifest del baseline include `balanced.py`, `baseline.py` e `kernels.py`, ma non `simulations/estimator/kernel.py`, usato dal calcolo dello spettro.

**Conclusione:** esiste un percorso di riuso di evidenza obsoleta. Non è stata dimostrata una contaminazione dei risultati pubblicati. Servono test di mutazione e invalidazione completa delle dipendenze pertinenti.

### 4.2 Il verificatore non ricostruisce tutti i numeri pubblicati

`diagnostics/verify.py` verifica diverse identità utili, ma non ricalcola sistematicamente tutti i momenti, le MCSE, le mediane, gli intervalli, le metriche OOS, i benchmark e le regressioni riportate nei CSV e nelle tabelle. Alcuni output sono accettati perché esistono o perché un precedente JSON dice `passed`.

`closeout.py` accetta il report dei test senza legarlo integralmente alla sorgente eseguita. I manifest di compilazione non vincolano tutti i file di figure e tabelle letti da LaTeX.

**Conclusione:** rafforzare la certificazione, non confondere `passed` con una ricostruzione indipendente completa.

### 4.3 Riproduzione completa non disponibile dal solo clone

Il `.gitignore` esclude i `.npz`; i checkpoint sintetici individuali e i momenti numerici non sono pubblicati. Sono presenti aggregati e seed. La rigenerazione completa è possibile in linea di principio, ma non è equivalente alla verifica immediata degli esperimenti storici.

Il comando `verify_final_outputs.py`, chiamato in CI sotto il titolo generico “Verify published scientific aggregates”, controlla soltanto risultati **empirici**. Lo smoke delle simulazioni è verificato separatamente. Serve un verificatore pubblico degli aggregati sintetici e un archivio versionato dei checkpoint, senza pubblicare dati JKP soggetti a licenza.

## 5. Cosa gli esperimenti mostrano e cosa non mostrano

### 5.1 Rate: manca un esperimento distinto sulla sequenza teorica

`reporting.py` stima rate della penalizzazione oracle ex post e delle scelte di validazione. Non esegue una sequenza separata fissata del tipo `lambda_T = a s_ref T^{-b/(b+1)}`.

Il teorema fornisce un bound sotto determinate condizioni; non impone a ogni DGP liscio e alla sua penalizzazione oracle di mostrare esattamente quella pendenza. I valori osservati non vanno etichettati automaticamente come errore del DGP o confutazione del teorema. Un confronto aggiuntivo lungo la sequenza teorica risponde a una domanda diversa e più precisa.

L'oracle ensemble è scelto e valutato sugli stessi campioni Monte Carlo. È un plug-in oracle descrittivo; una valutazione cross-fitted fra repliche separa l'ottimismo di selezione Monte Carlo. Bootstrap con riselezione non elimina da solo tale distinzione.

### 5.2 Approssimazione Nyström e regione ad alta complessità

Il rank principale è 256. Il confronto 256→512 controlla soltanto il regret medio oracle/selezionato, non l'intera curva né l'equivalenza con il kernel infinito. Il risultato riportato del 4,64% ha questa portata limitata.

Per `T>256`, la complessità empirica della rappresentazione resta al massimo 256. Pertanto i grafici quasi non regolarizzati possono riflettere anche il rank numerico. Aumentare soltanto T non costituisce un esperimento asintotico del kernel infinito. Servono confronto con solve esatto a piccole dimensioni e risoluzione crescente o certificata per le lambda rilevanti.

### 5.3 Spettro: prova analitica distinta dal fit numerico

Il confronto fra operatori è matematicamente pertinente. Il fit numerico circa 1,240, contro b teorico 1,5 nella finestra 30–200, è un diagnostico a risoluzione finita. La banda ±0,35 non è un intervallo di confidenza e non prova E5. Il codice la usa però anche come gate di accettazione della produzione.

Separare: ammissibilità analitica E5; correttezza matriciale del sandwich; risoluzione numerica; descrizione del fit. Non cambiare finestre o tolleranze per ottenere la pendenza desiderata. Lo stesso vale per le altre smoothness, dove il controllo usa una banda diversa.

La sensibilità della quadratura valuta 20 policy fissate con campioni annidati 8.192→32.768; non quantifica da sola l'errore di tutte le curve o dei fit spettrali. Aggiungere seed di quadratura indipendenti.

## 6. Validazione: funziona ma va diagnosticata meglio

A T=1440 i risultati pubblicati sono:

```text
oracle lambda                         1.000000000e-6
mediana lambda selezionata            1.274274986e-6
media lambda selezionata              8.655312405e-4
frequenza di selezione ai due estremi  8.4%
oracle regret                         0.00312118675
selected regret                       0.00615412393
rapporto selected/oracle              1.97172563686
```

La media della lambda selezionata è circa 679 volte la mediana. Non basta riassumere il problema come “lambda troppo grande”: occorre separare distribuzione, scelte ai due estremi e contributo degli estremi alla media. Il codice non presenta lo Sharpe alla lambda oracle-loss come un oracle-Sharpe distinto; rendere esplicita la distinzione nelle etichette.

La validazione corrente è cronologica e non ho individuato uso del futuro nella selezione. Il test che modifica un array OOS esterno mai passato al selettore è però troppo debole per costituire un audit end-to-end. Occorrono mutazioni reali dell'intera pipeline.

L'OOS è un percorso stazionario indipendente, non il seguito temporale della storia di training. È una scelta valida per il rischio di popolazione, ma aggiungere una valutazione futura contigua come esperimento diverso, non rinominare quella esistente.

## 7. Nuovo DGP proposto, senza scartare il baseline

Il baseline dipendente solo da z1 è intenzionale e corretto. Non è necessario eliminarlo. Si può aggiungere una seconda economia con

```text
u_d = (z_d+1)/2
theta_eta(u) = 2*pi*u_1 + eta * sum_{d=2}^6 sin(2*pi*(u_d-u_1))
eta = 0.35
beta_eta(z) = sqrt(c_beta) * (
 sqrt(2/3)*cos(theta_eta), sqrt(2/3)*sin(theta_eta), 1/sqrt(3)
)
```

Lo shift comune r/3 lascia invarianti le differenze dentro il seno e ruota l'angolo di 2*pi*r/3 modulo 2*pi. Norma e bilanciamento restano esatti; W*, q*, E2–E6 hanno la stessa struttura. Le derivate della policy rispetto a tutte le sei coordinate sono non nulle su insiemi di misura positiva. L'ampiezza è una proposta fissata prima dei nuovi risultati, non una calibrazione per centrare i rate.

Ho verificato numericamente queste identità in modo indipendente. Questa economia non è ancora implementata nel commit e non è automaticamente un esempio least-favourable/minimax: resta una policy liscia.

## 8. Ordine raccomandato

1. Proteggere lo snapshot valido e correggere provenienza, invalidazione e verificatori.
2. Pubblicare checkpoint sintetici e manifest necessari alla verifica.
3. Separare gli esperimenti theory-path, oracle e validation; certificare l'approssimazione.
4. Aggiungere `rich6d` preservando identità, dati originali e teoria.
5. Confrontare pochi selettori cronologici prospectivamente fissati e aggiungere OOS contiguo.
6. Rigenerare soltanto i risultati invalidati o nuovi e aggiornare soltanto la sezione simulazioni.

Non trasformare l'incarico in “forzare la U” o “ottenere pendenza −0,6”. Il risultato scientifico è descrivere ciò che emerge e stabilire quali affermazioni sono effettivamente sostenute.

## 9. Riferimenti esatti dello snapshot

Tutti i percorsi sopra sono relativi al repository al commit indicato. URL base:

```text
https://github.com/ENRICOBIGNOZZI/portfolio_learnability/blob/cdfbd11a5390c15c54cace9b95e7d5e40c2ec35e/
```

CI ispezionata: run `37759411343`, job `113251886202`.

L'evidenza locale allegata non contiene dati empirici, credenziali o font. `reviewed_ridge.py` è una copia del solo modulo controllato per riprodurre i test locali, **non** un secondo motore di simulazione da installare nel progetto.
