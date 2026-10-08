# Incarico Codex — audit mirato, correzioni verificabili e simulazioni di conferma

Lavora sul repository **`ENRICOBIGNOZZI/portfolio_learnability`**.

Snapshot iniziale già controllato:

```text
cdfbd11a5390c15c54cace9b95e7d5e40c2ec35e
```

Questo incarico prosegue il lavoro esistente. **Non ricostruire il progetto da zero. Non cancellare il nuovo baseline bilanciato. Non modificare la teoria.** Correggi le lacune di verifica, poi aggiungi esperimenti distinti che rispondano alle domande scientifiche mancanti.

Leggi prima `AUDIT_CODICE_E_RISULTATI.md` e i file di evidenza allegati. Le osservazioni sotto sono riferite allo snapshot indicato: se il branch è avanzato, verifica quali problemi sono già risolti prima di intervenire. Non applicare patch alla cieca e non sovrascrivere lavoro concorrente.

## 0. Regole di autorità, preservazione e onestà

1. Registra branch, SHA, stato del working tree, dipendenze e ambiente. Non fare force push, reset distruttivi, pulizie indiscriminate o riscritture della storia.
2. Proteggi byte-per-byte i nove file di `paper/theory/`, le figure teoriche protette e i 112 file empirici nel manifest esistente. Loss, estimator, definizione dei pesi, Sharpe, enunciati, ipotesi E1–E6 e dimostrazioni non sono oggetto di modifica.
3. `paper/main.tex` può cambiare soltanto nei collegamenti ai nuovi materiali di simulazione o in correzioni tecniche di compilazione indispensabili. Non riscrivere formule o prose teoriche per adattarle ai risultati.
4. Il contenuto teorico importato da `/Users/enrico/Downloads/Portfolio 8` è stato confrontato con `Portfolio_Paper_25_FINAL_MINIMAL.zip`: coincide nei nove file protetti. Il dubbio precedente sulla versione è risolto per questi byte. **Non sostituire la teoria e non bloccarti chiedendo conferma di una corrispondenza già verificabile.** Registra la verifica di contenuto, senza inventare una conferma personale dell'utente.
5. I vecchi motori a spettro prescritto R1/NL/L e J=1000/2000/4000 restano rimossi. Il nuovo baseline al commit iniziale è invece evidenza valida da preservare. Una versione storica dei risultati non è un secondo motore attivo.
6. Conserva gli output iniziali come snapshot immutabile o archivio versionato. Le correzioni e i nuovi esperimenti vanno in output versionati; non sostituire silenziosamente risultati precedenti.
7. Non selezionare seed, finestre spettrali, parametri economici o range dei grafici per ottenere una U, uno Sharpe desiderato o la pendenza −0,6.
8. Distingui esplicitamente: errore confermato, vulnerabilità di verifica, limite numerico, ipotesi analitica, risultato empirico e nuova proposta. Il ritrovamento di una cache invalidabile non prova che i risultati precedenti siano sbagliati.
9. Usa solo risorse già autorizzate. Non acquistare cloud/GPU, non modificare credenziali o infrastruttura, non pubblicare dati JKP soggetti a licenza. Se manca una risorsa indispensabile, consegna il lavoro verificato e un blocco preciso: mai dichiarare completo ciò che non ha girato.
10. Evita refactoring generali o nuovi framework: patch locali, una sola implementazione canonica, test mirati e risultati leggibili.

### Hash di riferimento della teoria

SHA-256 dell'archivio originale:

```text
a6419bc9cca08d70f67dbbf8cfc044cb8a4049e276b3b7a6586eb436770be86b
```

Git blob attesi nello snapshot iniziale:

```text
paper/theory/direct_port_learning.tex       f17df8e281c66b31e023ed11004ec21b290f295d
paper/theory/economomic_spectrum_final.tex   4bdc30ad2b317d127d98989b71d2744cea506769
paper/theory/introduction.tex               05d7c34dea1291ee0fb0d8dcce315166a0a053a9
paper/theory/population_problem.tex         34a73cfadec819e4a6ba0bfb865bf8fa9cec4eb0
paper/theory/proofs_learning.tex            5b6ace4a3125475c016995fb513fd8cfce0402e0
paper/theory/references.bib                 7726601494d55015f96d0d72b9c2dad4b3d11faa
paper/theory/representation_learnability.tex 99690b0bcca348eb238459149103d15ac6406fab
paper/theory/representation_tables.tex      5305869509301fcb918222f7f5243827a8d6f2b8
paper/theory/unbounded_concentration.tex    927bd05dcd555b094bb4f86c88a516089d0de7f8
```

La tabella di rappresentazione corrisponde al prefisso di 3.648 byte di `appendice_2.tex`. Controlla i byte, non i nomi delle cartelle. Il confronto non autorizza modifiche teoriche.

**Osservazione protetta da riportare, non correggere:** `direct_port_learning.tex`, `eq:estimator`, uguaglia nella stessa catena una policy `argmin` a un valore scalare `min`. La prima definizione dell'estimatore mediante argmin resta il contratto del codice. Segnala la discordanza e il segnaposto `table ....` in `simulations/docs/protected_theory_observations.md`; non modificare quei byte e non trasformare questa segnalazione in una modifica della loss. La correzione editoriale/matematica del testo protetto richiede un incarico separato dell'autore.

## 1. Prima fase: conferma i problemi nello stato attuale

Produci `simulations/docs/audit_v2_findings.md`, con una riga per problema: file/funzione, evidenza, gravità, stato, test riproducibile e intervento minimo.

I moduli da controllare integralmente sono `config/design.py`, `dgp/*`, `estimator/*`, `experiments/*`, `diagnostics/*`, `plotting/*`, `run.py`, `paper.py`, `manuscript.py`, i due test delle simulazioni, `.github/workflows/empirics.yml`, `.gitignore` e `verify_final_outputs.py`. Non estendere questo incarico a una riscrittura dell'empirics.

Priorità iniziali:

- **P0-cache:** dipendenze scientifiche e audit non completamente vincolati agli hash.
- **P0-verifica:** aggregati, OOS, MCSE, benchmark e slope non tutti ricostruiti indipendentemente.
- **P0-replay:** i checkpoint sintetici non sono disponibili dal solo clone.
- **P1-rate:** manca un esperimento separato con lambda fissata dalla legge teorica; il fit oracle non lo sostituisce.
- **P1-risoluzione:** il controllo di rank vale nella regione selezionata, non su tutta la curva o sul kernel infinito.
- **P1-selezione:** distribuzioni molto asimmetriche delle lambda, controllo degli estremi insufficiente, test end-to-end da rafforzare.
- **P2-scientifico:** aggiungere un DGP a sei coordinate, senza perdere le identità esatte né eliminare il baseline.

Non trasformare queste priorità in affermazioni che i risultati pubblicati siano falsi. Riproduci prima ciascun problema.

## 2. Correggi invalidazione, provenienza e stato di esecuzione

### 2.1 Dipendenze scientifiche complete

Nel vecchio `run.py`, `SCIENCE_SOURCES` esclude lo stesso `run.py`, mentre `context_for` sceglie basis e riferimento di popolazione. Una modifica scientifica lì può riutilizzare checkpoint con vecchio `run_hash`.

Implementa fingerprint completi per le fasi effettive, senza duplicare il motore:

```text
economic_state_hash
representation_hash
training_fit_hash
selection_hash
population_evaluation_hash
audit_hash
render_hash
```

Puoi usare un'unica fingerprint conservativa per le prime fasi se rende la patch più semplice, purché includa tutte le dipendenze scientifiche, compresa l'orchestrazione. Distingui invece l'evidenza storica da quella rigenerata: non basta riscrivere il manifest con gli hash correnti.

Vincola esplicitamente configurazione economica, kernel/normalizzazione, anchor e seed, algoritmo di fitting, selettore, quadratura, riferimento di popolazione, versioni di schema e dipendenze effettivamente chiamate. Registra un unico `effective_kernel_spec`: nelle configurazioni alternative non confondere `Environment.nu` effettivamente usato dal learner con il valore nu=1.5 rimasto nei parametri del DGP economico. I parametri di sola concorrenza non devono cambiare i dati scientifici.

### 2.2 Audit cache

`environment_audit` non deve accettare un JSON soltanto perché `environment_hash`, numero di stati e `passed` coincidono. Vincola codice, formule implementate, impostazioni, seed, soglie e kernel realmente usato.

Il controllo del baseline usa `simulations/estimator/kernel.py` per gli operatori spettrali. Questo file deve invalidare l'audit corrispondente; il vecchio manifest del baseline non lo include.

Aggiungi test che:

- mutano la versione scientifica di `context_for` e impediscono il riuso dei risultati incompatibili;
- mutano il kernel usato nella quadratura e invalidano E5;
- mutano una soglia o il codice di un audit e invalidano quel report;
- modificano solo un renderer e non certificano come nuovi i vecchi risultati numerici;
- cambiano `--workers` senza cambiare seed o risultati entro la tolleranza numerica dichiarata.

Usa directory temporanee nei test. Non modificare veri report di produzione per testare l'invalidazione. Una migrazione di cache richiede evidenza esplicita di compatibilità; altrimenti ricalcola in una nuova directory.

### 2.3 Manifest, test e compilazione

I report dei test devono includere comando realmente eseguito, return code, sorgenti/hash, versione ambiente e log. Un vecchio `tests['passed']=true` non basta per certificare la nuova sorgente.

Registra separatamente timestamp di avvio/fine del run originario, resume e rendering, tempo wall-clock e somma dei tempi delle repliche. Non presentare il tempo dell'ultimo resume come durata delle 3.500 valutazioni.

Scrivi manifest e checkpoint atomicamente. Rifiuta output incompleti o una seconda esecuzione concorrente incompatibile sulla stessa destinazione. I gate di produzione devono restare attivi anche con `python -O`: preferisci errori espliciti ad assert per i controlli critici.

Per LaTeX usa la lista delle dipendenze letta dal compilatore, o equivalente: hash di tutti i `.tex`, `.bib`, figure e tabelle effettivamente inclusi. La validità del PDF non è provata soltanto dall'hash del main e dei due file di testo generati.

## 3. Ricostruzione indipendente dei risultati e pubblicazione dei checkpoint sintetici

### 3.1 Verificatore numerico

Rafforza `diagnostics/verify.py` con una ricostruzione indipendente dai checkpoint, non una seconda chiamata allo stesso summarizer con successivo confronto tautologico.

Per ogni ambiente, T, lambda e selettore ricalcola e confronta:

```text
conteggio repliche, seed e identità del percorso
media, mediana, deviazione standard, MCSE
intervalli dichiarati e loro definizione
loss e Sharpe population e OOS
regret, bias, norma di estimation, cross term con segno
complessità campionaria e di popolazione
scelte di validation e penalità effettivamente refittate
oracle ensemble, oracle pathwise e benchmark
frequenze ai due estremi separatamente
regressioni di rate e bootstrap su percorsi interi
valori che compaiono nelle tabelle e nel testo generato
```

La ricostruzione degli intervalli deve seguire esattamente la procedura dichiarata. Distingui intervalli Monte Carlo condizionati a quadratura/anchor fissati dall'incertezza dovuta a tali approssimazioni. Non chiamarli intervalli simultanei se sono soltanto pointwise.

Test di mutazione obbligatori: modifica una cella di OOS loss, una MCSE, uno Sharpe, una selezione, un benchmark e una slope pubblicata; ogni modifica deve far fallire il controllo pertinente. Verifica anche NaN, Inf, repliche mancanti/duplicate, ordine delle lambda, forma delle matrici e identità errate dei seed.

### 3.2 Test del fitting e del protocollo

Mantieni e amplia i test esistenti. Verifica il fitting primal/dual, il secondo momento non centrato e il fattore T nella penalizzazione. Non sostituire la loss con una regressione sui ritorni condizionali stimati.

Il test OOS esistente modifica un array esterno inutilizzato: è utile ma non sufficiente. Crea un vero test end-to-end che mantiene identico il training e altera:

- tutti i ritorni di test;
- momenti di popolazione e oggetti oracle disponibili all'evaluator;
- W* disponibile ai diagnostici;
- la lunghezza dell'OOS.

Anchor, preprocessing di training, lambda candidate, scelta di validation e coefficienti stimati devono restare identici. Le metriche di test devono invece cambiare quando appropriato. Impedisci materialmente al selettore di accedere a questi oggetti.

### 3.3 Archivio accessibile e due livelli di CI

Pubblica o prepara un archivio immutabile dei soli dati **sintetici**: checkpoint per replica, momenti di quadratura, configurazioni, seed, hash e informazioni necessarie a rigenerare figure e tabelle. Usa GitHub Release/artifact o altro storage già autorizzato, non inserire grandi binari indiscriminatamente nella history Git.

L'archivio deve avere URL/identificatore stabile, checksum complessivo e manifest per file. Un manifest privo dei file non è una pubblicazione dei checkpoint. Non includere osservazioni JKP, credenziali o file di ambiente privati.

Offri due comandi distinti, anche con nomi diversi da quelli proposti:

```text
verify_published_aggregates   # rapido, dal clone, controlli possibili sugli aggregati
verify_archived_experiment   # dal pacchetto raw, ricostruzione completa
```

Documenta esattamente cosa ciascuno certifica. Aggiorna la CI: il vecchio `verify_final_outputs.py` verifica gli empirics, non gli aggregati delle simulazioni. Mantieni quel controllo e aggiungi quello sintetico. Lo smoke superato non deve diventare automaticamente “paper profile riprodotto”.

## 4. Baseline da preservare come riferimento

Mantieni il DGP corrente e i suoi parametri come `baseline_original` o equivalente identificatore stabile, senza alterarne retroattivamente i dati.

```text
N=600; G=N/3; D=6; K_F=3
rho_d=0.95
c_beta=0.04²
sigma_eps=0.08
mu_F=(0.10, 0.06, 0.03)
nu=1.5; ell=1; K(z,z)=1
```

Processo:

```text
S[g,d,0] iid N(0,1)
S[g,d,t+1] = rho_d*S[g,d,t] + sqrt(1-rho_d²)*eta[g,d,t+1]
U[g,d,t] = Phi(S[g,d,t])
Z[g,r,d,t] = 2*((U[g,d,t]+r/3) mod 1)-1, r=0,1,2
```

Le coordinate di un singolo stock hanno supporto continuo nel cubo, ma i tre stock di uno stesso gruppo non sono indipendenti. Non usare formule population per stock iid.

```text
theta_0(z) = pi*(z_1+1)
beta_0(z) = sqrt(c_beta)*(sqrt(2/3)*cos(theta_0),
                         sqrt(2/3)*sin(theta_0), 1/sqrt(3))'
F[t+1] iid N(mu_F, I_3-mu_F*mu_F')
epsilon[t+1] iid N(0,sigma_eps² I_N)
R[t+1] = B[t] F[t+1] + epsilon[t+1]
```

Indipendenza dei nuovi shock da informazione corrente e fra famiglie di shock. `E[FF']=I_3` è il **secondo momento**, non la covarianza. Non introdurre persistenza nei fattori lasciando immutati i momenti condizionali.

Identità da mantenere:

```text
||beta(z)||² = c_beta
B_t'B_t/N = Gamma_beta = (c_beta/3)I_3
m_t = B_t mu_F
S_t = E[R R' | F_t-info] = B_t B_t' + sigma_eps² I_N
w_t* = S_t^{-1}m_t
W*(z) = beta(z)'mu_F / (c_beta/3 + sigma_eps²/N)
w_t* = W*(Z_t)/N
```

Deriva W* dalla normal equation; non scegliere una target policy e ricostruire i premi per farla tornare.

```text
g = c_beta/3
q* = g/(g+sigma_eps²/N) * ||mu_F||²
Q* = 1-q*
SR* = sqrt(q*/(1-q*))
```

Non normalizzare i pesi a somma uno e non fissare a posteriori volatilità/leverage dentro la loss. Eventuali metriche a scala diversa vanno etichettate e separate.

## 5. Nuova economia `rich6d`: tutte le caratteristiche informative, E4 esatta

Aggiungi una seconda loading map, senza cambiare processo dei gruppi, marginali delle caratteristiche, fattori, idiosincratico o kernel.

Definisci sul cubo:

```text
u_d(z) = (z_d+1)/2
eta = 0.35

theta_eta(z) = 2*pi*u_1(z)
             + eta * sum_{d=2}^6 sin(2*pi*(u_d(z)-u_1(z)))

beta_eta(z) = sqrt(c_beta) * (
    sqrt(2/3)*cos(theta_eta(z)),
    sqrt(2/3)*sin(theta_eta(z)),
    1/sqrt(3)
)'
```

L'ampiezza eta=0.35 è fissata **prima dei nuovi risultati**. Eta=0 deve riprodurre il baseline. Non cambiare eta per migliorare slope, Sharpe o forma dei grafici.

### 5.1 Dimostrazione breve da includere nei materiali delle simulazioni

Per lo shift comune di ruolo `u -> (u+r/3) mod1`, le differenze `u_d-u_1` cambiano al più per interi. Quindi i seni restano uguali e

```text
theta_eta(shift_r(u)) = theta_eta(u)+2*pi*r/3   modulo 2*pi.
```

Somme di seni/coseni ai tre angoli implicano

```text
||beta_eta(z)||²=c_beta
sum_{r=0}^2 beta_eta(Z_gr) beta_eta(Z_gr)' = c_beta I_3.
```

Dunque `B'B/N=(c_beta/3)I_3`, la stessa W* derivata sopra con beta_eta e gli stessi q*/SR*. La prova non usa la scelta di mu_F per allineare artificialmente il target al kernel.

Tutte le sei coordinate influenzano realmente la policy. Implementa le derivate analitiche:

```text
d theta / d z_d = pi*eta*cos(2*pi*(u_d-u_1)), d>=2
d theta / d z_1 = pi - pi*eta*sum_{d=2}^6 cos(2*pi*(u_d-u_1)).
```

Confrontale con differenze finite su punti interni e riporta `E[(partial_d W*)²]` per ogni d. Non basta dire “sei caratteristiche” quando cinque non entrano nei loadings.

### 5.2 Audit senza modificare le ipotesi del paper

- E1: stessi AR stazionari e trasformazioni misurabili, N finito.
- E2: norma dei loadings invariata, shocks condizionatamente gaussiani, stesso bound sufficiente `B_X²=4(c_beta+sigma_eps²)` quando K(z,z)=1; verifica MGF non centrale e quarto momento K4=3.
- E3: le funzioni hanno estensione liscia su un intorno del cubo; W* è combinazione finita di tali funzioni. Documenta il dominio e l'ordine Sobolev, senza usare una norma Nyström finita come prova della norma RKHS infinita.
- E4: verifica `S_t w_t* = m_t` con un solve indipendente e la formula della policy.
- E5: conserva il sandwich `(sigma_eps²/N)T_K <= Sigma_H <= (c_beta+sigma_eps²/N)T_K`. I tre fattori economici **non** rendono automaticamente di rango tre la componente fattoriale integrata sugli stati.
- E6: stessi q*/SR* per eta0 e eta0.35 a parità di parametri.

La dipendenza da sei coordinate non prova che questa sia una hard family minimax. Non chiamarla tale. Rimane un benchmark liscio più ricco.

### 5.3 Implementazione canonica

Introduci un identificatore di loading map nei parametri effettivi e negli hash. Usa la stessa definizione in generazione, formula W*, quadratura e valutazione; verifica le identità anche con implementazioni algebriche indipendenti nei test.

Non lasciare chiamate hardcoded a `beta(z,p)` che usino il baseline durante la valutazione del nuovo DGP. Non creare un secondo `run.py` o duplicare l'intera pipeline.

## 6. Certifica prima l'approssimazione numerica

### 6.1 Solve esatto di riferimento a dimensioni piccole

Per N in `{30,60}` e T in `{60,120}`, costruisci il Gram temporale del **kernel completo**:

```text
G_ts = R[t+1]' K(Z_t,Z_s) R[s+1] / N²
alpha_lambda = (G + T*lambda*I_T)^{-1} 1_T
W_hat_lambda = sum_s alpha_lambda[s] X_s
C_hat_exact = Tr[G(G+T*lambda*I_T)^{-1}].
```

Confronta payoff, loss empirica, norma/penalizzazione e normal equation con un'implementazione indipendente. Il fattore T è indispensabile. Poi confronta Nyström sullo stesso campione per rank crescenti e medesima lambda. La coincidenza a rank finito non è garantita: quantificare l'errore, non impostare tolleranze inventate dopo i risultati.

### 6.2 Risoluzione nella regione selezionata e nell'intera curva

Mantieni il test storico 256→512 sul regret come evidenza delimitata. Aggiungi confronti su complessità, payoff e regret a **lambda fissate**, non solo dopo avere riselezionato l'oracle a ciascun rank. La riselezione può mascherare differenze della curva.

Per nuovi esperimenti usa anchor annidati e rank candidati `{256,512,1024}`; considera 2048 solo se il criterio numerico predefinito lo richiede e le risorse lo consentono. Confronta sulle stesse traiettorie e su quadratura comune sufficientemente risolta.

Registra separatamente:

```text
errore relativo/assoluto di regret nella regione theory/oracle/selected
errore di payoff quadratico fra policy approssimata e più risolta
errore della complessità
floor di proiezione
sensibilità del tratto quasi interpolante e del massimo di loss
```

Criterio proposto per nuovi risultati principali: floor di proiezione sotto il 5% del regret medio rilevante e differenza media fra due risoluzioni sotto il 5% nella regione principale. Mostra anche l'incertezza della differenza; un punto stimato entro il 5% non basta se la sua incertezza è troppo ampia. Congela il criterio prima della produzione. Se fallisce, aumenta risoluzione o etichetta l'esperimento come non risolto, senza alterare economia o rimuovere punti.

Non pretendere il 5% su qualunque lambda quasi zero; quella regione può rimanere diagnostica. Ma non presentarla come accuratezza certificata del kernel completo.

### 6.3 Quadratura e spettro

Conserva il calcolo per gruppi, che è corretto. Aggiungi repliche indipendenti della quadratura, non solo un confronto annidato con lo stesso seed. Su policy fissate confronta 8.192 e 32.768 gruppi e almeno quattro seed di quadratura indipendenti, con aggregazione e incertezza dichiarate.

Per lo spettro usa una scala di risoluzioni predefinita `{1024,2048,4096}` gruppi e tre seed. Prima stima memoria/runtime: 3M nodi e matrici dense possono essere costosi. Usa operatori matrix-free o autovalori parziali controllati se necessario, con confronto su una dimensione dove il solve denso è disponibile.

Mantieni le finestre descrittive 30–200, 60–400 e 120–800, quando realmente risolte; segnala gli indici non risolti. Non cambiare automaticamente finestre, cutoff o smoothness finché il fit non somiglia a b.

Separa nei report:

```text
analytic_E5_status
operator_identity_status
quadrature_resolution_status
spectral_fit_descriptive_statistics
```

Il fit empirico vicino a −b non è la prova di E5; il fit lontano non è automaticamente una violazione della prova analitica. La banda storica ±0.35, o ±0.5 per altri kernel, resta una scelta diagnostica storica: non farne il criterio di verità del teorema.

N resta fisso durante ogni esperimento di rate. Il lower bound `(sigma_eps²/N)T_K` non fornisce da solo costanti uniformi per un limite congiunto N→infinito.

## 7. Tre esperimenti distinti: theory path, oracle e validation

Questa distinzione è obbligatoria. Non limitarti a sovrapporre una retta teorica ai fit ex post.

### 7.1 Theory path senza tuning sui risultati

Per ciascun DGP e kernel, congela

```text
b = 1+2*nu/D
lambda_T(a) = a*s_ref*T^(-b/(b+1))
a in {0.25, 1, 4}.
```

`s_ref` è una scala positiva fissata da un pilot indipendente, per esempio l'autovalore principale del managed second-moment operator a risoluzione dichiarata. Il pilot può usare il DGP per calibrare unità numeriche, ma non le loss o gli Sharpe delle repliche di conferma. Etichetta chiaramente questa procedura teorica, non implementabile senza informazione di calibrazione population quando la usi.

Registra una volta s_ref, seed, quadratura, rank e hash. Non ricalibrarlo a ciascun T o per centrare l'oracle. Calcola i coefficienti alle lambda effettive della formula, senza snap alla griglia dei 96 valori.

Per ogni a e T mostra:

```text
regret medio e quantili a lambda_T(a)
T^(b/(b+1))*regret
C_pop(lambda_T(a))/T^(1/(b+1))
C_sample(lambda_T(a)) e C_sample/T
bias di regolarizzazione, floor numerico, estimation norm, cross term
Sharpe gap coerente con la definizione del paper
```

Non richiedere costanza esatta del regret riscalato: un upper bound e un target liscio possono dare convergenza più rapida. La sequenza di lambda non dimostra da sola il rate statistico; servono misure di rischio e approssimazione.

### 7.2 Oracle ex post

Mantieni l'oracle storico che minimizza la media Monte Carlo del regret. Chiamalo `ensemble_plugin_loss_oracle` o definiscilo altrettanto chiaramente. Il suo Sharpe è **lo Sharpe alla penalità oracle per la loss**, non un oracle per lo Sharpe.

Aggiungi un controllo cross-fitted a cinque fold sulle repliche: per ogni T scegli la lambda sui quattro fold di calibrazione e valutala sul quinto. I fold sono per percorso intero, mai per cella T/lambda. Non usare gli OOS della replica valutata per scegliere la lambda. Riporta l'ottimismo di selezione Monte Carlo e gli intervalli, non nasconderlo dentro una precisione apparente.

### 7.3 Validazione implementabile

Mantieni `holdout25` come benchmark immutato. Confronta prospectivamente solo pochi selettori:

1. `holdout25`: il metodo esistente.
2. `rolling3`: tre split cronologici; train `[0,.4T)`/validation `[.4T,.6T)`, train `[0,.6T)`/validation `[.6T,.8T)`, train `[0,.8T)`/validation `[.8T,T)`. Specifica arrotondamenti interi e lunghezze minime. Aggrega le loss con pesi pari al numero di osservazioni validate. Refit finale su tutti i T dati.
3. `rolling3_conservative`: variante prudente della stessa procedura, solo se definita prima delle nuove simulazioni. Una regola one-standard-error richiede una stima dell'incertezza che rispetti la dipendenza temporale; documentala come euristica, non come nuovo teorema. Altrimenti ometti questa terza variante e motivane l'omissione.

Il criterio resta la loss del paper. Non introdurre una promessa che la validazione dimezzerà il gap; confronta ciò che emerge. Il selettore non vede W*, momenti population, OOS, oracle o slope.

Preserva i tie-break del metodo storico per riprodurlo. Per nuovi metodi fissa una regola deterministica: tra minimi effettivamente pari scegli la lambda più alta. Non cambiare la tolleranza di parità dopo aver osservato i risultati.

Aggiungi il portafoglio zero come benchmark esplicito: Q=1, C=0, payoff nullo e convenzione Sharpe dichiarata. Se vuoi ammetterlo fra le scelte di validation, crea un metodo distinto; non cambiare implicitamente l'estimatore storico o inventare lambda finita equivalente a infinito.

### 7.4 Diagnostici di selezione obbligatori

Per ogni T/ambiente/selettore riporta media, mediana, IQR, media geometrica delle lambda positive, frequenze estremo inferiore ed estremo superiore, contributo delle scelte agli estremi alla media della lambda, frequenza zero-policy se ammessa, distribuzione di complessità e regret.

A T=1440 il baseline storico ha media lambda circa `8.6553e-4`, mediana circa `1.2743e-6` e frequenza combinata agli estremi 8.4%. Non usare la media come se descrivesse la scelta tipica. Non scambiare nei grafici pendenza della media, della mediana e della media geometrica.

## 8. OOS indipendente e OOS futuro: entrambi, senza confonderli

L'OOS corrente è un nuovo percorso stazionario indipendente. Mantienilo per misurare population risk.

Aggiungi una valutazione contigua: per una singola traiettoria sufficientemente lunga, stima al tempo T usando soltanto le prime T osservazioni e valuta da T+1 a T+H. Nessuna osservazione successiva a T entra in fitting o selezione. Per comparare T diversi, i blocchi possono sovrapporsi all'interno della replica: bootstrap e incertezza devono allora resamplare la replica intera.

Etichetta `independent_stationary_test` e `contiguous_future_test`. Non chiamare una replica riavviata “next year” o “continuazione futura” e non sostituire silenziosamente le vecchie metriche.

Usa H=720 come default per confrontabilità, con indicazione chiara che lo Sharpe è per periodo. Se presenti annualizzazione, dichiarala separatamente e non cambiare la loss o la definizione teorica dello Sharpe.

## 9. Disegno prospettico e risorse

Prima di osservare nuovi risultati di performance crea `simulations/config/confirmation_v2.json` con schema, timestamp, hash, tutte le scelte e un elenco dei criteri di accuratezza numerica.

Proposta concreta da congelare:

```text
master_seed_v2 = 2026100801
famiglie RNG distinte: stato, fattori, idiosincratico, anchor, pilot,
quadratura, test indipendente, bootstrap, selezione dei fold
loading maps: baseline_original (eta0), rich6d (eta0.35)
N headline: 600
T pratici: 60,90,120,180,240,360,540,720,1080,1440
T aggiuntivi: 2160,3240,4860,7290
kernel headline: Matern nu1.5, ell1
kernel sensitivity: nu0.5 e nu2.5, D6 invariato
H test: 720
```

Esegui prima un pilot **esclusivamente di costo e accuratezza numerica** su seed separati: tempo per data/fit, memoria, errori di proiezione e solver. Non usare il pilot per scegliere i parametri economici o le finestre che producono i rate migliori.

Obiettivi di produzione, se sostenibili sulle risorse autorizzate:

- 500 repliche nuove per ciascuna loading map nel confronto dei selettori sul T-grid pratico;
- prime 200 repliche coerentemente estese per ciascuna loading map nei T aggiuntivi e theory paths;
- 200 per le due smoothness alternative, con estensione temporale dichiarata;
- confronti di rank e quadratura appaiati su subset predefiniti, almeno 50 percorsi per controlli onerosi e più repliche se l'incertezza impedisce di certificare la soglia.

Questi numeri sono un protocollo per **nuovi** risultati. Non dichiararli preregistrati per il vecchio run già osservato. Se il costo stimato non consente il protocollo, documenta il ridimensionamento prima della produzione e mantieni l'obiettivo scientifico; non sostituire una run a quattro repliche con “paper complete”.

Usa common random numbers per comparare eta, selettori e rank quando appropriato. Registra il numero di traiettorie economiche distinte e il numero di valutazioni: le due robustness di rank storiche riutilizzano percorsi del baseline, quindi 3.500 valutazioni non sono 3.500 traiettorie tutte reciprocamente indipendenti.

## 10. Metriche, benchmark e grafici

### 10.1 Metriche con definizioni non ambigue

Mantieni separati `Q_hat_OOS`, rischio population valutato con quadratura, regret relativo a W* analitico, regret relativo a un optimum di sottospazio quando necessario, SR population e SR campionario.

La loss population alla ridge di popolazione non ha varianza di stima: non deve essere forzata a una U. La decomposizione corretta resta

```text
Q(ahat)-Q* = [Q(a_lambda)-Q*]
           + (ahat-a_lambda)' S (ahat-a_lambda)
           + 2(ahat-a_lambda)'(S a_lambda-m).
```

Il terzo termine ha segno. Non chiamare `Q(ahat)-Q(a_lambda)` una varianza nonnegativa.

Per una rappresentazione finita, `C_sample<=min(T,r_num)` e `C_sample/T<=1`. `C_population/T` può superare uno. Non censurare questi valori e non mescolare le due definizioni. Il numero di fattori economici resta tre, indipendente dal rank numerico.

### 10.2 Benchmark

Preserva zero/uguale-peso, affine, validation e oracle già descritti, mantenendo le scale chiaramente dichiarate. Se aggiungi il benchmark `beta_span_informed`, deve stimare i coefficienti dai dati di training, non usare mu_F vera. Etichettalo come benchmark privilegiato che conosce le vere funzioni beta; non è un concorrente con la stessa informazione del learner generico.

Per miglioramenti di selezione confronta differenze appaiate di regret/OOS e intervalli. Non scegliere un nuovo metodo sui risultati di conferma e presentarlo come vincitore validato su dati ancora intatti. Eventuali ulteriori modifiche richiedono una nuova coorte di seed.

### 10.3 Figure prioritarie

Non massimizzare il numero di figure. Produci queste figure scientificamente leggibili, più appendice riproducibile:

1. Quattro pannelli per ciascun DGP: loss OOS completa, Sharpe OOS, heatmap, distribuzione/scelta di C e lambda. Mantieni legenda coerente mean/median.
2. Regret population e regret normalizzato per `q*`, con CI, contro C e C/T. Conserva accanto la loss grezza. Nel baseline Q* è vicino a uno: un asse log della loss può nascondere il miglioramento.
3. Decomposizione con bias, floor, estimation norm e cross term con segno.
4. Theory path riscalati e oracle/validation distinti: lambda, C, regret, Sharpe gap e relativi intervalli.
5. Confronto baseline/rich6d e selettori, su percorsi appaiati.
6. Appendice: risoluzione kernel/rank, quadratura, finestre spettrali, estremi delle lambda, indipendente vs futuro contiguo, N/persistenza/kernel sensitivity.

Per OOS excess loss, valori negativi rispetto a Q* possono verificarsi per errore campionario: non tagliarli a zero e non metterli su un log impossibile; usa scala lineare/simmetrica e spiega.

Le heatmap su griglia comune di complessità possono interpolare soltanto dentro supporti effettivamente calcolati, con metodo documentato; niente extrapolazione mascherata. Una normalizzazione rispetto al minimo di riga è solo descrittiva, non una procedura di selezione.

Non disegnare esiti schematici spacciandoli per Monte Carlo. Niente anni fittizi o riferimenti JKP su risultati sintetici.

## 11. Integrazione nel paper e riepilogo delle modifiche

Modifica soltanto testo/tabelle/figure della sezione simulazioni e le parti tecniche indispensabili per includerli. La teoria protetta deve restare con gli stessi hash. La patch del nuovo DGP e i suoi audit appartengono alla documentazione delle simulazioni, non sono un'occasione per riscrivere E1–E6.

Rendi espliciti:

- baseline esatto vs robustness che violano alcune ipotesi;
- supporto continuo dei population ranks vs supporto finito degli empirical ranks a N fisso;
- kernel infinito teorico vs approssimazione numerica;
- dipendenza temporale delle caratteristiche vs fattori iid;
- source condition r=1 come ipotesi del teorema, non prova che un singolo target liscio sia least-favourable;
- rate teorico, fit descrittivo e intervallo Monte Carlo;
- riferimento analitico full-policy vs riferimento di sottospazio per eteroschedasticità.

Compila da checkout pulito con dipendenze documentate, verifica riferimenti e layout. Non scrivere `visual_review=passed` senza ispezione effettiva dei render. Un PDF compilato non certifica i numeri che contiene: occorrono entrambi i controlli.

## 12. Deliverable e criteri di completamento

Consegna, usando nomi coerenti e una sola pipeline:

```text
simulations/docs/audit_v2_findings.md
simulations/docs/confirmation_v2_protocol.md
simulations/config/confirmation_v2.json
simulations/docs/rich6d_derivation.md
simulations/docs/rate_interpretation.md
simulations/docs/reproduction_v2.md
simulations/outputs/confirmation_v2/audit/
simulations/outputs/confirmation_v2/data/
simulations/outputs/confirmation_v2/tables/
simulations/outputs/confirmation_v2/figures/
simulations/outputs/confirmation_v2/paper/
```

Nel report finale distingui le seguenti condizioni, senza comprimerle in un generico `complete`:

```text
source_preservation_verified
baseline_reproduced_or_archived_verified
cache_mutation_tests_passed
aggregate_mutation_tests_passed
raw_checkpoint_archive_accessible
exact_kernel_reference_checked
numerical_resolution_certified_for_claimed_region
rich6d_algebra_and_assumptions_checked
theory_path_executed
selector_comparison_executed
independent_and_forward_OOS_executed
all_reported_numbers_reconstructed
paper_compiled_and_render_reviewed
```

Per ciascuna: comando, SHA/schema/hash, evidenza, numero di repliche effettive, eventuale limite. Fornisci anche il diff dei file modificati e la conferma che teoria/empirics protetti sono invariati.

## 13. Ordine operativo vincolante

1. Ispeziona HEAD e salva il riferimento iniziale senza modificare sorgenti protette.
2. Conferma i gap, correggi cache/provenienza/verificatori con test di mutazione.
3. Verifica o recupera l'archivio sintetico e ricostruisci gli aggregati storici disponibili.
4. Aggiungi `rich6d`, dimostrazione breve e test indipendenti; baseline eta0 riproducibile.
5. Introduci confronto exact-kernel/Nyström e protocollo di risoluzione.
6. Congela pilot, protocollo di conferma, seed e budget prima delle nuove performance.
7. Esegui theory paths, oracle cross-fitted e selettori, con OOS separati.
8. Verifica quadratura/spettro senza criteri che impongano la pendenza teorica.
9. Rigenera figure, tabelle e soltanto la sezione simulazioni.
10. Compila, controlla render, verifica hash e consegna il report con risultati e limiti reali.

**Non fermarti dopo aver scritto codice o lanciato job. Non dichiarare un rate confermato perché una curva sembra parallela a una retta. Non dichiarare fallito il teorema perché un DGP liscio o una lambda oracle non hanno la stessa pendenza. Il compito è produrre evidenza riproducibile, non un risultato prestabilito.**
