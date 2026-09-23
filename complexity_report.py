"""Write an Italian HTML report from saved analysis tables. No model training."""

from html import escape
from pathlib import Path
import shlex

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/complexity_analysis"


def table(headers, rows):
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row)
                   + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(path, caption):
    return f'<figure><a href="{path}"><img src="{path}" loading="lazy" alt="{escape(caption)}"></a><figcaption>{caption}</figcaption></figure>'


def build_report(output=None, characteristic_name="all", characteristics=None):
    global OUTPUT
    OUTPUT = Path(output) if output is not None else ROOT / "results/complexity_analysis"
    comparison = pd.read_parquet(OUTPUT / "kernel_comparison.parquet")
    gaussian = comparison.set_index("kernel").loc["gaussian"]

    peaks = table(
        ["Kernel", "λ* OOS", "Average C/T*", "Sharpe OOS", "Intervallo puntuale 95%", "C/T*: percentili bootstrap"],
        [[r.label, f"{r.peak_lambda:.3e}", f"{r.peak_q:.3f}", f"{r.peak_sharpe:.3f}",
          f"[{r.peak_sharpe_ci_low:.2f}, {r.peak_sharpe_ci_high:.2f}]",
          f"[{r.peak_q_boot_p025:.3f}, {r.peak_q_boot_p975:.3f}]"] for r in comparison.itertuples()])
    decline = table(
        ["Kernel", "C/T medio più alto", "Sharpe al C/T più alto", "Calo dal massimo", "Intervallo del calo 95%", "Sharpe IS al C/T più alto"],
        [[r.label, f"{r.largest_q:.3f}", f"{r.largest_q_sharpe:.3f}", f"{r.drop_from_peak:.3f}",
          f"[{r.drop_ci_low:.2f}, {r.drop_ci_high:.2f}]", f"{r.train_sharpe_at_largest_q:.2f}"] for r in comparison.itertuples()])
    annual = table(
        ["Kernel", "C/T* 2024", "C* 2024", "λ* 2024", "Sharpe OOS 2024", "Finestre con massimo a un estremo"],
        [[r.label, f"{r.test2024_q:.3f}", f"{r.test2024_C:.2f}", f"{r.test2024_lambda:.3e}",
          f"{r.test2024_sharpe:.3f}", f"{round(r.annual_peak_at_endpoint_share * r.windows)}/{r.windows}"] for r in comparison.itertuples()])
    implemented = table(
        ["Kernel", "Sharpe OOS uncapped", "Crescita composta annualizzata¹", "Max drawdown¹",
         "Gross mediano", "Gross p95", "Gross p99", "Gross max", "Max |weight|",
         "Lengthscale", "b descrittivo, 2024"],
        [[r.label, f"{r.selected_sharpe:.3f}", f"{r.cagr:.1%}", f"{r.max_drawdown:.1%}",
          f"{r.median_gross_exposure:.2f}", f"{r.p95_gross_exposure:.2f}",
          f"{r.p99_gross_exposure:.2f}", f"{r.max_gross_exposure:.2f}",
          "—" if pd.isna(r.max_abs_weight) else f"{r.max_abs_weight:.3f}",
          "—" if r.kernel in ("linear", "ntk") else f"{r.lengthscale:.4f}",
          "—" if pd.isna(r.estimated_b) else f"{r.estimated_b:.3f}"] for r in comparison.itertuples()])

    gallery = ""
    for row in comparison.itertuples():
        folder = f"../{row.kernel}/{characteristic_name}"
        gallery += f'<details id="{row.kernel}"><summary>{row.label} · tutti i grafici</summary>'
        gallery += f'<p><a href="{folder}/pooled_oos_relative_complexity_3d.html">Ruota la curva OOS aggregata</a> · <a href="{folder}/relative_complexity_3d.html">Ruota la curva OOS 2024</a></p>'
        gallery += '<div class="gallery">'
        for filename, caption in [
            ("sharpe_vs_relative_complexity", "Sharpe IS e OOS vs Average C/T; scale verticali separate."),
            ("sharpe_vs_relative_complexity_2024", "Sharpe IS e OOS vs C/T, singola finestra 2024."),
            ("relative_complexity_vs_lambda", "Average C/T vs λ; unico punto evidenziato: massimo OOS."),
            ("relative_complexity_uncertainty", "Sharpe OOS aggregato e intervalli bootstrap puntuali."),
            ("relative_complexity_3d", "Curva OOS 2024: λ, C/T e Sharpe."),
            ("pooled_oos_relative_complexity_3d", "Curva OOS aggregata: λ, Average C/T e Sharpe dei rendimenti concatenati."),
            ("eigenvalue_spectrum", "Spettro esistente: istogramma normalizzato e fit per rango, 2024."),
        ]:
            gallery += figure(f"{folder}/{filename}.png", f"{row.label} — {caption}")
        gallery += "</div></details>"

    if characteristic_name != "all":
        gallery = ""
        for row in comparison.itertuples():
            folder = ROOT / "results" / row.kernel / characteristic_name
            gallery += f'<details><summary>{row.label} · tutti i grafici</summary><div class="gallery">'
            for path in sorted(folder.glob("*.png")):
                relative = f"../{row.kernel}/{characteristic_name}/{path.name}"
                gallery += figure(relative, path.stem.replace("_", " "))
                if path.with_suffix(".html").exists():
                    gallery += f'<p><a href="{relative[:-4]}.html">Grafico 3D interattivo</a></p>'
            gallery += "</div></details>"
        first = comparison.iloc[0]
        name = escape(characteristic_name)
        linear_spectrum = pd.read_parquet(ROOT / "results/linear" / characteristic_name / "kernel_eigenvalues.parquet")
        feature_count = int(linear_spectrum.groupby("test_year").size().max())
        rank_note = ("Anche l'NTK senza bias in una dimensione ha rango al massimo due: "
                     "le molte coordinate casuali non equivalgono ad altrettanti fattori indipendenti. "
                     "Due autovalori non identificano una legge di decadimento asintotica."
                     if feature_count == 2 else "Le coordinate casuali possono essere linearmente dipendenti.")
        command_selection = escape(shlex.join(characteristics or [characteristic_name]))
        content = f'''<!doctype html><html lang="it"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Complexity · {name}</title>
<style>body{{margin:0;background:#f3f1ec;color:#243343;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1160px;margin:auto;padding:32px 24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}
img{{width:100%;height:auto}}figure{{margin:16px 0}}figcaption{{font-size:13px;color:#65717b}}
table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #ddd}}
.table-wrap{{overflow-x:auto}}.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}
a{{color:#126e7e}}summary{{cursor:pointer;font-size:20px;padding:16px 0}}
@media(max-width:760px){{.gallery{{grid-template-columns:1fr}}}}</style><main>
<h1>Solo {name} · equity, spettro e complessità</h1>
<p>Sei kernel riallenati con la caratteristica {name}. {int(first.windows)} finestre expanding,
{int(first.months)} rendimenti OOS da {first.first_return_date} a {first.last_return_date};
{int(first.lambdas)} λ per kernel. Training iniziale di 10 anni, validation di 5 anni, test annuale.
Il refit include training e validation: T varia da {int(first.T_min)} a {int(first.T_max)} mesi.</p>
<p><a href="../lambda_scaling_{name}/report.html">Confronto delle regole di lambda</a> · <a href="#grafici">Tutti i grafici</a></p>
<section><h2>Portafogli con selezione annuale</h2>
<p>λ è selezionata sulla loss raw di validation. I pesi del test sono ridimensionati quando l'esposizione lorda supera 2.</p>
{figure(f'../equities_{characteristic_name}.png', 'Percorsi OOS uncapped dei sei kernel.')}{implemented}</section>
<section><h2>Sharpe e complessità relativa</h2>
<p>C(λ) = Σ μ/(μ+λ); C/T è calcolato per finestra e poi mediato. Gli Sharpe OOS delle curve
usano i rendimenti raw concatenati, mentre gli Sharpe IS sono medie tra finestre.
Le stelle identificano massimi ex post e non selezionano i portafogli implementati.
Le bande sono intervalli puntuali da 1.000 ricampionamenti appaiati di finestre test annuali.</p>
{figure('relative_complexity_all_kernels.png', 'Sharpe OOS vs complessità relativa, con intervalli bootstrap.')}
{peaks}{decline}</section><section><h2>Massimi nelle singole finestre</h2>{annual}
{figure('relative_complexity_optima_over_time.png', 'Massimi annuali ex post; scale verticali adattate al rango dei kernel.')}</section>
<section><h2>Spettro e interpretazione</h2>
<p>Lo spettro proviene dal secondo momento non centrato F′F/T. Il limite è C/T ≤ min(1, P/T).
Il lineare ha {feature_count} coordinate, inclusa la costante. {rank_note}
I fit dello spettro finito sono descrittivi.</p>
<p>Le equity capitalizzano i rendimenti excess salvati. Costi e turnover non sono inclusi.
La pulizia del dataset è quella della prova precedente e usa la disponibilità sull'intero campione.</p></section>
<section id="grafici"><h2>Galleria completa</h2>{gallery}</section>
<section><h2>Dati</h2><p><a href="kernel_comparison.parquet">Confronto kernel</a> ·
<a href="window_optima.parquet">Massimi per finestra</a></p>
<p>Rigenerazione: <code>python3 complexity_analysis.py --characteristics {command_selection}</code>.</p></section></main></html>'''
        OUTPUT.mkdir(exist_ok=True, parents=True)
        (OUTPUT / "report.html").write_text(content, encoding="utf-8")
        print("Report:", OUTPUT / "report.html")
        return

    content = f"""
<header>
<p class="eyebrow">PORTFOLIO KERNELS · RESEARCH NOTE · 08 SETTEMBRE 2026</p>
<h1>Complexity, regolarizzazione<br>e performance fuori campione</h1>
<p class="lead">Sei kernel, 47 finestre expanding e una lettura normalizzata in C/T.
Analisi delle run già salvate, senza nuovo training né modifica dei pesi.</p>
<div class="facts"><span><b>564</b> rendimenti mensili OOS</span><span><b>60</b> λ per kernel</span><span><b>1.000</b> ricampionamenti</span></div>
</header>
<nav><a href="#sintesi">Sintesi</a><a href="#metodo">Definizioni</a><a href="#risultati">Risultati</a><a href="#curva-3d">Curva 3D</a><a href="#limiti">Limiti</a><a href="#galleria">Galleria completa</a></nav>

<section id="sintesi"><h2>01 · La conclusione, prima dei dettagli</h2>
<p><strong>C/T è utile per confrontare finestre di stima di lunghezza diversa, ma non sostituisce C in ogni interpretazione teorica.</strong>
Nei risultati esistenti tutti e sei i kernel hanno un massimo dello Sharpe OOS aggregato seguito da un calo.
La normalizzazione non crea questo fenomeno: cambia soltanto l'ascissa, lasciando invariati rendimenti, Sharpe e λ del massimo.</p>
<p>Per i kernel non lineari il massimo si colloca a un Average C/T tra 0.281 e 0.399; per il lineare a 0.108.
Non è una soglia universale. I massimi annuali oscillano sensibilmente e le regioni vicine al massimo aggregato sono abbastanza piatte:
è più prudente parlare di una <em>zona di complexity favorevole</em> che di un numero preciso da usare in ogni periodo.</p>
<div class="callout">La figura principale per discutere il risultato è quella 2D qui sotto.
La nuova curva 3D mostra esattamente gli stessi Sharpe OOS concatenati, aggiungendo l'asse λ.
Le precedenti superfici basate su medie di Sharpe annuali sono escluse dai risultati principali.</div>
{figure('relative_complexity_all_kernels.png', 'Figura 1. Sharpe OOS aggregato vs Average C/T. Stelle: massimi osservati ex post. Bande: intervalli puntuali al 95%. L’asse x del lineare ha una scala diversa per rendere leggibile il suo intervallo più ristretto.')}
</section>

<section id="metodo"><h2>02 · Che cosa misuriamo</h2>
<h3>Campione, finestre e denominatore</h3>
<p>La run utilizza tutte le 132 caratteristiche disponibili dopo la pulizia. Il modello lineare ha 133 feature, inclusa la costante;
i kernel non lineari ne utilizzano 1.000. La procedura parte da 10 anni di training e 5 di validation, con un test di 12 mesi.
Il train si espande annualmente. Il modello usato nel test viene ristimato su train + validation:</p>
<div class="formula">T = numero effettivo di mesi del refit &nbsp; → &nbsp; da 180 a 732.</div>
<p>Il denominatore non è il numero di titoli e non è la lunghezza del test. Lo abbiamo ricostruito dalle date dei panel puliti
e confrontato con le dimensioni degli spettri salvati. Le 47 finestre hanno anni di formazione 1978–2024.
Poiché il rendimento è quello del mese successivo, la serie realizzata copre febbraio 1978–gennaio 2025.
La dicitura “test 2024” indica quindi formazione gennaio–dicembre 2024 e rendimenti febbraio 2024–gennaio 2025.</p>
<h3>Complexity empirica e normalizzazione</h3>
<p>Se F è la matrice mensile dei rendimenti delle feature, lo spettro salvato proviene da F′F/T.
Si tratta di una matrice di <strong>secondi momenti non centrati</strong>, non di una covarianza dopo sottrazione della media.
Indicando con μ<sub>j,w</sub> gli autovalori della finestra w:</p>
<div class="formula">C<sub>w</sub>(λ) = Σ<sub>j</sub> μ<sub>j,w</sub> / (μ<sub>j,w</sub> + λ)<br>
q<sub>w</sub>(λ) = C<sub>w</sub>(λ) / T<sub>w</sub><br>
Average C/T = (1/W) Σ<sub>w</sub> q<sub>w</sub>(λ).</div>
<p>La divisione avviene <strong>prima</strong> della media. Usare invece ΣC/ΣT attribuirebbe più peso alle finestre lunghe.
Gli autovalori contengono già il fattore 1/T del secondo momento; dividere C per T introduce una nuova misura relativa,
non corregge un fattore mancante nella matrice.</p>
<p>Nel campione finito, 0 ≤ C/T ≤ min(1, P/T), con P numero di feature. C/T indica quindi complexity rispetto alle osservazioni,
non la percentuale di tutte le feature utilizzate. Nel 2024 il limite del lineare è al massimo 133/732 ≈ 0.182;
i modelli con 1.000 feature possono avvicinarsi a 1. I rispettivi intervalli non devono coincidere.</p>
<h3>Collegamento al paper</h3>
<p>Il riferimento è <em>Portfolio_Paper.pdf</em>, pp. 14–15: Definizione 1, equazione (22), Proposizione 3, equazione (23), e Figura 2.
La complexity teorica è definita attraverso lo spettro di popolazione; quella qui calcolata usa lo spettro empirico e un numero finito di feature.</p>
<p>Nel caso di decadimento polinomiale e sotto le ipotesi della Proposizione 3, C*<sub>T</sub> cresce come T<sup>1/(br+1)</sup>.
Ne segue che C*<sub>T</sub>/T decresce come T<sup>−br/(br+1)</sup> per b,r positivi.
Non ci si deve dunque aspettare necessariamente un rapporto ottimale costante nel tempo.
La Figura 2 è schematica: non impone un unico massimo interno in ogni singolo campione osservato.
Il caso gaussiano con decadimento esponenziale richiede inoltre la sua specifica interpretazione asintotica.</p>
</section>

<section id="risultati"><h2>03 · Il massimo OOS e il successivo calo</h2>
<p>Per ogni λ concateno i 564 rendimenti mensili OOS non vincolati e calcolo Sharpe = √12 × media / deviazione standard campionaria.
Il fattore √12 è una convenzione di presentazione dello Sharpe: non introduce alcun obiettivo di volatilità nel portafoglio.
Per l'IS mostro invece la media degli Sharpe delle finestre di refit, che si sovrappongono:
non è una serie IS indipendente comparabile mese per mese alla serie OOS concatenata.</p>
{peaks}
<p>Il massimo gaussiano è {gaussian.peak_sharpe:.3f} a C/T medio {gaussian.peak_q:.3f}.
Matérn 5/2 è molto vicino. Questi valori sono massimi <strong>scelti guardando il test</strong>: misurano il profilo della curva,
non la performance che si sarebbe potuta selezionare anticipatamente. Le bande dei singoli kernel non costituiscono un test della differenza fra kernel.</p>
{decline}
<p>Aumentando C/T oltre la zona favorevole, lo Sharpe OOS cala mentre lo Sharpe IS raggiunge valori molto alti nei modelli non lineari.
Questo contrasto è compatibile con un forte adattamento al campione di stima. Non ho levigato le curve 2D,
imposto una parabola o aggiunto punti per ottenere una discesa: sono i risultati salvati sulla griglia originale.</p>
<h3>Quanto è preciso il massimo?</h3>
<p>Ho effettuato 1.000 ricampionamenti di intere finestre test da 12 mesi, con seed 2024.
Ogni estrazione usa gli stessi blocchi per tutti i λ, preservando il confronto appaiato e la dipendenza all'interno dell'anno.
Le bande dello Sharpe sono puntuali per λ. Per descrivere l'instabilità della posizione del massimo,
in ogni estrazione ricalcolo anche λ* e la media dei C/T delle finestre estratte.</p>
<p>I percentili della posizione sono ampi: per il gaussiano circa [{gaussian.peak_q_boot_p025:.3f}, {gaussian.peak_q_boot_p975:.3f}].
Non sono un intervallo di confidenza formalmente corretto per l'argmax dopo selezione.
Gli intervalli appaiati del calo confrontano il λ del massimo osservato con il λ a maggiore complexity;
anche questi non correggono la scelta ex post del primo punto.
Il bootstrap tratta le finestre annuali come ricampionabili e non ricostruisce il training, né conserva dipendenze fra anni:
è un controllo descrittivo di stabilità, non una validazione completa del processo di ricerca.</p>
</section>

<section><h2>04 · Il risultato non è soltanto una media?</h2>
{annual}
<p>Nella finestra 2024 tutti i kernel mostrano un massimo interno ai valori esaminati,
ma le posizioni non coincidono con i massimi aggregati. Dodici rendimenti producono stime di Sharpe e di argmax molto rumorose.
Nel complesso delle 47 finestre esistono anche massimi agli estremi della griglia, come quantificato in tabella.
È quindi corretto mostrare il 2024 come esempio aggiuntivo, non come prova che il fenomeno si manifesti identico ogni anno.</p>
{figure('relative_complexity_optima_over_time.png', 'Figura 2. Posizione del massimo OOS nelle singole finestre test. I punti sono ex post, senza smoothing né fit teorico.')}
<p>Per una singola finestra C/T è determinato da λ: il grafico (λ, C/T, Sharpe) è inevitabilmente una curva nello spazio.
Anche facendo la semplice media su tutte le finestre a ogni λ rimane un solo punto per λ, quindi ancora una curva.</p>
</section>

<section id="curva-3d"><h2>05 · La curva 3D dello Sharpe OOS aggregato</h2>
<p>Il grafico 3D usa ora <strong>esattamente lo stesso Sharpe OOS concatenato delle curve 2D</strong>.
A ogni λ corrisponde una coppia: la media dei C/T delle finestre e lo Sharpe calcolato sui 564 rendimenti mensili OOS.
La media riguarda soltanto la complexity; non gli Sharpe annuali.</p>
<div class="formula">λ → (Average C/T(λ), Sharpe OOS concatenato(λ)).</div>
<p>Un solo parametro produce una curva nello spazio, non una superficie.
I punti sono quelli della griglia salvata, uniti per facilitarne la lettura: nessuno smoothing,
nessuna interpolazione su una seconda dimensione e nessuna selezione locale di finestre.</p>
{figure('../gaussian/all/pooled_oos_relative_complexity_3d.png', 'Figura 3. Gaussian: curva 3D dello Sharpe OOS concatenato. Assi: λ in scala logaritmica, Average C/T e Sharpe OOS. La stella identifica lo stesso massimo ex post della figura 2D.')}
<p class="button-row"><a class="button" href="../gaussian/all/pooled_oos_relative_complexity_3d.html">Apri e ruota la curva gaussiana ↗</a></p>
<div class="callout"><strong>Per il gaussiano: {gaussian.peak_sharpe:.2f}, non 8.</strong>
Il massimo della curva è {gaussian.peak_sharpe:.2f} a Average C/T = {gaussian.peak_q:.3f},
con λ scelto ex post sull'intero test e rendimenti senza cap.
Lo Sharpe della strategia salvata, con λ scelto sulla validation e nessun cap lordo,
è invece {gaussian.selected_sharpe:.2f}. Nessuno dei due numeri è stato modificato dalla nuova visualizzazione.</div>
<h3>Perché la precedente superficie mostrava circa 8?</h3>
<p>Mostrava una media locale degli Sharpe stimati su finestre di soli dodici mesi.
Una media di rapporti non equivale al rapporto fra media e deviazione standard della serie concatenata.
Inoltre i pesi locali privilegiavano alcune finestre; cercare il massimo della superficie aggiungeva una selezione ex post.
Quel valore non rappresentava quindi la performance dell'intero backtest.</p>
<p>Le superfici precedenti e le relative diagnostiche sono conservate soltanto nelle sottocartelle
<code>archive_mean_sharpe</code>, escluse dalla galleria e non più generate dallo script.
Le curve della singola finestra 2024 rimangono disponibili, chiaramente separate dalla curva OOS aggregata.</p>
</section>

<section><h2>06 · Portafogli implementati e spettro</h2>
<p>Le curve di complexity usano rendimenti senza cap per descrivere il modello ridge.
Le equity già salvate usano invece il λ scelto sulla validation e il vincolo mensile Σ|w| ≤ 2,
ottenuto ridimensionando i pesi quando necessario. Un ridimensionamento variabile nel tempo può modificare lo Sharpe;
un fattore costante positivo non lo modifica. Non è quindi corretto aspettarsi che la classifica delle equity coincida
con quella dei massimi delle curve OOS non vincolate.</p>
{implemented}
<p class="small">¹ Crescita e drawdown sono calcolati meccanicamente su ∏(1 + portfolio_return), come equity dei rendimenti salvati.
Il campo di partenza è un rendimento excess lead: senza una riconciliazione del cash account e del finanziamento,
queste misure non vanno automaticamente lette come crescita della ricchezza totale netta investibile.</p>
<p>Matérn 1/2 ha lo Sharpe più alto tra le serie selezionate con cap, mentre il gaussiano ha il massimo più alto
tra le curve raw aggregate. Sono confronti diversi, non risultati contraddittori.
Le lengthscale sono quelle della run esistente: mediana calcolata sul primo train, con scelta fra i moltiplicatori
sulla prima validation e poi valore fissato. Non sono state riottimizzate in questa analisi.</p>
<p>Il b riportato è la pendenza negativa del fit log-log dello spettro empirico dell'ultimo refit,
dopo l'esclusione degli autovalori sotto 10<sup>−10</sup> del maggiore.
È una descrizione dello spettro finito, non una stima automaticamente consistente dell'esponente di popolazione del paper.
Normalizzare μ per μ<sub>1</sub> cambia la scala verticale del fit, non la pendenza.
Gli istogrammi esistenti mantengono l'asse x lineare e mostrano il 90% inferiore degli autovalori normalizzati,
con la coda superiore dichiarata fuori vista e il fit nel pannello destro; non sono stati alterati per introdurre C/T.</p>
</section>

<section id="limiti"><h2>07 · Che cosa manca per una conclusione da paper</h2>
<p><strong>Preprocessing point-in-time.</strong> Nel loader attuale la selezione delle caratteristiche per quota di valori mancanti
è effettuata sull'intero periodo 1963–2024. L'universo delle caratteristiche può quindi incorporare informazioni future sulla disponibilità.
Il fatto che la regressione usi finestre expanding non elimina questa forma di look-ahead nella preparazione dei dati.
Non ho cambiato il filtro o rifatto le run: per verificare performance rigorosamente OOS occorre un esperimento separato
con universo fissato ex ante o filtro calcolato solo sul campione di training.</p>
<p><strong>Ricerca ripetuta sul test.</strong> Kernel, griglie, grafici e massimi sono stati esplorati più volte sugli stessi periodi.
Il punto OOS ottimale è un oggetto descrittivo ex post. Non va inserito nella selezione operativa dei pesi,
né presentato come risultato di una scelta effettuata senza conoscere il test.</p>
<p><strong>Costi e robustezza.</strong> Questi risultati non verificano costi di transazione, turnover, liquidità, prestito titoli
o finanziamento. Gli Sharpe elevati richiedono ulteriori audit prima di una lettura economica forte.
Le 1.000 random feature restano una sola approssimazione salvata per kernel: non è stata misurata la sensibilità a seed e numero di feature.</p>
<p><strong>Stessa λ non significa stessa C/T.</strong> Ogni punto della curva media tiene λ uguale tra le finestre
e media le rispettive complexity normalizzate. Non è il backtest di una strategia che mantenga C/T costante:
quest'ultima richiederebbe una regola per scegliere λ in ciascun refit usando solo lo spettro allora disponibile.</p>
<p>In sintesi, i risultati sono coerenti con un costo dell'eccesso di complexity, ma non dimostrano da soli
la legge asintotica del paper, la superiorità statistica di un kernel o una performance netta implementabile.</p>
</section>

<section id="galleria"><h2>08 · Galleria completa · tutti i kernel</h2>
<p>Apri una scheda per vedere tutti i grafici. Ogni immagine è cliccabile a piena risoluzione.
Le curve OOS aggregate e le curve 2024 sono disponibili anche in HTML interattivo, con JavaScript incluso localmente.</p>
{gallery}
</section>

<section><h2>09 · Riproducibilità e controlli</h2>
<p>Per rigenerare analisi, nuovi grafici C/T e questo report dai risultati salvati:</p>
<pre>MPLCONFIGDIR=/private/tmp/paper_matplotlib /usr/local/bin/python3 complexity_analysis.py</pre>
<p>Per aggiornare soltanto il report:</p>
<pre>/usr/local/bin/python3 complexity_report.py</pre>
<p>I controlli automatici verificano: uguaglianza dello Sharpe prima e dopo la normalizzazione;
0 ≤ C/T ≤ 1; numero di autovalori uguale a min(T,P); esposizione lorda massima non superiore a 2;
completezza dei blocchi test di 12 mesi.
Nessun modello è stato riallenato, nessun peso è stato modificato e i grafici originali in C sono conservati.</p>
<p>Dati riassuntivi: <a href="kernel_comparison.parquet">confronto kernel</a>,
<a href="window_optima.parquet">massimi per finestra</a>.
In ogni cartella del kernel sono salvati anche diagnostiche normalizzate, intervalli ed estrazioni bootstrap.</p>
<p class="small">Fonti: risultati locali results/&lt;kernel&gt;/all, script train_model.py, complexity.py,
complexity_analysis.py, download_JKP/read_dataset.py e Portfolio_Paper.pdf pp. 14–15.
Report descrittivo delle run disponibili, non certificazione indipendente dei dati.</p>
</section>
<footer>PORTFOLIO KERNELS · C/T ANALYSIS · PNG + HTML, NESSUN PDF AGGIUNTO</footer>
"""

    style = """
    :root { color-scheme: light; --ink:#192d40; --muted:#536578; --accent:#147d83; }
    * { box-sizing:border-box; } html { scroll-behavior:smooth; }
    body { margin:0; background:#f5f6f7; color:var(--ink); font:17px/1.75 Georgia,serif; }
    main { max-width:1120px; margin:36px auto; padding:58px 68px; background:#fff; }
    h1 { font-size:44px; line-height:1.15; font-weight:500; letter-spacing:-1px; margin:18px 0; }
    h2 { font-size:29px; line-height:1.25; font-weight:500; margin:0 0 24px; }
    h3 { font-size:20px; margin-top:30px; } p { margin:0 0 18px; }
    .eyebrow,nav,.facts,table,figcaption,.small,footer,summary,.button { font-family:Arial,sans-serif; }
    .eyebrow { font-size:11px; letter-spacing:2px; color:var(--accent); }
    .lead { font-size:21px; color:var(--muted); max-width:840px; }
    .facts { display:flex; gap:36px; padding:22px 0; font-size:13px; color:var(--muted); }
    .facts b { display:block; font-size:30px; color:var(--ink); font-weight:500; }
    nav { display:flex; flex-wrap:wrap; gap:10px 23px; border-block:1px solid #dde5e9; padding:18px 0; font-size:13px; }
    a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
    section { padding:48px 0 10px; border-bottom:1px solid #e2e7eb; scroll-margin-top:20px; }
    .callout { background:#edf5f5; border-left:3px solid var(--accent); padding:22px 26px; margin:24px 0; }
    .formula { text-align:center; background:#f7f8fa; padding:22px; margin:22px 0; font-size:20px; }
    figure { margin:30px 0; } img { max-width:100%; height:auto; display:block; margin:auto; }
    figcaption { font-size:12px; line-height:1.6; color:var(--muted); margin-top:12px; }
    .table-wrap { overflow-x:auto; margin:26px 0; } table { border-collapse:collapse; width:100%; font-size:12px; line-height:1.5; }
    th { text-align:left; background:#edf2f5; font-weight:600; } th,td { padding:12px 9px; border-bottom:1px solid #dce3e8; }
    td { font-variant-numeric:tabular-nums; } td:first-child { white-space:nowrap; }
    .small { font-size:12px; color:var(--muted); line-height:1.7; }
    .button { background:var(--ink); color:white; padding:13px 22px; display:inline-block; font-size:14px; border-radius:3px; }
    details { margin:18px 0; border:1px solid #dce3e8; padding:20px; }
    summary { cursor:pointer; font-size:17px; font-weight:600; } details>p { margin:18px 0; }
    .gallery { display:grid; grid-template-columns:1fr 1fr; gap:22px; align-items:start; }
    .gallery figure { margin:12px 0; } pre { overflow:auto; padding:18px; background:#f4f6f8; font-size:12px; }
    footer { padding-top:30px; font-size:10px; letter-spacing:1px; color:var(--muted); }
    @media(max-width:760px) { main { margin:0; padding:28px 20px; } h1 { font-size:32px; } h2 {font-size:25px;} .gallery {grid-template-columns:1fr;} .facts {gap:18px;} }
    @media print { body {background:white;} main {margin:0;padding:0;} nav,.button-row {display:none;} figure,table {break-inside:avoid;} }
    """
    document = f'<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Portfolio kernels — Complexity C/T</title><style>{style}</style></head><body><main>{content}</main></body></html>'
    OUTPUT.mkdir(exist_ok=True, parents=True)
    destination = OUTPUT / "report.html"
    destination.write_text(document, encoding="utf-8")
    print("Report:", destination)


if __name__ == "__main__":
    build_report()
