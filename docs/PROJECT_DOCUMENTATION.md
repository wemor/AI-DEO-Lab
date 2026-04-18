# Vollständige Projektdokumentation: AI-DOE-Lab

## 1. Einleitung & Projektziel
In der modernen Produktentwicklung und dem Maschinenbau gewinnt Maschinelles Lernen (ML) zunehmend an Bedeutung. Eine große Herausforderung beim Einsatz von Data Science im Engineering ist jedoch die Qualität und Struktur der Daten. "Standard"-Datensätze in Unternehmen sind oft von historischem Bias und willkürlichen Versuchsparametern geprägt (sogenanntes "Bad DoE"). 

Dieses Projekt demonstriert rein analytisch in einer "In-Silico"-Umgebung (Einfeldträger-Simulation unter Punktlast), warum strukturierte Versuchsplanung (*Design of Experiments*, kurz DoE) wichtiger ist als pure Datenmenge, und wie die Wahl des richtigen ML-Algorithmus physikalisches Wissen replizieren kann.

---

## 2. Der Ground-Truth Simulator (Analytische Mechanik)
Anstatt echte, verrauschte Labortests durchzuführen, haben wir eine makellose mathematische Berechnung in Python implementiert (`src/physics_sim.py`). Diese "Ground Truth" erzeugt für beliebige Geometrien ($L, b, h$) und Materialien ($\rho, E, \sigma_y$) fehlerfreie Zielgrößen:

- **Biegemoment ($M_{max}$):** $M_{max} = \frac{F \cdot L}{4}$
- **Flächenträgheitsmoment ($I$):** $I = \frac{b \cdot h^3}{12}$
- **Gewicht ($m$):** $m = L \cdot b \cdot h \cdot \rho$
- **Maximale Randfaserspannung ($\sigma_{max}$):** $\sigma_{max} = \frac{M_{max} \cdot \left(\frac{h}{2}\right)}{I}$
- **Maximale Durchbiegung ($w_{max}$):** $w_{max} = \frac{F \cdot L^3}{48 \cdot E \cdot I}$
- **1. Eigenfrequenz ($f_1$):** $f_1 = \frac{\pi}{2 L^2} \sqrt{\frac{E \cdot I}{\rho \cdot b \cdot h}}$

Wir haben zudem die Zielgröße **Sicherheitsfaktor ($SF$)** als $SF = \frac{\sigma_y}{\sigma_{max}}$ definiert, was sich später als didaktisch höchst wertvolle Falle für Machine Learning Algorithmen erwiesen hat.

---

## 3. Data Engineering: Die Macht des DoE

### Phase A: Der "Bad DoE" Ansatz
Um ein historisch gewachsenes, unstrukturiertes Datenarchiv (z.B. aus 10 Jahren Labor-Experimenten ohne Plan) zu simulieren, haben wir einen Zufallsgenerator gebaut, der extreme Cluster bildet. Konstrukteure bauten in der Vergangenheit entweder "ganz kleine", "mittlere" oder "riesige" Träger. Der Parameterraum zwischen diesen Clustern wurde nie erforscht. 
**Ergebnis für die KI:** Die Vorhersage von Target-Variablen für mittlere bis neue Parameter schlägt massiv fehl, das Modell rät blind. R^2 Werte stürzten vielfach ins Negative.

### Phase B: Smart Domain DoE (Latin Hypercube Sampling)
Ein moderner Versuchsplan nutzt mathematische Algorithmen (z.B. *LHS*), um den $N$-dimensionalen Design-Raum mit möglichst wenigen Punkten perfekt und gleichmäßig abzudecken. Wir haben mit LHS einen exklusiven Mini-Datensatz von nur **120 Trägern** generiert.

---

## 4. Algorithmen-Vergleich: Warum Bäume keine Runden Kurven mögen

Wir leiteten den LHS-Datensatz durch zwei völlig unterschiedliche KI-Algorithmen und untersuchten die Performance auf einem harten, ungesehenen Test-Set.

### 4.1 Der Entscheidungsbaum (XGBoost)
XGBoost ist eine der stärksten tabellarischen KIs der Welt. Er zerschneidet Datenräume mithilfe stufenartiger Bins. 
Unsere Formeln (wie $L^3$ oder $E \cdot h^3$) sind jedoch hochgradig stetig und extrem nicht-linear. 
* **Ergebnis:** XGBoost verbesserte sich durch das Smart DoE drastisch, zeigte aber trotzdem grobe Schnitzer an den Rändern. Er versucht eine sehr steile Parabel $x^3$ mit wenigen harten "Treppenstufen" aufzubauen, wodurch er zwischen den 120 Datenpunkten teilweise abrutschte (z. B. 16.3 % MAPE beim Gewicht, > 50 % bei der Durchbiegung).

### 4.2 Die physikalische Synthese (Neuronales Netz / MLP)
Wir ersetzten den XGBoost durch ein Multi-Layer Perceptron (MLP) mit 2 Hidden-Layern à 64 Neuronen, trainiert via `lbfgs`.
* **Ergebnis:** Da Neuronale Netze kontinuierliche Aktivierungsfunktionen nutzen (die nicht-lineare "Kurven" direkt abbilden können), interpolierten sie die physikalischen Formeln gigantisch gut. 
* **Performance-Sprung:** Der Fehler der Spannung ($\sigma_{max}$) fiel auf unglaubliche 1.59 %, der R^2-Score kletterte durch die Bank auf > 0.99. 

---

## 5. Das Rätsel des Safety Factors (Asymptoten-Fehler)

Bei der Auswertung des besten Modells stießen wir auf ein Paradoxon:
Die KI konnte die auftretende Spannung (z. B. $0.10$ MPa) der Träger mit astronomischer Präzision (z.B. Fehler von nur $0.05$ MPa) vorhersagen. 
Dennoch versagte die KI grandios (Fehler von teilweise > 1000 Punkten) bei der Vorhersage des exakt davon abhängigen Sicherheitsfaktors ($SF = \sigma_y / \sigma_{max}$). 

### Die Interpretation: 
Das Modell leidet an der Division durch asymptotische Null-Werte ($1/x$). Bei extrem massiven Trägern ist die Spannung so klein, dass sie nahe der Null liegt. Ein winziger KI-Abstand von $0.05$ im Vorhersagefehler ist in absoluten Zahlen ein Traumwert. Dividiert man aber eine Streckgrenze von 250 MPa durch den winzigen absoluten Fehler, explodiert die Bruch-Differenz. 

Es zeigten sich hier zwei essenzielle Data-Science Lektionen für Ingenieure:
1. **Machine Learning sollte niemals isoliert $1/x$-Verhältnisse mit Null-Äquivalenz prognostizieren.**
2. **Post-Processing schlägt End-to-End ML:** Wenn das Modell Größe A ($\sigma_{max}$) verstanden hat, und B ($SF$) einfach nur eine konstante Division ist, lässt man das Modell ausschließlich A ausgeben und errechnet B analytisch in Millisekunden.

### Konsequenz für die finale Pipeline:
Die Zielgröße `safety_factor` wird mit sofortiger Wirkung aus der KI-Pipeline verbannt. Die Modelle prognostizieren nur noch Gewicht, Durchbiegung, Spannung und Eigenfrequenz. Der Sicherheitsfaktor wird danach händisch klassisch nachgerechnet.

---

---
 
 ## 7. Das Live Learning Lab (Die interaktive App)
 
 Die finale Komponente des Projekts ist die App `app.py`. Sie überführt die wissenschaftlichen Erkenntnisse in ein interaktives Experimentierfeld.
 
 ### 7.1 Die Vergleichs-Matrix
 Der Ingenieur kann in der App aus vier Basiskombinationen wählen, um sie gegen die physikalische Wahrheit (Ground Truth) antreten zu lassen:
 
 | Kombination | DoE-Strategie | AI-Solver | Kern-Aussage |
 | :--- | :--- | :--- | :--- |
 | **1. Worst Case** | Bad Domain DoE | XGBoost | Zeigt das Scheitern bei schlechten Daten (Cluster-Bias). |
 | **2. Data Shift** | Smart Domain DoE | XGBoost | Zeigt den massiven Gewinn durch Struktur (DoE-Effekt). |
 | **3. Solver Test**| Bad Domain DoE | Neural Network | Guter Algorithmus kann fehlende Daten nicht heilen. |
 | **4. Gold Standard**| Smart Domain DoE | Neural Network | Maximale Präzision durch optimale Synergie. |
 
 ### 7.2 Interaktives Training
 Über die Sidebar kann die Anzahl der Trainingspunkte für beide DoE-Strategien live angepasst werden. Per Knopfdruck werden alle vier KI-Modelle simultan neu trainiert, was den direkten Vergleich der Lernkurven und Fehlerbalken ermöglicht.
 
 ### 7.3 Generative Design (Inverse Optimization)
 Die App nutzt das "Gold Standard" Modell als Ersatzmodell für einen Optimierungs-Algorithmus, um basierend auf Nutzer-Nebenbedingungen (z.B. maximale Durchbiegung) automatisch die optimale Geometrie zu finden.
 
 ---
 
 **Modell-Status:** Die App ist lokal lauffähig und bietet volle Transparenz über die Vorhersagequalität der verschiedenen KI-Strategien.
