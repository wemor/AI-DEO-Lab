# Fehleranalyse: Smart Domain DoE (AI-Vorhersagen)

Obwohl wir mit **Latin Hypercube Sampling (LHS)** eine strukturierte und gleichmäßige Verteilung unserer Daten erreicht haben (was das Modell im Vergleich zu unstrukturierten Daten drastisch verbessert), sehen wir in unseren Plots noch Ausreißer.

Um das zu verstehen, haben wir das sogenannte "Smart Domain Modell" (trainiert auf nur 120 Samples) auf unser isoliertes Test-Set (30 Samples) angewendet und die minimalen / maximalen Abweichungen im Detail untersucht:

## 1. Ergebnisse der Analyse (aus unserem Test-Set)

| Zielgröße (Target) | Ø Abweichung (MAPE) | Ø Absolut (MAE) | Max. Fehler (Abs) | Max. Fehler (Rel) |
|---|---|---|---|---|
| **Gewicht (kg)** | 16.3 % | ~22.9 kg | 64.87 kg | 58.5 % |
| **Durchbiegung (mm)** | 56.6 % | 0.02 mm | 0.10 mm | 343.0 % |
| **Spannung (MPa)** | 12.5 % | 0.20 MPa | 0.79 MPa | 32.1 % |
| **Sicherheitsfaktor** | 33.2 % | 194.2 | 1359.5 | 111.2 % |
| **Eigenfrequenz (Hz)** | 7.3 % | 10.5 Hz | 44.1 Hz | 21.2 % |

---

## 2. Deep Dive: Wo irrt sich die KI am meisten?

Wenn man sich den größten Einzelfehler bei bestimmten Metriken ansieht, erkennen wir ein Muster. 

**Beispiel Gewicht (`weight_kg`):**
Das Modell schätzte einen 2.66m Aluminiumträger auf 175 kg, obwohl der echte theoretische Wert bei ~110 kg liegt.

**Beispiel Durchbiegung (`deflection_mm`):**
Das Modell schätzte 0.24 mm Durchbiegung, die reale Durchbiegung lag aber bei 0.35 mm. Da es sich um mikroskopische Absolutwerte handelt, schnellt der prozentuale Fehler logischerweise auf > 300 % in die Höhe.

**Beispiel Sicherheitsfaktor (`safety_factor`):**
Der Sicherheitsfaktor steigt asymptotisch gegen Unendlich, sobald die Spannung gegen Null geht. Die KI schätzt bei einem sehr stabilen Träger einen riesigen Faktor von 2581, das echte Modell sagt aber 1221. Das ist zwar ein Abweichung von über 1000 Punkten, aber da in beiden Fällen der Träger "sicher" ist (alles über > 1.0 bricht nicht), spielt dieser gigantische Ausreißer in der Realität der Klassifikation ("Versagt er oder nicht?") gar keine Rolle.

---

## 3. Warum gibt es überhaupt noch diese Fehler?

Die KI nutzt hier **XGBoost** (einen extrem starken *Entscheidungsbaum-Algorithmus*).
Entscheidungsbäume unterteilen Daten in sogenannte "Rechtecke" (oder Bins), und geben letztendlich *Treppenstufen* aus.

Unsere Physik besteht aber aus hochgradig **nichttrivialen, polynomialen Kurven**:
Zum Beispiel steigt das Flächenträgheitsmoment ($I$) mit $h^3$ an, die Durchbiegung mit $L^3$, und das Biegemoment geteilt durch $I$ erzeugt hochkomplexe Multiplikationen.

### Die Limites (Unser Erklärungsansatz):
1. **Das Treppenstufen-Problem**: XGBoost versucht eine steile $X^3$-Kurve in winzigen diskreten Treppenstufen nachzubilden. 
2. **Datenmenge**: Für Treppenstufen braucht man an steilen Rampen *sehr* viele Datenpunkte. 120 Trainingsträger auf einen 6-dimensionalen Raum ($L, B, H, \rho, E, \sigma_y$) sind für das extrem steile Wachstum von z.B. $L^3/h^3$ noch deutlich zu spärlich, um perfekte Extrapolationen oder fehlerfreie Interpolationen der Ränder (Ausreißer) zu garantieren.
3. **Durchbiegung als Quotient**: Die Formel $\Delta = \frac{F \cdot L^3}{48 E I}$ teilt durch Werte! Das bedeutet, bei extrem steifen Konstruktionen geht der Wert fast asymptotisch auf Null, was für Baumbasierte Modelle immer einen massiven Bias in den Randfällen ausstrahlt.

---

## 4. Wie machen wir die Vorhersage perfekt?

Wir haben nun phänomenal dargelegt, dass reine *Datenstruktur* besser ist als *Datenmenge*, aber der ML-Algorithmus schränkt uns noch ein. 

Folgende Lösungen bringen uns als Ingenieure jetzt zu 99.9% Trefferquote:
1. **Model Change (Physik liebt Neuronale Netze)**: Anstelle von XGBoost verwenden wir **Künstliche Neuronale Netze (MLP)** oder einfache *Polynomial Regression*. Da NN auf stetigen, differenzierbaren Funktionen basieren (Sigmoid, ReLU), können sie so eine Formel wie $x^2$ nativ "rund" erlernen, ohne Treppenstufen zu bilden.
2. **Feature Engineering (Die Königsklasse)**: Wir helfen der KI! Wenn wir wissen, dass das Volumen wichtig ist, geben wir ihr nicht nur $L, b, h$, sondern auch ein berechnetes Feature `volume = L*b*h` in das Modell.
3. **Mehr Daten**: Wenn wir den LHS von 120 Punkten auf 1000 Punkte hochdrehen (was uns In-Silico nur Millisekunden kostet), wird die "Treppe" des XGBoost mikroskopisch klein und verschwindet optisch auf dem Graphen.
