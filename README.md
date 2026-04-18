# AI-DOE-Lab: Virtual Engineering & Machine Learning

Dieses Projekt dient als **Blue Print (Blaupause) und Proof of Concept** für den Einsatz von Künstlicher Intelligenz (Speziell Regression) in der Produktentwicklung, Materialforschung und im Virtual Engineering.

Anhand eines theoretischen Einfeldträgers (Biegeträger unter Mittellast) simulieren wir die Brücke zwischen realem physikalischen Verhalten, unzureichenden Versuchsplänen (Bad DoE) und der algorithmischen Lösung durch "Smart Data" (Latin Hypercube Sampling).

Das Projekt ist methodisch in 4 sukzessive Phasen aufgebaut:

---

## 🏗️ Phase 1: Der Datensimulator (Ground Truth)
In der ersten Phase wurde ein rein physikalischer Simulator (`src/physics_sim.py`) programmiert. Anhand klassischer festigkeitslehrmechanischer Formeln errechnet dieser Simulator für beliebige Geometrie- (Länge, Breite, Höhe) und Materialaspekte (E-Modul, Streckgrenze, Dichte) hochpräzise Ausgabewerte:
- Gewicht des Trägers
- Durchbiegung (Deflection)
- Maximale Spannung (Max Stress)
- Sicherheitsfaktor gegen Bruch

**Kernziel:** Schaffung einer in-silico "Prüfmaschine", um beliebig viele kostenfreie Datensätze erzeugen zu können, die der absoluten mechanischen Wahrheit entsprechen.

## 📉 Phase 2: Design of Experiments - Mangelhafte Daten
In der Praxis liegen oft nur historisch gewachsene, lückenhafte Versuchsdaten vor ("Bad DoE"). Dies wurde in `src/doe_generator.py` simuliert:
- Es wurden bewusst nur sehr wenige Daten erzeugt (**N=30**).
- Die Geometrien wurden stur in Extremwerten geclustert (z.B. nur 500mm oder 2000mm Länge, ohne Zwischenwerte).
- Natürliches Labor-Rauschen (Ungenauigkeiten) wurde hinzugegeben.

**Kernziel:** Absichtliche Erzeugung eines qualitativ schlechten Datensatzes, an dem die KI zwangsläufig scheitern muss.

## 🤖 Phase 3: Die KI-Pipeline & das erwartete Scheitern
Diese stark lückenhaften 30 Datenpunkte wurden in eine moderne **XGBoost Machine Learning Pipeline** (`src/pipeline.py`) eingespeist. Die KI sollte versuchen, die mechanische Physik aus diesem schlechten Datensatz zu abstrahieren.

**Ergebnis:** Katastrophales Versagen.
Die Vorhersage der Durchbiegung ($L^3$) erreichte ein R² von **-29.69**. Die KI riet völlig blind und produzierte Fehlparameter, die im realen Ingenieurswesen zu einstürzenden Tragwerken geführt hätten.
**Fazit:** Selbst die modernsten KI-Algorithmen (XGBoost, Cross-Validation, Hyperparameter-GridSearch) sind nutzlos, wenn der Lösungsraum (Datenstruktur) blind und lückenhaft ist.

## 💡 Phase 4: Die Lösung durch Smart DoE
Statt nun zehntausende neue Dummy-Daten generieren zu lassen ("Big Data"), wurde das Projekt durch hochstrukturiertes Sammeln gelöst ("Smart Data").
Mittels **Latin Hypercube Sampling (LHS)** (`src/smart_doe_generator.py`) wurden lediglich **200 intelligente, extrem gleichmäßig verteilte Datenpunkte** erzeugt.

**Ergebnis:** Der direkte Siegeszug der KI.
Ohne eine einzige Code-Änderung am KI-Modell schoss das R² auf dem neuen LHS-Datensatz für dieselbe nicht-lineare Durchbiegung von -29.69 auf exzellente **+0.81** hinauf. 

## 📊 Phase 5: Visualisierung & Log-Log Skalierung
Im direkten Vergleich auf identischen "Hold-Out" Testträgern zeigte sich die AI-Überlegenheit visuell brillant. Da hochpotente Formeln ($w \propto L^3$) riesige numerische Streuungen und ein optisches "Verklumpen" im unteren Wertebereich verursachen, wurde methodisch die **doppellogarithmische Skalierung** genutzt. So wird die absolute Vorhersage-Perfektion des LHS-Modells über alle Zehnerpotenzen und Dezimalbereiche in Echtzeit analytisch sichtbar gemacht.

## 🛠️ Phase 6: Domain Specific AI (Die Ingenieurs-Realität)
In der echten Industrie wird niemals "alles Mögliche" konstruiert, sondern hochspezifisch in einem begrenzten `Design Space` gearbeitet (z.B. exakt 1m - 3m Länge und feste, diskrete Katalog-Materialien wie Stahl, Aluminium, Titan).
Als das DoE-Setup auf exakt so einen diskreten Bereich begrenzt wurde, verschwanden die exponentiellen Verzerrungen. Mit nur 150 stur systematisch generierten Punktdaten wurde die KI zum hochpräzisen, linear berechenbaren **Fachexperten** (R² > 0.91 für Spannung).

## 🚀 Phase 7: Skalierbarkeit (Erste Eigenfrequenz)
Als Beweis für die Skalierbarkeit (Return on Investment) dieser Struktur wurde im Nachgang die *erste Eigenfrequenz ($f_1$)* eingefordert. Da die Ground-Truth-Pipeline und LHS-Struktur bereits existierten, musste hierfür nur eine einzige Formel zum Simulator hinzugefügt werden. Die XGBoost-KI lernte die hochkomplexe Schwingungslehre völlig automatisch mit und schoss aus dem Stand auf ein **R² = 0.981**. Ein Nachweis dafür, dass ein einmal solide aufgespannter Design-Space nachträglich unendlich viele Zielvariablen adaptieren kann.

---

### 🎓 Zentrale Erkenntnis & Entmystifizierung der Black Box

> *In der Künstlichen Intelligenz schlägt eine saubere Datenqualität und geometrische Strukturierung immer schiere Masse.*
> Nur rund 160 (LHS) raumfüllend verteilte Trainingspunkte genügen, um ein ML-Modell zu erschaffen, das komplexe, mehrdimensionale Materialmechanik versteht und verlässliche Voraussagen trifft.
> 
> **Ingenieurswesen trifft Data Science:** 
> Die größte Einstiegshürde ist oft der Glaube, KI sei eine magische "Black Box". Dieses Projekt beweist radikal das Gegenteil: **KI kann nicht zaubern.** Sie erfindet keine Physik aus dem Nichts, sondern sie interpoliert ausschließlich die Datenräume, die man ihr aufspannt.
> Das wahre Erfolgsgeheimnis moderner ML-Modelle in der Entwicklung liegt nicht in immer komplexeren Algorithmen, sondern in der direkten **Domain Expertise des Ingenieurs**. Wenn ein klassischer Konstrukteur seinen physikalischen Gültigkeitsbereich ("Design Space") versteht und diesen sauber mittels Design of Experiments (LHS) abtastet, bricht er den Mythos der Black Box. Aus der vermeintlichen Magie wird dann ein exzellentes, logisches und hochpräzises Werkzeug für das Virtual Engineering.
