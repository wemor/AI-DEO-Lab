# Phase 2: Design of Experiments (DoE) & Daten-Generierung

## Zielsetzung
Bevor wir die KI große Datenmengen analysieren lassen, wollen wir gezielt aufzeigen, woran Maschinelles Lernen im Ingenieurswesen in der Regel (zu Beginn) scheitert: **Einem schlechten, unstrukturierten Datensatz**. 

Anstelle eines perfekten *Space Filling Designs* (z.B. LHS - Latin Hypercube Sampling), welches den gesamten Parameterräumen abdeckt, haben wir absichtlich einen Datensatz erzeugt, wie er oft historisch aus Excel-Listen von vergangenen Projektversuchen herausgekratzt wird.

## Umsetzung
Das Skript `src/doe_generator.py` fungiert als Brücke zwischen Zufallsparametern und dem Simulator aus Phase 1. Es generiert den rohen Datensatz `data/raw/unstructured_beam_data.csv`.

Dabei haben wir **drei eklatante Fehler** eingebaut, die der späteren KI das Leben extrem schwer machen werden:

### 1. Minimaler Umfang (Undersampling)
Das Skript erzeugt **nur 30 Datenpunkte**. Die KI versucht damit, physikalische 3D-Geometrie und komplexe Materialbrüche aus nur einem Bruchteil der statistischen Realität "auswendig" zu erraten. Modelle wie Random Forests oder neuronale Netze verfallen hier zwingend dem *Overfitting* (sie lernen die Trainingspunkte auswendig, versagen aber bei Vorhersagen völlig).

### 2. Geclusterte Parameter (Lücken im Wissen)
Erlaubt wurde von uns keine gleichmäßige Verteilung. 
Anstatt Längen von z.B. 100 mm bis 5000 mm stetig zu variieren, zieht der Code nur gezielt diskrete "Standard-Maße" (z.B. Testreihen mit 500 mm und Testreihen mit 2000 mm, aber absolut leere Varianzen dazwischen). Das bedeutet: Fragt man die KI später nach einem Träger mit 1200 mm Länge, bewegt sich die Maschine im puren Blindflug.

### 3. Rauschen & Uneindeutigkeit (Messungenauigkeiten)
Tatsächliches Stahl hat einen E-Modul von ca. 210 GPa. Im realen Labor schwankt die Auswertung jedoch je nach Messmaschine, Temperatur und Charge leicht.
Wir haben ganz bewusst *Gauß-Rauschen (Normalverteilung)* über die Eigenschaften von Aluminium und Stahl gelegt. Für das physikalisch reine Trainingsmodell bedeutet dies, dass zwei geometrisch absolut identische Stahlträger plötzlich um wenige Millimeter unterschiedlich stark bei gleicher Last durchbiegen können (= Messfehler/Laborunterschiede). Die KI darf diese Fehler nicht "mitlernen", ansonsten wird sie ungenau.

## Resultat
Das Output-Ergebnis liegt in `unstructured_beam_data.csv` und gleicht exakt der Herausforderung unseres vorherigen `TestMix` Ansatzes in der Chemie-Software. Nun liegt die Hürde vor "Phase 3", mit diesem mangelhaften Datensatz ein erstes Basis-Schätzmodell (Baseline) aufzubauen.
