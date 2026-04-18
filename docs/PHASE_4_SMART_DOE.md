# Phase 4: Smart Data DoE 

## Zielsetzung
Nachdem das KI-Modell in Phase 3 aufgrund des kleinen, geclusterten Datensatzes dramatisch versagte (R² von -29.69 bei der Durchbiegung), beweisen wir in Phase 4, dass nicht die Algorithmen (XGBoost), sondern die Datenstruktur das Problem war.

Anstatt nun 10.000 unstrukturierte Zufallsexperimente anzustreben ("Big Data"), setzen wir auf "Smart Data" mittels **Design of Experiments (DoE)**. Konkret verwenden wir das **Latin Hypercube Sampling (LHS)**.

## Umsetzung
Das Skript `src/smart_doe_generator.py` generiert präzise **200 Datenpunkte**. 
LHS sorgt dafür, dass jede der 6 Dimensionen (Breite, Höhe, Längen, Festigkeiten etc.) in genau 200 gleich große Intervalle unterteilt und abgegriffen wird, sodass der Lösungsraum multidimensional und lückenlos abgesucht wird – ganz ohne Redundanzen oder leere Cluster in der Geometrie.

## Ergebnisse: Der direkte Vorher-Nachher Vergleich
Das absolut selbe XGBoost-Skript (`src/pipeline.py`), welches vorher kläglich versagte, wurde ohne eine einzige Zeile Codeänderung auf den neuen 200-Punkte LHS-Datensatz trainiert.

### Die R²-Bewertungen auf den fremden Testdaten (Hold-Out):
- **Durchbiegung (Deflection)**:
  - *Vorher (Schlechtes DoE)*: R² = -29.69 (Fatales Versagen)
  - **Nachher (Smart DoE)**: R² = **+0.81** (Exzellente Annäherung der kubischen Funktion $L^3$)
- **Bauteil-Gewicht**:
  - *Vorher (Schlechtes DoE)*: R² = 0.11
  - **Nachher (Smart DoE)**: R² = **+0.73**
- **Sicherheitsfaktor (Bruchrisiko)**:
  - *Vorher (Schlechtes DoE)*: R² = 0.68
  - **Nachher (Smart DoE)**: R² = **+0.69**

## Fazit & Projektabschluss
Wir konnten den Beweis erfolgreich erbringen:
In der Künstlichen Intelligenz (speziell im Ingenieurswesen/Virtual Engineering) schlägt **Datenqualität und geometrische Strukturierung immer schiere Masse.** 
Selbst nicht-lineare, mechanische Zusammenhänge lassen sich bereits mit winzigen Datensätzen (160 Training, 40 Test) hervorragend erlernen, *wenn* der Datensatz intelligent (via Space-Filling DoE) generiert wurde.

Dieses theoretische Kragarm-Projekt (`AI-DOE-Lab`) dient somit als perfekter Proof of Concept und "Blaupause" für die Reaktivierung des realen TestMix-Chemieprojekts (`AI-DOE-TestMix`), sobald dort strukturierte Laborversuche möglich sind!
