# Phase 3: AI Pipeline und Limitierungen

## Umsetzung
In Phase 3 wurde die Datei `src/pipeline.py` implementiert. Das Skript lädt unseren defizitären, unsystematischen Datensatz (`unstructured_beam_data.csv`).
Das Modell verwendet **XGBoost Regressoren** gebettet in einen `MultiOutputRegressor`. Um bestmögliche Settings zu filtern und Overfitting zu bekämpfen, durchlaufen die Bäume eine `GridSearchCV` (Hyperparameter-Validierung) auch trotz der geringen Datenmenge von 30 Samples.

Für den Training-und-Test-Zyklus trennt das Skript 80% (24 Datensätze) zum reinen Trainieren ab und nutzt die restlichen ungesehenen 6 Datensätze zur Bewertung der KI (R²-Wert).

## Vorhersage-Ergebnisse (Metrics)

Die Testergebnisse bestätigen auf dramatische und beabsichtigte Weise unsere Erwartungen:
```json
{
  "weight_kg": {
    "MAE": 25.85,
    "R2": 0.1182
  },
  "deflection_mm": {
    "MAE": 6.52,
    "R2": -29.6913
  },
  "max_stress_mpa": {
    "MAE": 5.19,
    "R2": 0.9464
  },
  "safety_factor": {
    "MAE": 37.63,
    "R2": 0.6894
  }
}
```

### Interpretation des Scheiterns
- **Durchbiegung (Deflection R² = -29.69)**: Die physikalische Formel der Durchbiegung hat eine hoch potenziell nicht-lineare Beziehung zur Länge ($L^3$). Die nur 24 Trainigsdaten besaßen Cluster um extreme Längen (nur 500mm vs 2000mm, nichts dazwischen). Dies zwingt das Modell zum Raten. Ein R² von -29.69 bedeutet, das Modell ist gigantisch schlechter als hätte es einfach immer *stur den Mittelwert* aller Durchbiegungen geraten.
- **Max Stress (R² = 0.94)**: Offenbar gab es Cluster, aus denen das Modell zumindest einen flachen Zusammenhang zwischen Kraft, Trägheit und Spannung ziehen konnte, weshalb es hier erstaunlich gut abschneidet.
- **Weight (R² = 0.11)**: Der lineare physische Zusammenhang des Gewichtes konnte mit den Lücken kaum verstanden werden. 

## Fazit: Ohne ein echtes DoE ist KI blind
Das Projekt zeigt makellos unser Problemkind: Haben Datensätze keinen statistisch relevanten, gleichmäßigen Formraum (Space Filling Design / *LHS* Design of Experiments) und sind in ihrer Menge zu klein, so werden Vorhersagen unweigerlich fatal, unabhängig davon, ob wir modernste Architekturen wie XGBoost einsetzen. In der Realität hätte die Nutzung dieser KI in der Projektplanung fatale ingenieurstechnische Ausfälle (z.B. verbogene Balken, zerstörte Maschinen) zur Folge.

Um die KI erfolgreich einzusetzen, muss das Projekt zwingend mit Phase 4 ("Erstellung eines systematischen, mathematisch sauberen 10.000 Punkte DoE") fortgeführt werden.
