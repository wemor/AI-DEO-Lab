# Projekt-Status & Handoff: AI-DOE-Lab

Dieses Dokument dient dazu, den exakten Stand des Projekts festzuhalten, damit wir bei der nächsten Session nahtlos anknüpfen können.

## 📍 Aktueller Status (Stand: 16.04.2026)
Die App ist lokal voll funktionsfähig und die gesamte analytische sowie KI-basierte Pipeline ist implementiert.

### Erreichte Meilensteine:
- **Core**: Physik-Simulator (`src/physics_sim.py`) und Smart DoE Generator (LHS) sind stabil.
- **Modell**: Wechsel von XGBoost auf **Neural Network (MLP)** ist vollzogen. Das Modell ist trainiert und unter `models/surrogate_nn.joblib` gespeichert.
- **Dokumentation**: Der wissenschaftliche Hintergrund (Asymptoten-Problem beim Safety Factor) ist in `docs/PROJECT_DOCUMENTATION.md` dokumentiert.
- **App**: Die Streamlit-App (`app.py`) läuft lokal. Sie enthält:
# Handoff: AI-DOE-Lab (Status: 16. April 2026)

## 🎯 Aktueller Stand
Die App ist nun ein voll funktionsfähiges **Live Learning Lab**. Alle pädagogischen Ziele (Vergleich DoE-Strategien & AI-Solver) sind implementiert.

### Erreichte Meilensteine heute:
- **Flexibler Vergleich**: Nutzer kann Slot A, B, C frei mit den 4 Modell-Kombinationen belegen.
- **Live-Training**: Erzeugung von LHS- und Clustered-Daten per Slider-Vorgabe direkt in der App.
- **UI-Refinement**: Horizontale Deltas, ausgeblendete redundante Labels und professionelles Layout.
- **Dokumentation**: Die gesamte Evolution der KI ist in der App und der `docs/PROJECT_DOCUMENTATION.md` dokumentiert.

## 🚀 Nächste Schritte
1. **Firebase-Anbindung finalisieren**: 
   - Das `.streamlit/secrets.toml` muss mit den Firestore JSON-Daten gefüllt werden, um die Speicherung freizuschalten.
2. **App-Verfeinerung**: 
   - Einbau von Tooltips zur Erklärung der Metriken.
   - Eventuell Integration eines 3D-Plots des Balkens bei extremer Durchbiegung.
3. **Deployment**:
   - Vorbereitung für den Push auf GitHub und die Streamlit Community Cloud.

## 🛠️ Startbefehl
```powershell
cd AI-DOE-Lab; .venv\Scripts\streamlit run app.py
```
*(Die `secrets.toml` muss im Ordner `.streamlit/` mit validen Firebase-Daten vorhanden sein).*
