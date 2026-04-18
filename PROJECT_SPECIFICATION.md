# AI-DOE: Biegeträger (Virtual Engineering)

## 1. Projektziel
Das Ziel dieses Projekts ist es, die Funktionsweise und den Mehrwert von maschinellem Lernen (ML) sowie *Design of Experiments (DoE)* an einem leicht verständlichen, theoretischen Use-Case zu demonstrieren. 

Anstelle von chemischen Versuchen mit Rauschen und begrenzter Datenmenge nutzen wir hier die analytische Mechanik eines **Biegeträgers** unter konstanter Einzelpunktlast. Die physikalischen Formeln ersetzen das Labor ("In-Silico Environment") und generieren fehlerfreie Daten. Darauf aufbauend lernt ein KI-Modell (z.B. XGBoost), die physikalischen Zusammenhänge eigenständig abzubilden.

## 2. Der Anwendungsfall: Der Einfeldträger
Wir betrachten einen statisch bestimmten Biegeträger (rechteckiger Querschnitt) der Länge $L$, der an beiden Enden gelagert ist und in der Mitte mit einer konstanten Kraft $F$ belastet wird.

### 2.1 Inputs (Features für das KI-Modell)
Die KI bekommt für jeden generierten (künstlichen) Träger die folgenden Variablen übergeben:
- **Geometrie**:
  - `length_mm` ($L$)
  - `width_mm` ($b$)
  - `height_mm` ($h$)
- **Materialeigenschaften**:
  - `density_kg_m3` ($\rho$)
  - `youngs_modulus_gpa` ($E$)
  - `yield_strength_mpa` ($\sigma_y$ - die Streckgrenze / theoretische Maximalfestigkeit)

### 2.2 Outputs (Targets / "Laborergebnisse")
Unsere Simulation errechnet anhand von physikalischen Formeln die Zielgrößen:
1. `weight_kg`: Gesamtgewicht des Trägers.
2. `deflection_mm`: Maximale Durchbiegung in der Mitte des Trägers.
3. `max_stress_mpa`: Die maximale auftretende Randfaserspannung.
4. `safety_factor`: Quotient aus Streckgrenze ($\sigma_y$) und maximaler Spannung.
5. `failure` (Ziel der Klassifikation): Ein binärer Wert (True/False). Der Träger bricht oder verbiegt sich plastisch, wenn `safety_factor` < 1.0.

## 3. Physikalische Formeln (Die "Simulation")
Die Datenbasis wird mit folgenden klassischen mechanischen Grundlagen erzeugt:

- **Flächenträgheitsmoment ($I$)**:  
  $I = \frac{b \cdot h^3}{12}$
- **Biegemoment ($M_{max}$)** bei einer mittigen Punktlast $F$:  
  $M_{max} = \frac{F \cdot L}{4}$
- **Maximale Spannung ($\sigma_{max}$)**:  
  $\sigma_{max} = \frac{M_{max} \cdot \left(\frac{h}{2}\right)}{I}$
- **Durchbiegung ($w_{max}$)** bei mittiger Kraft:  
  $w_{max} = \frac{F \cdot L^3}{48 \cdot E \cdot I}$
- **Trägergewicht ($m$)**:  
  $m = L \cdot b \cdot h \cdot \rho$

*Anmerkung: Alle Einheiten müssen im Skript für die Formeln korrekt umgerechnet werden (z. B. mm in m für das Volumen).*

## 4. Geplanter Projektablauf
Das Projekt gliedert sich in folgende meilensteinbasierte Phasen:

1. **Datensimulator (Ground Truth)**
   - Programmierung eines kleinen Python-Moduls, das für beliebige Inputs (Geometrie & Material) die Outputs berechnet.
   - Festlegen sinnvoller physikalischer Grenzwerte (z.B. Stahl, Aluminium, Holz) als Vorlage.
2. **DoE (Design of Experiments) Datengenerator**
   - Aufsetzen eines Skripts, das automatisiert Tausende Konstruktionsträger zufällig oder per *Latin Hypercube Sampling* zusammenwürfelt und simuliert. (Unser virtuelles "Experiment-Labor").
3. **Training des KI-Modells**
   - Ein fiktiver Ingenieur hat nur ein begrenztes Testbudget (z.B. nur 100 Labortests). Wir entnehmen 100 Samples aus den Tausenden von Daten und trainieren unsere KI (*Machine Learning Pipeline*).
4. **Validierung und In-Silico Screening**
   - Die KI muss nun die restlichen 10.000 generierten Träger in Sekunden bewerten und raten, ob sie brechen (`failure`) oder nicht. Wir überprüfen die Treffergenauigkeit der KI mit der echten Physik-Simulation.
