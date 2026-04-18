# Phase 1: Datensimulator (Die "Physikalische Wahrheit")

## Zielsetzung
Das KI-Modell benötigt zum Lernen eine "Ground Truth", also physikalisch korrekte Zieldaten (Targets) für jegliche Eingangsgeometrien (Features). In einem echten Konstruktionsbüro stammen diese aus dem Labor oder teuren FEM-Analysen. Um unbegrenzt und fehlerfrei Daten generieren zu können, haben wir die klassischen Gleichungen der Mechanik für einen einfachen Biegeträger in Python übersetzt. Diese bilden unsere virtuelle "Prüfmaschine".

## Umsetzung
Das Skript `src/physics_sim.py` nimmt unformatierte Parameter entgegen und wandelt diese nach SI-Einheiten um. Anschließend wird das physikalische Modell eines zweiseitig gelagerten Trägers (Einfeldträger) unter einer Punktlast in der Mitte (hier pauschal auf $F = 1000 \text{ N}$ festgelegt) errechnet.

### 1. Die Eingangsparameter (Inputs)
- **Länge** (`length_mm`)
- **Breite** (`width_mm`)
- **Höhe** (`height_mm`)
- **Materialdichte** (`density_kg_m3`)
- **Elastizitätsmodul** (`youngs_modulus_gpa`)
- **Streckgrenze / Festigkeit** (`yield_strength_mpa`)

### 2. Der Berechnungs-Algorithmus
Das Skript berechnet die mechanischen Reaktionen auf die Last:
- **Flächenträgheitsmoment ($I$)**: $I = (b \cdot h^3) / 12$
- **Maximales Biegemoment ($M$)**: $M = (F \cdot L) / 4$
- **Durchbiegung ($w$)**: $w = (F \cdot L^3) / (48 \cdot E \cdot I)$
- **Maximale Randfaserspannung ($\sigma$)**: $\sigma = (M \cdot h / 2) / I$
- **Bauteilgewicht ($m$)**: $m = L \cdot b \cdot h \cdot \rho$

### 3. Rückgabewerte (Targets für die KI)
Die Simulation gibt folgende "Prüfberichte" für den Träger zurück:
- `weight_kg` (Das Gewicht entscheidet über Werkstoffkosten und Transport)
- `deflection_mm` (Die Steifigkeit des Konstrukts)
- `max_stress_mpa` (Die auftretende Betriebsspannung)
- `safety_factor` (Wie sicher ist der Balken vor Plastifizierung? Streckgrenze / maximale Spannung)
- `failure` (Ging der Träger kaputt? `True` wenn `safety_factor < 1.0`, ansonsten `False`)

## Test und Validierung
Der Simulator wurde mit einem herkömmlichen 50x50 mm Stahlrohr (Vollmaterial) auf 1 Meter Länge validiert.
Die Simulation ergibt bei 100 kg ($1000 \text{ N}$) Last korrekterweise nur 0,19 mm Durchbiegung bei einem Sicherheitsfaktor gegen Bruch von 29. Das physikalische Grundgerüst funktioniert somit fehlerfrei.
