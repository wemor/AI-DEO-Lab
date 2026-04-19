import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import firebase_admin
from firebase_admin import credentials, firestore
from scipy.optimize import minimize
from pathlib import Path
from scipy.stats import qmc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Local imports
from src.physics_sim import simulate_beam
from src.visualization import plot_beam_3d, plot_doe_distribution_plotly, plot_accuracy_comparison_plotly

# --- 1. Page Config & Theme ---
st.set_page_config(page_title="AI-DOE Lab", page_icon="🏗️", layout="wide")

# Initialize Firebase
@st.cache_resource
def get_firestore_client():
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
        except Exception as e:
            st.warning(f"Firebase not initialized. Ensure st.secrets['firebase'] is set. Error: {e}")
            return None
    try:
        return firestore.client()
    except Exception:
        return None

db = get_firestore_client()

# --- 2. CSS Styling ---
st.markdown(
    """
    <style>
    .footer-msg {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #161b22;
        color: #8b949e;
        text-align: left;
        padding: 8px 20px;
        font-size: 14px;
        border-top: 1px solid #30363d;
        z-index: 1000;
    }
    /* Container für Wert und Delta (das Zwischen-Div in Streamlit) */
    [data-testid="stMetric"] > div {
        display: flex !important;
        flex-direction: row !important;
        align-items: baseline !important;
        gap: 0.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #8b949e !important;
        margin-bottom: -5px !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        width: auto !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        margin-top: 0 !important;
    }
    
    /* Dedicated styling for the first column (Labels) */
    [data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    [data-testid="column"]:nth-of-type(1) [data-testid="stMetricLabel"] {
        padding-bottom: 12px !important;
    }

    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 8px;
        border-radius: 8px;
        height: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. AI Training Logic (4 Variants) ---

FEATURES = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
TARGETS = ["weight_kg", "deflection_mm", "max_stress_mpa", "eigenfrequency_hz"]

def generate_live_data(n_samples, strategy="smart", seed=None):
    materials = [
        {"density_kg_m3": 7850.0, "youngs_modulus_gpa": 210.0, "yield_strength_mpa": 350.0}, # Steel
        {"density_kg_m3": 2700.0, "youngs_modulus_gpa": 69.0,  "yield_strength_mpa": 250.0}, # Aluminum
        {"density_kg_m3": 4500.0, "youngs_modulus_gpa": 110.0, "yield_strength_mpa": 880.0}  # Titanium
    ]

    rng = np.random.default_rng(seed)

    if strategy == "smart":
        l_bounds, w_bounds, h_bounds = [500, 3000], [20, 300], [20, 500]
        bounds = np.array([l_bounds, w_bounds, h_bounds, [0.0, 3.0]])
        sampler = qmc.LatinHypercube(d=4, seed=seed)
        sample = sampler.random(n=n_samples)
        scaled = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        rows = []
        for s in scaled:
            mat_idx = min(2, int(np.floor(s[3])))
            mat = materials[mat_idx]
            rows.append({"length_mm": s[0], "width_mm": s[1], "height_mm": s[2], "density_kg_m3": mat["density_kg_m3"], "youngs_modulus_gpa": mat["youngs_modulus_gpa"], "yield_strength_mpa": mat["yield_strength_mpa"]})
    else:
        rows = []
        for _ in range(n_samples):
            l = rng.choice([500.0, 3000.0]) + rng.normal(0, 50)
            w = rng.choice([20.0, 300.0]) + rng.normal(0, 5)
            h = rng.choice([20.0, 500.0]) + rng.normal(0, 10)
            mat = materials[rng.integers(0, 3)]
            rows.append({"length_mm": max(500, min(3000, l)), "width_mm": max(20, min(300, w)), "height_mm": max(20, min(500, h)), "density_kg_m3": mat["density_kg_m3"], "youngs_modulus_gpa": mat["youngs_modulus_gpa"], "yield_strength_mpa": mat["yield_strength_mpa"]})
            
    df = pd.DataFrame(rows)
    return simulate_beam(df)

def train_model(df, model_type="nn"):
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), FEATURES)])
    if model_type == "nn":
        regressor = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42, solver='lbfgs')
    else:
        regressor = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(regressor))])
    pipeline.fit(df[FEATURES], df[TARGETS])
    return pipeline

# --- 4. Session State & Sidebar ---
if "models" not in st.session_state:
    st.session_state.models = None
    st.session_state.training_df_smart = None
    st.session_state.training_df_bad = None
    st.session_state.evaluation_df = None

if st.session_state.models is None:
    with st.spinner("🚀 Training all 4 AI variants..."):
        df_smart = generate_live_data(120, "smart")
        df_bad = generate_live_data(30, "bad")
        st.session_state.evaluation_df = generate_live_data(50, "smart", seed=42) # Fixed seed for reproducibility
        st.session_state.training_df_smart = df_smart
        st.session_state.training_df_bad = df_bad
        st.session_state.models = {
            "bad_xgb": train_model(df_bad, "xgb"),
            "bad_nn": train_model(df_bad, "nn"),
            "smart_xgb": train_model(df_smart, "xgb"),
            "smart_nn": train_model(df_smart, "nn"),
            "n_smart": 120, "n_bad": 30
        }

st.sidebar.markdown("### 🧪 AI-DOE Lab")
st.sidebar.markdown("---")

with st.sidebar.expander("🎓 AI Training Setup", expanded=True):
    n_bad_in = st.slider("Häufigkeit Bad DoE (Clustered)", 10, 200, st.session_state.models["n_bad"])
    n_smart_in = st.slider("Häufigkeit Smart DoE (LHS)", 50, 1000, st.session_state.models["n_smart"])
    if st.button("🚀 Alle Modelle neu trainieren", use_container_width=True):
        with st.spinner("Training 4 variants..."):
            df_smart_new = generate_live_data(n_smart_in, "smart")
            df_bad_new = generate_live_data(n_bad_in, "bad")
            st.session_state.evaluation_df = generate_live_data(50, "smart", seed=42) # Reproducible test set
            st.session_state.training_df_smart = df_smart_new
            st.session_state.training_df_bad = df_bad_new
            st.session_state.models = {
                "bad_xgb": train_model(df_bad_new, "xgb"),
                "bad_nn": train_model(df_bad_new, "nn"),
                "smart_xgb": train_model(df_smart_new, "xgb"),
                "smart_nn": train_model(df_smart_new, "nn"),
                "n_smart": n_smart_in, "n_bad": n_bad_in
            }
            st.success("Aktualisiert!")
            st.rerun()

with st.sidebar.expander("🏗️ Beam Configuration", expanded=True):
    materials_dict = {"Steel": {"d": 7850, "y": 210, "s": 350}, "Aluminum": {"d": 2700, "y": 69, "s": 250}, "Titanium": {"d": 4500, "y": 110, "s": 880}}
    mat_choice = st.selectbox("Material Choice", list(materials_dict.keys()))
    mat = materials_dict[mat_choice]
    length = st.slider("Length [mm]", 500, 3000, 1000)
    width = st.slider("Width [mm]", 20, 300, int(st.session_state.get('opt_w', 50)))
    height = st.slider("Height [mm]", 20, 500, int(st.session_state.get('opt_h', 100)))
    density, youngs, yield_str = st.number_input("Density", value=float(mat["d"])), st.number_input("Youngs", value=float(mat["y"])), st.number_input("Yield", value=float(mat["s"]))

with st.sidebar.expander("🎯 Optimization Constraints", expanded=False):
    max_defl = st.slider("Max Deflection [mm]", 0.1, 20.0, 5.0, help="Maximale zulässige Durchbiegung in der Mitte.")
    min_freq = st.slider("Min Frequency [Hz]", 1, 500, 30, help="Minimale Eigenfrequenz zur Vermeidung von Resonanz.")
    min_sf = st.slider("Min Safety Factor", 1.0, 10.0, 3.0, help="Sicherheitsfaktor gegenüber der Streckgrenze.")

# --- 5. Main View (Tab 1) ---
tab_sim, tab_file, tab_opt, tab_docs = st.tabs(["📊 Live Simulation", "💾 Storage", "🚀 Inverse Design", "📚 Docs"])

input_df = pd.DataFrame([{ "length_mm": float(length), "width_mm": float(width), "height_mm": float(height), "density_kg_m3": float(density), "youngs_modulus_gpa": float(youngs), "yield_strength_mpa": float(yield_str) }])

with tab_sim:
    st.markdown("### 📊 Flexibler AI-Vergleich")
    
    # Comparison Slot Configuration
    st.markdown("#### Vergleichs-Konfiguration")
    slot_cols = st.columns(4)
    
    OPTIONS = {
        "Bad DoE + XGBoost": "bad_xgb",
        "Bad DoE + NN": "bad_nn",
        "Smart DoE + XGBoost": "smart_xgb",
        "Smart DoE + NN": "smart_nn"
    }
    
    with slot_cols[0]:
        choice_a = st.selectbox("Slot A (Spalte 2)", list(OPTIONS.keys()), index=0)
    with slot_cols[1]:
        choice_b = st.selectbox("Slot B (Spalte 3)", list(OPTIONS.keys()), index=1)
    with slot_cols[2]:
        choice_c = st.selectbox("Slot C (Spalte 4)", list(OPTIONS.keys()), index=2)
    with slot_cols[3]:
        choice_d = st.selectbox("Slot D (Spalte 5)", list(OPTIONS.keys()), index=3)
    
    st.markdown("---")
    
    # Physics Calculation
    phys_res = simulate_beam(input_df)
    p_vals = [phys_res['weight_kg'].iloc[0], phys_res['deflection_mm'].iloc[0], phys_res['max_stress_mpa'].iloc[0], phys_res['safety_factor'].iloc[0], phys_res['eigenfrequency_hz'].iloc[0]]

    # 3D Visualization
    fig = plot_beam_3d(length, width, height, phys_res['deflection_mm'].iloc[0])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Display 6-Column Tabular Metric View
    # [Labels | Physics | Model A | Model B | Model C | Model D]
    c_lab, c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1, 1])
    
    def get_preds(key):
        preds = st.session_state.models[OPTIONS[key]].predict(input_df[FEATURES])[0]
        sf = yield_str / (abs(preds[2]) + 1e-9)
        return [preds[0], preds[1], preds[2], sf, preds[3]]

    p_a = get_preds(choice_a)
    p_b = get_preds(choice_b)
    p_c = get_preds(choice_c)
    p_d = get_preds(choice_d)

    lbls = ["Weight [kg]", "Deflection [mm]", "Stress [MPa]", "Safety Factor", "Frequency [Hz]"]
    helps = [
        "Das Gesamtgewicht des Trägers basierend auf Geometrie und Materialdichte.",
        "Die maximale vertikale Absenkung in der Mitte des Trägers unter Last.",
        "Die maximale mechanische Spannung in der Randfaser (Sollte unter der Streckgrenze liegen).",
        "Verhältnis von Streckgrenze zu auftretender Spannung (SF > 1.0 ist sicher).",
        "Die erste natürliche Eigenfrequenz. Wichtig für die Vibrationsanalyse."
    ]

    with c_lab:
        st.markdown("<p style='font-size: 0.9rem; font-weight: bold; margin-bottom: 10px;'>📏 Metric</p>", unsafe_allow_html=True)
        for i in range(5):
            # Symmetric st.metric for perfect alignment
            # Using label for spacing and value for the actual name
            st.metric(" ", lbls[i])
    
    with c1:
        st.markdown("<p style='font-size: 0.9rem; font-weight: bold; margin-bottom: 10px;'>🛠️ Physik</p>", unsafe_allow_html=True)
        for i in range(5): st.metric("Ref", f"{p_vals[i]:.2f}", help=helps[i])

    with c2:
        st.markdown(f"<p style='font-size: 0.9rem; font-weight: bold; margin-bottom: 10px;'>A: {choice_a}</p>", unsafe_allow_html=True)
        for i in range(5): st.metric("Pred", f"{p_a[i]:.2f}", delta=f"{p_a[i]-p_vals[i]:.3f}", delta_color="normal", help=helps[i])

    with c3:
        st.markdown(f"<p style='font-size: 0.9rem; font-weight: bold; margin-bottom: 10px;'>B: {choice_b}</p>", unsafe_allow_html=True)
        for i in range(5): st.metric("Pred", f"{p_b[i]:.2f}", delta=f"{p_b[i]-p_vals[i]:.3f}", delta_color="normal", help=helps[i])

    with c4:
        st.markdown(f"<p style='font-size: 0.9rem; font-weight: bold; margin-bottom: 10px;'>C: {choice_c}</p>", unsafe_allow_html=True)
        for i in range(5): st.metric("Pred", f"{p_c[i]:.2f}", delta=f"{p_c[i]-p_vals[i]:.3f}", delta_color="normal", help=helps[i])

    with c5:
        st.markdown(f"<p style='font-size: 0.9rem; font-weight: bold; margin-bottom: 10px;'>D: {choice_d}</p>", unsafe_allow_html=True)
        for i in range(5): st.metric("Pred", f"{p_d[i]:.2f}", delta=f"{p_d[i]-p_vals[i]:.3f}", delta_color="normal", help=helps[i])

    # Error Chart (All 4 variants)
    st.markdown("---")
    st.markdown("#### 📈 Abweichungen im Vergleich [%]")
    
    def err_perc(preds): return [(abs(a - p) / (p + 1e-9)) * 100 for a, p in zip(preds, p_vals)]
    
    # Calculate errors for all 4 available variants
    error_data = []
    for name, key in OPTIONS.items():
        # Get raw ML preds (4 targets)
        raw_p = st.session_state.models[key].predict(input_df[FEATURES])[0]
        # Insert safety factor calculation (derived) to match p_vals index
        # p_vals: [Wgt, Def, Str, SF, Frq]
        # raw_p:  [Wgt, Def, Str, Frq]
        sf_pred = input_df['yield_strength_mpa'].iloc[0] / (raw_p[2] + 1e-9)
        full_p = [raw_p[0], raw_p[1], raw_p[2], sf_pred, raw_p[3]]
        
        errors = err_perc(full_p)
        for metric, err in zip(["Wgt", "Def", "Str", "SF", "Frq"], errors):
            error_data.append({"Metric": metric, "Error [%]": err, "Source": name})
            
    comp_df = pd.DataFrame(error_data)
    
    import plotly.express as px
    fig_err = px.bar(
        comp_df, x="Metric", y="Error [%]", color="Source", 
        barmode="group",
        title="Abweichungen im Vergleich [%] (Log-Skala)",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hover_data={"Error [%]": ":.2f"},
        log_y=True
    )
    fig_err.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgb(220, 220, 220)',
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_err.update_yaxes(
        gridcolor="white", 
        zerolinecolor="white",
        tickfont=dict(color="white"),
        title_font=dict(color="white")
    )
    fig_err.update_xaxes(
        gridcolor="white", 
        zerolinecolor="white",
        tickfont=dict(color="white"),
        title_font=dict(color="white")
    )
    st.plotly_chart(fig_err, use_container_width=True)
    st.info("ℹ️ **Hinweis zur Log-Skala:** Da das schlechte Modell Fehler von über 1000% produziert, das gute Modell aber im Bereich von 0.1% liegt, nutzen wir hier eine logarithmische Skalierung. So bleiben auch kleine Präzisionsunterschiede sichtbar.")

    # --- 🎓 AI Insight: Design of Experiments (DoE) Analyse ---
    with st.expander("🎓 AI Insight: Design of Experiments (DoE) Analyse", expanded=False):
        st.markdown("""
        **Was sehen wir hier?**  
        Diese Grafik zeigt den "Wissensraum", mit dem die KI trainiert wurde. 
        
        **Wichtig für Ingenieure:**  
        Während die Geometrie (z.B. Länge) kontinuierlich variiert, sind die Materialeigenschaften (wie der *Young's Modulus*) **diskret** angesetzt. Wir nutzen einen Katalog aus realen Materialien (Stahl, Aluminium, Titan) anstatt beliebiger theoretischer Zwischenwerte.
        
        **Warum scheitern einige Modelle?**  
        * **Links (Bad DoE):** Die Datenpunkte sind eng geclustert (z.B. nur bei extremen Längen). Die KI lernt dort zwar perfekt, hat aber „keine Ahnung“, was dazwischen passiert (Interpolationsfehler).  
        * **Rechts (Smart DoE):** Dank *Latin Hypercube Sampling (LHS)* sind die Datenpunkte strategisch über den gesamten Raum verteilt. Die KI lernt die Physik für alle Kombinationen, was sie zu einem robusten Werkzeug für das Engineering macht.
        """)
        if st.session_state.training_df_smart is not None and st.session_state.training_df_bad is not None:
            fig_doe = plot_doe_distribution_plotly(st.session_state.training_df_smart, st.session_state.training_df_bad)
            st.plotly_chart(fig_doe, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Performance-Check: True vs. Predicted")
        st.markdown("""
        Hier prüfen wir die Modelle auf Herz und Nieren. Die Diagonale (gestrichelt) stellt die **physikalische Wahrheit** (Ground Truth) dar. 
        
        **Methodik der Test-Zusammenstellung:**
        *   **Unabhängiger Blind-Test:** Wir nutzen 50 komplett neue virtuelle Experimente, die kein Teil des Trainings waren. So testen wir die echte „Intelligenz“.
        *   **Maximale Abdeckung (LHS):** Die Punkte sind mittels *Latin Hypercube Sampling* über den gesamten Designraum verteilt, um Lücken im Wissen aufzudecken.
        *   **Fixierter Benchmark:** Damit du die Verbesserung der Modelle direkt vergleichen kannst, nutzen wir einen festen Startwert (Seed). So bleibt der „Prüfstand“ immer gleich, während du die KI neu trainierst.
        *   **Physik-Referenz:** Jeder Testpunkt wurde im Hintergrund mit dem echten Physik-Modell berechnet.
        
        **Interpretation:** Je näher die Punkte an der Diagonale liegen, desto besser hat die KI die Physik verstanden statt nur Daten auswendig zu lernen.
        """)
        
        if st.session_state.evaluation_df is not None:
            # Create tabs for each target metric
            tabs = st.tabs([t.replace("_", " ").title() for t in TARGETS])
            
            for i, target in enumerate(TARGETS):
                with tabs[i]:
                    # Predict all variants for the evaluation set for this specific target
                    y_preds_eval = {}
                    for name, key in OPTIONS.items():
                        # TARGETS maps to the indices of the prediction output
                        y_preds_eval[name] = st.session_state.models[key].predict(st.session_state.evaluation_df[FEATURES])[:, i]
                    
                    fig_acc = plot_accuracy_comparison_plotly(st.session_state.evaluation_df, y_preds_eval, target)
                    st.plotly_chart(fig_acc, use_container_width=True, key=f"acc_plot_{target}")

# Tabs Storage, Opt and Docs (Brief)
with tab_file:
    st.markdown("### 💾 Cloud Storage (Firestore)")
    if db:
        st.success("✅ Firebase-Verbindung aktiv.")
        
        # Data preparation for saving
        save_data = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "inputs": input_df.iloc[0].to_dict(),
            "results": {
                "weight_kg": float(p_vals[0]),
                "deflection_mm": float(p_vals[1]),
                "max_stress_mpa": float(p_vals[2]),
                "safety_factor": float(p_vals[3]),
                "eigenfrequency_hz": float(p_vals[4])
            },
            "material": mat_choice,
            "models_trained": {
                "n_smart": st.session_state.models["n_smart"],
                "n_bad": st.session_state.models["n_bad"]
            }
        }
        
        if st.button("🚀 Design in Cloud speichern", use_container_width=True):
            try:
                db.collection("simulations").add(save_data)
                st.balloons()
                st.success("Erfolgreich in Firebase gespeichert!")
            except Exception as e:
                st.error(f"Fehler beim Speichern: {e}")
                
        st.divider()
        st.markdown("#### Letzte Simulationen")
        try:
            docs = db.collection("simulations").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
            for doc in docs:
                d = doc.to_dict()
                st.write(f"📅 {d.get('timestamp')} | 🏗️ {d.get('material')} | {d['results']['weight_kg']:.2f} kg")
        except:
            st.info("Noch keine Daten vorhanden oder Index wird erstellt.")
            
    else:
        st.warning("⚠️ Firebase ist noch nicht konfiguriert.")
        st.info("Bitte füge deine Service Account Daten in `.streamlit/secrets.toml` ein, um die Cloud-Speicherung zu aktivieren.")

with tab_opt:
    st.markdown("### 🚀 Generative Design using Smart+NN Gold Standard")
    st.write("Finde die leichteste Geometrie (Breite/Höhe), die alle mechanischen Grenzwerte einhält.")
    
    if st.button("🚀 Design optimieren", use_container_width=True):
        with st.spinner("KI-Optimierung läuft..."):
            model = st.session_state.models["smart_nn"]
            
            def objective(x):
                # x = [width, height]
                test_df = pd.DataFrame([{
                    "length_mm": float(length), "width_mm": x[0], "height_mm": x[1],
                    "density_kg_m3": float(density), "youngs_modulus_gpa": float(youngs), "yield_strength_mpa": float(yield_str)
                }])
                preds = model.predict(test_df[FEATURES])[0]
                return preds[0] # weight_kg
            
            def constraint_defl(x):
                test_df = pd.DataFrame([{
                    "length_mm": float(length), "width_mm": x[0], "height_mm": x[1],
                    "density_kg_m3": float(density), "youngs_modulus_gpa": float(youngs), "yield_strength_mpa": float(yield_str)
                }])
                preds = model.predict(test_df[FEATURES])[0]
                return max_defl - preds[1] # max_defl - deflection >= 0

            def constraint_stress(x):
                test_df = pd.DataFrame([{
                    "length_mm": float(length), "width_mm": x[0], "height_mm": x[1],
                    "density_kg_m3": float(density), "youngs_modulus_gpa": float(youngs), "yield_strength_mpa": float(yield_str)
                }])
                preds = model.predict(test_df[FEATURES])[0]
                sf = yield_str / (abs(preds[2]) + 1e-9)
                return sf - min_sf # sf - min_sf >= 0

            def constraint_freq(x):
                test_df = pd.DataFrame([{
                    "length_mm": float(length), "width_mm": x[0], "height_mm": x[1],
                    "density_kg_m3": float(density), "youngs_modulus_gpa": float(youngs), "yield_strength_mpa": float(yield_str)
                }])
                preds = model.predict(test_df[FEATURES])[0]
                return preds[3] - min_freq # freq - min_freq >= 0

            cons = [
                {'type': 'ineq', 'fun': constraint_defl},
                {'type': 'ineq', 'fun': constraint_stress},
                {'type': 'ineq', 'fun': constraint_freq}
            ]
            
            res = minimize(objective, x0=[width, height], bounds=[(20, 300), (20, 500)], constraints=cons, method='COBYLA')
            
            if res.success:
                opt_w, opt_h = res.x
                st.success(f"Optimale Geometrie gefunden: Breite={opt_w:.1f}mm, Höhe={opt_h:.1f}mm")
                
                # Compare with current
                opt_df = pd.DataFrame([{
                    "length_mm": float(length), "width_mm": opt_w, "height_mm": opt_h,
                    "density_kg_m3": float(density), "youngs_modulus_gpa": float(youngs), "yield_strength_mpa": float(yield_str)
                }])
                opt_res = simulate_beam(opt_df)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write("**Aktuelles Design:**")
                    st.write(f"Gewicht: {phys_res['weight_kg'].iloc[0]:.2f} kg")
                with col_res2:
                    st.write("**Optimiertes Design:**")
                    st.write(f"Gewicht: {opt_res['weight_kg'].iloc[0]:.2f} kg")
                
                st.info(f"Einsparungspotenzial: {((phys_res['weight_kg'].iloc[0] - opt_res['weight_kg'].iloc[0]) / phys_res['weight_kg'].iloc[0]) * 100:.1f}%")
                
                if st.button("Optimierte Maße übernehmen"):
                    st.session_state.opt_w = opt_w
                    st.session_state.opt_h = opt_h
                    st.rerun()
            else:
                st.error("Keine gültige Lösung gefunden. Constraints lockern?")

with tab_docs:
    try: st.markdown(open("docs/PROJECT_DOCUMENTATION.md", "r", encoding="utf-8").read())
    except: st.warning("Docs missing.")

st.markdown(f'<div class="footer-msg">AI-DOE Lab | Flexible 4-Model Comparison | Active Project: {st.session_state.get("project_name", "Lab")}</div>', unsafe_allow_html=True)
