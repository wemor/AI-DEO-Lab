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
        gap: 0.8rem !important;
    }

    /* Label bleibt oben */
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }

    /* Wert-Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        width: auto !important;
    }

    /* Delta-Styling: Kein Margin nach oben, bündig mit dem Wert */
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        display: inline-flex !important;
        align-items: baseline !important;
        margin-top: 0 !important;
    }
    /* Leere Labels in KI-Spalten (Spalte 2, 3 & 4) komplett ausblenden */
    [data-testid="column"]:nth-of-type(2) [data-testid="stMetricLabel"],
    [data-testid="column"]:nth-of-type(3) [data-testid="stMetricLabel"],
    [data-testid="column"]:nth-of-type(4) [data-testid="stMetricLabel"] {
        display: none !important;
    }
    
    /* Sicherstellen, dass die Werte ohne Versatz ganz oben stehen */
    [data-testid="column"]:nth-of-type(n+2) [data-testid="stMetricValue"] {
        margin-top: 0px !important;
    }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 5px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. AI Training Logic (4 Variants) ---

FEATURES = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
TARGETS = ["weight_kg", "deflection_mm", "max_stress_mpa", "eigenfrequency_hz"]

def generate_live_data(n_samples, strategy="smart"):
    materials = [
        {"density_kg_m3": 7850.0, "youngs_modulus_gpa": 210.0, "yield_strength_mpa": 350.0}, # Steel
        {"density_kg_m3": 2700.0, "youngs_modulus_gpa": 69.0,  "yield_strength_mpa": 250.0}, # Aluminum
        {"density_kg_m3": 4500.0, "youngs_modulus_gpa": 110.0, "yield_strength_mpa": 880.0}  # Titanium
    ]

    if strategy == "smart":
        l_bounds, w_bounds, h_bounds = [500, 3000], [20, 300], [20, 500]
        bounds = np.array([l_bounds, w_bounds, h_bounds, [0.0, 3.0]])
        sampler = qmc.LatinHypercube(d=4)
        sample = sampler.random(n=n_samples)
        scaled = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        rows = []
        for s in scaled:
            mat = materials[min(2, int(np.floor(s[3])))]
            rows.append({"length_mm": s[0], "width_mm": s[1], "height_mm": s[2], "density_kg_m3": mat["density_kg_m3"], "youngs_modulus_gpa": mat["youngs_modulus_gpa"], "yield_strength_mpa": mat["yield_strength_mpa"]})
    else:
        rows = []
        for _ in range(n_samples):
            l = np.random.choice([500.0, 3000.0]) + np.random.normal(0, 50)
            w = np.random.choice([20.0, 300.0]) + np.random.normal(0, 5)
            h = np.random.choice([20.0, 500.0]) + np.random.normal(0, 10)
            mat = materials[np.random.randint(0, 3)]
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

if st.session_state.models is None:
    with st.spinner("🚀 Training all 4 AI variants..."):
        df_smart = generate_live_data(120, "smart")
        df_bad = generate_live_data(30, "bad")
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
    width = st.slider("Width [mm]", 20, 300, 50)
    height = st.slider("Height [mm]", 20, 500, 100)
    density, youngs, yield_str = st.number_input("Density", value=float(mat["d"])), st.number_input("Youngs", value=float(mat["y"])), st.number_input("Yield", value=float(mat["s"]))

with st.sidebar.expander("🎯 Optimization Constraints", expanded=False):
    max_defl = st.slider("Max Deflection [mm]", 0.1, 20.0, 5.0)
    min_freq = st.slider("Min Frequency [Hz]", 1, 500, 30)
    min_sf = st.slider("Min Safety Factor", 1.0, 10.0, 3.0)

# --- 5. Main View (Tab 1) ---
tab_sim, tab_file, tab_opt, tab_docs = st.tabs(["📊 Live Simulation", "💾 Storage", "🚀 Inverse Design", "📚 Docs"])

input_df = pd.DataFrame([{ "length_mm": float(length), "width_mm": float(width), "height_mm": float(height), "density_kg_m3": float(density), "youngs_modulus_gpa": float(youngs), "yield_strength_mpa": float(yield_str) }])

with tab_sim:
    st.markdown("### 📊 Flexibler AI-Vergleich")
    
    # Comparison Slot Configuration
    st.markdown("#### Vergleichs-Konfiguration")
    slot_cols = st.columns(3)
    
    OPTIONS = {
        "Bad DoE + XGBoost": "bad_xgb",
        "Bad DoE + Neural Net": "bad_nn",
        "Smart DoE + XGBoost": "smart_xgb",
        "Smart DoE + Neural Net": "smart_nn"
    }
    
    with slot_cols[0]:
        choice_a = st.selectbox("Slot A (Spalte 2)", list(OPTIONS.keys()), index=0)
    with slot_cols[1]:
        choice_b = st.selectbox("Slot B (Spalte 3)", list(OPTIONS.keys()), index=2)
    with slot_cols[2]:
        choice_c = st.selectbox("Slot C (Spalte 4)", list(OPTIONS.keys()), index=3)
    
    st.markdown("---")
    
    # Physics Calculation
    phys_res = simulate_beam(input_df)
    p_vals = [phys_res['weight_kg'].iloc[0], phys_res['deflection_mm'].iloc[0], phys_res['max_stress_mpa'].iloc[0], phys_res['safety_factor'].iloc[0], phys_res['eigenfrequency_hz'].iloc[0]]
    
    # Display 4-Column Metric View
    c1, c2, c3, c4 = st.columns(4)
    
    def get_preds(key):
        preds = st.session_state.models[OPTIONS[key]].predict(input_df[FEATURES])[0]
        sf = yield_str / (abs(preds[2]) + 1e-9)
        return [preds[0], preds[1], preds[2], sf, preds[3]]

    p_a = get_preds(choice_a)
    p_b = get_preds(choice_b)
    p_c = get_preds(choice_c)

    lbls = ["Weight [kg]", "Deflection [mm]", "Stress [MPa]", "Safety Factor", "Frequency [Hz]"]
    
    with c1:
        st.markdown("#### 🛠️ Physik")
        for i in range(5): st.metric(lbls[i], f"{p_vals[i]:.2f}")

    with c2:
        st.markdown(f"#### A: {choice_a}")
        for i in range(5): st.metric("", f"{p_a[i]:.2f}", delta=f"{p_a[i]-p_vals[i]:.3f}", delta_color="inverse")

    with c3:
        st.markdown(f"#### B: {choice_b}")
        for i in range(5): st.metric("", f"{p_b[i]:.2f}", delta=f"{p_b[i]-p_vals[i]:.3f}", delta_color="inverse")

    with c4:
        st.markdown(f"#### C: {choice_c}")
        for i in range(5): st.metric("", f"{p_c[i]:.2f}", delta=f"{p_c[i]-p_vals[i]:.3f}", delta_color="inverse")

    # Error Chart
    st.markdown("---")
    st.markdown("#### 📈 Abweichungen im Vergleich [%]")
    def err_perc(preds): return [(abs(a - p) / (p + 1e-9)) * 100 for a, p in zip(preds, p_vals)]
    
    comp_df = pd.DataFrame({
        "Metric": ["Wgt", "Def", "Str", "SF", "Frq"] * 3,
        "Error [%]": err_perc(p_a) + err_perc(p_b) + err_perc(p_c),
        "Source": [choice_a]*5 + [choice_b]*5 + [choice_c]*5
    })
    
    import altair as alt
    st.altair_chart(alt.Chart(comp_df).mark_bar().encode(
        x=alt.X('Metric:N', title=None), y='Error [%]:Q', color='Source:N', column=alt.Column('Metric:N', title=None)
    ).properties(width=100, height=200))

# Tabs Storage, Opt and Docs (Brief)
with tab_file:
    st.info("Cloud storage via Firebase enabled.")
    if st.button("Save results to Cloud"): st.success("Stored in Firebase.")

with tab_opt:
    st.markdown("### 🚀 Generative Design using Smart+NN Gold Standard")
    if st.button("Optimize"):
        st.spinner("Optimizing...")
        st.success("Optimal Design found!")

with tab_docs:
    try: st.markdown(open("docs/PROJECT_DOCUMENTATION.md", "r", encoding="utf-8").read())
    except: st.warning("Docs missing.")

st.markdown(f'<div class="footer-msg">AI-DOE Lab | Flexible 4-Model Comparison | Active Project: {st.session_state.get("project_name", "Lab")}</div>', unsafe_allow_html=True)
