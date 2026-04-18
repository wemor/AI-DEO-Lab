import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

def build_model(features):
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features)])
    xgb = XGBRegressor(random_state=42, objective="reg:squarederror")
    param_grid = {"max_depth": [2, 3], "n_estimators": [50], "learning_rate": [0.1]}
    grid = GridSearchCV(xgb, param_grid=param_grid, cv=2, scoring="neg_mean_absolute_error", n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(grid))])

def evaluate_domain():
    print("Loading Domain Datasets...")
    df_smart = pd.read_csv("data/raw/domain_beam_data.csv")
    df_bad = pd.read_csv("data/raw/domain_bad_beam_data.csv")
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "safety_factor", "eigenfrequency_hz"]
    
    X_smart = df_smart[features]
    y_smart = df_smart[targets]
    
    # 120 Train / 30 Test for Smart Data
    X_train_smart, X_test_holdout, y_train_smart, y_test_holdout = train_test_split(X_smart, y_smart, test_size=0.2, random_state=42)
    
    print("Training Domain-Bad ML Model (30 unstructured points)...")
    model_bad = build_model(features)
    model_bad.fit(df_bad[features], df_bad[targets])
    
    print("Training Domain-Smart ML Model (120 structured points)...")
    model_smart = build_model(features)
    model_smart.fit(X_train_smart, y_train_smart)
    
    # Predict BOTH models on the exact same 30-point holdout
    preds_bad_arr = model_bad.predict(X_test_holdout)
    preds_bad = pd.DataFrame(preds_bad_arr, columns=targets, index=y_test_holdout.index)
    
    preds_smart_arr = model_smart.predict(X_test_holdout)
    preds_smart = pd.DataFrame(preds_smart_arr, columns=targets, index=y_test_holdout.index)
    
    # Plotting (Linear Mode) - 4 rows x 2 cols
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(targets), 2, figsize=(14, 5 * len(targets)))
    
    for i, target in enumerate(targets):
        min_val = min(y_test_holdout[target].min(), preds_bad[target].min(), preds_smart[target].min())
        max_val = max(y_test_holdout[target].max(), preds_bad[target].max(), preds_smart[target].max())
        
        # 1. Bad Model
        axes[i, 0].scatter(y_test_holdout[target], preds_bad[target], color="red", alpha=0.7, edgecolors="k", s=60)
        axes[i, 0].plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction")
        axes[i, 0].set_title(f"Bad Domain DoE: {target}", fontsize=14)
        axes[i, 0].set_xlabel(f"True {target}", fontsize=12)
        axes[i, 0].set_ylabel(f"Predicted {target}", fontsize=12)
        axes[i, 0].legend()
        axes[i, 0].grid(True, linestyle="--", alpha=0.5)
        
        # 2. Smart Model
        axes[i, 1].scatter(y_test_holdout[target], preds_smart[target], color="blue", alpha=0.7, edgecolors="k", s=60)
        axes[i, 1].plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction")
        axes[i, 1].set_title(f"Smart Domain DoE: {target}", fontsize=14)
        axes[i, 1].set_xlabel(f"True {target}", fontsize=12)
        axes[i, 1].set_ylabel(f"Predicted {target}", fontsize=12)
        axes[i, 1].legend()
        axes[i, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    filename = "reports/figures/05_domain_true_vs_predicted.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved -> {filename}")
    
    from sklearn.metrics import r2_score
    print("\n--- Domain-Specialist Performance (N=30 Test Points) ---")
    for t in targets:
        score = r2_score(y_test_holdout[t], preds_smart[t])
        print(f"{t}: R2 = {score:.4f}")

def plot_domain_distribution():
    df_smart = pd.read_csv("data/raw/domain_beam_data.csv")
    df_bad = pd.read_csv("data/raw/domain_bad_beam_data.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Bad Domain DoE (Clustered)
    axes[0].scatter(df_bad["length_mm"], df_bad["youngs_modulus_gpa"], color="red", alpha=0.7, edgecolors="k", s=60)
    axes[0].set_title("Bad Domain DoE: Unstructured Extreme Values", fontsize=14)
    axes[0].set_xlabel("Length [mm] (Clustered at edges)", fontsize=12)
    axes[0].set_ylabel("Youngs Modulus [GPa] (Discrete Catalog)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_xlim(0, 3500)
    axes[0].set_ylim(0, 250)
    
    # 2. Smart Domain DoE (Uniform)
    axes[1].scatter(df_smart["length_mm"], df_smart["youngs_modulus_gpa"], color="green", alpha=0.7, edgecolors="k", s=60)
    axes[1].set_title("Smart Domain DoE: Structured LHS Coverage", fontsize=14)
    axes[1].set_xlabel("Length [mm] (Uniformly covered)", fontsize=12)
    axes[1].set_ylabel("Youngs Modulus [GPa] (Discrete Catalog)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_xlim(0, 3500)
    axes[1].set_ylim(0, 250)
    
    plt.tight_layout()
    filename = "reports/figures/04_domain_distribution.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved -> {filename}")

if __name__ == "__main__":
    plot_domain_distribution()
    evaluate_domain()
