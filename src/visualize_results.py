import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_true_vs_pred():
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "safety_factor", "eigenfrequency_hz"]
    
    print("Loading datasets...")
    bad_df = pd.read_csv("data/raw/unstructured_beam_data.csv")
    smart_df = pd.read_csv("data/raw/smart_beam_data.csv")
    
    # Smart Data split -> 160 train, 40 hold-out test
    X_smart = smart_df[features]
    y_smart = smart_df[targets]
    X_train_smart, X_test_holdout, y_train_smart, y_test_holdout = train_test_split(X_smart, y_smart, test_size=0.2, random_state=42)
    
    # Bad Data -> Train on all 30
    X_train_bad = bad_df[features]
    y_train_bad = bad_df[targets]
    
    print("Training Bad DoE Model (30 pts)...")
    model_bad = build_model(features)
    model_bad.fit(X_train_bad, y_train_bad)
    
    print("Training Smart DoE Model (160 pts)...")
    model_smart = build_model(features)
    model_smart.fit(X_train_smart, y_train_smart)
    
    # Evaluate BOTH on the exact same 40-point holdout set
    preds_bad_arr = model_bad.predict(X_test_holdout)
    preds_bad = pd.DataFrame(preds_bad_arr, columns=targets, index=y_test_holdout.index)
    
    preds_smart_arr = model_smart.predict(X_test_holdout)
    preds_smart = pd.DataFrame(preds_smart_arr, columns=targets, index=y_test_holdout.index)
    
    _create_scatter_plot(y_test_holdout, preds_bad, preds_smart, targets, use_log_scale=False)
    _create_scatter_plot(y_test_holdout, preds_bad, preds_smart, targets, use_log_scale=True)

def _create_scatter_plot(y_test_holdout, preds_bad, preds_smart, targets, use_log_scale):
    fig, axes = plt.subplots(len(targets), 2, figsize=(14, 5 * len(targets)))
    
    for i, target in enumerate(targets):
        min_val = min(
            y_test_holdout[target].min(), 
            preds_bad[target].min(),
            preds_smart[target].min()
        )
        max_val = max(
            y_test_holdout[target].max(), 
            preds_bad[target].max(),
            preds_smart[target].max()
        )
        
        # If log scale, we avoid negative/zero boundaries for min_val to plot cleanly
        if use_log_scale and min_val <= 0:
            min_val = y_test_holdout[target][y_test_holdout[target] > 0].min() * 0.5
            if pd.isna(min_val): min_val = 1e-3
            
        for col_idx, (preds, title_prefix) in enumerate([(preds_bad, "Bad DoE Model"), (preds_smart, "Smart DoE Model")]):
            ax = axes[i, col_idx]
            color = "red" if col_idx == 0 else "green"
            
            ax.scatter(y_test_holdout[target], preds[target], color=color, alpha=0.7, edgecolors="k", s=60)
            ax.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction")
            
            mode_str = "Log-Log" if use_log_scale else "Linear"
            ax.set_title(f"{title_prefix}: {target} ({mode_str})", fontsize=14)
            ax.set_xlabel(f"True {target}", fontsize=12)
            ax.set_ylabel(f"Predicted {target}", fontsize=12)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            
            if use_log_scale:
                # symlog allows handling negative ai predictions cleanly without crashing the log scale
                ax.set_xscale("symlog", linthresh=max(1.0, min_val))
                ax.set_yscale("symlog", linthresh=max(1.0, min_val))

    plt.tight_layout()
    suffix = "log" if use_log_scale else "linear"
    filename = f"reports/figures/01_true_vs_predicted_{suffix}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved -> {filename}")

def plot_doe_distribution():
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    
    bad_df = pd.read_csv("data/raw/unstructured_beam_data.csv")
    smart_df = pd.read_csv("data/raw/smart_beam_data.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bad Data Input Dist
    axes[0].scatter(bad_df["length_mm"], bad_df["youngs_modulus_gpa"], color="red", alpha=0.7, edgecolors="k", s=100)
    axes[0].set_title("Unstructured DoE Input Distribution", fontsize=14)
    axes[0].set_xlabel("Length [mm]", fontsize=12)
    axes[0].set_ylabel("Youngs Modulus [GPa]", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    # Force identical axis limits to show how clustered it is
    axes[0].set_xlim(400, 2600)
    axes[0].set_ylim(0, 260)
    
    # Smart Data Input Dist
    axes[1].scatter(smart_df["length_mm"], smart_df["youngs_modulus_gpa"], color="green", alpha=0.7, edgecolors="k", s=50)
    axes[1].set_title("LHS (Smart DoE) Input Distribution", fontsize=14)
    axes[1].set_xlabel("Length [mm]", fontsize=12)
    axes[1].set_ylabel("Youngs Modulus [GPa]", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_xlim(400, 2600)
    axes[1].set_ylim(0, 260)
    
    plt.tight_layout()
    plt.savefig("reports/figures/02_doe_distribution_space.png", dpi=300)
    print("Saved -> reports/figures/02_doe_distribution_space.png")

if __name__ == "__main__":
    # Create the graphics for the user presentation
    plot_doe_distribution()
    plot_true_vs_pred()
