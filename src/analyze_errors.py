import pandas as pd
import numpy as np
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
    # Increase estimators slightly for better representation if desired, but keep same grid for now
    param_grid = {"max_depth": [3, 4], "n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    grid = GridSearchCV(xgb, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(grid))])

def analyze_smart_domain_errors():
    print("Loading Smart Domain Dataset...")
    df_smart = pd.read_csv("data/raw/domain_beam_data.csv")
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "safety_factor", "eigenfrequency_hz"]
    
    X_smart = df_smart[features]
    y_smart = df_smart[targets]
    
    # Use standard 80/20 train/test split (same as in evaluate_domain.py)
    X_train, X_test, y_train, y_test = train_test_split(X_smart, y_smart, test_size=0.2, random_state=42)
    
    print(f"Training Smart Model on {len(X_train)} samples...")
    model = build_model(features)
    model.fit(X_train, y_train)
    
    print(f"Predicting on {len(X_test)} unseen holdout samples...\n")
    preds_arr = model.predict(X_test)
    preds = pd.DataFrame(preds_arr, columns=targets, index=y_test.index)
    
    results = []
    
    for target in targets:
        y_true = y_test[target]
        y_pred = preds[target]
        
        # Absolute Errors
        abs_err = np.abs(y_pred - y_true)
        # Percentage Errors (prevent division by zero issues, add small epsilon)
        perc_err = (abs_err / (y_true + 1e-9)) * 100
        
        # Find index of max absolute Error
        idx_max_error = abs_err.idxmax()
        
        # Metrics
        mae = abs_err.mean()
        mape = perc_err.mean()
        max_err_val = abs_err.max()
        max_perc_err_val = perc_err.max()
        
        # Details of the worst predicted sample
        worst_true = y_true.loc[idx_max_error]
        worst_pred = y_pred.loc[idx_max_error]
        worst_features = X_test.loc[idx_max_error].to_dict()
        
        res = {
            "Target Variable": target,
            "Mean Absolute Error (MAE)": f"{mae:.3f}",
            "Mean Perc. Error (MAPE)": f"{mape:.2f}%",
            "Max Dev (Absolute)": f"{max_err_val:.3f}",
            "Max Dev (Relative)": f"{max_perc_err_val:.2f}%",
            "Worst Sample True Value": worst_true,
            "Worst Sample Pred Value": worst_pred,
            "Worst Sample Index": idx_max_error,
            "Worst Sample Features": worst_features
        }
        results.append(res)
        
    df_results = pd.DataFrame(results)
    
    print("="*60)
    print(" ERROR ANALYSIS RESULTS (Smart Domain DoE)")
    print("="*60)
    for res in results:
        t = res["Target Variable"]
        print(f"\n--- {t.upper()} ---")
        print(f"MAPE (Ø Fehler): {res['Mean Perc. Error (MAPE)']} | MAE: {res['Mean Absolute Error (MAE)']}")
        print(f"Maximaler Fehler: {res['Max Dev (Absolute)']} (entspricht {res['Max Dev (Relative)']})")
        print(f"Detail zum schlechtesten Sample (Index {res['Worst Sample Index']}):")
        print(f"  Echter Wert: {res['Worst Sample True Value']:.4f}")
        print(f"  Vorhersage : {res['Worst Sample Pred Value']:.4f}")
        print("  Geometrie / Material dieses Samples:")
        for k, v in res['Worst Sample Features'].items():
            print(f"    - {k}: {v:.2f}")

    # Plot Error Distribution
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(targets), figsize=(20, 5))
    
    for i, target in enumerate(targets):
        y_true = y_test[target]
        y_pred = preds[target]
        perc_err = ((y_pred - y_true) / (y_true + 1e-9)) * 100
        
        axes[i].hist(perc_err, bins=15, color="purple", alpha=0.7, edgecolor="black")
        axes[i].set_title(f"Error Dist: {target}", fontsize=12)
        axes[i].set_xlabel("Relative Deviation [%]")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        
        # Highlight mean error
        axes[i].axvline(perc_err.mean(), color="red", linestyle="--", label=f"Mean Diff: {perc_err.mean():.2f}%")
        axes[i].legend()

    plt.tight_layout()
    filename = "reports/figures/06_smart_error_distribution.png"
    plt.savefig(filename, dpi=300)
    print(f"\nSaved Error Distribution Plot -> {filename}")


if __name__ == "__main__":
    analyze_smart_domain_errors()
