import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

def build_nn_model(features):
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features)])
    
    # MLP Regressor for capturing non-linear physics smoothly
    mlp = MLPRegressor(random_state=42, max_iter=3000)
    
    # GridSearch for best architecture and regularization
    # lbfgs is generally better for small datasets
    param_grid = {
        "hidden_layer_sizes": [(32, 32), (64, 64), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "solver": ["lbfgs"],
        "alpha": [0.001, 0.01, 0.1]
    }
    
    grid = GridSearchCV(mlp, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    
    return Pipeline([("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(grid))])

def evaluate_neural_network():
    print("Loading Smart Domain Dataset...")
    df_smart = pd.read_csv("data/raw/domain_beam_data.csv")
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "safety_factor", "eigenfrequency_hz"]
    
    X_smart = df_smart[features]
    y_smart = df_smart[targets]
    
    # 120 Train / 30 Test
    X_train, X_test, y_train, y_test = train_test_split(X_smart, y_smart, test_size=0.2, random_state=42)
    
    print("Training Neural Network (MLP) on 120 structured points... (This may take a minute with GridSearch)")
    model = build_nn_model(features)
    model.fit(X_train, y_train)
    
    print("Predicting on 30 Test points...")
    preds_arr = model.predict(X_test)
    preds = pd.DataFrame(preds_arr, columns=targets, index=y_test.index)
    
    # Print R2 Scores
    print("\n--- NEURAL NETWORK PERFORMANCE (R2 Score) ---")
    for t in targets:
        score = r2_score(y_test[t], preds[t])
        print(f"{t}: R2 = {score:.4f}")
        
    print("\n--- ERROR METRICS ---")
    results = []
    for target in targets:
        y_true = y_test[target]
        y_pred = preds[target]
        
        abs_err = np.abs(y_pred - y_true)
        perc_err = (abs_err / (y_true + 1e-9)) * 100
        
        mae = abs_err.mean()
        mape = perc_err.mean()
        max_perc_err_val = perc_err.max()
        
        results.append({
            "Target": target,
            "MAE": mae,
            "MAPE (%)": mape,
            "Max Dev (%)": max_perc_err_val
        })
        
        print(f"{target.ljust(20)} | MAPE: {mape:6.2f}% | MAE: {mae:6.2f} | Max Error: {max_perc_err_val:6.2f}%")
        
    # Plotting True vs Predicted for Neural Network
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(targets), figsize=(20, 4))
    
    for i, target in enumerate(targets):
        min_val = min(y_test[target].min(), preds[target].min())
        max_val = max(y_test[target].max(), preds[target].max())
        
        axes[i].scatter(y_test[target], preds[target], color="darkorange", alpha=0.8, edgecolors="k", s=60)
        axes[i].plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect")
        axes[i].set_title(f"NN (MLP): {target}", fontsize=12)
        axes[i].set_xlabel("True")
        axes[i].set_ylabel("Predicted")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    filename = "reports/figures/07_nn_true_vs_predicted.png"
    plt.savefig(filename, dpi=300)
    print(f"\nSaved Distribution Plot -> {filename}")

if __name__ == "__main__":
    evaluate_neural_network()
