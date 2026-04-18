import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

def build_xgb_model(features):
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features)])
    xgb = XGBRegressor(random_state=42, objective="reg:squarederror")
    param_grid = {"max_depth": [2, 3], "n_estimators": [50], "learning_rate": [0.1]}
    grid = GridSearchCV(xgb, param_grid=param_grid, cv=2, scoring="neg_mean_absolute_error", n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(grid))])

def build_nn_model(features):
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features)])
    mlp = MLPRegressor(random_state=42, max_iter=3000)
    # Reduced grid search space for faster computation while retaining best parameters
    param_grid = {
        "hidden_layer_sizes": [(64, 64)],
        "activation": ["relu"],
        "solver": ["lbfgs"],
        "alpha": [0.01]
    }
    grid = GridSearchCV(mlp, param_grid=param_grid, cv=2, scoring="neg_mean_absolute_error", n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(grid))])

def main():
    print("Loading Datasets...")
    df_smart = pd.read_csv("data/raw/domain_beam_data.csv")
    df_bad = pd.read_csv("data/raw/domain_bad_beam_data.csv")
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "safety_factor", "eigenfrequency_hz"]
    
    X_smart = df_smart[features]
    y_smart = df_smart[targets]
    X_train_smart, X_test_holdout, y_train_smart, y_test_holdout = train_test_split(X_smart, y_smart, test_size=0.2, random_state=42)
    
    print("1. Training Bad XGBoost Model...")
    model_bad_xgb = build_xgb_model(features)
    model_bad_xgb.fit(df_bad[features], df_bad[targets])
    preds_bad_xgb = pd.DataFrame(model_bad_xgb.predict(X_test_holdout), columns=targets, index=y_test_holdout.index)
    
    print("2. Training Smart XGBoost Model...")
    model_smart_xgb = build_xgb_model(features)
    model_smart_xgb.fit(X_train_smart, y_train_smart)
    preds_smart_xgb = pd.DataFrame(model_smart_xgb.predict(X_test_holdout), columns=targets, index=y_test_holdout.index)
    
    print("3. Training Smart Neural Network Model...")
    model_smart_nn = build_nn_model(features)
    model_smart_nn.fit(X_train_smart, y_train_smart)
    preds_smart_nn = pd.DataFrame(model_smart_nn.predict(X_test_holdout), columns=targets, index=y_test_holdout.index)
    
    # Plotting 5 rows (targets) x 3 cols (models)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(targets), 3, figsize=(20, 5 * len(targets)))
    
    titles = ["Bad DoE (XGBoost)", "Smart DoE (XGBoost)", "Smart DoE (Neural Network)"]
    preds_list = [preds_bad_xgb, preds_smart_xgb, preds_smart_nn]
    colors = ["red", "blue", "darkorange"]
    
    for i, target in enumerate(targets):
        # Calculate consistent limits for all 3 plots of this target
        min_val = min(y_test_holdout[target].min(), preds_bad_xgb[target].min(), preds_smart_xgb[target].min(), preds_smart_nn[target].min())
        max_val = max(y_test_holdout[target].max(), preds_bad_xgb[target].max(), preds_smart_xgb[target].max(), preds_smart_nn[target].max())
        
        for j in range(3):
            ax = axes[i, j]
            preds = preds_list[j]
            ax.scatter(y_test_holdout[target], preds[target], color=colors[j], alpha=0.7, edgecolors="k", s=60)
            ax.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction")
            
            if i == 0:
                ax.set_title(f"{titles[j]}\n{target}", fontsize=16, fontweight="bold")
            else:
                ax.set_title(f"{titles[j]}\n{target}", fontsize=14)
                
            ax.set_xlabel(f"True {target}", fontsize=12)
            ax.set_ylabel(f"Predicted {target}", fontsize=12)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    fig.suptitle("Evolution of AI Capabilities: Unstructured Data vs. Structured Data vs. Best Algorithm", fontsize=24, fontweight="bold")
    
    filename = "reports/figures/08_model_evolution_comparison.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved -> {filename}")

if __name__ == "__main__":
    main()
