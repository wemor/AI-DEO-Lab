import pandas as pd
import json
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

import sys

def main():
    data_file = "data/raw/unstructured_beam_data.csv"
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        df = pd.read_csv(f"../{data_file}")
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "safety_factor", "eigenfrequency_hz"]
    
    X = df[features]
    y = df[targets]
    
    print(f"Dataset shape: {X.shape}")
    
    # 80/20 Split forces the model to learn from ~24 points and test on 6 points
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples...")
    print(f"Testing on {len(X_test)} samples...\n")
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), features)
    ])
    
    xgb = XGBRegressor(random_state=42, objective="reg:squarederror")
    # Tiny dataset needs simple trees, but gridsearch still cross-validates
    param_grid = {
        "max_depth": [2, 3],
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
    }
    
    # CV=2 because data is too small for standard 5-fold CV
    grid = GridSearchCV(xgb, param_grid=param_grid, cv=2, scoring="neg_mean_absolute_error")
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", MultiOutputRegressor(grid))
    ])
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    preds_df = pd.DataFrame(preds, columns=targets, index=y_test.index)
    
    # Evaluate
    metrics = {}
    for col in targets:
        mae = mean_absolute_error(y_test[col], preds_df[col])
        r2 = r2_score(y_test[col], preds_df[col])
        metrics[col] = {"MAE": round(mae, 2), "R2": round(r2, 4)}
        
    print("--- Validation Metrics (Test Data) ---")
    print(json.dumps(metrics, indent=2))
    
    print("\n[AI LIMITATION OBSERVED]")
    print("Notice the R2 scores. Values around 0 or negative mean the AI prediction")
    print("is worse than simply guessing the average value. This proves that an ML model")
    print("CANNOT magically deduce physics from 30 badly clustered DoE experiments.")

if __name__ == "__main__":
    main()
