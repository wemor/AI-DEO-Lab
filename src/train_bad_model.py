import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

def train_and_save_bad_model():
    print("Loading BAD Domain Dataset (Unstructured Clustered Data)...")
    df_bad = pd.read_csv("data/raw/domain_bad_beam_data.csv")
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    # Targets (same as final model for consistency)
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "eigenfrequency_hz"]
    
    X = df_bad[features]
    y = df_bad[targets]
    
    print(f"Training 'BAD' XGBoost Model on {len(X)} unstructured points...")
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features)])
    
    # We use a simple XGBoost that is likely to overfit or fail outside the clusters
    xgb = XGBRegressor(
        n_estimators=50, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42, 
        objective="reg:squarederror"
    )
    
    model = Pipeline([
        ("preprocessor", preprocessor), 
        ("regressor", MultiOutputRegressor(xgb))
    ])
    
    model.fit(X, y)
    
    # Save the model
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/bad_doe_xgb.joblib")
    print("Successfully trained and saved the BAD DoE XGBoost Model to models/bad_doe_xgb.joblib")

if __name__ == "__main__":
    train_and_save_bad_model()
