import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings("ignore")

def train_and_save_final_model():
    print("Loading Smart Domain Dataset...")
    df_smart = pd.read_csv("data/raw/domain_beam_data.csv")
    
    features = ["length_mm", "width_mm", "height_mm", "density_kg_m3", "youngs_modulus_gpa", "yield_strength_mpa"]
    # REMOVED safety_factor as requested (calculated analytically instead)
    targets = ["weight_kg", "deflection_mm", "max_stress_mpa", "eigenfrequency_hz"]
    
    X = df_smart[features]
    y = df_smart[targets]
    
    print("Training FINAL Neural Network Surrogate Model on all 150 Smart DoE Points...")
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features)])
    
    # We use the best params found during grid search
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64), 
        activation='relu', 
        solver='lbfgs', 
        alpha=0.01,
        max_iter=3000,
        random_state=42
    )
    
    model = Pipeline([
        ("preprocessor", preprocessor), 
        ("regressor", MultiOutputRegressor(mlp))
    ])
    
    model.fit(X, y)
    
    # Save the model
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "models/surrogate_nn.joblib")
    
    # Also save the specific column names to ensure input shapes are correct
    joblib.dump({"features": features, "targets": targets}, "models/model_meta.joblib")
    print("Successfully trained and saved the AI Physics Surrogate Model to models/surrogate_nn.joblib")

if __name__ == "__main__":
    train_and_save_final_model()
