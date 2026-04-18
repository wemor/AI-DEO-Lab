import numpy as np
import pandas as pd
from pathlib import Path

from physics_sim import simulate_beam

def generate_unstructured_data(n_samples: int = 30, output_path: str = None) -> pd.DataFrame:
    """
    Intentionally creates a "bad", unstructured dataset (similar to the legacy chemistry data).
    It uses a very small N, lots of randomness, and limited clustered ranges to ensure 
    the AI will struggle to generalize properly.
    """
    np.random.seed(42)
    
    # We create clustered geometry parameters rather than an even DoE distribution
    # E.g. mostly clustered around 500mm and 2000mm length, ignoring everything in between.
    lengths = np.random.choice([500.0, 550.0, 2000.0, 2100.0], size=n_samples)
    widths = np.random.uniform(20.0, 100.0, size=n_samples)
    heights = np.random.uniform(20.0, 150.0, size=n_samples)
    
    # Let's say they only tested 2 materials randomly (Aluminum and Steel variants)
    # Steel ~7850kg/m3, 210GPa E, 350MPa Yield
    # Alum ~2700kg/m3, 70GPa E, 200MPa Yield
    material_choice = np.random.choice(["Steel", "Aluminum"], size=n_samples, p=[0.7, 0.3])
    
    density = np.where(material_choice == "Steel", 7850.0, 2700.0)
    # Add some random noise to materials to simulate "varying grades" or "messy data"
    youngs = np.where(material_choice == "Steel", 210.0, 70.0) + np.random.normal(0, 5.0, size=n_samples)
    yield_strength = np.where(material_choice == "Steel", 350.0, 200.0) + np.random.normal(0, 10.0, size=n_samples)
    
    df = pd.DataFrame({
        "experiment_id": [f"TEST_{i:03d}" for i in range(1, n_samples + 1)],
        "length_mm": lengths,
        "width_mm": widths,
        "height_mm": heights,
        "density_kg_m3": density,
        "youngs_modulus_gpa": youngs,
        "yield_strength_mpa": yield_strength
    })
    
    # Calculate the labels / ground truth
    df_simulated = simulate_beam(df)
    
    # Round to realistic precision found in labs
    df_simulated = df_simulated.round(3)
    
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_simulated.to_csv(out_path, index=False)
        print(f"[{n_samples} Unstructured Samples] generated and saved to {output_path}")
        
    return df_simulated

if __name__ == "__main__":
    # Generate the small, problematic dataset for Phase 1 of our teaching exercise
    generate_unstructured_data(n_samples=30, output_path="data/raw/unstructured_beam_data.csv")
