import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc
from physics_sim import simulate_beam

def generate_smart_data(n_samples: int = 200, output_path: str = None) -> pd.DataFrame:
    """
    Creates a "Smart Data" dataset using Latin Hypercube Sampling (LHS).
    LHS evenly spans the full multi-dimensional parameter space without overlaps,
    meaning we need vastly fewer samples (N=200) to get excellent generalization.
    """
    # 6 continuous parameter combinations
    l_bounds = [500.0, 2500.0]
    w_bounds = [20.0, 100.0]
    h_bounds = [20.0, 150.0]
    d_bounds = [500.0, 8000.0] # From light wood to heavy steel
    e_bounds = [10.0, 250.0]   # GPa
    y_bounds = [50.0, 600.0]   # MPa
    
    bounds = np.array([l_bounds, w_bounds, h_bounds, d_bounds, e_bounds, y_bounds])
    
    # Generate LHS
    sampler = qmc.LatinHypercube(d=6, seed=42)
    sample = sampler.random(n=n_samples)
    
    # Scale from [0, 1] to our physical bounds
    scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    
    df = pd.DataFrame(scaled_sample, columns=[
        "length_mm", "width_mm", "height_mm", "density_kg_m3", 
        "youngs_modulus_gpa", "yield_strength_mpa"
    ])
    df.insert(0, "experiment_id", [f"SMART_{i:03d}" for i in range(1, n_samples + 1)])
    
    # Calculate the labels / ground truth
    df_simulated = simulate_beam(df)
    
    # Rounding for cleanliness
    df_simulated = df_simulated.round(3)
    
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_simulated.to_csv(out_path, index=False)
        print(f"[{n_samples} Smart LHS Samples] generated and saved to {output_path}")
        
    return df_simulated

if __name__ == "__main__":
    # Generate the Smart LHS DoE Dataset
    generate_smart_data(n_samples=200, output_path="data/raw/smart_beam_data.csv")
