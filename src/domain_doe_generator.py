import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc
from physics_sim import simulate_beam

def generate_domain_data(n_samples: int = 150, output_path: str = None) -> pd.DataFrame:
    """
    Creates a "Domain" Dataset: Valid geometries but STRICT, discrete materials.
    """
    # 3 Continuous Geometric Parameters
    l_bounds = [1000.0, 3000.0]
    w_bounds = [50.0, 150.0]
    h_bounds = [100.0, 300.0]
    
    # 1 Discrete Material Parameter mapped between [0, 3) 
    bounds = np.array([l_bounds, w_bounds, h_bounds, [0.0, 3.0]])
    
    sampler = qmc.LatinHypercube(d=4, seed=123)
    sample = sampler.random(n=n_samples)
    scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    
    # Real-world material catalogs: Density [kg/m3], Youngs [GPa], Yield [MPa]
    materials = [
        {"material": "Steel",    "density_kg_m3": 7850.0, "youngs_modulus_gpa": 210.0, "yield_strength_mpa": 350.0},
        {"material": "Aluminum", "density_kg_m3": 2700.0, "youngs_modulus_gpa": 69.0,  "yield_strength_mpa": 250.0},
        {"material": "Titanium", "density_kg_m3": 4500.0, "youngs_modulus_gpa": 110.0, "yield_strength_mpa": 880.0}
    ]
    
    rows = []
    for i, s in enumerate(scaled_sample):
        mat_idx = int(np.floor(s[3]))
        if mat_idx >= 3: mat_idx = 2  # edge case catch
        
        # Grab physical properties based on discrete selection
        mat_props = materials[mat_idx]
        
        row = {
            "length_mm": s[0],
            "width_mm": s[1],
            "height_mm": s[2],
            "density_kg_m3": mat_props["density_kg_m3"],
            "youngs_modulus_gpa": mat_props["youngs_modulus_gpa"],
            "yield_strength_mpa": mat_props["yield_strength_mpa"]
            # I can omit mapping the string name for the AI, it only needs numerical properties
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.insert(0, "experiment_id", [f"DOMAIN_{i:03d}" for i in range(1, n_samples + 1)])
    
    # Calculate physics
    df_simulated = simulate_beam(df).round(3)
    
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_simulated.to_csv(out_path, index=False)
        print(f"[{n_samples} Domain LHS Samples] generated and saved to {output_path}")
        
    return df_simulated

def generate_domain_bad_data(n_samples: int = 30, output_path: str = None) -> pd.DataFrame:
    """Creates a 'Bad DoE' dataset within the specific domain (highly clustered)."""
    l_vals = np.random.choice([1000.0, 3000.0], size=n_samples) + np.random.normal(0, 50, n_samples)
    w_vals = np.random.choice([50.0, 150.0], size=n_samples)   + np.random.normal(0, 5, n_samples)
    h_vals = np.random.choice([100.0, 300.0], size=n_samples)  + np.random.normal(0, 10, n_samples)
    
    mat_idxs = np.random.randint(0, 3, size=n_samples)
    materials = [
        {"material": "Steel",    "density_kg_m3": 7850.0, "youngs_modulus_gpa": 210.0, "yield_strength_mpa": 350.0},
        {"material": "Aluminum", "density_kg_m3": 2700.0, "youngs_modulus_gpa": 69.0,  "yield_strength_mpa": 250.0},
        {"material": "Titanium", "density_kg_m3": 4500.0, "youngs_modulus_gpa": 110.0, "yield_strength_mpa": 880.0}
    ]
    
    rows = []
    for i in range(n_samples):
        mat_props = materials[mat_idxs[i]]
        row = {
            "length_mm": max(1000.0, min(3000.0, l_vals[i])),
            "width_mm": max(50.0, min(150.0, w_vals[i])),
            "height_mm": max(100.0, min(300.0, h_vals[i])),
            "density_kg_m3": mat_props["density_kg_m3"],
            "youngs_modulus_gpa": mat_props["youngs_modulus_gpa"] + np.random.normal(0, 2), 
            "yield_strength_mpa": mat_props["yield_strength_mpa"] + np.random.normal(0, 10)
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.insert(0, "experiment_id", [f"DOMAIN_BAD_{i:03d}" for i in range(1, n_samples + 1)])
    df_simulated = simulate_beam(df).round(3)
    
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_simulated.to_csv(out_path, index=False)
        print(f"[{n_samples} Bad Domain Samples] generated and saved to {output_path}")
        
    return df_simulated

if __name__ == "__main__":
    generate_domain_data(n_samples=150, output_path="data/raw/domain_beam_data.csv")
    generate_domain_bad_data(n_samples=30, output_path="data/raw/domain_bad_beam_data.csv")
