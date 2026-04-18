import numpy as np
import pandas as pd

def simulate_beam(df: pd.DataFrame, force_n: float = 1000.0) -> pd.DataFrame:
    """
    Computes the mechanical properties (ground truth) of a simply supported beam
    under a central point load based on geometry and material inputs.
    
    Inputs in df:
    - length_mm
    - width_mm
    - height_mm
    - density_kg_m3
    - youngs_modulus_gpa
    - yield_strength_mpa
    
    Outputs generated:
    - weight_kg
    - deflection_mm
    - max_stress_mpa
    - safety_factor
    - failure (boolean)
    """
    # Create a copy to store results
    result = df.copy()
    
    # 1. Unit conversions to SI units (meters, Pascals)
    L_m = df["length_mm"] * 1e-3
    b_m = df["width_mm"] * 1e-3
    h_m = df["height_mm"] * 1e-3
    
    E_pa = df["youngs_modulus_gpa"] * 1e9
    rho = df["density_kg_m3"]
    yield_pa = df["yield_strength_mpa"] * 1e6
    
    # 2. Mechanics Formulas (Simply Supported Beam)
    # Area moment of inertia (I) [m^4]
    I_m4 = (b_m * h_m**3) / 12.0
    
    # Max Bending Moment (M) [Nm] at center of simply supported beam
    M_max_nm = (force_n * L_m) / 4.0
    
    # Max Stress (Sigma) [Pa]
    sigma_max_pa = (M_max_nm * (h_m / 2.0)) / I_m4
    
    # Max Deflection (w) [m] at center
    deflection_m = (force_n * L_m**3) / (48.0 * E_pa * I_m4)
    
    # Weight [kg]
    weight_kg = L_m * b_m * h_m * rho
    
    # First Natural Frequency [Hz]
    mu_kg_m = rho * b_m * h_m
    f1_hz = (np.pi / (2.0 * L_m**2)) * np.sqrt((E_pa * I_m4) / mu_kg_m)
    
    # 3. Output features (converting back to usable dimensions)
    result["weight_kg"] = weight_kg
    result["deflection_mm"] = deflection_m * 1000.0
    result["max_stress_mpa"] = sigma_max_pa / 1e6
    result["safety_factor"] = yield_pa / sigma_max_pa
    result["eigenfrequency_hz"] = f1_hz
    
    # Failure condition: Safety factor < 1.0 means it breaks/yields
    result["failure"] = result["safety_factor"] < 1.0
    
    return result

if __name__ == "__main__":
    # Quick test for a single standard steel beam: 
    # 1m length, 50x50mm cross-section
    test_data = pd.DataFrame([{
        "length_mm": 1000.0,
        "width_mm": 50.0,
        "height_mm": 50.0,
        "density_kg_m3": 7850.0,
        "youngs_modulus_gpa": 210.0,
        "yield_strength_mpa": 350.0
    }])
    
    print("--- Test: 1000N Load on 50x50mm Steel Beam (1m) ---")
    out = simulate_beam(test_data, force_n=1000.0)
    for col in test_data.columns:
        out = out.drop(columns=[col])  # drop inputs to only print outputs
    print(out.T)
