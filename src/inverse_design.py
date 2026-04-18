import joblib
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Load AI Surrogate Model globally for fast evaluation
MODEL = joblib.load("models/surrogate_nn.joblib")
META = joblib.load("models/model_meta.joblib")
FEATURES = META["features"]

def evaluate_surrogate(x, length_mm, density, youngs, yield_str):
    # x = [width_mm, height_mm]
    w, h = x[0], x[1]
    
    # Create DF matching the feature columns exactly
    df = pd.DataFrame([{
        "length_mm": length_mm,
        "width_mm": w,
        "height_mm": h,
        "density_kg_m3": density,
        "youngs_modulus_gpa": youngs,
        "yield_strength_mpa": yield_str
    }])[FEATURES]
    
    # Predict ["weight_kg", "deflection_mm", "max_stress_mpa", "eigenfrequency_hz"]
    preds = MODEL.predict(df)[0]
    return preds

def objective(x, length_mm, density, youngs, yield_str):
    # We want to minimize weight (index 0 of preds)
    preds = evaluate_surrogate(x, length_mm, density, youngs, yield_str)
    return preds[0] # weight_kg

def constraint_deflection(x, length_mm, density, youngs, yield_str, max_deflection):
    # SLSQP constraint format: must be >= 0
    # constraint: max_deflection - true_deflection >= 0
    preds = evaluate_surrogate(x, length_mm, density, youngs, yield_str)
    return max_deflection - preds[1] # preds[1] is deflection_mm

def constraint_safety(x, length_mm, density, youngs, yield_str, min_sf):
    preds = evaluate_surrogate(x, length_mm, density, youngs, yield_str)
    stress = abs(preds[2])
    
    # Analytical Safety Factor! (Replacing the AI error)
    # True Safety Factor = yield_strength / stress
    # Constraint: SF >= min_sf  --> SF - min_sf >= 0
    calculated_sf = yield_str / (stress + 1e-9)
    return calculated_sf - min_sf

def constraint_eigen(x, length_mm, density, youngs, yield_str, min_hz):
    preds = evaluate_surrogate(x, length_mm, density, youngs, yield_str)
    # Constraint: hz >= min_hz --> hz - min_hz >= 0
    return preds[3] - min_hz

def run_inverse_design():
    print("--- INVERSE DESIGN OPTIMIZATION STARTING ---")
    print("Using AI Surrogate Model (NN) for hyper-fast evaluations...\n")
    
    # Context (e.g. we need a 3-meter bridge beam out of steel)
    LENGTH = 3000.0   # mm
    DENSITY = 7850.0  # kg/m3 (Steel)
    YOUNGS = 210.0    # GPa
    YIELD = 350.0     # MPa
    
    # Requirements
    MAX_DEFLECTION = 5.0  # mm
    MIN_SAFETY_FACTOR = 3.0
    MIN_FREQ = 30.0       # Hz
    
    print(f"Goal: Find the LEAST HEAVY beam possible with Length {LENGTH}mm")
    print(f"Rule 1: Deflection MUST be <= {MAX_DEFLECTION} mm")
    print(f"Rule 2: Safety Factor MUST be >= {MIN_SAFETY_FACTOR} (Analytically computed from AI Stress!)")
    print(f"Rule 3: Eigenfrequency MUST be >= {MIN_FREQ} Hz\n")
    
    # Initial Guess (e.g. width=100, height=200)
    x0 = np.array([100.0, 200.0])
    
    # Bounds for the geometry (e.g., width 20-300mm, height 20-500mm)
    bnds = ((20, 300), (20, 500))
    
    # SLSQP Constraints (all must be non-negative)
    cons = [
        {'type': 'ineq', 'fun': constraint_deflection, 'args': (LENGTH, DENSITY, YOUNGS, YIELD, MAX_DEFLECTION)},
        {'type': 'ineq', 'fun': constraint_safety, 'args': (LENGTH, DENSITY, YOUNGS, YIELD, MIN_SAFETY_FACTOR)},
        {'type': 'ineq', 'fun': constraint_eigen, 'args': (LENGTH, DENSITY, YOUNGS, YIELD, MIN_FREQ)}
    ]
    
    result = minimize(
        objective, 
        x0, 
        args=(LENGTH, DENSITY, YOUNGS, YIELD),
        method='SLSQP', 
        bounds=bnds, 
        constraints=cons,
        options={'disp': True, 'maxiter': 100}
    )
    
    print("\n--- OPTIMIZATION RESULTS ---")
    if result.success:
        print("SUCCESS! Scipy Optimizer has found the optimal geometric configuration.")
    else:
        print(f"TERMINATED: {result.message}")
        
    opt_w, opt_h = result.x
    preds = evaluate_surrogate(result.x, LENGTH, DENSITY, YOUNGS, YIELD)
    final_sf = YIELD / abs(preds[2])
    
    print(f"\n======== OPTIMAL GEOMETRY FOUND ========")
    print(f"Width (b):  {opt_w:.1f} mm")
    print(f"Height (h): {opt_h:.1f} mm")
    print("========================================")
    print("\nPERFORMANCE OF THIS BEAM (VIA AI SURROGATE):")
    print(f"Weight:            {preds[0]:.2f} kg (Minimized!)")
    print(f"Deflection:        {preds[1]:.2f} mm    (Limit: <= {MAX_DEFLECTION})")
    print(f"Max Stress:        {preds[2]:.2f} MPa")
    print(f"Safety Factor:     {final_sf:.2f}         (Limit: >= {MIN_SAFETY_FACTOR})")
    print(f"Eigenfrequency:    {preds[3]:.2f} Hz    (Limit: >= {MIN_FREQ})")
    
if __name__ == "__main__":
    run_inverse_design()
