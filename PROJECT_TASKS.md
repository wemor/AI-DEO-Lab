# AI-DOE-Lab Project Tasks

## 1. Project Initialization
- [x] Create `PROJECT_SPECIFICATION.md`.
- [x] Initialize folder structure (`src`, `data/raw`, `data/processed`).
- [x] Create `requirements.txt`.
- [x] Initialize Python Virtual Environment (`.venv`) and install requirements.

## 2. Phase 1: Data Simulator
- [x] Implement `src/physics_sim.py` using mechanics formulas.
- [x] Test the simulator with a standard steel beam.

## 3. Phase 2: DoE Data Generation
- [x] Implement `src/doe_generator.py` to create a *small, unstructured* dataset (e.g. N=30) to intentionally force bad ML performance.

## 4. Phase 3: AI Pipeline (Testing Bad Data)
- [x] Implement `src/pipeline.py` (XGBoost, GridSearchCV).
- [x] Train against simulated data and log failure.

## 5. Phase 4: Smart DoE & Perfecting the Model
- [x] Implement `src/smart_doe_generator.py` using Latin Hypercube Sampling (LHS) for N=200 points.
- [x] Update pipeline to evaluate the smart dataset.
- [x] Document the successful prediction metrics as proof.

## 6. Phase 5: Visualizing the Results
- [x] Implement `src/visualize_results.py` using matplotlib.
- [x] Generate "True vs Predicted AI Accuracies" graphs.
- [x] Generate "DoE Parameter Distribution" graphs.

## 7. Phase 6: Domain-Specific AI (Engineering Reality)
- [x] Implement `src/domain_doe_generator.py` mapping continuous geometry spaces with discrete materials (Steel, Aluminum, Titanium).
- [x] Train AI strictly on this localized valid space.
- [x] Document how limiting the "Design Space" removes non-linear spread and provides hyper-accurate AI tools.

## 8. Phase 7: Erweiterung der physikalischen Zielgrößen
- [x] Integration der ersten Eigenfrequenz ($f_1$) in die Ground-Truth-Simulation.
- [x] Evaluation der automatisierten Propagation durch alle ML-Pipelines und Visualisierungen als 5. Zielvariable.
