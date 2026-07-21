# Quantum Success Prediction

**Predicting the success probability of Deutsch-Jozsa, Bernstein-Vazirani, and Simon's algorithms under noise using classical machine learning.**

## Overview

This project combines quantum computing simulation with classical machine learning to:

1. **Simulate** three foundational quantum algorithms (DJ, BV, Simon's) across varying qubit counts and noise conditions using Qiskit Aer.
2. **Build a structured dataset** linking circuit parameters (qubit count, depth, gate count) and noise parameters (type, strength) to measured success probability.
3. **Train interpretable ML models** (Decision Tree, Random Forest) to predict success probability from circuit/noise parameters — without re-running simulations.
4. **Analyze feature importance** to identify which parameters drive noise-induced degradation the most.

## Project Structure

```
├── notebooks/
│   ├── 01_circuit_building.ipynb         # Build & visualize quantum circuits
│   ├── 02_noise_simulation.ipynb         # Run noise sweep, generate dataset
│   ├── 03_eda_feature_engineering.ipynb   # Explore data, engineer features
│   ├── 04_ml_training.ipynb              # Train & tune ML models
│   └── 05_results_analysis.ipynb         # Feature importance & final analysis
│
├── src/
│   ├── circuits.py              # DJ, BV, Simon's oracle & circuit builders
│   ├── noise_models.py          # Noise model configurations
│   ├── simulation.py            # Simulation runner
│   ├── feature_engineering.py   # Feature table construction
│   └── ml_pipeline.py           # ML training & evaluation utilities
│
├── data/                        # Generated datasets
├── outputs/
│   ├── figures/                 # Saved plots
│   └── models/                  # Trained model files
├── tests/                       # Unit tests
├── setup_env.bat                # One-click environment setup (Windows)
└── requirements.txt             # Python dependencies
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Adhavan1801/quantum-success-prediction.git
cd quantum-success-prediction
```

### 2. Set Up the Environment
**Windows:**
```bash
setup_env.bat
```
**Manual setup:**
```bash
python -m venv venv
venv\Scripts\activate.bat       # Windows
pip install -r requirements.txt
```

### 3. Run the Notebooks
```bash
jupyter notebook
```
Open and run notebooks in order: `01` → `02` → `03` → `04` → `05`

## Algorithms Studied

| Algorithm | Purpose | Key Property |
|-----------|---------|-------------|
| **Deutsch-Jozsa** | Determines if a function is constant or balanced | Single query, deterministic |
| **Bernstein-Vazirani** | Recovers a hidden bitstring | Single query, deterministic |
| **Simon's** | Finds hidden period of a 2-to-1 function | Few queries, deterministic |

## Noise Models

| Noise Type | Description |
|------------|-------------|
| **Depolarizing** | Random replacement with mixed state |
| **Amplitude Damping** | Energy loss (T1 decay) |
| **Thermal Relaxation** | Combined T1/T2 effects |

## Tools & Technologies

- **Quantum**: Qiskit, Qiskit Aer
- **ML**: scikit-learn (DecisionTree, RandomForest, GridSearchCV)
- **Data**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Python 3.11, Jupyter Notebook

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
