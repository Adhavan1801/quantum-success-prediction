# Quantum Success Prediction

**Predicting the success probability of Deutsch-Jozsa, Bernstein-Vazirani, and Simon's algorithms under noise using classical machine learning.**

## Overview

This project combines quantum computing simulation with classical machine learning to predict the success probability of foundational quantum algorithms under various noise conditions.

### Project Progress
- [x] Stage 1: Circuit building (DJ, BV, Simon's)
- [x] Stage 2: Noise simulation (Depolarizing, Amplitude Damping, Thermal Relaxation)
- [x] Stage 3: EDA & feature engineering
- [x] Stage 4: ML model training (Decision Tree, Random Forest)
- [x] Stage 5: Results analysis & benchmark comparisons
- [x] Stage 6: Interactive Web Application & Local Deployment

## Project Structure

```
├── data/
│   └── results.csv                         # Simulation results (1,386 experiments)
├── notebooks/
│   ├── 01_circuit_building.ipynb            # Build & visualize quantum circuits
│   ├── 02_noise_simulation.ipynb            # Simulate noise and export results
│   ├── 03_eda_feature_engineering.ipynb     # Explore data and engineer features
│   ├── 04_ml_training.ipynb                 # Train Decision Tree & Random Forest models
│   ├── 05_results_analysis.ipynb            # Model comparison and feature importance
│   └── 06_circuit_visualization.ipynb       # Visualize circuits from simulation results
├── outputs/
│   ├── figures/                             # Evaluation plots and comparison figures
│   ├── models/                              # Trained .joblib models (RF & DT)
│   └── model_comparison.csv                 # Test set R², MAE, RMSE metrics
├── static/
│   ├── index.html                           # Single-page web interface
│   ├── styles.css                           # Quantum dark aesthetic design system
│   └── app.js                               # Interactive frontend logic & API client
├── src/
│   ├── app.py / server.py                   # Local Flask backend server & REST API
│   ├── circuits.py                          # DJ, BV, Simon's circuit builders
│   ├── feature_engineering.py               # Dataset and ML feature utilities
│   ├── ml_pipeline.py                       # ML model training and evaluation
│   ├── noise_models.py                      # Quantum noise model definitions
│   └── simulation.py                        # Circuit execution and parameter sweeps
├── app.py                                   # Root entrypoint to run the web application
├── tests/
│   └── test_circuits.py                     # Unit tests for circuit builders
└── requirements.txt                         # Python dependencies
```

## Algorithms Studied

| Algorithm | Purpose | Key Property |
|-----------|---------|-------------|
| **Deutsch-Jozsa** | Determines if a function is constant or balanced | Single query, deterministic |
| **Bernstein-Vazirani** | Recovers a hidden bitstring | Single query, deterministic |
| **Simon's** | Finds hidden period of a 2-to-1 function | Few queries, deterministic |

## Quick Start

### 1. Set Up the Environment
```bash
python -m venv venv
venv\Scripts\activate.bat       # Windows
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Application
```bash
python app.py
```
Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)** to:
- Enter quantum algorithm and noise parameters with real-time range guidance.
- Predict the algorithm success probability and view the ensemble confidence score.
- Visualize individual tree predictions across the Random Forest.
- Compare Random Forest vs Decision Tree predictions.
- Optionally verify predictions against live Qiskit Aer simulations (4096 shots).

### 3. Run the Jupyter Notebooks (Optional)
```bash
jupyter notebook
```
Open `notebooks/` to inspect data exploration, model training, and analysis.

## Tools & Technologies

- **Quantum Computing**: Qiskit 1.4.2, Qiskit Aer 0.17.0
- **Machine Learning**: Scikit-Learn 1.9, Joblib
- **Web Frontend & API**: Flask, Vanilla HTML5/CSS3/JavaScript (Quantum Design System)
- **Data & Visualization**: Pandas, NumPy, Matplotlib, Seaborn

- **Quantum**: Qiskit, Qiskit Aer
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Python 3.11, Jupyter Notebook

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
