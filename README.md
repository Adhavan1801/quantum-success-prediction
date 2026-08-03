# Quantum Success Prediction

**Predicting the success probability of Deutsch-Jozsa, Bernstein-Vazirani, and Simon's algorithms under noise using classical machine learning.**

## Overview

This project combines quantum computing simulation with classical machine learning to predict the success probability of foundational quantum algorithms under various noise conditions.

### Current Progress: Stage 1 — Circuit Building ✅

> Building and visualizing quantum circuits for Deutsch-Jozsa, Bernstein-Vazirani, and Simon's algorithms.

**Upcoming stages:**
- [ ] Stage 2: Noise simulation
- [ ] Stage 3: EDA & feature engineering
- [ ] Stage 4: ML model training
- [ ] Stage 5: Results analysis

## Project Structure

```
├── notebooks/
│   ├── 01_circuit_building.ipynb         # Build & visualize quantum circuits
│   └── 06_circuit_visualization.ipynb    # Circuit diagrams & analysis
│
├── src/
│   └── circuits.py              # DJ, BV, Simon's oracle & circuit builders
│
├── tests/
│   └── test_circuits.py         # Unit tests for circuit builders
│
└── requirements.txt             # Python dependencies
```

## Algorithms Studied

| Algorithm | Purpose | Key Property |
|-----------|---------|-------------|
| **Deutsch-Jozsa** | Determines if a function is constant or balanced | Single query, deterministic |
| **Bernstein-Vazirani** | Recovers a hidden bitstring | Single query, deterministic |
| **Simon's** | Finds hidden period of a 2-to-1 function | Few queries, deterministic |

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Adhavan1801/quantum-success-prediction.git
cd quantum-success-prediction
```

### 2. Set Up the Environment
```bash
python -m venv venv
venv\Scripts\activate.bat       # Windows
pip install -r requirements.txt
```

### 3. Run the Notebooks
```bash
jupyter notebook
```
Open `01_circuit_building.ipynb` to explore the quantum circuits.

## Tools & Technologies

- **Quantum**: Qiskit, Qiskit Aer
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Python 3.11, Jupyter Notebook

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
