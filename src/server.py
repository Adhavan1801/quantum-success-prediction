# =============================================================
# server.py — Local Web Server & API for Quantum ML Prediction
# =============================================================
"""
Flask server providing REST API and single-page web application
for quantum circuit success probability prediction and confidence scoring.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.circuits import (
    build_dj_oracle,
    build_dj_circuit,
    build_bv_circuit,
    build_simon_circuit,
    get_circuit_stats,
    validate_circuit_parameters,
    check_circuit_constructibility,
)
from src.noise_models import (
    get_depolarizing_model,
    get_amplitude_damping_model,
    get_thermal_relaxation_model,
)
from src.simulation import run_circuit, compute_success_probability

# Initialize Flask app
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
app = Flask(__name__, static_folder=STATIC_DIR)

# -------------------------------------------------------------
# LOAD TRAINED MODELS
# -------------------------------------------------------------
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
RF_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")
DT_PATH = os.path.join(MODEL_DIR, "decision_tree_model.joblib")

print("Loading ML models...")
rf_model = joblib.load(RF_PATH)
dt_model = joblib.load(DT_PATH)
print("Models loaded successfully!")

# Expected feature names
FEATURE_NAMES = [
    "n_qubits",
    "circuit_depth",
    "gate_count_1q",
    "gate_count_2q",
    "total_gate_count",
    "noise_strength",
    "algo_bernstein_vazirani",
    "algo_deutsch_jozsa",
    "algo_simons",
    "noise_amplitude_damping",
    "noise_depolarizing",
    "noise_none",
    "noise_thermal_relaxation",
]

# Dataset bounds & valid ranges derived from results.csv
VALID_RANGES = {
    "n_qubits": {"min": 2, "max": 8, "default": 3, "step": 1, "label": "Number of Qubits"},
    "circuit_depth": {"min": 3, "max": 12, "default": 5, "step": 1, "label": "Circuit Depth"},
    "gate_count_1q": {"min": 4, "max": 27, "default": 7, "step": 1, "label": "1-Qubit Gate Count"},
    "gate_count_2q": {"min": 0, "max": 11, "default": 2, "step": 1, "label": "2-Qubit Gate Count (CNOT)"},
    "total_gate_count": {"min": 5, "max": 35, "default": 9, "step": 1, "label": "Total Gate Count"},
    "noise_strengths": {
        "none": {"min": 0.0, "max": 0.0, "default": 0.0, "step": 0.0, "label": "No Noise (Ideal)"},
        "depolarizing": {"min": 0.001, "max": 0.10, "default": 0.01, "step": 0.001, "label": "Depolarizing (0.001 - 0.10)"},
        "amplitude_damping": {"min": 0.001, "max": 0.10, "default": 0.01, "step": 0.001, "label": "Amplitude Damping (0.001 - 0.10)"},
        "thermal_relaxation": {"min": 0.01, "max": 0.10, "default": 0.033, "step": 0.001, "label": "Thermal Relaxation (0.01 - 0.10)"},
    },
    "algorithms": ["deutsch_jozsa", "bernstein_vazirani", "simons"],
    "noise_types": ["none", "depolarizing", "amplitude_damping", "thermal_relaxation"],
}

# Pre-defined realistic benchmark presets
BENCHMARK_PRESETS = [
    {
        "id": "preset_dj_ideal",
        "name": "Deutsch-Jozsa (3 Qubits, Ideal)",
        "description": "Baseline 3-qubit Deutsch-Jozsa balanced circuit with zero noise.",
        "algorithm": "deutsch_jozsa",
        "n_qubits": 3,
        "circuit_depth": 6,
        "gate_count_1q": 8,
        "gate_count_2q": 3,
        "total_gate_count": 11,
        "noise_type": "none",
        "noise_strength": 0.0,
    },
    {
        "id": "preset_bv_depol",
        "name": "Bernstein-Vazirani (5 Qubits, Depolarizing p=0.01)",
        "description": "5-qubit secret bitstring recovery under typical Pauli depolarizing noise.",
        "algorithm": "bernstein_vazirani",
        "n_qubits": 5,
        "circuit_depth": 6,
        "gate_count_1q": 11,
        "gate_count_2q": 3,
        "total_gate_count": 14,
        "noise_type": "depolarizing",
        "noise_strength": 0.01,
    },
    {
        "id": "preset_simon_damping",
        "name": "Simon's Algorithm (4 Qubits, Amplitude Damping γ=0.02)",
        "description": "4-qubit hidden period search experiencing energy loss / T1 decay.",
        "algorithm": "simons",
        "n_qubits": 4,
        "circuit_depth": 6,
        "gate_count_1q": 8,
        "gate_count_2q": 5,
        "total_gate_count": 13,
        "noise_type": "amplitude_damping",
        "noise_strength": 0.02,
    },
    {
        "id": "preset_scaled_thermal",
        "name": "Scaled 7-Qubit Circuit (Thermal Relaxation)",
        "description": "Deep 7-qubit circuit with realistic T1/T2 thermal relaxation decoherence.",
        "algorithm": "deutsch_jozsa",
        "n_qubits": 7,
        "circuit_depth": 10,
        "gate_count_1q": 20,
        "gate_count_2q": 7,
        "total_gate_count": 27,
        "noise_type": "thermal_relaxation",
        "noise_strength": 0.033,
    },
    {
        "id": "preset_heavy_noise",
        "name": "Stress Test (High Depolarizing p=0.10)",
        "description": "Extreme noise condition testing algorithm degradation boundary.",
        "algorithm": "bernstein_vazirani",
        "n_qubits": 6,
        "circuit_depth": 7,
        "gate_count_1q": 13,
        "gate_count_2q": 4,
        "total_gate_count": 17,
        "noise_type": "depolarizing",
        "noise_strength": 0.10,
    },
]


def build_feature_vector(params):
    """
    Construct the 13-feature array for ML model prediction.
    """
    algo = params.get("algorithm", "deutsch_jozsa").lower()
    noise_t = params.get("noise_type", "none").lower()

    row = {
        "n_qubits": int(params.get("n_qubits", 3)),
        "circuit_depth": int(params.get("circuit_depth", 5)),
        "gate_count_1q": int(params.get("gate_count_1q", 7)),
        "gate_count_2q": int(params.get("gate_count_2q", 2)),
        "total_gate_count": int(params.get("total_gate_count", 9)),
        "noise_strength": float(params.get("noise_strength", 0.0)),
        # One-hot encoded algorithm
        "algo_bernstein_vazirani": 1 if algo == "bernstein_vazirani" else 0,
        "algo_deutsch_jozsa": 1 if algo == "deutsch_jozsa" else 0,
        "algo_simons": 1 if algo == "simons" else 0,
        # One-hot encoded noise type
        "noise_amplitude_damping": 1 if noise_t == "amplitude_damping" else 0,
        "noise_depolarizing": 1 if noise_t == "depolarizing" else 0,
        "noise_none": 1 if noise_t == "none" else 0,
        "noise_thermal_relaxation": 1 if noise_t == "thermal_relaxation" else 0,
    }

    # Order columns exactly matching training data
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    return df


def estimate_circuit_metrics(algorithm, n_qubits):
    """
    Generate the standard transpiled circuit statistics for a given algorithm and qubit count.
    """
    n_qubits = max(2, min(8, int(n_qubits)))
    algorithm = algorithm.lower()

    if algorithm == "deutsch_jozsa":
        oracle = build_dj_oracle(n_qubits, "balanced", seed=42)
        qc = build_dj_circuit(n_qubits, oracle)
    elif algorithm == "bernstein_vazirani":
        # Alternating secret string
        secret = "".join("1" if i % 2 == 0 else "0" for i in range(n_qubits))
        qc = build_bv_circuit(n_qubits, secret)
    elif algorithm == "simons":
        # Period string
        secret = "1" + "0" * (n_qubits - 1)
        qc = build_simon_circuit(n_qubits, secret)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    stats = get_circuit_stats(qc)
    return stats


# -------------------------------------------------------------
# STATIC ROUTES
# -------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(STATIC_DIR, path)


# -------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------
@app.route("/api/ranges", methods=["GET"])
def get_ranges():
    """Return dataset metadata, boundaries, and validation parameters."""
    return jsonify({
        "status": "success",
        "ranges": VALID_RANGES,
        "presets": BENCHMARK_PRESETS,
    })


@app.route("/api/circuit_estimate", methods=["GET"])
def get_circuit_estimate():
    """Estimate typical circuit metrics for algorithm and qubit count."""
    algo = request.args.get("algorithm", "deutsch_jozsa")
    qubits = request.args.get("n_qubits", 3)

    try:
        qubits = int(qubits)
        metrics = estimate_circuit_metrics(algo, qubits)
        return jsonify({
            "status": "success",
            "algorithm": algo,
            "n_qubits": qubits,
            "metrics": metrics,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/validate_circuit", methods=["GET", "POST"])
def validate_circuit():
    """
    Validate whether the input parameters are physically and logically
    valid to construct a real quantum circuit.
    """
    if request.method == "POST":
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify({"status": "error", "message": "Invalid JSON body"}), 400
    else:
        data = request.args.to_dict()

    algo = data.get("algorithm", "deutsch_jozsa")
    n_qubits = data.get("n_qubits", 3)
    depth = data.get("circuit_depth")
    g1 = data.get("gate_count_1q")
    g2 = data.get("gate_count_2q")
    g_total = data.get("total_gate_count")
    noise_t = data.get("noise_type", "none")
    noise_str = data.get("noise_strength", 0.0)

    res = validate_circuit_parameters(
        algorithm=algo,
        n_qubits=n_qubits,
        circuit_depth=depth,
        gate_count_1q=g1,
        gate_count_2q=g2,
        total_gate_count=g_total,
        noise_type=noise_t,
        noise_strength=noise_str,
    )
    return jsonify({
        "status": "success",
        "validation": res,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict quantum circuit success probability and compute confidence metrics.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON body"}), 400

    # Extract inputs
    algorithm = data.get("algorithm", "deutsch_jozsa")
    n_qubits = int(data.get("n_qubits", 3))
    circuit_depth = int(data.get("circuit_depth", 5))
    gate_count_1q = int(data.get("gate_count_1q", 7))
    gate_count_2q = int(data.get("gate_count_2q", 2))
    total_gate_count = int(data.get("total_gate_count", gate_count_1q + gate_count_2q))
    noise_type = data.get("noise_type", "none")
    noise_strength = float(data.get("noise_strength", 0.0))

    # Validate whether parameters are valid to construct a real quantum circuit
    validation = validate_circuit_parameters(
        algorithm=algorithm,
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        gate_count_1q=gate_count_1q,
        gate_count_2q=gate_count_2q,
        total_gate_count=total_gate_count,
        noise_type=noise_type,
        noise_strength=noise_strength,
    )

    if not validation["is_valid"]:
        return jsonify({
            "status": "error",
            "error_type": "circuit_validation_failed",
            "message": "Parameters cannot construct a valid quantum circuit: " + "; ".join(validation["errors"]),
            "validation_errors": validation["errors"],
            "warnings": validation["warnings"],
            "details": validation["details"],
        }), 400

    warnings = list(validation["warnings"])

    # Build input feature matrix
    params = {
        "algorithm": algorithm,
        "n_qubits": n_qubits,
        "circuit_depth": circuit_depth,
        "gate_count_1q": gate_count_1q,
        "gate_count_2q": gate_count_2q,
        "total_gate_count": total_gate_count,
        "noise_type": noise_type,
        "noise_strength": noise_strength,
    }

    X = build_feature_vector(params)

    # Predictions
    rf_pred = float(rf_model.predict(X)[0])
    dt_pred = float(dt_model.predict(X)[0])

    # Clamp predictions to physical quantum bounds [0.0, 1.0]
    rf_pred_clamped = max(0.0, min(1.0, rf_pred))
    dt_pred_clamped = max(0.0, min(1.0, dt_pred))

    # Ensemble tree analysis across the Random Forest (e.g. 100/200 trees)
    # Using X.values avoids scikit-learn feature name warnings
    tree_preds = np.array([tree.predict(X.values)[0] for tree in rf_model.estimators_])
    tree_preds_clamped = np.clip(tree_preds, 0.0, 1.0)

    mean_tree = float(np.mean(tree_preds_clamped))
    std_tree = float(np.std(tree_preds_clamped))
    min_tree = float(np.min(tree_preds_clamped))
    max_tree = float(np.max(tree_preds_clamped))

    # 95% Confidence Interval (± 1.96 * sigma)
    ci_margin = 1.96 * std_tree
    ci_lower = max(0.0, float(rf_pred_clamped - ci_margin))
    ci_upper = min(1.0, float(rf_pred_clamped + ci_margin))

    # Tree Consensus Confidence Score (0% to 100%)
    # std_tree in dataset ranges from 0.0 to 0.162 (mean ~0.013)
    # A standard deviation of 0.01 gives ~97.5% confidence
    confidence_score = max(0.0, min(1.0, 1.0 - (2.5 * std_tree))) * 100.0

    # Categorical confidence level rating
    if confidence_score >= 95.0:
        confidence_level = "Very High Confidence"
        confidence_color = "#10b981"  # Emerald
    elif confidence_score >= 90.0:
        confidence_level = "High Confidence"
        confidence_color = "#00f5d4"  # Cyan
    elif confidence_score >= 80.0:
        confidence_level = "Moderate Confidence"
        confidence_color = "#f59e0b"  # Amber
    else:
        confidence_level = "Low Confidence (Elevated Variance)"
        confidence_color = "#ef4444"  # Crimson

    # Success probability assessment
    if dt_pred_clamped >= 0.95:
        success_status = "Very High Success Probability"
        success_color = "#10b981"
        success_desc = "Near-ideal algorithm execution with high probability of measuring the correct state."
    elif dt_pred_clamped >= 0.85:
        success_status = "High Success Probability"
        success_color = "#00f5d4"
        success_desc = "Reliable algorithm execution with low noise impact on output bitstrings."
    elif dt_pred_clamped >= 0.70:
        success_status = "Moderate Success Probability"
        success_color = "#f59e0b"
        success_desc = "Noticeable noise impact. Probability of erroneous measurement is elevated."
    else:
        success_status = "Low Success Probability"
        success_color = "#ef4444"
        success_desc = "High noise degradation. Measurements are dispersed toward random bitstrings."

    # Distribution histogram of trees (10 bins for visualization)
    hist, bin_edges = np.histogram(tree_preds_clamped, bins=10, range=(0.0, 1.0))
    distribution_bins = [
        {
            "bin_start": round(float(bin_edges[i]), 3),
            "bin_end": round(float(bin_edges[i+1]), 3),
            "count": int(hist[i]),
            "percentage": round(float(hist[i] / len(tree_preds_clamped) * 100), 1),
        }
        for i in range(len(hist))
    ]

    # Circuit density metrics
    depth_density = round(total_gate_count / max(1, circuit_depth), 2)
    entanglement_ratio = round(gate_count_2q / max(1, total_gate_count) * 100, 1)

    return jsonify({
        "status": "success",
        "inputs": params,
        "prediction": {
            "success_probability": round(rf_pred_clamped, 6),
            "success_probability_pct": round(rf_pred_clamped * 100, 2),
            "decision_tree_probability": round(dt_pred_clamped, 6),
            "decision_tree_probability_pct": round(dt_pred_clamped * 100, 2),
        },
        "confidence": {
            "score": round(confidence_score, 2),
            "level": confidence_level,
            "color": confidence_color,
            "std_error": round(std_tree, 5),
            "ci_95": {
                "lower": round(ci_lower, 5),
                "upper": round(ci_upper, 5),
                "lower_pct": round(ci_lower * 100, 2),
                "upper_pct": round(ci_upper * 100, 2),
                "margin_pct": round(ci_margin * 100, 2),
            },
            "tree_stats": {
                "total_trees": len(tree_preds),
                "min": round(min_tree, 4),
                "max": round(max_tree, 4),
                "mean": round(mean_tree, 4),
            },
            "distribution": distribution_bins,
        },
        "success_assessment": {
            "status": success_status,
            "color": success_color,
            "description": success_desc,
        },
        "fidelity": {
            "status": success_status,
            "color": success_color,
            "description": success_desc,
        },
        "circuit_telemetry": {
            "depth_density": depth_density,
            "entanglement_ratio_pct": entanglement_ratio,
        },
        "warnings": warnings,
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Execute real Qiskit Aer simulation with 4096 shots to verify
    the classical ML prediction against real quantum execution.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid JSON body"}), 400

    algorithm = data.get("algorithm", "deutsch_jozsa").lower()
    n_qubits = max(2, min(8, int(data.get("n_qubits", 3))))
    noise_type = data.get("noise_type", "none").lower()
    noise_strength = float(data.get("noise_strength", 0.0))

    try:
        # Build circuit
        if algorithm == "deutsch_jozsa":
            oracle = build_dj_oracle(n_qubits, "balanced", seed=42)
            qc = build_dj_circuit(n_qubits, oracle)
            expected = "non-zero"
            secret = None
        elif algorithm == "bernstein_vazirani":
            secret = "".join("1" if i % 2 == 0 else "0" for i in range(n_qubits))
            qc = build_bv_circuit(n_qubits, secret)
            expected = secret
        elif algorithm == "simons":
            secret = "1" + "0" * (n_qubits - 1)
            qc = build_simon_circuit(n_qubits, secret)
            expected = secret
        else:
            return jsonify({"status": "error", "message": f"Unsupported algorithm: {algorithm}"}), 400

        # Build noise model
        noise_model = None
        if noise_type == "depolarizing" and noise_strength > 0:
            noise_model = get_depolarizing_model(noise_strength)
        elif noise_type == "amplitude_damping" and noise_strength > 0:
            noise_model = get_amplitude_damping_model(noise_strength)
        elif noise_type == "thermal_relaxation" and noise_strength > 0:
            # 1 / noise_strength gives T1 in microsec
            t1 = max(5.0, 1.0 / max(0.001, noise_strength))
            t2 = t1 * 0.8
            noise_model = get_thermal_relaxation_model(t1, t2)

        # Execute simulation
        counts = run_circuit(qc, noise_model=noise_model, shots=4096, seed=42)

        # Compute probability
        sim_prob = compute_success_probability(
            counts=counts,
            expected_output=expected,
            algorithm=algorithm,
            n_qubits=n_qubits,
            secret_string=secret,
        )

        # Top 5 measurement bitstrings
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_counts = [{"state": k, "count": v, "pct": round(v / 4096 * 100, 2)} for k, v in sorted_counts]

        return jsonify({
            "status": "success",
            "simulated_success_probability": round(float(sim_prob), 6),
            "simulated_success_probability_pct": round(float(sim_prob) * 100, 2),
            "shots": 4096,
            "top_measurements": top_counts,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Return model performance metrics comparison."""
    comp_path = os.path.join(PROJECT_ROOT, "outputs", "model_comparison.csv")
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        data = df_comp.to_dict(orient="records")
    else:
        data = [
            {"Model": "Decision Tree", "R²": 0.9693, "MAE": 0.0070, "RMSE": 0.0200},
            {"Model": "Random Forest", "R²": 0.9675, "MAE": 0.0094, "RMSE": 0.0206},
        ]
    return jsonify({"status": "success", "models": data})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================")
    print(f" Quantum Success Prediction Server")
    print(f" Running locally at: http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
