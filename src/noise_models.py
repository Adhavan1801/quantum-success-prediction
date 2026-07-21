# =============================================================
# noise_models.py — Noise Model Configurations
# =============================================================
"""
Create and configure Qiskit Aer noise models for simulation:
    - No noise (ideal)
    - Depolarizing noise
    - Amplitude damping
    - Thermal relaxation

Each function returns a qiskit_aer.noise.NoiseModel ready for use
with the AerSimulator.
"""

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    thermal_relaxation_error,
)


def get_depolarizing_model(error_rate):
    """
    Create a depolarizing noise model.

    Depolarizing noise randomly replaces a qubit's state with the
    maximally mixed state with probability `error_rate`.

    Parameters
    ----------
    error_rate : float
        Probability of depolarization per gate (0 to 1).

    Returns
    -------
    NoiseModel
        Noise model with depolarizing error on all single-qubit
        and two-qubit gates.
    """
    noise_model = NoiseModel()

    # Single-qubit depolarizing error
    error_1q = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3", "h", "x"])

    # Two-qubit depolarizing error (typically higher)
    error_2q = depolarizing_error(error_rate * 2, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model


def get_amplitude_damping_model(gamma):
    """
    Create an amplitude damping noise model.

    Amplitude damping simulates energy loss: a qubit in state |1⟩
    decays to |0⟩ with probability gamma. This models the physical
    T1 relaxation process.

    Parameters
    ----------
    gamma : float
        Damping parameter (0 to 1). Higher = more damping.

    Returns
    -------
    NoiseModel
        Noise model with amplitude damping on all gates.
    """
    noise_model = NoiseModel()

    # Single-qubit amplitude damping
    error_1q = amplitude_damping_error(gamma)
    noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3", "h", "x"])

    # For two-qubit gates, apply damping to each qubit independently
    error_2q = amplitude_damping_error(gamma).tensor(
        amplitude_damping_error(gamma)
    )
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model


def get_thermal_relaxation_model(t1_us, t2_us, gate_time_ns=50):
    """
    Create a thermal relaxation noise model.

    This is the most realistic noise model, combining both amplitude
    damping (T1) and phase damping (T2). It uses physical parameters
    from real quantum hardware.

    Parameters
    ----------
    t1_us : float
        T1 relaxation time in microseconds.
    t2_us : float
        T2 coherence time in microseconds. Must satisfy T2 <= 2*T1.
    gate_time_ns : float
        Gate execution time in nanoseconds (default: 50 ns).

    Returns
    -------
    NoiseModel
        Noise model with thermal relaxation on all gates.
    """
    noise_model = NoiseModel()

    # Convert units to nanoseconds for consistency
    t1_ns = t1_us * 1000
    t2_ns = t2_us * 1000

    # Ensure physical constraint: T2 <= 2*T1
    t2_ns = min(t2_ns, 2 * t1_ns)

    # Single-qubit thermal relaxation error
    error_1q = thermal_relaxation_error(t1_ns, t2_ns, gate_time_ns)
    noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3", "h", "x"])

    # Two-qubit gate typically takes longer (~300-500 ns)
    cx_gate_time_ns = gate_time_ns * 6
    error_2q = thermal_relaxation_error(t1_ns, t2_ns, cx_gate_time_ns).tensor(
        thermal_relaxation_error(t1_ns, t2_ns, cx_gate_time_ns)
    )
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model


def get_all_noise_configs():
    """
    Generate the complete set of noise configurations for the sweep.

    Returns a list of dictionaries, each describing one noise configuration.

    Returns
    -------
    list of dict
        Each dict has keys:
            - noise_type : str ("none", "depolarizing", "amplitude_damping",
                               "thermal_relaxation")
            - noise_strength : float (the key parameter value)
            - noise_model : NoiseModel or None
            - description : str (human-readable description)
    """
    configs = []

    # --- 1. No noise (ideal) ---
    configs.append({
        "noise_type": "none",
        "noise_strength": 0.0,
        "noise_model": None,
        "description": "Ideal (no noise)",
    })

    # --- 2. Depolarizing noise ---
    depol_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    for rate in depol_rates:
        configs.append({
            "noise_type": "depolarizing",
            "noise_strength": rate,
            "noise_model": get_depolarizing_model(rate),
            "description": f"Depolarizing (p={rate})",
        })

    # --- 3. Amplitude damping ---
    gamma_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    for gamma in gamma_values:
        configs.append({
            "noise_type": "amplitude_damping",
            "noise_strength": gamma,
            "noise_model": get_amplitude_damping_model(gamma),
            "description": f"Amplitude damping (γ={gamma})",
        })

    # --- 4. Thermal relaxation ---
    # Parameterized by T1 (in microseconds); T2 is set to T1 * 0.8
    # Shorter T1 = more noise; strength = 1/T1 for consistency
    thermal_configs = [
        {"t1": 100, "t2": 80,  "label": "loose",    "strength": 0.01},
        {"t1": 50,  "t2": 40,  "label": "moderate",  "strength": 0.02},
        {"t1": 30,  "t2": 24,  "label": "tight",     "strength": 0.033},
        {"t1": 20,  "t2": 16,  "label": "strong",    "strength": 0.05},
        {"t1": 10,  "t2": 8,   "label": "very_strong", "strength": 0.1},
    ]
    for tc in thermal_configs:
        configs.append({
            "noise_type": "thermal_relaxation",
            "noise_strength": tc["strength"],
            "noise_model": get_thermal_relaxation_model(tc["t1"], tc["t2"]),
            "description": f"Thermal relaxation (T1={tc['t1']}μs, {tc['label']})",
        })

    return configs


def describe_noise_configs():
    """
    Print a summary table of all noise configurations.

    Useful for documentation and notebook markdown cells.
    """
    configs = get_all_noise_configs()
    print(f"{'#':<4} {'Type':<22} {'Strength':<12} {'Description'}")
    print("-" * 70)
    for i, cfg in enumerate(configs):
        print(f"{i:<4} {cfg['noise_type']:<22} {cfg['noise_strength']:<12.4f} "
              f"{cfg['description']}")
    print(f"\nTotal configurations: {len(configs)}")
