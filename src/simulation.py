# =============================================================
# simulation.py — Circuit Execution and Success Measurement
# =============================================================
"""
Run quantum circuits with optional noise models and measure
success probability. Includes both single-experiment and
full parameter sweep functions.
"""

import time
from tqdm import tqdm
from qiskit import transpile
from qiskit_aer import AerSimulator

from src.circuits import (
    generate_all_circuit_configs,
    get_circuit_stats,
    check_simon_success,
)
from src.noise_models import get_all_noise_configs


def run_circuit(circuit, noise_model=None, shots=4096, seed=42):
    """
    Execute a quantum circuit on the Aer simulator.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to execute.
    noise_model : NoiseModel or None
        Noise model to apply. None = ideal simulation.
    shots : int
        Number of measurement shots.
    seed : int
        Simulator seed for reproducibility.

    Returns
    -------
    dict
        Measurement counts, e.g., {"00": 2048, "11": 2048}.
    """
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()

    transpiled = transpile(circuit, simulator)
    result = simulator.run(transpiled, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()

    return counts


def compute_success_probability(counts, expected_output, algorithm, n_qubits=None,
                                 secret_string=None):
    """
    Compute the fraction of shots that produced the correct output.

    Parameters
    ----------
    counts : dict
        Measurement counts from circuit execution.
    expected_output : str
        The expected correct bitstring.
        For DJ balanced oracles, use "non-zero" to check for any non-zero result.
    algorithm : str
        Algorithm name ("deutsch_jozsa", "bernstein_vazirani", "simons").
    n_qubits : int, optional
        Number of qubits (needed for Simon's).
    secret_string : str, optional
        Secret string (needed for Simon's).

    Returns
    -------
    float
        Success probability between 0.0 and 1.0.
    """
    total_shots = sum(counts.values())

    if algorithm == "deutsch_jozsa":
        if expected_output == "non-zero":
            # Balanced oracle: success = NOT measuring all zeros
            zero_string = "0" * len(list(counts.keys())[0])
            zero_count = counts.get(zero_string, 0)
            return (total_shots - zero_count) / total_shots
        else:
            # Constant oracle: success = measuring all zeros
            correct_count = counts.get(expected_output, 0)
            return correct_count / total_shots

    elif algorithm == "bernstein_vazirani":
        # BV: success = measuring the exact secret string
        correct_count = counts.get(expected_output, 0)
        return correct_count / total_shots

    elif algorithm == "simons":
        # Simon's: success = each measurement z satisfies z · s = 0 (mod 2)
        return check_simon_success(counts, secret_string or expected_output,
                                    n_qubits or len(expected_output))

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_sweep(qubit_range=range(2, 9), shots=4096, seed=42):
    """
    Run the full parameter sweep across all algorithms and noise configs.

    This is the main simulation function. For each combination of:
        - Algorithm (DJ, BV, Simon's)
        - Qubit count (2 to 8)
        - Oracle instance (multiple per algorithm/qubit)
        - Noise configuration (none + 3 types × multiple strengths)

    It runs the circuit and records the result.

    Parameters
    ----------
    qubit_range : range
        Range of qubit counts to sweep.
    shots : int
        Number of shots per circuit execution.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Each dict contains all circuit parameters, noise parameters,
        and the measured success probability.
    """
    print("=" * 60)
    print("QUANTUM NOISE SIMULATION SWEEP")
    print("=" * 60)

    # Generate all circuit configurations
    print("\n[1/3] Generating circuit configurations...")
    circuit_configs = generate_all_circuit_configs(qubit_range)
    print(f"  → {len(circuit_configs)} circuit configurations generated")

    # Generate all noise configurations
    print("\n[2/3] Loading noise configurations...")
    noise_configs = get_all_noise_configs()
    print(f"  → {len(noise_configs)} noise configurations loaded")

    total_experiments = len(circuit_configs) * len(noise_configs)
    print(f"\n[3/3] Running {total_experiments} experiments "
          f"({len(circuit_configs)} circuits × {len(noise_configs)} noise configs)")
    print(f"  → Shots per experiment: {shots}")
    print(f"  → Estimated time: {total_experiments * 0.3:.0f}–{total_experiments * 0.8:.0f} seconds")
    print()

    results = []
    start_time = time.time()

    for config in tqdm(circuit_configs, desc="Circuits", unit="circuit"):
        circuit = config["circuit"]

        # Get transpiled circuit stats (same for all noise configs of this circuit)
        stats = get_circuit_stats(circuit)

        for noise_cfg in noise_configs:
            # Run the circuit with this noise configuration
            counts = run_circuit(
                circuit,
                noise_model=noise_cfg["noise_model"],
                shots=shots,
                seed=seed,
            )

            # Compute success probability
            success_prob = compute_success_probability(
                counts=counts,
                expected_output=config["expected_output"],
                algorithm=config["algorithm"],
                n_qubits=config["n_qubits"],
                secret_string=config.get("expected_output"),
            )

            # Record the result
            results.append({
                # Circuit parameters
                "algorithm": config["algorithm"],
                "n_qubits": config["n_qubits"],
                "oracle_desc": config["oracle_desc"],
                "circuit_depth": stats["circuit_depth"],
                "gate_count_1q": stats["gate_count_1q"],
                "gate_count_2q": stats["gate_count_2q"],
                "total_gate_count": stats["total_gate_count"],

                # Noise parameters
                "noise_type": noise_cfg["noise_type"],
                "noise_strength": noise_cfg["noise_strength"],

                # Target variable
                "success_probability": success_prob,
            })

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE — {len(results)} results in {elapsed:.1f} seconds")
    print(f"{'=' * 60}")

    return results
