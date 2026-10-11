# =============================================================
# circuits.py — Quantum Circuit Builders
# =============================================================
"""
Build oracle circuits and full algorithm circuits for:
    - Deutsch-Jozsa (DJ)
    - Bernstein-Vazirani (BV)
    - Simon's Algorithm

Each function returns a Qiskit QuantumCircuit ready for simulation.
"""

import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# =============================================================
# DEUTSCH-JOZSA
# =============================================================

def build_dj_oracle(n_qubits, oracle_type="balanced", seed=None):
    """
    Build a Deutsch-Jozsa oracle circuit.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits (total circuit has n_qubits + 1 for ancilla).
    oracle_type : str
        "constant_0", "constant_1", or "balanced".
    seed : int, optional
        Random seed for reproducible balanced oracles.

    Returns
    -------
    QuantumCircuit
        The oracle as a quantum circuit on n_qubits + 1 qubits.
    """
    total_qubits = n_qubits + 1  # +1 for ancilla (output) qubit
    oracle = QuantumCircuit(total_qubits, name=f"DJ_Oracle_{oracle_type}")

    if oracle_type == "constant_0":
        # f(x) = 0 for all x → oracle does nothing
        pass

    elif oracle_type == "constant_1":
        # f(x) = 1 for all x → flip the ancilla unconditionally
        oracle.x(n_qubits)

    elif oracle_type == "balanced":
        # f(x) = 1 for exactly half the inputs
        # Strategy: apply CNOT from a random subset of input qubits to ancilla,
        # optionally preceded by X gates to vary which half is "1"
        rng = random.Random(seed)

        # Randomly flip some input qubits to create variety
        for i in range(n_qubits):
            if rng.random() < 0.5:
                oracle.x(i)

        # Apply CNOT from each input qubit to the ancilla
        for i in range(n_qubits):
            oracle.cx(i, n_qubits)

        # Un-flip the input qubits (so oracle is self-contained)
        for i in range(n_qubits):
            if rng.random() < 0.5:
                oracle.x(i)
    else:
        raise ValueError(f"Unknown oracle_type: {oracle_type}")

    return oracle


def build_dj_circuit(n_qubits, oracle):
    """
    Build the full Deutsch-Jozsa circuit.

    Steps:
        1. Initialize ancilla to |1⟩
        2. Apply Hadamard to all qubits
        3. Apply oracle
        4. Apply Hadamard to input qubits
        5. Measure input qubits

    Parameters
    ----------
    n_qubits : int
        Number of input qubits.
    oracle : QuantumCircuit
        The DJ oracle circuit.

    Returns
    -------
    QuantumCircuit
        Complete DJ circuit with measurements.
    """
    total_qubits = n_qubits + 1
    circuit = QuantumCircuit(total_qubits, n_qubits)

    # Step 1: Initialize ancilla to |1⟩
    circuit.x(n_qubits)

    # Step 2: Apply Hadamard to ALL qubits (input + ancilla)
    for i in range(total_qubits):
        circuit.h(i)

    # Step 3: Apply the oracle
    circuit.barrier()
    circuit.compose(oracle, inplace=True)
    circuit.barrier()

    # Step 4: Apply Hadamard to input qubits only
    for i in range(n_qubits):
        circuit.h(i)

    # Step 5: Measure input qubits
    circuit.measure(range(n_qubits), range(n_qubits))

    return circuit


def get_dj_expected_output(n_qubits, oracle_type):
    """
    Return the expected measurement outcome for DJ.

    - Constant oracle → all zeros "000...0"
    - Balanced oracle → any non-zero string (we check for NOT all zeros)

    Returns
    -------
    str
        Expected bitstring, or "non-zero" for balanced.
    """
    if oracle_type.startswith("constant"):
        return "0" * n_qubits
    else:
        return "non-zero"


# =============================================================
# BERNSTEIN-VAZIRANI
# =============================================================

def build_bv_oracle(n_qubits, secret_string):
    """
    Build a Bernstein-Vazirani oracle for a given secret string.

    The oracle computes f(x) = s · x (mod 2), where s is the secret string.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits.
    secret_string : str
        The hidden bitstring, e.g., "101".

    Returns
    -------
    QuantumCircuit
        The BV oracle circuit on n_qubits + 1 qubits.
    """
    assert len(secret_string) == n_qubits, \
        f"Secret string length {len(secret_string)} != n_qubits {n_qubits}"

    total_qubits = n_qubits + 1
    oracle = QuantumCircuit(total_qubits, name=f"BV_Oracle_{secret_string}")

    # For each bit in the secret string that is '1',
    # apply CNOT from that input qubit to the ancilla
    # Note: Qiskit uses little-endian ordering, so we reverse
    for i, bit in enumerate(reversed(secret_string)):
        if bit == "1":
            oracle.cx(i, n_qubits)

    return oracle


def build_bv_circuit(n_qubits, secret_string):
    """
    Build the full Bernstein-Vazirani circuit.

    Steps:
        1. Initialize ancilla to |1⟩
        2. Apply Hadamard to all qubits
        3. Apply oracle
        4. Apply Hadamard to input qubits
        5. Measure input qubits

    Parameters
    ----------
    n_qubits : int
        Number of input qubits.
    secret_string : str
        The hidden bitstring.

    Returns
    -------
    QuantumCircuit
        Complete BV circuit with measurements.
    """
    total_qubits = n_qubits + 1
    circuit = QuantumCircuit(total_qubits, n_qubits)

    # Step 1: Initialize ancilla to |1⟩
    circuit.x(n_qubits)

    # Step 2: Apply Hadamard to ALL qubits
    for i in range(total_qubits):
        circuit.h(i)

    # Step 3: Apply the oracle
    circuit.barrier()
    oracle = build_bv_oracle(n_qubits, secret_string)
    circuit.compose(oracle, inplace=True)
    circuit.barrier()

    # Step 4: Apply Hadamard to input qubits
    for i in range(n_qubits):
        circuit.h(i)

    # Step 5: Measure input qubits
    circuit.measure(range(n_qubits), range(n_qubits))

    return circuit


def get_bv_expected_output(secret_string):
    """
    Return the expected measurement outcome for BV.

    The BV algorithm should recover the secret string exactly.

    Returns
    -------
    str
        The secret bitstring.
    """
    return secret_string


# =============================================================
# SIMON'S ALGORITHM
# =============================================================

def build_simon_oracle(n_qubits, secret_string):
    """
    Build a Simon's algorithm oracle for a given secret string.

    Creates a 2-to-1 function f such that f(x) = f(y) iff x ⊕ y = s,
    where s is the secret string.

    For s = "00...0", the function is 1-to-1 (identity-like).
    For s ≠ "00...0", we build a proper 2-to-1 mapping.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits (circuit uses 2*n_qubits total).
    secret_string : str
        The hidden period string, e.g., "110".

    Returns
    -------
    QuantumCircuit
        The Simon's oracle on 2 * n_qubits qubits.
    """
    assert len(secret_string) == n_qubits, \
        f"Secret string length {len(secret_string)} != n_qubits {n_qubits}"

    total_qubits = 2 * n_qubits
    oracle = QuantumCircuit(total_qubits, name=f"Simon_Oracle_{secret_string}")

    # Step 1: Copy input register to output register (f(x) = x initially)
    for i in range(n_qubits):
        oracle.cx(i, n_qubits + i)

    # Step 2: If secret string is non-trivial, make it 2-to-1
    # Find the highest bit position where s has a '1'
    s_reversed = secret_string[::-1]  # Little-endian for Qiskit
    one_positions = [i for i, bit in enumerate(s_reversed) if bit == "1"]

    if one_positions:
        # Use the first '1' position as the control
        control_qubit = one_positions[0]

        # When the control qubit is |1⟩, XOR the output with s
        for i in one_positions:
            oracle.cx(control_qubit, n_qubits + i)

    return oracle


def build_simon_circuit(n_qubits, secret_string):
    """
    Build one iteration of Simon's circuit.

    Steps:
        1. Apply Hadamard to input register
        2. Apply oracle
        3. Apply Hadamard to input register
        4. Measure input register

    Note: Simon's algorithm needs multiple runs + classical post-processing
    to determine the secret string. Each run produces one equation.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits.
    secret_string : str
        The hidden period string.

    Returns
    -------
    QuantumCircuit
        Complete Simon's circuit with measurements on input register.
    """
    total_qubits = 2 * n_qubits
    circuit = QuantumCircuit(total_qubits, n_qubits)

    # Step 1: Apply Hadamard to input register
    for i in range(n_qubits):
        circuit.h(i)

    # Step 2: Apply the oracle
    circuit.barrier()
    oracle = build_simon_oracle(n_qubits, secret_string)
    circuit.compose(oracle, inplace=True)
    circuit.barrier()

    # Step 3: Apply Hadamard to input register
    for i in range(n_qubits):
        circuit.h(i)

    # Step 4: Measure input register only
    circuit.measure(range(n_qubits), range(n_qubits))

    return circuit


def check_simon_success(counts, secret_string, n_qubits):
    """
    Check if Simon's circuit outputs are consistent with the secret string.

    Each measurement result z should satisfy z · s = 0 (mod 2).

    Parameters
    ----------
    counts : dict
        Measurement counts from circuit execution.
    secret_string : str
        The hidden period string.
    n_qubits : int
        Number of input qubits.

    Returns
    -------
    float
        Fraction of measurement results satisfying z · s = 0 (mod 2).
    """
    s_bits = [int(b) for b in reversed(secret_string)]
    total_shots = sum(counts.values())
    correct_shots = 0

    for bitstring, count in counts.items():
        z_bits = [int(b) for b in reversed(bitstring)]

        # Ensure z and s have same length (pad if needed)
        while len(z_bits) < n_qubits:
            z_bits.append(0)

        # Check z · s = 0 (mod 2)
        dot_product = sum(z * s for z, s in zip(z_bits, s_bits)) % 2
        if dot_product == 0:
            correct_shots += count

    return correct_shots / total_shots


# =============================================================
# CIRCUIT GENERATION UTILITIES
# =============================================================

def generate_secret_strings(n_qubits, count=3, seed=42):
    """
    Generate random secret bitstrings for BV and Simon's.

    Parameters
    ----------
    n_qubits : int
        Length of each bitstring.
    count : int
        Number of strings to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of str
        List of random bitstrings.
    """
    rng = random.Random(seed + n_qubits)
    strings = set()

    while len(strings) < count:
        s = "".join(rng.choice("01") for _ in range(n_qubits))
        # Avoid all-zeros for Simon's (trivial case)
        if s != "0" * n_qubits:
            strings.add(s)

    return sorted(strings)


def generate_all_circuit_configs(qubit_range=range(2, 9)):
    """
    Generate all circuit configurations for the parameter sweep.

    For each qubit count:
        - DJ: 1 constant_0 + 1 constant_1 + 3 balanced oracles
        - BV: 3 random secret strings
        - Simon's: 3 random secret strings (non-trivial)

    Parameters
    ----------
    qubit_range : range
        Range of qubit counts to use.

    Returns
    -------
    list of dict
        Each dict has keys: algorithm, n_qubits, oracle_params, circuit, expected_output
    """
    configs = []

    for n in qubit_range:
        # --- Deutsch-Jozsa ---
        for otype in ["constant_0", "constant_1"]:
            oracle = build_dj_oracle(n, otype)
            circuit = build_dj_circuit(n, oracle)
            configs.append({
                "algorithm": "deutsch_jozsa",
                "n_qubits": n,
                "oracle_desc": otype,
                "circuit": circuit,
                "expected_output": get_dj_expected_output(n, otype),
            })

        for seed_i in range(3):
            oracle = build_dj_oracle(n, "balanced", seed=seed_i + n * 100)
            circuit = build_dj_circuit(n, oracle)
            configs.append({
                "algorithm": "deutsch_jozsa",
                "n_qubits": n,
                "oracle_desc": f"balanced_{seed_i}",
                "circuit": circuit,
                "expected_output": get_dj_expected_output(n, "balanced"),
            })

        # --- Bernstein-Vazirani ---
        bv_secrets = generate_secret_strings(n, count=3, seed=42)
        for secret in bv_secrets:
            circuit = build_bv_circuit(n, secret)
            configs.append({
                "algorithm": "bernstein_vazirani",
                "n_qubits": n,
                "oracle_desc": f"secret_{secret}",
                "circuit": circuit,
                "expected_output": get_bv_expected_output(secret),
            })

        # --- Simon's Algorithm ---
        simon_secrets = generate_secret_strings(n, count=3, seed=99)
        for secret in simon_secrets:
            circuit = build_simon_circuit(n, secret)
            configs.append({
                "algorithm": "simons",
                "n_qubits": n,
                "oracle_desc": f"period_{secret}",
                "circuit": circuit,
                "expected_output": secret,  # The hidden period
            })

    return configs


def get_circuit_stats(circuit):
    """
    Get transpiled circuit statistics.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to analyze.

    Returns
    -------
    dict
        Keys: circuit_depth, gate_count_1q, gate_count_2q, total_gate_count
    """
    # Transpile to basis gates for consistent depth/gate counts
    transpiled = transpile(circuit, basis_gates=["u1", "u2", "u3", "cx"],
                           optimization_level=1)

    ops = transpiled.count_ops()
    gate_count_2q = ops.get("cx", 0)
    gate_count_1q = sum(v for k, v in ops.items()
                        if k not in ["cx", "measure", "barrier"])

    return {
        "circuit_depth": transpiled.depth(),
        "gate_count_1q": gate_count_1q,
        "gate_count_2q": gate_count_2q,
        "total_gate_count": gate_count_1q + gate_count_2q,
    }


# =============================================================
# CIRCUIT VALIDATION & CONSTRUCTIBILITY
# =============================================================

def validate_circuit_parameters(
    algorithm,
    n_qubits,
    circuit_depth=None,
    gate_count_1q=None,
    gate_count_2q=None,
    total_gate_count=None,
    noise_type="none",
    noise_strength=0.0,
):
    """
    Validate whether the input parameters are physically and logically
    valid to construct a real quantum circuit for the given algorithm.

    Checks:
        1. Algorithm support ('deutsch_jozsa', 'bernstein_vazirani', 'simons')
        2. Qubit count bounds and register sizing (ancilla requirements)
        3. Gate non-negativity and total gate count consistency (1Q + 2Q = total)
        4. Minimum gate requirements for algorithm interference & oracle realizability
        5. Physical scheduling feasibility:
           - Depth cannot exceed total gates
           - Total gates cannot exceed Depth * Total Qubits
           - 2Q gates cannot exceed Depth * floor(Total Qubits / 2)
        6. Noise channel physical boundaries

    Returns
    -------
    dict
        {
            'is_valid': bool,
            'constructible': bool,
            'errors': list of str,
            'warnings': list of str,
            'details': dict
        }
    """
    errors = []
    warnings = []

    # 1. Algorithm validation
    valid_algos = ["deutsch_jozsa", "bernstein_vazirani", "simons"]
    algo = str(algorithm).strip().lower()
    if algo not in valid_algos:
        errors.append(f"Invalid algorithm '{algorithm}'. Must be one of: {', '.join(valid_algos)}.")
        return {
            "is_valid": False,
            "constructible": False,
            "errors": errors,
            "warnings": warnings,
            "details": {},
        }

    # 2. Qubit count validation
    try:
        n_qubits = int(n_qubits)
    except (ValueError, TypeError):
        errors.append("Number of qubits must be an integer.")
        return {
            "is_valid": False,
            "constructible": False,
            "errors": errors,
            "warnings": warnings,
            "details": {},
        }

    if n_qubits < 1:
        errors.append(f"Qubit count must be at least 1, received {n_qubits}.")
    elif algo == "simons" and n_qubits < 2:
        errors.append(f"Simon's algorithm requires at least 2 input qubits to construct a non-trivial period, received {n_qubits}.")
    elif n_qubits < 2 or n_qubits > 8:
        warnings.append(f"Qubit count ({n_qubits}) is outside tested benchmark range (2–8).")

    # Total physical qubits in circuit
    # DJ & BV: n input qubits + 1 ancilla qubit = n + 1
    # Simon's: n input qubits + n output/ancilla qubits = 2 * n
    total_qubits = (n_qubits + 1) if algo in ["deutsch_jozsa", "bernstein_vazirani"] else (2 * n_qubits)

    # 3. Gate Counts & Depth validation (if provided)
    if circuit_depth is not None:
        try:
            circuit_depth = int(circuit_depth)
            if circuit_depth < 1:
                errors.append(f"Circuit depth must be at least 1, received {circuit_depth}.")
        except (ValueError, TypeError):
            errors.append("Circuit depth must be an integer.")

    if gate_count_1q is not None:
        try:
            gate_count_1q = int(gate_count_1q)
            if gate_count_1q < 0:
                errors.append(f"Single-qubit gate count cannot be negative, received {gate_count_1q}.")
        except (ValueError, TypeError):
            errors.append("Single-qubit gate count must be an integer.")

    if gate_count_2q is not None:
        try:
            gate_count_2q = int(gate_count_2q)
            if gate_count_2q < 0:
                errors.append(f"Two-qubit gate count cannot be negative, received {gate_count_2q}.")
        except (ValueError, TypeError):
            errors.append("Two-qubit gate count must be an integer.")

    if total_gate_count is not None:
        try:
            total_gate_count = int(total_gate_count)
            if total_gate_count < 0:
                errors.append(f"Total gate count cannot be negative, received {total_gate_count}.")
        except (ValueError, TypeError):
            errors.append("Total gate count must be an integer.")

    # Consistency between gate counts
    if gate_count_1q is not None and gate_count_2q is not None:
        computed_total = gate_count_1q + gate_count_2q
        if total_gate_count is not None and total_gate_count != computed_total:
            errors.append(
                f"Total gate count ({total_gate_count}) is inconsistent with 1Q ({gate_count_1q}) + 2Q ({gate_count_2q}) = {computed_total}."
            )
        else:
            total_gate_count = computed_total

    # 4. Constructibility & Minimal Gate Requirements for Oracle/Interference
    if algo in ["deutsch_jozsa", "bernstein_vazirani"]:
        # Ancilla initialization (X + H = 2) + input initialization (n H) + measurement interference (n H) = 2n + 2
        min_1q_required = 2 * n_qubits + 2
        # Non-trivial oracle requires at least 1 CNOT from input to ancilla
        min_2q_required = 1
    elif algo == "simons":
        # Input superposition (n H) + measurement interference (n H) = 2n
        min_1q_required = 2 * n_qubits
        # Copy input register to output register requires n CNOT gates
        min_2q_required = n_qubits
    else:
        min_1q_required = 0
        min_2q_required = 0

    if gate_count_1q is not None and gate_count_1q < min_1q_required:
        errors.append(
            f"Insufficient 1-qubit gates for {algo.replace('_', ' ').title()}: Requires at least {min_1q_required} "
            f"gates for state initialization & interference, but only {gate_count_1q} specified."
        )

    if gate_count_2q is not None and gate_count_2q < min_2q_required:
        errors.append(
            f"Insufficient 2-qubit gates for {algo.replace('_', ' ').title()}: Requires at least {min_2q_required} "
            f"CNOT gates to construct the oracle, but only {gate_count_2q} specified."
        )

    if gate_count_2q is not None and gate_count_2q > 0 and total_qubits < 2:
        errors.append("Two-qubit gates cannot be constructed on a circuit with fewer than 2 total qubits.")

    # 5. Physical Scheduling & Depth Constraints
    if circuit_depth is not None and total_gate_count is not None:
        # Depth cannot exceed total gate count (+2 allowance for barriers / alignment)
        if circuit_depth > total_gate_count + 2:
            errors.append(
                f"Circuit depth ({circuit_depth}) cannot exceed total gate count ({total_gate_count}). "
                f"A quantum circuit cannot have more depth layers than total gates applied."
            )

        # In depth D on N qubits, each qubit can execute at most 1 gate per unit depth.
        max_gates_in_depth = circuit_depth * total_qubits
        if total_gate_count > max_gates_in_depth:
            errors.append(
                f"Impossible circuit packing: Cannot fit {total_gate_count} gates into depth {circuit_depth} on {total_qubits} qubits. "
                f"Maximum theoretical capacity is {max_gates_in_depth} gates."
            )

    # 2-Qubit gate parallelism bound:
    # At most floor(total_qubits / 2) non-overlapping 2-qubit gates can execute in a single depth slice
    if circuit_depth is not None and gate_count_2q is not None and gate_count_2q > 0:
        max_parallel_2q_per_layer = max(1, total_qubits // 2)
        min_depth_for_2q = (gate_count_2q + max_parallel_2q_per_layer - 1) // max_parallel_2q_per_layer
        if circuit_depth < min_depth_for_2q:
            errors.append(
                f"Depth too shallow for 2-qubit gates: {gate_count_2q} CNOT gates on {total_qubits} qubits "
                f"require a depth of at least {min_depth_for_2q}, but depth is set to {circuit_depth}."
            )

    # 6. Noise channel physical boundaries
    valid_noise_types = ["none", "depolarizing", "amplitude_damping", "thermal_relaxation"]
    noise_t = str(noise_type).strip().lower()
    if noise_t not in valid_noise_types:
        errors.append(f"Invalid noise type '{noise_type}'. Must be one of: {', '.join(valid_noise_types)}.")
    else:
        try:
            noise_str = float(noise_strength)
            if noise_str < 0.0:
                errors.append(f"Noise strength cannot be negative, received {noise_str}.")
            elif noise_t == "none" and noise_str != 0.0:
                warnings.append("Noise strength reset to 0.0 for ideal 'none' noise model.")
            elif noise_t in ["depolarizing", "amplitude_damping"] and (noise_str > 0.75):
                errors.append(f"Noise strength ({noise_str}) exceeds physical maximum for quantum channel (0.75).")
            elif noise_t in ["depolarizing", "amplitude_damping"] and (noise_str < 0.001 or noise_str > 0.10):
                warnings.append(f"Noise strength ({noise_str}) is outside recommended calibration range (0.001–0.10).")
            elif noise_t == "thermal_relaxation" and (noise_str < 0.01 or noise_str > 0.10):
                warnings.append(f"Thermal relaxation strength ({noise_str}) is outside recommended range (0.01–0.10).")
        except (ValueError, TypeError):
            errors.append("Noise strength must be a numerical float.")

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "constructible": is_valid,
        "errors": errors,
        "warnings": warnings,
        "details": {
            "algorithm": algo,
            "n_qubits": n_qubits,
            "total_qubits_required": total_qubits,
            "min_1q_gates_required": min_1q_required,
            "min_2q_gates_required": min_2q_required,
            "max_gates_for_depth": (circuit_depth * total_qubits) if circuit_depth else None,
        },
    }


def check_circuit_constructibility(algorithm, n_qubits, circuit_depth=None, gate_count_1q=None, gate_count_2q=None):
    """
    Convenience helper returning (is_constructible, error_messages).
    """
    res = validate_circuit_parameters(
        algorithm=algorithm,
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        gate_count_1q=gate_count_1q,
        gate_count_2q=gate_count_2q,
    )
    return res["constructible"], res["errors"]

