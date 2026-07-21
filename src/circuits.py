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
