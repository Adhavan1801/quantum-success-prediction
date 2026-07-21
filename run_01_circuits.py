"""
=============================================================
Script 01 — Circuit Building
=============================================================
Build and verify quantum circuits for DJ, BV, and Simon's
algorithms. Visualize circuit stats across qubit counts.
=============================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import transpile
from qiskit_aer import AerSimulator

from src.circuits import (
    build_dj_oracle, build_dj_circuit,
    build_bv_circuit,
    build_simon_circuit, check_simon_success,
    generate_secret_strings,
    generate_all_circuit_configs, get_circuit_stats,
)
from src.simulation import run_circuit, compute_success_probability

sns.set_theme(style='whitegrid', palette='deep')
os.makedirs('outputs/figures', exist_ok=True)


def run_ideal(circuit, shots=4096):
    """Run a circuit on the ideal Aer simulator."""
    sim = AerSimulator()
    transpiled = transpile(circuit, sim)
    result = sim.run(transpiled, shots=shots).result()
    return result.get_counts()


# =============================================================
print("=" * 65)
print("  NOTEBOOK 01 — CIRCUIT BUILDING")
print("=" * 65)

# --- 1.2 Deutsch-Jozsa Algorithm ---
print("\n" + "=" * 65)
print("  1.2 DEUTSCH-JOZSA ALGORITHM")
print("=" * 65)

# Constant-0 oracle
n = 3
oracle_const = build_dj_oracle(n, 'constant_0')
circuit_const = build_dj_circuit(n, oracle_const)
print(f"\n[DJ] Constant-0 Oracle (n={n}):")
print(f"  Total qubits: {circuit_const.num_qubits} ({n} input + 1 ancilla)")
print(f"  Classical bits: {circuit_const.num_clbits}")

counts = run_ideal(circuit_const)
print(f"  Measurement: {counts}")
print(f"  Expected: {{'000': 4096}}")
print(f"  {'✓ PASS' if counts.get('000', 0) == 4096 else '✗ FAIL'}")

# Constant-1 oracle
oracle_const1 = build_dj_oracle(n, 'constant_1')
circuit_const1 = build_dj_circuit(n, oracle_const1)
counts1 = run_ideal(circuit_const1)
print(f"\n[DJ] Constant-1 Oracle (n={n}):")
print(f"  Measurement: {counts1}")
print(f"  {'✓ PASS' if counts1.get('000', 0) == 4096 else '✗ FAIL'}")

# Balanced oracle
oracle_bal = build_dj_oracle(n, 'balanced', seed=42)
circuit_bal = build_dj_circuit(n, oracle_bal)
counts_bal = run_ideal(circuit_bal)
print(f"\n[DJ] Balanced Oracle (n={n}):")
print(f"  Measurement: {counts_bal}")
print(f"  Expected: any non-zero bitstring (NOT '000')")
print(f"  {'✓ PASS' if '000' not in counts_bal else '✗ FAIL'}")


# --- 1.3 Bernstein-Vazirani Algorithm ---
print("\n" + "=" * 65)
print("  1.3 BERNSTEIN-VAZIRANI ALGORITHM")
print("=" * 65)

test_secrets = ['10', '101', '1010', '11011']
print("\nTesting BV with multiple secret strings:")
for secret in test_secrets:
    n = len(secret)
    circuit = build_bv_circuit(n, secret)
    counts = run_ideal(circuit)
    recovered = max(counts, key=counts.get)
    status = '✓' if recovered == secret else '✗'
    print(f"  {status} secret=\"{secret}\" → recovered=\"{recovered}\" "
          f"(probability: {counts[recovered]/4096:.2%})")


# --- 1.4 Simon's Algorithm ---
print("\n" + "=" * 65)
print("  1.4 SIMON'S ALGORITHM")
print("=" * 65)

test_simons = [('10', 2), ('110', 3), ('101', 3)]
print("\nTesting Simon's with known periods:")
for secret, n in test_simons:
    circuit = build_simon_circuit(n, secret)
    counts = run_ideal(circuit)
    success = check_simon_success(counts, secret, n)
    status = '✓' if success >= 0.99 else '✗'
    print(f"  {status} period=\"{secret}\" (n={n})")
    print(f"    Outputs: {dict(sorted(counts.items(), key=lambda x: -x[1]))}")
    print(f"    z·s=0 satisfaction rate: {success:.2%}")


# --- 1.5 Correctness Verification — All Algorithms ---
print("\n" + "=" * 65)
print("  1.5 CORRECTNESS VERIFICATION — ALL ALGORITHMS")
print("=" * 65)

configs = generate_all_circuit_configs(range(2, 6))  # 2 to 5 qubits
results = []
for cfg in configs:
    counts = run_circuit(cfg['circuit'], noise_model=None, shots=4096)
    success = compute_success_probability(
        counts, cfg['expected_output'], cfg['algorithm'],
        n_qubits=cfg['n_qubits'], secret_string=cfg.get('expected_output')
    )
    results.append({
        'algorithm': cfg['algorithm'],
        'n_qubits': cfg['n_qubits'],
        'oracle': cfg['oracle_desc'],
        'success': success,
    })

df_verify = pd.DataFrame(results)
print(f"\nTotal circuits tested: {len(df_verify)}")
print(f"All passed (success ≥ 0.99): {(df_verify['success'] >= 0.99).all()}")
print(f"\nSuccess probability by algorithm:")
print(df_verify.groupby('algorithm')['success'].agg(['mean', 'min', 'max']).round(4).to_string())


# --- 1.6 Circuit Statistics ---
print("\n" + "=" * 65)
print("  1.6 CIRCUIT STATISTICS (Transpiled, qubits 2-8)")
print("=" * 65)

configs_full = generate_all_circuit_configs(range(2, 9))
stats_data = []
for cfg in configs_full:
    stats = get_circuit_stats(cfg['circuit'])
    stats['algorithm'] = cfg['algorithm']
    stats['n_qubits'] = cfg['n_qubits']
    stats_data.append(stats)

df_stats = pd.DataFrame(stats_data)
print("\nMean circuit statistics by algorithm and qubit count:")
print(df_stats.groupby(['algorithm', 'n_qubits'])[
    ['circuit_depth', 'gate_count_2q', 'total_gate_count']
].mean().round(1).to_string())

# Plot: Circuit Depth vs Qubit Count
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for algo in df_stats['algorithm'].unique():
    subset = df_stats[df_stats['algorithm'] == algo]
    grouped = subset.groupby('n_qubits')['circuit_depth'].mean()
    label = algo.replace('_', ' ').title()
    axes[0].plot(grouped.index, grouped.values, marker='o', linewidth=2,
                 markersize=7, label=label)
    grouped_2q = subset.groupby('n_qubits')['gate_count_2q'].mean()
    axes[1].plot(grouped_2q.index, grouped_2q.values, marker='s', linewidth=2,
                 markersize=7, label=label)

axes[0].set_xlabel('Number of Qubits', fontsize=12)
axes[0].set_ylabel('Circuit Depth', fontsize=12)
axes[0].set_title('Circuit Depth vs Qubit Count', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[1].set_xlabel('Number of Qubits', fontsize=12)
axes[1].set_ylabel('Two-Qubit Gate Count', fontsize=12)
axes[1].set_title('Two-Qubit Gates vs Qubit Count', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
plt.suptitle('Transpiled Circuit Complexity', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/circuit_complexity.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Figure saved: outputs/figures/circuit_complexity.png")

print("\n" + "=" * 65)
print("  ✅ NOTEBOOK 01 COMPLETE")
print("=" * 65)

