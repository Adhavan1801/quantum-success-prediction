"""
=============================================================
Script 02 — Noise Simulation
=============================================================
Configure noise models, run the full parameter sweep across
all algorithms and noise conditions, save dataset.
=============================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.noise_models import (
    get_all_noise_configs, describe_noise_configs,
)
from src.circuits import generate_all_circuit_configs
from src.simulation import run_sweep
from src.feature_engineering import build_feature_table, save_dataset

sns.set_theme(style='whitegrid', palette='deep')
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("=" * 65)
print("  NOTEBOOK 02 — NOISE SIMULATION")
print("=" * 65)

# --- 2.2 Noise Model Explanation ---
print("\n" + "=" * 65)
print("  2.2 NOISE CONFIGURATIONS")
print("=" * 65)
describe_noise_configs()

# --- 2.3 Sweep Configuration ---
print("\n" + "=" * 65)
print("  2.3 SWEEP CONFIGURATION")
print("=" * 65)

qubit_range = range(2, 9)
circuit_configs = generate_all_circuit_configs(qubit_range)
noise_configs = get_all_noise_configs()

total = len(circuit_configs) * len(noise_configs)
print(f"\nCircuit configurations: {len(circuit_configs)}")
print(f"Noise configurations:  {len(noise_configs)}")
print(f"Total experiments:     {total}")
print(f"Shots per experiment:  4096")

# --- 2.4 Run the Full Sweep ---
print("\n" + "=" * 65)
print("  2.4 RUNNING FULL PARAMETER SWEEP")
print("=" * 65)
print("\n⚠️  This will take ~10-15 minutes. Progress bar below.\n")

raw_results = run_sweep(
    qubit_range=range(2, 9),
    shots=4096,
    seed=42,
)

# --- 2.5 Sanity Check ---
print("\n" + "=" * 65)
print("  2.5 SANITY CHECK")
print("=" * 65)

df = build_feature_table(raw_results)
print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head().to_string())

noiseless = df[df['noise_type'] == 'none']
print(f"\n--- Noiseless Results ---")
print(f"  Mean success probability: {noiseless['success_probability'].mean():.4f}")
print(f"  Min success probability:  {noiseless['success_probability'].min():.4f}")
print(f"  All ≥ 0.99: {(noiseless['success_probability'] >= 0.99).all()}")

noisy = df[df['noise_type'] != 'none']
print(f"\n--- Noisy Results ---")
print(f"  Mean success probability: {noisy['success_probability'].mean():.4f}")
print(f"  Min success probability:  {noisy['success_probability'].min():.4f}")

# Quick plot
fig, ax = plt.subplots(figsize=(10, 6))
for ntype in df['noise_type'].unique():
    if ntype == 'none':
        continue
    subset = df[df['noise_type'] == ntype]
    grouped = subset.groupby('noise_strength')['success_probability'].mean()
    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2,
            markersize=7, label=ntype.replace('_', ' ').title())
ax.set_xlabel('Noise Strength', fontsize=12)
ax.set_ylabel('Mean Success Probability', fontsize=12)
ax.set_title('Success Probability vs Noise Strength (Quick Check)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/noise_quick_check.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n📊 Figure saved: outputs/figures/noise_quick_check.png")

# --- 2.6 Export Dataset ---
print("\n" + "=" * 65)
print("  2.6 EXPORT DATASET")
print("=" * 65)

save_dataset(df, 'data/results.csv')
print(f"\nDataset columns: {list(df.columns)}")
print(f"\nDataset statistics:")
print(df.describe().round(4).to_string())

print("\n" + "=" * 65)
print("  ✅ NOTEBOOK 02 COMPLETE")
print("=" * 65)

