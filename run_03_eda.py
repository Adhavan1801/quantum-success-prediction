"""
=============================================================
Script 03 — Exploratory Data Analysis & Feature Engineering
=============================================================
Explore the dataset, visualize distributions, correlations,
key trends, and prepare features for ML.
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

from src.feature_engineering import load_dataset, encode_features, get_dataset_summary

sns.set_theme(style='whitegrid', palette='deep')
os.makedirs('outputs/figures', exist_ok=True)

print("=" * 65)
print("  NOTEBOOK 03 — EDA & FEATURE ENGINEERING")
print("=" * 65)

# --- 3.1 Load Dataset ---
print("\n" + "=" * 65)
print("  3.1 LOAD DATASET")
print("=" * 65)

df = load_dataset('data/results.csv')

# --- 3.2 Dataset Overview ---
print("\n" + "=" * 65)
print("  3.2 DATASET OVERVIEW")
print("=" * 65)

get_dataset_summary(df)

print(f"\nData types:")
print(df.dtypes.to_string())
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"\nMissing values per column:")
print(df.isnull().sum().to_string())
print(f"\nFirst 10 rows:")
print(df.head(10).to_string())
print(f"\nStatistical summary:")
print(df.describe().round(4).to_string())

# --- 3.3 Distribution Plots ---
print("\n" + "=" * 65)
print("  3.3 DISTRIBUTION PLOTS")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['success_probability'], bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
axes[0].axvline(df['success_probability'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df["success_probability"].mean():.3f}')
axes[0].set_xlabel('Success Probability', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Overall Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)

for algo in df['algorithm'].unique():
    subset = df[df['algorithm'] == algo]
    axes[1].hist(subset['success_probability'], bins=30, alpha=0.6,
                 label=algo.replace('_', ' ').title())
axes[1].set_xlabel('Success Probability', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Distribution by Algorithm', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=9)

for ntype in df['noise_type'].unique():
    subset = df[df['noise_type'] == ntype]
    axes[2].hist(subset['success_probability'], bins=30, alpha=0.6,
                 label=ntype.replace('_', ' ').title())
axes[2].set_xlabel('Success Probability', fontsize=12)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('Distribution by Noise Type', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=9)

plt.suptitle('Success Probability Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/success_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figure saved: outputs/figures/success_distributions.png")

# --- 3.4 Correlation Analysis ---
print("\n" + "=" * 65)
print("  3.4 CORRELATION ANALYSIS")
print("=" * 65)

numeric_cols = ['n_qubits', 'circuit_depth', 'gate_count_1q', 'gate_count_2q',
                'total_gate_count', 'noise_strength', 'success_probability']
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nCorrelation with success_probability:")
print(corr_matrix['success_probability'].sort_values(ascending=False).to_string())
print("\n📊 Figure saved: outputs/figures/correlation_heatmap.png")

# --- 3.5 Key Trends ---
print("\n" + "=" * 65)
print("  3.5 KEY TRENDS")
print("=" * 65)

# Success vs Qubits
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for algo in df['algorithm'].unique():
    subset = df[df['algorithm'] == algo]
    grouped = subset.groupby('n_qubits')['success_probability'].mean()
    label = algo.replace('_', ' ').title()
    axes[0].plot(grouped.index, grouped.values, marker='o', linewidth=2.5,
                 markersize=8, label=label)
axes[0].set_xlabel('Number of Qubits', fontsize=12)
axes[0].set_ylabel('Mean Success Probability', fontsize=12)
axes[0].set_title('Success vs Qubit Count (by Algorithm)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_ylim(-0.05, 1.05)

for ntype in df['noise_type'].unique():
    subset = df[df['noise_type'] == ntype]
    grouped = subset.groupby('n_qubits')['success_probability'].mean()
    label = ntype.replace('_', ' ').title()
    axes[1].plot(grouped.index, grouped.values, marker='s', linewidth=2.5,
                 markersize=8, label=label)
axes[1].set_xlabel('Number of Qubits', fontsize=12)
axes[1].set_ylabel('Mean Success Probability', fontsize=12)
axes[1].set_title('Success vs Qubit Count (by Noise Type)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_ylim(-0.05, 1.05)
plt.suptitle('How Qubit Count Affects Success Probability', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/success_vs_qubits.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figure saved: outputs/figures/success_vs_qubits.png")

print("\nMean success probability by algorithm × qubits:")
print(df.groupby(['algorithm', 'n_qubits'])['success_probability'].mean().round(4).unstack().to_string())

# Noise degradation curves per algorithm
df_noisy = df[df['noise_type'] != 'none']
algorithms = df_noisy['algorithm'].unique()
fig, axes = plt.subplots(1, len(algorithms), figsize=(6 * len(algorithms), 5))
if len(algorithms) == 1:
    axes = [axes]
for ax, algo in zip(axes, algorithms):
    subset = df_noisy[df_noisy['algorithm'] == algo]
    for ntype in subset['noise_type'].unique():
        nt_subset = subset[subset['noise_type'] == ntype]
        grouped = nt_subset.groupby('noise_strength')['success_probability'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2,
                label=ntype.replace('_', ' ').title())
    ax.set_xlabel('Noise Strength', fontsize=11)
    ax.set_ylabel('Mean Success Probability', fontsize=11)
    ax.set_title(algo.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
plt.suptitle('Noise Degradation Curves Per Algorithm', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/noise_degradation_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figure saved: outputs/figures/noise_degradation_curves.png")

# Heatmaps
noise_types = [nt for nt in df['noise_type'].unique() if nt != 'none']
fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5))
if len(noise_types) == 1:
    axes = [axes]
for ax, ntype in zip(axes, noise_types):
    subset = df[df['noise_type'] == ntype]
    pivot = subset.pivot_table(values='success_probability', index='n_qubits',
                                columns='noise_strength', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title(ntype.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Noise Strength', fontsize=11)
    ax.set_ylabel('Qubits', fontsize=11)
plt.suptitle('Success Probability Heatmaps', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/noise_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figure saved: outputs/figures/noise_heatmaps.png")

# Depth vs Success scatter
fig, ax = plt.subplots(figsize=(10, 6))
for algo in df['algorithm'].unique():
    subset = df[(df['algorithm'] == algo) & (df['noise_type'] != 'none')]
    ax.scatter(subset['circuit_depth'], subset['success_probability'],
               alpha=0.3, s=20, label=algo.replace('_', ' ').title())
ax.set_xlabel('Circuit Depth (Transpiled)', fontsize=12)
ax.set_ylabel('Success Probability', fontsize=12)
ax.set_title('Circuit Depth vs Success Probability', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('outputs/figures/depth_vs_success.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figure saved: outputs/figures/depth_vs_success.png")

# Box plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df, x='algorithm', y='success_probability', ax=axes[0], palette='Set2')
axes[0].set_title('Success by Algorithm', fontsize=14, fontweight='bold')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=15)
sns.boxplot(data=df, x='noise_type', y='success_probability', ax=axes[1], palette='Set3')
axes[1].set_title('Success by Noise Type', fontsize=14, fontweight='bold')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=15)
plt.suptitle('Success Probability Box Plots', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/success_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figure saved: outputs/figures/success_boxplots.png")

# --- 3.6 Feature Engineering ---
print("\n" + "=" * 65)
print("  3.6 FEATURE ENGINEERING")
print("=" * 65)

X, y, feature_names = encode_features(df)
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape:  {y.shape}")
print(f"\nFeature names ({len(feature_names)}):")
for i, name in enumerate(feature_names):
    print(f"  {i+1}. {name}")

print(f"\nEncoded feature matrix (first 5 rows):")
print(X.head().to_string())

print("\n" + "=" * 65)
print("  ✅ NOTEBOOK 03 COMPLETE")
print("=" * 65)

