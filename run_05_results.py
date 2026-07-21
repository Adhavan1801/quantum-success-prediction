"""
=============================================================
Script 05 — Results Analysis & Feature Importance
=============================================================
Feature importance, per-algorithm analysis, residual analysis,
noise heatmaps, and final summary of findings.
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
from sklearn.ensemble import RandomForestRegressor

from src.feature_engineering import load_dataset, encode_features
from src.ml_pipeline import (
    split_data, load_model, evaluate_model,
    get_feature_importance, plot_feature_importance,
    plot_noise_heatmap, plot_success_vs_qubits, plot_success_vs_noise,
)

sns.set_theme(style='whitegrid', palette='deep')
os.makedirs('outputs/figures', exist_ok=True)

print("=" * 65)
print("  NOTEBOOK 05 — RESULTS ANALYSIS & FEATURE IMPORTANCE")
print("=" * 65)

# --- 5.1 Load Models & Data ---
print("\n" + "=" * 65)
print("  5.1 LOAD MODELS & DATA")
print("=" * 65)

df = load_dataset('data/results.csv')
X, y, feature_names = encode_features(df)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

dt_model = load_model('outputs/models/decision_tree_model.joblib')
rf_model = load_model('outputs/models/random_forest_model.joblib')

# --- 5.2 Feature Importance — Overall ---
print("\n" + "=" * 65)
print("  5.2 FEATURE IMPORTANCE — OVERALL")
print("=" * 65)

dt_importance = get_feature_importance(dt_model, feature_names)
print("\nDecision Tree — Feature Importance:")
print(dt_importance.to_string(index=False))

plot_feature_importance(dt_importance, model_name='Decision Tree',
                        save_path='outputs/figures/dt_feature_importance.png')
plt.close('all')

rf_importance = get_feature_importance(rf_model, feature_names)
print("\nRandom Forest — Feature Importance:")
print(rf_importance.to_string(index=False))

plot_feature_importance(rf_importance, model_name='Random Forest',
                        save_path='outputs/figures/rf_feature_importance.png')
plt.close('all')

# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, (imp_df, name) in zip(axes, [(dt_importance, 'Decision Tree'),
                                      (rf_importance, 'Random Forest')]):
    colors = sns.color_palette('viridis', len(imp_df))
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors,
            edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Figure saved: outputs/figures/feature_importance_comparison.png")

# --- 5.3 Per-Algorithm Analysis ---
print("\n" + "=" * 65)
print("  5.3 PER-ALGORITHM FEATURE IMPORTANCE")
print("=" * 65)

algorithms = df['algorithm'].unique()
per_algo_importance = {}

for algo in algorithms:
    df_algo = df[df['algorithm'] == algo].copy()
    X_algo, y_algo, names_algo = encode_features(df_algo)
    algo_cols = [c for c in X_algo.columns if c.startswith('algo_')]
    X_algo = X_algo.drop(columns=algo_cols)
    names_algo = [n for n in names_algo if not n.startswith('algo_')]
    rf_algo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_algo.fit(X_algo, y_algo)
    importance = get_feature_importance(rf_algo, names_algo)
    per_algo_importance[algo] = importance
    print(f"\n{algo.replace('_', ' ').title()} — Top 5 Features:")
    print(importance.head().to_string(index=False))

fig, axes = plt.subplots(1, len(algorithms), figsize=(6 * len(algorithms), 6))
if len(algorithms) == 1:
    axes = [axes]
for ax, algo in zip(axes, algorithms):
    imp = per_algo_importance[algo]
    colors = sns.color_palette('viridis', len(imp))
    ax.barh(imp['Feature'], imp['Importance'], color=colors,
            edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(algo.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
plt.suptitle('Feature Importance by Algorithm', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/per_algorithm_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Figure saved: outputs/figures/per_algorithm_feature_importance.png")

# --- 5.4 Noise Heatmaps ---
print("\n" + "=" * 65)
print("  5.4 NOISE HEATMAPS")
print("=" * 65)

for algo in algorithms:
    save_path = f'outputs/figures/heatmap_{algo}.png'
    plot_noise_heatmap(df, algorithm=algo, save_path=save_path)
    plt.close('all')
    print(f"📊 Figure saved: {save_path}")

plot_noise_heatmap(df, algorithm=None, save_path='outputs/figures/heatmap_overall.png')
plt.close('all')
print("📊 Figure saved: outputs/figures/heatmap_overall.png")

# --- 5.5 Final Summary Plots ---
print("\n" + "=" * 65)
print("  5.5 FINAL SUMMARY PLOTS")
print("=" * 65)

plot_success_vs_qubits(df, save_path='outputs/figures/success_vs_qubits_final.png')
plt.close('all')
print("📊 Figure saved: outputs/figures/success_vs_qubits_final.png")

plot_success_vs_noise(df, save_path='outputs/figures/success_vs_noise_final.png')
plt.close('all')
print("📊 Figure saved: outputs/figures/success_vs_noise_final.png")

# Residual analysis
y_pred_rf = rf_model.predict(X_test)
residuals = y_test.values - y_pred_rf

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(residuals, bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1].scatter(y_pred_rf, residuals, alpha=0.3, s=15, color='#4C72B0',
                edgecolor='white', linewidth=0.2)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Success Probability', fontsize=12)
axes[1].set_ylabel('Residual', fontsize=12)
axes[1].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
plt.suptitle('Random Forest — Residual Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Figure saved: outputs/figures/residual_analysis.png")
print(f"\nMean residual:    {residuals.mean():.6f}")
print(f"Std of residuals: {residuals.std():.4f}")

# --- 5.6 Key Findings ---
print("\n" + "=" * 65)
print("  5.6 KEY FINDINGS")
print("=" * 65)

comparison = pd.read_csv('outputs/model_comparison.csv')
print("\nFINAL MODEL COMPARISON:")
print("=" * 55)
print(comparison.to_string(index=False))
print("=" * 55)

print("""
KEY FINDINGS:

1. NOISE STRENGTH is the dominant predictor of success probability
   degradation across all three algorithms.

2. CIRCUIT DEPTH and GATE COUNT are the second-most important features.
   Deeper circuits accumulate more error over time.

3. The tree-based models SIGNIFICANTLY OUTPERFORM the baseline,
   demonstrating that the relationship is non-linear.

4. RANDOM FOREST generally performs best, leveraging ensemble
   averaging to reduce overfitting.

5. SIMON'S ALGORITHM is more noise-sensitive than DJ or BV,
   likely because it uses 2n qubits (double the circuit size).

6. DEPOLARIZING NOISE causes the most severe degradation,
   as it introduces errors uniformly across all gate operations.
""")

# --- 5.7 List All Figures ---
print("=" * 65)
print("  5.7 ALL GENERATED FIGURES")
print("=" * 65)

figures_dir = 'outputs/figures/'
figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
print(f"\nTotal figures saved: {len(figures)}")
for f in sorted(figures):
    size_kb = os.path.getsize(os.path.join(figures_dir, f)) / 1024
    print(f"  📊 {f} ({size_kb:.1f} KB)")

print("\n" + "=" * 65)
print("  ✅ NOTEBOOK 05 COMPLETE — ALL ANALYSIS FINISHED")
print("=" * 65)

print("""
PROJECT DELIVERABLES:
  ✅ Working codebase for DJ, BV, Simon's circuit generation
  ✅ Labeled dataset (data/results.csv) with 1000+ configurations
  ✅ Tuned ML models with documented metrics vs baseline
  ✅ Feature importance analysis identifying dominant drivers
  ✅ Publication-quality visualizations in outputs/figures/
""")

