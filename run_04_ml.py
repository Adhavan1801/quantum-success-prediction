"""
=============================================================
Script 04 — ML Model Training & Evaluation
=============================================================
Train Baseline, Decision Tree, and Random Forest with
GridSearchCV. Evaluate and compare all models.
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

from src.feature_engineering import load_dataset, encode_features
from src.ml_pipeline import (
    split_data,
    train_baseline, train_decision_tree, train_random_forest,
    evaluate_model, compare_models,
    plot_predicted_vs_actual, plot_model_comparison,
    save_model,
)

sns.set_theme(style='whitegrid', palette='deep')
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

print("=" * 65)
print("  NOTEBOOK 04 — ML MODEL TRAINING & EVALUATION")
print("=" * 65)

# --- 4.1 Data Preparation ---
print("\n" + "=" * 65)
print("  4.1 DATA PREPARATION")
print("=" * 65)

df = load_dataset('data/results.csv')
X, y, feature_names = encode_features(df)
print(f"\nFeatures: {len(feature_names)} columns")
print(f"Samples:  {len(X)}")

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# --- 4.2 Baseline Model ---
print("\n" + "=" * 65)
print("  4.2 BASELINE MODEL (Linear Regression on depth only)")
print("=" * 65)

baseline_model = train_baseline(X_train, y_train)
baseline_results = evaluate_model(
    baseline_model, X_test, y_test,
    model_name='Baseline (Linear Reg.)',
    depth_only=True
)

# --- 4.3 Decision Tree ---
print("\n" + "=" * 65)
print("  4.3 DECISION TREE (GridSearchCV, 5-fold CV)")
print("=" * 65)
print("\nHyperparameter grid:")
print("  max_depth:        [3, 5, 8, 10, 15, None]")
print("  min_samples_split: [2, 5, 10]")
print("  min_samples_leaf:  [1, 2, 5]")
print("  Total combinations: 54\n")

dt_model = train_decision_tree(X_train, y_train, cv=5)
dt_results = evaluate_model(dt_model, X_test, y_test, model_name='Decision Tree')

# Predicted vs Actual
plot_predicted_vs_actual(
    y_test, dt_results['y_pred'],
    model_name='Decision Tree',
    save_path='outputs/figures/dt_predicted_vs_actual.png'
)
plt.close('all')

# --- 4.4 Random Forest ---
print("\n" + "=" * 65)
print("  4.4 RANDOM FOREST (GridSearchCV, 5-fold CV)")
print("=" * 65)
print("\nHyperparameter grid:")
print("  n_estimators:      [50, 100, 200]")
print("  max_depth:         [5, 10, 15, None]")
print("  min_samples_split: [2, 5]")
print("  min_samples_leaf:  [1, 2]")
print("  Total combinations: 48\n")

rf_model = train_random_forest(X_train, y_train, cv=5)
rf_results = evaluate_model(rf_model, X_test, y_test, model_name='Random Forest')

plot_predicted_vs_actual(
    y_test, rf_results['y_pred'],
    model_name='Random Forest',
    save_path='outputs/figures/rf_predicted_vs_actual.png'
)
plt.close('all')

# --- 4.5 Model Comparison ---
print("\n" + "=" * 65)
print("  4.5 MODEL COMPARISON")
print("=" * 65)

all_results = [baseline_results, dt_results, rf_results]
comparison_df = compare_models(all_results)

plot_model_comparison(comparison_df, save_path='outputs/figures/model_comparison.png')
plt.close('all')

# Improvement over baseline
baseline_r2 = baseline_results['r2']
dt_r2 = dt_results['r2']
rf_r2 = rf_results['r2']
print('\nImprovement over Baseline:')
print(f'  Decision Tree: R² improved by {dt_r2 - baseline_r2:+.4f} '
      f'({(dt_r2 - baseline_r2) / max(abs(baseline_r2), 0.001) * 100:+.1f}%)')
print(f'  Random Forest: R² improved by {rf_r2 - baseline_r2:+.4f} '
      f'({(rf_r2 - baseline_r2) / max(abs(baseline_r2), 0.001) * 100:+.1f}%)')

# --- 4.6 All Models Predicted vs Actual ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_data = [
    ('Baseline (Linear Reg.)', baseline_results),
    ('Decision Tree', dt_results),
    ('Random Forest', rf_results),
]
for ax, (name, results) in zip(axes, models_data):
    ax.scatter(y_test, results['y_pred'], alpha=0.3, s=15, color='#4C72B0',
               edgecolor='white', linewidth=0.2)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{name}\nR²={results["r2"]:.4f}', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
plt.suptitle('Predicted vs Actual — All Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/all_models_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Figure saved: outputs/figures/all_models_predicted_vs_actual.png")

# --- 4.7 Save Models ---
print("\n" + "=" * 65)
print("  4.7 SAVE MODELS")
print("=" * 65)

save_model(baseline_model, 'outputs/models/baseline_model.joblib')
save_model(dt_model, 'outputs/models/decision_tree_model.joblib')
save_model(rf_model, 'outputs/models/random_forest_model.joblib')
comparison_df.to_csv('outputs/model_comparison.csv', index=False)
print("  Model comparison saved to outputs/model_comparison.csv")

print("\n" + "=" * 65)
print("  ✅ NOTEBOOK 04 COMPLETE")
print("=" * 65)

