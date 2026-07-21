# =============================================================
# ml_pipeline.py — Machine Learning Training & Evaluation
# =============================================================
"""
Train, evaluate, and analyze machine learning models for
predicting quantum circuit success probability.

Models:
    - Baseline: Linear regression on circuit_depth only
    - Decision Tree: Tuned with GridSearchCV
    - Random Forest: Tuned with GridSearchCV
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =============================================================
# DATA SPLITTING
# =============================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into train/test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Data split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


# =============================================================
# MODEL TRAINING
# =============================================================

def train_baseline(X_train, y_train, depth_col="circuit_depth"):
    """
    Train a baseline linear regression model using only circuit_depth.

    This provides a simple reference point to demonstrate that
    the tuned tree-based models add genuine value.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    depth_col : str
        Name of the circuit depth column.

    Returns
    -------
    LinearRegression
        Trained baseline model.
    """
    print("\n--- Training Baseline Model (Linear Regression on depth only) ---")
    X_depth = X_train[[depth_col]]
    model = LinearRegression()
    model.fit(X_depth, y_train)
    print(f"  Coefficient: {model.coef_[0]:.6f}")
    print(f"  Intercept:   {model.intercept_:.6f}")
    return model


def train_decision_tree(X_train, y_train, cv=5):
    """
    Train a Decision Tree Regressor with GridSearchCV hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    DecisionTreeRegressor
        Best model from GridSearchCV.
    """
    print("\n--- Training Decision Tree with GridSearchCV ---")

    param_grid = {
        "max_depth": [3, 5, 8, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    }

    dt = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        dt, param_grid, cv=cv, scoring="r2",
        n_jobs=-1, verbose=1, return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV R² score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_random_forest(X_train, y_train, cv=5):
    """
    Train a Random Forest Regressor with GridSearchCV hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    RandomForestRegressor
        Best model from GridSearchCV.
    """
    print("\n--- Training Random Forest with GridSearchCV ---")

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring="r2",
        n_jobs=-1, verbose=1, return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV R² score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# =============================================================
# MODEL EVALUATION
# =============================================================

def evaluate_model(model, X_test, y_test, model_name="Model",
                   depth_only=False, depth_col="circuit_depth"):
    """
    Evaluate a trained model on the test set.

    Parameters
    ----------
    model : estimator
        Trained sklearn model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    model_name : str
        Name for display purposes.
    depth_only : bool
        If True, only use the circuit_depth column (for baseline).
    depth_col : str
        Name of the circuit depth column.

    Returns
    -------
    dict
        Metrics: R², MAE, RMSE, and predictions.
    """
    if depth_only:
        X_eval = X_test[[depth_col]]
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  {model_name} — Test Set Performance:")
    print(f"    R²:   {r2:.4f}")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")

    return {
        "model_name": model_name,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "y_pred": y_pred,
    }


def compare_models(results_list):
    """
    Create a comparison table of model performance metrics.

    Parameters
    ----------
    results_list : list of dict
        List of results from evaluate_model().

    Returns
    -------
    pd.DataFrame
        Comparison table with R², MAE, RMSE for each model.
    """
    comparison = pd.DataFrame([
        {
            "Model": r["model_name"],
            "R²": r["r2"],
            "MAE": r["mae"],
            "RMSE": r["rmse"],
        }
        for r in results_list
    ])

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)
    print(comparison.to_string(index=False))
    print("=" * 55)

    return comparison


# =============================================================
# FEATURE IMPORTANCE
# =============================================================

def get_feature_importance(model, feature_names):
    """
    Extract feature importance scores from a tree-based model.

    Parameters
    ----------
    model : DecisionTreeRegressor or RandomForestRegressor
        Trained tree-based model.
    feature_names : list of str
        Feature names.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted by importance.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return importance_df


# =============================================================
# VISUALIZATION FUNCTIONS
# =============================================================

def plot_predicted_vs_actual(y_test, y_pred, model_name="Model",
                              save_path=None):
    """
    Scatter plot of predicted vs actual success probabilities.

    Parameters
    ----------
    y_test : array-like
        True values.
    y_pred : array-like
        Predicted values.
    model_name : str
        Model name for the title.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="#4C72B0", edgecolor="white",
               linewidth=0.3)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Success Probability", fontsize=12)
    ax.set_ylabel("Predicted Success Probability", fontsize=12)
    ax.set_title(f"{model_name} — Predicted vs Actual", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.show()


def plot_feature_importance(importance_df, model_name="Model",
                             save_path=None):
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance table from get_feature_importance().
    model_name : str
        Model name for the title.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("viridis", len(importance_df))
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"{model_name} — Feature Importance", fontsize=14,
                 fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.show()


def plot_model_comparison(comparison_df, save_path=None):
    """
    Bar chart comparing R², MAE, RMSE across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison table from compare_models().
    save_path : str, optional
        Path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["R²", "MAE", "RMSE"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metric, color in zip(axes, metrics, colors):
        ax.bar(comparison_df["Model"], comparison_df[metric],
               color=color, edgecolor="white", linewidth=1.5)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(comparison_df[metric]):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold",
                 y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.show()


def plot_success_vs_qubits(df, save_path=None):
    """
    Line plot of success probability vs qubit count, grouped by algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in df["algorithm"].unique():
        subset = df[df["algorithm"] == algo]
        grouped = subset.groupby("n_qubits")["success_probability"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", linewidth=2,
                markersize=8, label=algo.replace("_", " ").title())

    ax.set_xlabel("Number of Qubits", fontsize=12)
    ax.set_ylabel("Mean Success Probability", fontsize=12)
    ax.set_title("Success Probability vs Qubit Count", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_success_vs_noise(df, save_path=None):
    """
    Line plot of success probability vs noise strength, grouped by noise type.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    save_path : str, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Exclude "none" noise type
    df_noisy = df[df["noise_type"] != "none"]

    for ntype in df_noisy["noise_type"].unique():
        subset = df_noisy[df_noisy["noise_type"] == ntype]
        grouped = subset.groupby("noise_strength")["success_probability"].mean()
        ax.plot(grouped.index, grouped.values, marker="s", linewidth=2,
                markersize=8, label=ntype.replace("_", " ").title())

    ax.set_xlabel("Noise Strength", fontsize=12)
    ax.set_ylabel("Mean Success Probability", fontsize=12)
    ax.set_title("Success Probability vs Noise Strength", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_noise_heatmap(df, algorithm=None, save_path=None):
    """
    Heatmap of success probability: qubits × noise strength.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    algorithm : str, optional
        Filter to a specific algorithm. None = all combined.
    save_path : str, optional
        Path to save the figure.
    """
    if algorithm:
        df_plot = df[df["algorithm"] == algorithm].copy()
        title = f"Noise Heatmap — {algorithm.replace('_', ' ').title()}"
    else:
        df_plot = df.copy()
        title = "Noise Heatmap — All Algorithms"

    pivot = df_plot.pivot_table(
        values="success_probability",
        index="n_qubits",
        columns="noise_strength",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.5)

    ax.set_xlabel("Noise Strength", fontsize=12)
    ax.set_ylabel("Number of Qubits", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================
# MODEL PERSISTENCE
# =============================================================

def save_model(model, path):
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : estimator
        Trained sklearn model.
    path : str
        File path (should end in .joblib).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved: {path}")


def load_model(path):
    """
    Load a trained model from disk.

    Parameters
    ----------
    path : str
        File path to the saved model.

    Returns
    -------
    estimator
        The loaded sklearn model.
    """
    model = joblib.load(path)
    print(f"  Model loaded: {path}")
    return model
