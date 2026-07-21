# =============================================================
# feature_engineering.py — Dataset Construction & Feature Encoding
# =============================================================
"""
Build the feature table from raw simulation results and prepare
features for machine learning.
"""

import os
import pandas as pd
import numpy as np


def build_feature_table(raw_results):
    """
    Convert raw simulation results into a structured pandas DataFrame.

    Parameters
    ----------
    raw_results : list of dict
        Output from simulation.run_sweep().

    Returns
    -------
    pd.DataFrame
        Structured dataset with all features and target variable.
    """
    df = pd.DataFrame(raw_results)

    # Ensure correct data types
    df["algorithm"] = df["algorithm"].astype("category")
    df["noise_type"] = df["noise_type"].astype("category")
    df["n_qubits"] = df["n_qubits"].astype(int)
    df["circuit_depth"] = df["circuit_depth"].astype(int)
    df["gate_count_1q"] = df["gate_count_1q"].astype(int)
    df["gate_count_2q"] = df["gate_count_2q"].astype(int)
    df["total_gate_count"] = df["total_gate_count"].astype(int)
    df["noise_strength"] = df["noise_strength"].astype(float)
    df["success_probability"] = df["success_probability"].astype(float)

    return df


def encode_features(df):
    """
    Prepare features for machine learning.

    Steps:
        1. One-hot encode 'algorithm' column
        2. One-hot encode 'noise_type' column
        3. Drop the oracle_desc column (not a predictive feature)
        4. Separate features (X) and target (y)

    Parameters
    ----------
    df : pd.DataFrame
        The feature table from build_feature_table().

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except success_probability).
    y : pd.Series
        Target variable (success_probability).
    feature_names : list of str
        Names of all feature columns.
    """
    # Make a copy to avoid modifying the original
    df_ml = df.copy()

    # Drop non-predictive columns
    cols_to_drop = ["oracle_desc"]
    df_ml = df_ml.drop(columns=[c for c in cols_to_drop if c in df_ml.columns])

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_ml, columns=["algorithm", "noise_type"],
                                 prefix=["algo", "noise"], dtype=int)

    # Separate features and target
    target_col = "success_probability"
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    feature_names = list(X.columns)

    return X, y, feature_names


def save_dataset(df, path="data/results.csv"):
    """
    Save the dataset to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to save.
    path : str
        File path for the CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")
    print(f"  → Shape: {df.shape[0]} rows × {df.shape[1]} columns")


def load_dataset(path="data/results.csv"):
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    path : str
        File path to the CSV.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    df = pd.read_csv(path)
    print(f"Dataset loaded from {path}")
    print(f"  → Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def get_dataset_summary(df):
    """
    Print a comprehensive summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarize.
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    print(f"\nAlgorithms: {df['algorithm'].unique().tolist()}")
    print(f"Qubit range: {df['n_qubits'].min()} to {df['n_qubits'].max()}")
    print(f"Noise types: {df['noise_type'].unique().tolist()}")

    print(f"\nSuccess probability statistics:")
    print(f"  Mean:   {df['success_probability'].mean():.4f}")
    print(f"  Median: {df['success_probability'].median():.4f}")
    print(f"  Std:    {df['success_probability'].std():.4f}")
    print(f"  Min:    {df['success_probability'].min():.4f}")
    print(f"  Max:    {df['success_probability'].max():.4f}")

    print(f"\nRows per algorithm:")
    for algo, count in df["algorithm"].value_counts().items():
        print(f"  {algo}: {count}")

    print(f"\nRows per noise type:")
    for noise, count in df["noise_type"].value_counts().items():
        print(f"  {noise}: {count}")

    print(f"\nMissing values: {df.isnull().sum().sum()}")
