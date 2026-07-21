# =============================================================
# test_ml.py — Unit Tests for ML Pipeline
# =============================================================
"""
Tests verify that the ML pipeline functions work correctly
on a small synthetic dataset.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from src.feature_engineering import build_feature_table, encode_features
from src.ml_pipeline import (
    split_data,
    train_baseline,
    train_decision_tree,
    evaluate_model,
)


def _create_sample_data():
    """Create a small synthetic dataset for testing."""
    np.random.seed(42)
    rows = []
    for _ in range(100):
        n_qubits = np.random.choice([2, 3, 4, 5])
        noise_strength = np.random.choice([0.0, 0.01, 0.05, 0.1])
        depth = int(n_qubits * 3 + np.random.randint(0, 5))
        gate_2q = int(n_qubits * 2 + np.random.randint(0, 3))

        # Simple synthetic relationship for testing
        success = max(0, min(1,
            1.0 - noise_strength * 5 - n_qubits * 0.02 + np.random.normal(0, 0.05)
        ))

        rows.append({
            "algorithm": np.random.choice(["deutsch_jozsa", "bernstein_vazirani", "simons"]),
            "n_qubits": n_qubits,
            "oracle_desc": "test",
            "circuit_depth": depth,
            "gate_count_1q": depth - gate_2q,
            "gate_count_2q": gate_2q,
            "total_gate_count": depth,
            "noise_type": np.random.choice(["none", "depolarizing", "amplitude_damping"]),
            "noise_strength": noise_strength,
            "success_probability": success,
        })
    return rows


class TestFeatureEngineering(unittest.TestCase):
    """Test feature table construction and encoding."""

    def test_build_feature_table(self):
        """Feature table should have correct shape and columns."""
        raw = _create_sample_data()
        df = build_feature_table(raw)
        self.assertEqual(len(df), 100)
        self.assertIn("success_probability", df.columns)
        self.assertIn("algorithm", df.columns)
        self.assertIn("noise_type", df.columns)

    def test_encode_features(self):
        """Encoding should produce numeric features and remove oracle_desc."""
        raw = _create_sample_data()
        df = build_feature_table(raw)
        X, y, names = encode_features(df)

        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertNotIn("oracle_desc", X.columns)
        self.assertNotIn("success_probability", X.columns)

        # Check one-hot columns exist
        algo_cols = [c for c in X.columns if c.startswith("algo_")]
        noise_cols = [c for c in X.columns if c.startswith("noise_")]
        self.assertGreater(len(algo_cols), 0)
        self.assertGreater(len(noise_cols), 0)


class TestMLPipeline(unittest.TestCase):
    """Test model training and evaluation on synthetic data."""

    @classmethod
    def setUpClass(cls):
        """Prepare data once for all tests."""
        raw = _create_sample_data()
        df = build_feature_table(raw)
        X, y, names = encode_features(df)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_data(X, y)
        cls.feature_names = names

    def test_split_sizes(self):
        """Train/test split should have correct proportions."""
        total = len(self.X_train) + len(self.X_test)
        self.assertEqual(total, 100)
        self.assertAlmostEqual(len(self.X_test) / total, 0.2, places=1)

    def test_baseline_trains(self):
        """Baseline model should train without error."""
        model = train_baseline(self.X_train, self.y_train)
        self.assertIsNotNone(model)

    def test_decision_tree_trains(self):
        """Decision tree should train without error."""
        # Use small param grid for speed
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import GridSearchCV

        dt = DecisionTreeRegressor(random_state=42)
        grid = GridSearchCV(dt, {"max_depth": [3, 5]}, cv=3, scoring="r2")
        grid.fit(self.X_train, self.y_train)
        self.assertIsNotNone(grid.best_estimator_)

    def test_evaluate_returns_metrics(self):
        """Evaluation should return R², MAE, RMSE."""
        model = train_baseline(self.X_train, self.y_train)
        results = evaluate_model(model, self.X_test, self.y_test,
                                  model_name="Test", depth_only=True)
        self.assertIn("r2", results)
        self.assertIn("mae", results)
        self.assertIn("rmse", results)


if __name__ == "__main__":
    unittest.main()
