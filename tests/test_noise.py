# =============================================================
# test_noise.py — Unit Tests for Noise Models
# =============================================================
"""
Tests verify that noise models are valid Qiskit NoiseModel objects
and that noisy simulation degrades success probability.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit_aer.noise import NoiseModel

from src.noise_models import (
    get_depolarizing_model,
    get_amplitude_damping_model,
    get_thermal_relaxation_model,
    get_all_noise_configs,
)
from src.circuits import build_bv_circuit
from src.simulation import run_circuit, compute_success_probability


class TestNoiseModels(unittest.TestCase):
    """Test noise model construction."""

    def test_depolarizing_is_noise_model(self):
        model = get_depolarizing_model(0.01)
        self.assertIsInstance(model, NoiseModel)

    def test_amplitude_damping_is_noise_model(self):
        model = get_amplitude_damping_model(0.01)
        self.assertIsInstance(model, NoiseModel)

    def test_thermal_relaxation_is_noise_model(self):
        model = get_thermal_relaxation_model(50, 40)
        self.assertIsInstance(model, NoiseModel)

    def test_all_configs_have_required_keys(self):
        """Every noise config should have type, strength, model, description."""
        configs = get_all_noise_configs()
        for cfg in configs:
            self.assertIn("noise_type", cfg)
            self.assertIn("noise_strength", cfg)
            self.assertIn("noise_model", cfg)
            self.assertIn("description", cfg)

    def test_no_noise_config_exists(self):
        """There should be at least one 'none' noise config."""
        configs = get_all_noise_configs()
        none_configs = [c for c in configs if c["noise_type"] == "none"]
        self.assertGreater(len(none_configs), 0)


class TestNoiseDegradation(unittest.TestCase):
    """Test that noise actually degrades success probability."""

    def test_noise_reduces_success(self):
        """Noisy BV circuit should have lower success than noiseless."""
        secret = "101"
        circuit = build_bv_circuit(3, secret)

        # Noiseless run
        counts_ideal = run_circuit(circuit, noise_model=None, shots=4096)
        success_ideal = compute_success_probability(
            counts_ideal, secret, "bernstein_vazirani"
        )

        # Noisy run (strong depolarizing)
        noise_model = get_depolarizing_model(0.05)
        counts_noisy = run_circuit(circuit, noise_model=noise_model, shots=4096)
        success_noisy = compute_success_probability(
            counts_noisy, secret, "bernstein_vazirani"
        )

        self.assertAlmostEqual(success_ideal, 1.0, places=2)
        self.assertLess(success_noisy, success_ideal,
                        f"Noisy ({success_noisy}) should be < ideal ({success_ideal})")


if __name__ == "__main__":
    unittest.main()
