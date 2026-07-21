# =============================================================
# test_circuits.py — Unit Tests for Quantum Circuit Builders
# =============================================================
"""
Tests verify that DJ, BV, and Simon's circuits produce correct
outputs under ideal (noiseless) conditions.
"""

import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qiskit_aer import AerSimulator
from qiskit import transpile

from src.circuits import (
    build_dj_oracle,
    build_dj_circuit,
    build_bv_circuit,
    build_simon_circuit,
    check_simon_success,
    generate_secret_strings,
    generate_all_circuit_configs,
    get_circuit_stats,
)


class TestDeutschJozsa(unittest.TestCase):
    """Test Deutsch-Jozsa circuit correctness."""

    def _run_noiseless(self, circuit, shots=4096):
        """Helper: run circuit on ideal simulator."""
        sim = AerSimulator()
        transpiled = transpile(circuit, sim)
        result = sim.run(transpiled, shots=shots).result()
        return result.get_counts()

    def test_constant_0_oracle(self):
        """Constant-0 oracle → should measure all zeros."""
        for n in [2, 3, 4]:
            oracle = build_dj_oracle(n, "constant_0")
            circuit = build_dj_circuit(n, oracle)
            counts = self._run_noiseless(circuit)
            expected = "0" * n
            self.assertIn(expected, counts,
                          f"n={n}: Expected '{expected}' in counts {counts}")
            self.assertEqual(counts[expected], 4096,
                             f"n={n}: Expected 100% '{expected}'")

    def test_constant_1_oracle(self):
        """Constant-1 oracle → should also measure all zeros."""
        for n in [2, 3, 4]:
            oracle = build_dj_oracle(n, "constant_1")
            circuit = build_dj_circuit(n, oracle)
            counts = self._run_noiseless(circuit)
            expected = "0" * n
            self.assertIn(expected, counts,
                          f"n={n}: Expected '{expected}' in counts {counts}")
            self.assertEqual(counts[expected], 4096,
                             f"n={n}: Expected 100% '{expected}'")

    def test_balanced_oracle(self):
        """Balanced oracle → should NOT measure all zeros."""
        for n in [2, 3, 4]:
            oracle = build_dj_oracle(n, "balanced", seed=42)
            circuit = build_dj_circuit(n, oracle)
            counts = self._run_noiseless(circuit)
            zero_string = "0" * n
            self.assertNotIn(zero_string, counts,
                             f"n={n}: Balanced should NOT produce '{zero_string}', "
                             f"got {counts}")


class TestBernsteinVazirani(unittest.TestCase):
    """Test Bernstein-Vazirani circuit correctness."""

    def _run_noiseless(self, circuit, shots=4096):
        sim = AerSimulator()
        transpiled = transpile(circuit, sim)
        result = sim.run(transpiled, shots=shots).result()
        return result.get_counts()

    def test_recovers_secret_string(self):
        """BV should recover the secret string with 100% probability."""
        test_cases = [
            (2, "10"),
            (3, "101"),
            (4, "1101"),
            (5, "10110"),
        ]
        for n, secret in test_cases:
            circuit = build_bv_circuit(n, secret)
            counts = self._run_noiseless(circuit)
            self.assertIn(secret, counts,
                          f"n={n}, secret={secret}: Not found in {counts}")
            self.assertEqual(counts[secret], 4096,
                             f"n={n}, secret={secret}: Expected 100% recovery")


class TestSimons(unittest.TestCase):
    """Test Simon's circuit correctness."""

    def _run_noiseless(self, circuit, shots=4096):
        sim = AerSimulator()
        transpiled = transpile(circuit, sim)
        result = sim.run(transpiled, shots=shots).result()
        return result.get_counts()

    def test_outputs_satisfy_dot_product(self):
        """All Simon's outputs z should satisfy z·s = 0 (mod 2)."""
        test_cases = [
            (2, "10"),
            (3, "110"),
            (3, "101"),
        ]
        for n, secret in test_cases:
            circuit = build_simon_circuit(n, secret)
            counts = self._run_noiseless(circuit)
            success = check_simon_success(counts, secret, n)
            self.assertAlmostEqual(success, 1.0, places=2,
                                   msg=f"n={n}, s={secret}: success={success}")


class TestUtilities(unittest.TestCase):
    """Test circuit utility functions."""

    def test_generate_secret_strings(self):
        """Should generate non-trivial, unique bitstrings."""
        strings = generate_secret_strings(4, count=3)
        self.assertEqual(len(strings), 3)
        for s in strings:
            self.assertEqual(len(s), 4)
            self.assertNotEqual(s, "0000")  # Non-trivial

    def test_generate_all_configs(self):
        """Should generate configs for all algorithms."""
        configs = generate_all_circuit_configs(range(2, 4))
        self.assertGreater(len(configs), 0)
        algos = set(c["algorithm"] for c in configs)
        self.assertIn("deutsch_jozsa", algos)
        self.assertIn("bernstein_vazirani", algos)
        self.assertIn("simons", algos)

    def test_circuit_stats(self):
        """Circuit stats should return positive integers."""
        circuit = build_bv_circuit(3, "101")
        stats = get_circuit_stats(circuit)
        self.assertGreater(stats["circuit_depth"], 0)
        self.assertGreaterEqual(stats["gate_count_2q"], 0)
        self.assertGreater(stats["total_gate_count"], 0)


if __name__ == "__main__":
    unittest.main()
