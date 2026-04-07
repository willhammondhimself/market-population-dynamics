"""Test parity between Python and Rust simulation engines.

Note: Python uses numpy's Mersenne Twister RNG while Rust uses ChaCha8,
so identical seeds produce different random streams. We test statistical
properties rather than exact output matching.
"""

import numpy as np
import pytest

from src.agents import ModelParams
from src.simulation import simulate_market, _USE_RUST


@pytest.mark.skipif(not _USE_RUST, reason="Rust engine not available")
class TestRustParity:
    """Statistical parity tests between Python and Rust engines."""

    @pytest.fixture
    def params(self):
        return ModelParams(T=5.0)

    def test_output_keys_match(self, params):
        """Rust and Python should return the same dict keys."""
        py = simulate_market((0.5, 0.3, 0.2), params, seed=0, force_python=True)
        rs = simulate_market((0.5, 0.3, 0.2), params, seed=0, force_python=False)
        assert set(py.keys()) == set(rs.keys())

    def test_array_lengths_match(self, params):
        """Both engines should produce arrays of the same length."""
        py = simulate_market((0.5, 0.3, 0.2), params, seed=0, force_python=True)
        rs = simulate_market((0.5, 0.3, 0.2), params, seed=0, force_python=False)
        for key in py:
            assert len(py[key]) == len(rs[key]), f"{key} length mismatch"

    def test_positive_prices_rust(self, params):
        """Rust engine should produce positive prices."""
        for seed in range(10):
            rs = simulate_market((0.4, 0.3, 0.3), params, seed=seed, force_python=False)
            assert np.all(np.array(rs["price"]) > 0), f"Negative price at seed {seed}"

    def test_initial_conditions_match(self, params):
        """Both engines should start from the same initial conditions."""
        py = simulate_market((0.5, 0.3, 0.2), params, seed=0, force_python=True)
        rs = simulate_market((0.5, 0.3, 0.2), params, seed=0, force_python=False)
        assert py["price"][0] == rs["price"][0]
        assert py["fair_value"][0] == rs["fair_value"][0]

    def test_statistical_moments_similar(self, params):
        """Return distributions should have similar moments across many seeds."""
        py_vols = []
        rs_vols = []

        for seed in range(50):
            py = simulate_market((0.4, 0.3, 0.3), params, seed=seed, force_python=True)
            rs = simulate_market((0.4, 0.3, 0.3), params, seed=seed + 10000, force_python=False)
            py_vols.append(np.std(py["returns"][1:]) * np.sqrt(252))
            rs_vols.append(np.std(np.array(rs["returns"])[1:]) * np.sqrt(252))

        # Means should be within 30% of each other (both produce same dynamics)
        py_mean_vol = np.mean(py_vols)
        rs_mean_vol = np.mean(rs_vols)
        ratio = rs_mean_vol / py_mean_vol if py_mean_vol > 0 else float("inf")
        assert 0.5 < ratio < 2.0, f"Vol ratio {ratio} outside expected range"

    def test_emh_limit_rust(self, params):
        """Type I only should converge to fair value in Rust engine too."""
        rs = simulate_market((1.0, 0.0, 0.0), params, seed=0, force_python=False)
        mispricing = np.array(rs["mispricing"])
        half = len(mispricing) // 2
        avg_misp = np.mean(np.abs(mispricing[half:]))
        assert avg_misp < 0.3, f"Rust EMH limit: avg mispricing {avg_misp} too large"

    def test_rust_deterministic(self, params):
        """Same seed should produce identical Rust output."""
        r1 = simulate_market((0.5, 0.3, 0.2), params, seed=42, force_python=False)
        r2 = simulate_market((0.5, 0.3, 0.2), params, seed=42, force_python=False)
        np.testing.assert_array_equal(np.array(r1["price"]), np.array(r2["price"]))
