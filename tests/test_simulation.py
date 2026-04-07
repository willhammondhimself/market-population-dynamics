"""Integration tests for the simulation engine."""

import numpy as np
import pytest

from src.agents import ModelParams
from src.simulation import simulate_market


@pytest.fixture
def short_params():
    """Quick simulation params for testing (1 year, daily)."""
    return ModelParams(T=1.0, dt=1 / 252)


class TestSimulateMarket:
    def test_output_keys(self, short_params):
        result = simulate_market((0.5, 0.3, 0.2), short_params, seed=0)
        expected_keys = {
            "time", "price", "fair_value", "sentiment", "momentum",
            "demand_value", "demand_emotion", "demand_noise",
            "returns", "mispricing",
        }
        assert set(result.keys()) == expected_keys

    def test_array_lengths(self, short_params):
        result = simulate_market((0.5, 0.3, 0.2), short_params, seed=0)
        n = short_params.n_steps + 1
        for key in result:
            assert len(result[key]) == n, f"{key} has wrong length"

    def test_positive_prices(self, short_params):
        result = simulate_market((0.3, 0.4, 0.3), short_params, seed=42)
        assert np.all(result["price"] > 0), "Prices must be positive (log-normal dynamics)"

    def test_deterministic_with_seed(self, short_params):
        r1 = simulate_market((0.5, 0.3, 0.2), short_params, seed=123)
        r2 = simulate_market((0.5, 0.3, 0.2), short_params, seed=123)
        np.testing.assert_array_equal(r1["price"], r2["price"])

    def test_population_sum_assertion(self, short_params):
        with pytest.raises(AssertionError):
            simulate_market((0.5, 0.3, 0.3), short_params, seed=0, force_python=True)


class TestBoundaryCases:
    """Verify model behavior at simplex vertices."""

    def test_emh_limit(self):
        """Type I only: price should converge to fair value."""
        params = ModelParams(T=10.0, kappa=2.0)
        result = simulate_market((1.0, 0.0, 0.0), params, seed=0)

        # Final mispricing should be small
        final_mispricing = abs(result["mispricing"][-1])
        assert final_mispricing < 0.5, f"EMH limit: final mispricing {final_mispricing} too large"

        # Second-half average mispricing should be small
        half = len(result["mispricing"]) // 2
        avg = np.mean(np.abs(result["mispricing"][half:]))
        assert avg < 0.2, f"EMH limit: avg mispricing {avg} too large"

    def test_noise_only_random_walk(self):
        """Type III only: realized vol should be close to noise vol, not explosive."""
        params = ModelParams(T=5.0)
        result = simulate_market((0.0, 0.0, 1.0), params, seed=0)

        # Price should remain in a reasonable range (not blow up)
        log_price_range = np.max(np.log(result["price"])) - np.min(np.log(result["price"]))
        assert log_price_range < 5.0, f"Noise-only price range {log_price_range} too large"

    def test_emotion_only_volatility(self):
        """Type II only: should produce excess volatility."""
        params = ModelParams(T=5.0)
        result = simulate_market((0.0, 1.0, 0.0), params, seed=0)

        # Excess vol should be > 1 (more volatile than fundamentals)
        price_vol = np.std(result["returns"][1:]) * np.sqrt(252)
        fair_returns = np.diff(np.log(result["fair_value"]))
        fair_vol = np.std(fair_returns) * np.sqrt(252)
        if fair_vol > 1e-10:
            excess_vol = price_vol / fair_vol
            assert excess_vol > 1.0, f"Emotion-only excess vol {excess_vol} should be > 1"
