"""Tests for endogenous population dynamics (replicator equation)."""

import numpy as np
import pytest

from src.agents import ModelParams
from src.replicator import simulate_market_endogenous, ReplicatorParams
from src.simulation import simulate_market


@pytest.fixture
def params():
    return ModelParams(T=5.0)


@pytest.fixture
def rep_params():
    return ReplicatorParams(eta_fast=0.5, eta_slow=0.05)


class TestConservation:
    """Population fractions must always sum to 1."""

    def test_sum_to_one(self, params, rep_params):
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep_params, seed=42)
        pop_sums = result["population"].sum(axis=1)
        np.testing.assert_allclose(pop_sums, 1.0, atol=1e-12)

    def test_non_negative(self, params, rep_params):
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep_params, seed=42)
        assert np.all(result["population"] >= 0), "Population fractions must be non-negative"

    def test_minimum_population(self, params):
        """p_min constraint should prevent extinction."""
        rep = ReplicatorParams(eta_fast=5.0, eta_slow=0.5, p_min=0.01)
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep, seed=42)
        assert np.all(result["population"] >= rep.p_min - 1e-12)

    def test_conservation_aggressive_adaptation(self, params):
        """Even with very aggressive adaptation, sum should be 1."""
        rep = ReplicatorParams(eta_fast=10.0, eta_slow=1.0)
        result = simulate_market_endogenous((0.5, 0.3, 0.2), params, rep, seed=0)
        pop_sums = result["population"].sum(axis=1)
        np.testing.assert_allclose(pop_sums, 1.0, atol=1e-10)


class TestBoundaryRecovery:
    """eta=0 should recover the static model exactly."""

    def test_zero_adaptation_matches_static(self):
        params = ModelParams(T=2.0)
        p = (0.4, 0.35, 0.25)

        static = simulate_market(p, params, seed=42, force_python=True)
        endo = simulate_market_endogenous(
            p, params, ReplicatorParams(eta_fast=0.0, eta_slow=0.0), seed=42,
        )

        np.testing.assert_array_equal(static["price"], endo["price"])
        np.testing.assert_array_equal(static["fair_value"], endo["fair_value"])

    def test_zero_adaptation_constant_population(self):
        params = ModelParams(T=2.0)
        p = (0.4, 0.35, 0.25)
        endo = simulate_market_endogenous(
            p, params, ReplicatorParams(eta_fast=0.0, eta_slow=0.0), seed=42,
        )

        for t in range(len(endo["population"])):
            np.testing.assert_allclose(endo["population"][t], np.array(p), atol=1e-12)


class TestSingleTimescaleRecovery:
    """Setting one eta to 0 should reduce to single-timescale replicator."""

    def test_fast_only(self, params):
        rep = ReplicatorParams(eta_fast=5.0, eta_slow=0.0)
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep, seed=42)
        # Population should still evolve (not static)
        pop_range = np.max(result["population"], axis=0) - np.min(result["population"], axis=0)
        assert np.any(pop_range > 1e-4), "Fast-only should cause population changes"

    def test_slow_only(self, params):
        rep = ReplicatorParams(eta_fast=0.0, eta_slow=0.5)
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep, seed=42)
        pop_range = np.max(result["population"], axis=0) - np.min(result["population"], axis=0)
        assert np.any(pop_range > 0.0001), "Slow-only should cause population changes"


class TestKnownLimits:
    def test_adaptation_rate_effect(self):
        """Higher adaptation rates should produce larger population changes."""
        params = ModelParams(T=10.0)

        # Low adaptation
        rep_low = ReplicatorParams(eta_fast=0.1, eta_slow=0.01)
        r_low = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep_low, seed=42)
        drift_low = np.max(np.abs(r_low["population"][-1] - r_low["population"][0]))

        # High adaptation
        rep_high = ReplicatorParams(eta_fast=10.0, eta_slow=1.0)
        r_high = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep_high, seed=42)
        drift_high = np.max(np.abs(r_high["population"][-1] - r_high["population"][0]))

        assert drift_high > drift_low, (
            f"Higher adaptation should cause more drift: {drift_high} vs {drift_low}"
        )


class TestOutputStructure:
    def test_has_population_fields(self, params, rep_params):
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep_params, seed=0)
        assert "population" in result
        assert "p1" in result
        assert "p2" in result
        assert "p3" in result
        assert "pnl_short" in result
        assert "pnl_long" in result

    def test_population_shape(self, params, rep_params):
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep_params, seed=0)
        n = params.n_steps + 1
        assert result["population"].shape == (n, 3)
        assert len(result["p1"]) == n

    def test_has_standard_fields(self, params, rep_params):
        result = simulate_market_endogenous((0.33, 0.34, 0.33), params, rep_params, seed=0)
        standard_keys = {"time", "price", "fair_value", "sentiment", "momentum",
                         "demand_value", "demand_emotion", "demand_noise",
                         "returns", "mispricing"}
        assert standard_keys.issubset(set(result.keys()))
