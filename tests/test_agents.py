"""Unit tests for agent demand functions and prospect theory."""

import numpy as np
import pytest

from src.agents import ModelParams, prospect_value


class TestProspectValue:
    def test_positive_returns(self):
        r = np.array([0.01, 0.05, 0.10])
        v = prospect_value(r)
        assert np.all(v > 0), "Positive returns should give positive value"

    def test_negative_returns(self):
        r = np.array([-0.01, -0.05, -0.10])
        v = prospect_value(r)
        assert np.all(v < 0), "Negative returns should give negative value"

    def test_loss_aversion(self):
        """Losses hurt more than equivalent gains feel good."""
        r = np.array([0.05])
        v_gain = prospect_value(r)[0]
        v_loss = prospect_value(-r)[0]
        assert abs(v_loss) > abs(v_gain), "Loss aversion: |v(-r)| > |v(r)|"

    def test_zero_return(self):
        r = np.array([0.0])
        v = prospect_value(r)
        assert v[0] == 0.0, "v(0) should be 0"

    def test_symmetry_ratio(self):
        """The loss/gain ratio should approximate lambda_PT for small returns."""
        r = np.array([0.001])
        v_gain = prospect_value(r, lam_pt=2.25, a=1.0)[0]
        v_loss = prospect_value(-r, lam_pt=2.25, a=1.0)[0]
        ratio = abs(v_loss) / abs(v_gain)
        assert abs(ratio - 2.25) < 0.01, f"Expected ratio ~2.25, got {ratio}"

    def test_concavity_gains(self):
        """Value function should be concave for gains (a < 1)."""
        r1 = np.array([0.01])
        r2 = np.array([0.02])
        v1 = prospect_value(r1)[0]
        v2 = prospect_value(r2)[0]
        # Concavity: v(2r) < 2*v(r) for a < 1
        assert v2 < 2 * v1, "Gains should be concave"


class TestModelParams:
    def test_defaults(self):
        p = ModelParams()
        assert p.T == 10.0
        assert p.n_steps == 2520
        assert abs(p.dt - 1 / 252) < 1e-10

    def test_ewma_decay(self):
        p = ModelParams()
        assert 0 < p.ewma_decay < 1, "EWMA decay should be in (0, 1)"

    def test_custom_params(self):
        p = ModelParams(T=5.0, dt=1 / 252, kappa=3.0)
        assert p.T == 5.0
        assert p.kappa == 3.0
        assert p.n_steps == 1260
