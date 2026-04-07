"""Agent type definitions and demand functions.

Type I  (Value):   D1(t) = kappa * (V* - P(t)) / P(t)
Type II (Emotion): D2(t) = alpha * m(t) + beta * S(t)
Type III (Noise):  D3(t) = sigma_n * epsilon(t)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ModelParams:
    """Parameters for the market population dynamics model."""

    # Simulation
    T: float = 10.0          # total time (years)
    dt: float = 1 / 252      # time step (1 trading day)
    V_star: float = 100.0    # fair value
    P0: float = 100.0        # initial price

    # Market microstructure
    lam: float = 0.5         # price impact (Kyle's lambda)
    sigma_f: float = 0.01    # fundamental news volatility (daily)

    # Type I: Data-driven / Value
    kappa: float = 2.0       # mean-reversion speed

    # Type II: Intuition / Emotion
    alpha: float = 5.0       # momentum sensitivity
    beta: float = 3.0        # sentiment sensitivity
    tau_m: float = 20 / 252  # momentum lookback (~20 days)
    gamma: float = 5.0       # sentiment mean-reversion
    delta: float = 10.0      # sentiment response to returns
    sigma_s: float = 0.5     # sentiment noise
    lam_pt: float = 2.25     # prospect theory loss aversion
    a_pt: float = 0.88       # prospect theory curvature

    # Type III: Noise
    sigma_n: float = 1.0     # noise trader demand volatility

    @property
    def n_steps(self) -> int:
        return int(self.T / self.dt)

    @property
    def ewma_decay(self) -> float:
        """EWMA decay factor for momentum calculation."""
        return np.exp(-self.dt / self.tau_m)


def prospect_value(r: np.ndarray, lam_pt: float = 2.25, a: float = 0.88) -> np.ndarray:
    """Kahneman-Tversky value function.

    v(r) = r^a            if r >= 0
    v(r) = -lam_pt * |r|^a  if r < 0

    Key property: steeper for losses than gains (loss aversion).
    """
    return np.where(
        r >= 0,
        np.power(np.maximum(r, 0), a),
        -lam_pt * np.power(np.maximum(-r, 0), a),
    )
