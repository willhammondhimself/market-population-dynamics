"""Two-timescale replicator dynamics with P&L-driven population switching.

The key novelty: agents switch types based on relative P&L performance.
Population evolves via:
  dp_i/dt = eta_fast * p_i * (pi_i^short - pi_bar^short)
          + eta_slow * p_i * (pi_i^long  - pi_bar^long)

where pi_i^short/long are exponentially-weighted P&L over short/long windows.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

from src.agents import ModelParams, prospect_value


@dataclass
class ReplicatorParams:
    """Parameters for two-timescale replicator dynamics."""

    # Adaptation speeds
    eta_fast: float = 0.5        # fast (retail) adaptation rate
    eta_slow: float = 0.05       # slow (institutional) adaptation rate

    # P&L tracking windows
    tau_short: float = 20 / 252  # short P&L window (~20 days)
    tau_long: float = 60 / 252   # long P&L window (~60 days / ~1 quarter)

    # Population constraints
    p_min: float = 0.01          # minimum population fraction (prevents extinction)

    @property
    def short_decay(self) -> float:
        """EWMA decay for short P&L window."""
        return np.exp(-1.0 / (self.tau_short * 252))

    @property
    def long_decay(self) -> float:
        """EWMA decay for long P&L window."""
        return np.exp(-1.0 / (self.tau_long * 252))


def compute_agent_pnl(
    price: float,
    prev_price: float,
    fair_value: float,
    momentum: float,
    sentiment: float,
    noise_demand: float,
    params: ModelParams,
) -> Tuple[float, float, float]:
    """Compute instantaneous P&L for each agent type.

    Type I:  profits when price reverts to fair value
    Type II: profits from momentum (position aligned with trend)
    Type III: zero expected P&L (random positions)

    Returns:
        (pnl_1, pnl_2, pnl_3) instantaneous P&L for each type.
    """
    ret = np.log(price / prev_price) if prev_price > 0 else 0.0

    # Type I: holds position proportional to (V* - P)/P, profits from reversion
    d1 = params.kappa * (fair_value - prev_price) / prev_price
    pnl_1 = d1 * ret

    # Type II: holds position from momentum + sentiment, profits from continuation
    d2 = params.alpha * momentum + params.beta * sentiment
    pnl_2 = d2 * ret

    # Type III: random position, zero expected P&L
    pnl_3 = noise_demand * ret

    return pnl_1, pnl_2, pnl_3


def simulate_market_endogenous(
    p0: Tuple[float, float, float],
    params: ModelParams = None,
    rep_params: ReplicatorParams = None,
    seed: int = None,
) -> dict:
    """Simulate market with endogenous population dynamics.

    The population mix evolves over time via two-timescale replicator dynamics
    driven by relative P&L performance.

    Args:
        p0: initial population fractions (p1, p2, p3), must sum to 1.
        params: market model parameters.
        rep_params: replicator dynamics parameters.
        seed: random seed.

    Returns:
        dict with all standard fields plus population trajectories and P&L.
    """
    if params is None:
        params = ModelParams()
    if rep_params is None:
        rep_params = ReplicatorParams()
    if seed is not None:
        np.random.seed(seed)

    p1_0, p2_0, p3_0 = p0
    assert abs(p1_0 + p2_0 + p3_0 - 1.0) < 1e-10

    n = params.n_steps
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    # Arrays
    P = np.zeros(n + 1)
    V = np.zeros(n + 1)
    S = np.zeros(n + 1)
    m = np.zeros(n + 1)
    D1 = np.zeros(n + 1)
    D2 = np.zeros(n + 1)
    D3 = np.zeros(n + 1)
    returns = np.zeros(n + 1)

    # Population trajectories
    pop = np.zeros((n + 1, 3))
    pop[0] = [p1_0, p2_0, p3_0]

    # P&L tracking (short and long EWMA)
    pnl_short = np.zeros((n + 1, 3))
    pnl_long = np.zeros((n + 1, 3))

    # Initial conditions
    P[0] = params.P0
    V[0] = params.V_star

    # Random draws
    dW_f = np.random.randn(n) * sqrt_dt
    dW_s = np.random.randn(n) * sqrt_dt
    dW_n = np.random.randn(n) * sqrt_dt

    ewma_decay = params.ewma_decay
    short_decay = rep_params.short_decay
    long_decay = rep_params.long_decay

    for t in range(n):
        p1, p2, p3 = pop[t]

        # Fair value: GBM
        V[t + 1] = V[t] * np.exp(-0.5 * params.sigma_f**2 * dt + params.sigma_f * dW_f[t])

        # Demands
        D1[t] = params.kappa * (V[t] - P[t]) / P[t]
        D2[t] = params.alpha * m[t] + params.beta * S[t]
        D3[t] = params.sigma_n * dW_n[t] / sqrt_dt

        # Aggregate demand with current population
        D_total = p1 * D1[t] + p2 * D2[t] + p3 * D3[t]

        # Price update
        dP = params.lam * D_total * dt
        P[t + 1] = P[t] * np.exp(dP - 0.5 * (params.lam * D_total)**2 * dt**2)

        # Return
        returns[t + 1] = np.log(P[t + 1] / P[t])

        # Momentum
        m[t + 1] = ewma_decay * m[t] + (1 - ewma_decay) * returns[t + 1]

        # Sentiment
        v_r = prospect_value(
            np.array([returns[t + 1]]),
            lam_pt=params.lam_pt,
            a=params.a_pt,
        )[0]
        dS = -params.gamma * S[t] * dt + params.delta * v_r * dt + params.sigma_s * dW_s[t]
        S[t + 1] = S[t] + dS

        # Compute P&L for each agent type
        pnl_1, pnl_2, pnl_3 = compute_agent_pnl(
            P[t + 1], P[t], V[t], m[t], S[t], D3[t], params,
        )
        instant_pnl = np.array([pnl_1, pnl_2, pnl_3])

        # Update EWMA P&L trackers
        pnl_short[t + 1] = short_decay * pnl_short[t] + (1 - short_decay) * instant_pnl
        pnl_long[t + 1] = long_decay * pnl_long[t] + (1 - long_decay) * instant_pnl

        # Two-timescale replicator dynamics
        p_vec = pop[t]

        # Short-timescale (fast/retail)
        pi_bar_short = np.dot(p_vec, pnl_short[t + 1])
        dp_fast = rep_params.eta_fast * p_vec * (pnl_short[t + 1] - pi_bar_short)

        # Long-timescale (slow/institutional)
        pi_bar_long = np.dot(p_vec, pnl_long[t + 1])
        dp_slow = rep_params.eta_slow * p_vec * (pnl_long[t + 1] - pi_bar_long)

        # Update population
        p_new = p_vec + (dp_fast + dp_slow) * dt

        # Enforce constraints: non-negative, sum to 1, minimum population
        p_new = np.maximum(p_new, rep_params.p_min)
        p_new = p_new / p_new.sum()  # re-normalize

        pop[t + 1] = p_new

    time = np.arange(n + 1) * dt

    return {
        "time": time,
        "price": P,
        "fair_value": V,
        "sentiment": S,
        "momentum": m,
        "demand_value": D1,
        "demand_emotion": D2,
        "demand_noise": D3,
        "returns": returns,
        "mispricing": np.log(P / V),
        # Endogenous dynamics outputs
        "population": pop,              # (n+1, 3) array
        "p1": pop[:, 0],
        "p2": pop[:, 1],
        "p3": pop[:, 2],
        "pnl_short": pnl_short,         # (n+1, 3) EWMA short P&L
        "pnl_long": pnl_long,           # (n+1, 3) EWMA long P&L
    }
