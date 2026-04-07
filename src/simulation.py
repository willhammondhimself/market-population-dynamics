"""Core simulation engine.

simulate_market(p, params, seed) -> dict

Produces time series of prices, fair value, sentiment, demands, and returns
for a given population mix on the 2-simplex.
"""

import numpy as np
from typing import Tuple

from src.agents import ModelParams, prospect_value

# Try to use the Rust engine if available
_USE_RUST = False
try:
    from market_pop_dynamics_rust import simulate_market as _rust_simulate
    _USE_RUST = True
except ImportError:
    pass


def simulate_market(
    p: Tuple[float, float, float],
    params: ModelParams = None,
    seed: int = None,
    force_python: bool = False,
) -> dict:
    """Simulate market dynamics for a given population mix.

    Args:
        p: (p1, p2, p3) population fractions, must sum to 1.
        params: model parameters (defaults to ModelParams()).
        seed: random seed for reproducibility.
        force_python: skip Rust engine even if available.

    Returns:
        dict with time series: prices, fair_value, sentiment, demands, returns, mispricing.
    """
    if params is None:
        params = ModelParams()

    if _USE_RUST and not force_python:
        return _simulate_rust(p, params, seed)
    return _simulate_python(p, params, seed)


def _simulate_python(
    p: Tuple[float, float, float],
    params: ModelParams,
    seed: int = None,
) -> dict:
    """Pure-Python/NumPy simulation engine."""
    if seed is not None:
        np.random.seed(seed)

    p1, p2, p3 = p
    assert abs(p1 + p2 + p3 - 1.0) < 1e-10, f"Population fractions must sum to 1, got {p1 + p2 + p3}"

    n = params.n_steps
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    # Pre-allocate arrays
    P = np.zeros(n + 1)
    V = np.zeros(n + 1)
    S = np.zeros(n + 1)
    m = np.zeros(n + 1)
    D1 = np.zeros(n + 1)
    D2 = np.zeros(n + 1)
    D3 = np.zeros(n + 1)
    returns = np.zeros(n + 1)

    # Initial conditions
    P[0] = params.P0
    V[0] = params.V_star
    S[0] = 0.0
    m[0] = 0.0

    # Random draws
    dW_f = np.random.randn(n) * sqrt_dt
    dW_s = np.random.randn(n) * sqrt_dt
    dW_n = np.random.randn(n) * sqrt_dt

    decay = params.ewma_decay

    for t in range(n):
        # Fair value: GBM with zero drift
        V[t + 1] = V[t] * np.exp(-0.5 * params.sigma_f**2 * dt + params.sigma_f * dW_f[t])

        # Type I: mean-revert to fair value (normalized by price)
        D1[t] = params.kappa * (V[t] - P[t]) / P[t]

        # Type II: momentum + sentiment
        D2[t] = params.alpha * m[t] + params.beta * S[t]

        # Type III: pure noise
        D3[t] = params.sigma_n * dW_n[t] / sqrt_dt

        # Aggregate demand -> price change
        D_total = p1 * D1[t] + p2 * D2[t] + p3 * D3[t]

        # Price update (log-normal to prevent negative prices)
        dP = params.lam * D_total * dt
        P[t + 1] = P[t] * np.exp(dP - 0.5 * (params.lam * D_total) ** 2 * dt**2)

        # Log return
        returns[t + 1] = np.log(P[t + 1] / P[t])

        # Update momentum (EWMA of returns)
        m[t + 1] = decay * m[t] + (1 - decay) * returns[t + 1]

        # Update sentiment with prospect-theoretic asymmetry
        v_r = prospect_value(
            np.array([returns[t + 1]]),
            lam_pt=params.lam_pt,
            a=params.a_pt,
        )[0]
        dS = -params.gamma * S[t] * dt + params.delta * v_r * dt + params.sigma_s * dW_s[t]
        S[t + 1] = S[t] + dS

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
    }


def _simulate_rust(
    p: Tuple[float, float, float],
    params: ModelParams,
    seed: int = None,
) -> dict:
    """Dispatch to Rust engine and convert result to Python dict."""
    from market_pop_dynamics_rust import simulate_market as rust_sim

    result = rust_sim(
        p[0], p[1], p[2],
        params.T, params.dt, params.V_star, params.P0,
        params.lam, params.sigma_f, params.kappa,
        params.alpha, params.beta, params.tau_m,
        params.gamma, params.delta, params.sigma_s,
        params.lam_pt, params.a_pt, params.sigma_n,
        seed if seed is not None else -1,
    )
    return result
