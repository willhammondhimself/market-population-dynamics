"""Simplex sweep and metrics computation.

Functions for running Monte Carlo sweeps across the 2-simplex
and computing summary statistics (mispricing, excess vol, convergence).
"""

import numpy as np
from typing import Tuple, List
from time import time as timer

from src.agents import ModelParams
from src.simulation import simulate_market


def compute_metrics(
    p: Tuple[float, float, float],
    params: ModelParams = None,
    n_mc: int = 50,
    convergence_threshold: float = 0.1,
) -> dict:
    """Compute summary statistics for a given population mix via Monte Carlo.

    Args:
        p: (p1, p2, p3) population fractions.
        params: model parameters.
        n_mc: number of Monte Carlo paths.
        convergence_threshold: max |log(P/V*)| for convergence.

    Returns:
        dict with mean_mispricing, std_mispricing, excess_vol, convergence_prob.
    """
    if params is None:
        params = ModelParams()

    mispricings = []
    vol_ratios = []
    converged = 0

    for i in range(n_mc):
        result = simulate_market(p, params, seed=i * 1000 + hash(p) % 1000)

        # Mean absolute mispricing (second half to allow equilibration)
        half = len(result["mispricing"]) // 2
        mean_mispricing = np.mean(np.abs(result["mispricing"][half:]))
        mispricings.append(mean_mispricing)

        # Excess volatility ratio
        price_vol = np.std(result["returns"][1:]) * np.sqrt(252)
        fair_returns = np.diff(np.log(result["fair_value"]))
        fair_vol = np.std(fair_returns) * np.sqrt(252)
        if fair_vol > 1e-10:
            vol_ratios.append(price_vol / fair_vol)

        # Convergence check (final mispricing)
        if np.abs(result["mispricing"][-1]) < convergence_threshold:
            converged += 1

    return {
        "mean_mispricing": np.mean(mispricings),
        "std_mispricing": np.std(mispricings),
        "excess_vol": np.mean(vol_ratios) if vol_ratios else np.nan,
        "convergence_prob": converged / n_mc,
    }


def simplex_grid(n_grid: int = 20) -> List[Tuple[float, float, float]]:
    """Generate a triangular grid of points on the 2-simplex.

    Args:
        n_grid: number of divisions along each edge.

    Returns:
        list of (p1, p2, p3) tuples.
    """
    points = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            points.append((i / n_grid, j / n_grid, k / n_grid))
    return points


def sweep_simplex(
    n_grid: int = 20,
    n_mc: int = 30,
    params: ModelParams = None,
    verbose: bool = True,
) -> List[dict]:
    """Run Monte Carlo sweep across the full 2-simplex.

    Args:
        n_grid: grid resolution.
        n_mc: Monte Carlo paths per grid point.
        params: model parameters.
        verbose: print progress updates.

    Returns:
        list of dicts with p1, p2, p3 and metric fields.
    """
    if params is None:
        params = ModelParams()

    grid_points = simplex_grid(n_grid)
    if verbose:
        total = len(grid_points) * n_mc
        print(f"Sweeping {len(grid_points)} points with {n_mc} MC paths each ({total} total)...")

    results = []
    t_start = timer()

    for idx, p in enumerate(grid_points):
        metrics = compute_metrics(p, params, n_mc=n_mc)
        results.append({"p1": p[0], "p2": p[1], "p3": p[2], **metrics})

        if verbose and (idx + 1) % 50 == 0:
            elapsed = timer() - t_start
            rate = (idx + 1) / elapsed
            remaining = (len(grid_points) - idx - 1) / rate
            print(f"  {idx + 1}/{len(grid_points)} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    if verbose:
        elapsed = timer() - t_start
        print(f"Done! {len(grid_points)} points in {elapsed:.1f}s")

    return results
