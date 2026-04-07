"""Market Population Dynamics — Agent-based market simulation framework.

Three behaviorally-distinct trader populations on the 2-simplex:
  Type I  (Value)   — mean-reversion to fair value
  Type II (Emotion) — momentum + prospect-theoretic sentiment
  Type III (Noise)  — zero-intelligence random order flow
"""

from src.agents import prospect_value, ModelParams
from src.simulation import simulate_market
from src.analysis import compute_metrics
from src.replicator import simulate_market_endogenous, ReplicatorParams

__all__ = [
    "prospect_value",
    "ModelParams",
    "simulate_market",
    "compute_metrics",
    "simulate_market_endogenous",
    "ReplicatorParams",
]
