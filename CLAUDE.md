# Market Population Dynamics

Agent-based market simulation: three behaviorally-distinct trader populations on the 2-simplex.
**Author**: Will Hammond | **Advisor**: Alan Kahn

<!-- AUTO-MANAGED: build-commands -->
## Build & Test Commands

```bash
# Run all tests
pytest tests/

# Run single test module
pytest tests/test_replicator.py -v

# Build Rust extension (from rust_core/)
cd rust_core && maturin develop

# Build Rust in release mode
cd rust_core && maturin develop --release
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Architecture

```
src/
  __init__.py          # Exports: ModelParams, prospect_value, simulate_market,
                       #          compute_metrics, simulate_market_endogenous, ReplicatorParams
  agents.py            # ModelParams dataclass, prospect_value() (vectorized K-T)
  simulation.py        # simulate_market() — auto-selects Rust or Python backend
  replicator.py        # simulate_market_endogenous(), ReplicatorParams, compute_agent_pnl()
  analysis.py          # compute_metrics(), simplex_grid(), sweep_simplex()
  visualization.py     # ternary_coords(), plot_ternary_heatmap(), plot_price_paths(),
                       #   plot_demand_decomposition(), plot_mispricing_distributions()

rust_core/
  Cargo.toml           # cdylib crate: market_pop_dynamics_rust
  pyproject.toml       # maturin build config
  src/
    lib.rs             # PyO3 bindings: simulate_market, simulate_market_endogenous_py
    types.rs           # SimParams, SimResult structs
    agents.rs          # prospect_value() scalar, unit tests
    simulation.rs      # simulate_market() — matches Python _simulate_python exactly
    replicator.rs      # simulate_market_endogenous(), ReplicatorParams, EndogenousResult

tests/
  test_agents.py       # ModelParams, prospect_value
  test_simulation.py   # Output structure, boundary cases (EMH limit)
  test_replicator.py   # Conservation, boundary recovery, single-timescale recovery, known limits
  test_rust_parity.py  # Statistical parity between Python and Rust engines

notebooks/
  01_base_model_prototype.ipynb   # Static simplex sweep, phase diagrams
  02_endogenous_dynamics.ipynb    # Two-timescale replicator, bifurcation diagrams
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: conventions -->
## Conventions

- **Imports**: `from src.agents import ModelParams, prospect_value` — always absolute `src.*` imports
- **Simulation return**: All engines return a dict with keys: `time, price, fair_value, sentiment, momentum, demand_value, demand_emotion, demand_noise, returns, mispricing`
- **Endogenous extras**: `simulate_market_endogenous` additionally returns `population (n+1,3), p1, p2, p3, pnl_short (n+1,3), pnl_long (n+1,3)`
- **Population constraint**: fractions must sum to 1 (assert tolerance 1e-10); p_min=0.01 prevents extinction
- **RNG**: Python uses NumPy Mersenne Twister; Rust uses ChaCha8 — different streams, only statistical parity tested across engines
- **Price dynamics**: log-normal updates (`P * exp(...)`) to prevent negative prices
- **force_python=True**: bypass Rust engine in simulate_market() for exact-match tests against Python reference
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: patterns -->
## Patterns

- **Dual-backend pattern**: `simulate_market()` checks `_USE_RUST` flag at import time, falls back silently to Python if Rust extension absent
- **Flat-args PyO3 pattern**: Rust functions accept individual scalar parameters (not structs) to avoid PyO3 object overhead; structs assembled inside Rust
- **EWMA decay precomputation**: `ewma_decay`, `short_decay`, `long_decay` computed once before loop, not per-step
- **Pre-generated noise**: All Wiener increments drawn before simulation loop (`dw_f, dw_s, dw_n` arrays) — enables reproducibility and avoids RNG interleaving
- **Replicator normalization**: After each replicator step, `p_new = max(p_i, p_min)` then renormalize by sum — ensures simplex constraint even under aggressive adaptation
<!-- END AUTO-MANAGED -->

<!-- MANUAL -->
## Research Context

### Model Equations

**Demands**:
- D₁(t) = κ(V* − P(t))/P(t)  — Type I mean-reversion
- D₂(t) = α·m(t) + β·S(t)   — Type II momentum + sentiment
- D₃(t) = σₙ·ε(t)           — Type III noise

**Price**: dP = λ[p₁D₁ + p₂D₂ + p₃D₃]dt + σ_f dW_f (log-normal)

**Sentiment**: dS = −γS dt + δ·v(r) dt + σ_s dW_s
where v(r) is Kahneman-Tversky value function with λ_PT = 2.25 (loss aversion), a = 0.88 (curvature)

**Replicator**:
dp_i/dt = η_fast · p_i · (π_i^short − π̄^short) + η_slow · p_i · (π_i^long − π̄^long)

### Default Parameters

| Param | Value | Meaning |
|-------|-------|---------|
| κ | 2.0 | Type I mean-reversion speed |
| α | 5.0 | Momentum sensitivity |
| β | 3.0 | Sentiment sensitivity |
| τ_m | 20/252 | Momentum EWMA window |
| γ | 5.0 | Sentiment mean-reversion |
| δ | 10.0 | Sentiment return response |
| λ | 0.5 | Price impact (Kyle's λ) |
| σ_f | 0.01 | Fundamental news vol (daily) |
| λ_PT | 2.25 | Prospect theory loss aversion |
| η_fast | 0.5 | Retail adaptation rate |
| η_slow | 0.05 | Institutional adaptation rate |
| τ_short | 20/252 | Short P&L window |
| τ_long | 60/252 | Long P&L window (~1 quarter) |

### Key References
- Bouchaud (2021) IMH Microstructure [arXiv:2108.00242]
- Kurth, Majewski & Bouchaud (2026) Excess Volatility via Chiarella [PLoS ONE]
- Maitrier, Loeper & Bouchaud (2025) Impact & Order Imbalance [arXiv:2509.05065]
- Bouchaud (2024) Self-organized criticality in economics & finance [SSRN]
- Chiarella (1992) Original speculative dynamics model [Annals of Operations Research]
- Gabaix & Koijen (2021) Inelastic Markets Hypothesis
- Kahneman & Tversky (1979) Prospect Theory
- Cont & Bouchaud (2000) Herd Behavior
<!-- END MANUAL -->
