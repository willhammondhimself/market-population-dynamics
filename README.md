# Market Population Dynamics: Agent Heterogeneity and Price Convergence

**Author**: Will Hammond
**Advisor**: Alan Kahn
**Started**: March 2026

## Research Question

How do the relative proportions of three behaviorally-distinct trader populations determine whether a market converges to fair value, and what observable signatures characterize each regime?

## Agent Taxonomy

### Type I — Data-Driven / Value Traders
Agents who estimate fundamental value from observable data and trade mean-revertingly toward it. In the limit where 100% of the population is Type I, the market should converge to fair value (the EMH limit).

**Modeling approach**: Demand proportional to (V* - P), where V* is the estimated fair value. Ornstein-Uhlenbeck-like dynamics with heterogeneous estimation horizons.

### Type II — Intuition / Emotion Traders
Agents driven by behavioral factors: momentum-chasing, loss aversion (prospect theory), herding/sentiment contagion, and fear/greed cycles. Unlike Chiarella's mechanical trend followers, these agents exhibit *asymmetric* behavior — they panic-sell faster than they FOMO-buy, and their strategies are contagious across the population.

**Modeling approach**: Demand driven by recent returns (momentum) + sentiment field (mean-field herding) + prospect-theoretic value function (Kahneman-Tversky). Asymmetric response to gains vs. losses.

### Type III — Noise Traders
Zero-intelligence agents contributing pure noise to order flow. No information, no strategy.

**Modeling approach**: Standard Brownian motion demand. Can also explore Levy-stable noise for fat tails, or simple Poisson arrival of random orders.

## Core Analysis

### Population Simplex
The state space is the 2-simplex: **p** = (p1, p2, p3) with p1 + p2 + p3 = 1.

Key questions at each point in the simplex:
1. **Convergence**: Does the market price converge to fair value?
2. **Excess volatility**: How much larger is realized vol vs. fundamental vol?
3. **Mispricing distribution**: Unimodal (efficient) or bimodal (bubble/crash prone)?
4. **Price impact**: What is the GK multiplier M as a function of **p**?

### Phase Diagram
Map the simplex into regions:
- **Efficient zone**: Price tracks fair value, low excess vol
- **Momentum zone**: Trend-following dominates, bubble/crash dynamics
- **Noise zone**: Random walk dominates, high vol, no convergence
- **Critical boundaries**: Phase transitions between regimes

### Convergence Analysis
- Analytical: Mean-field limit as N → ∞, derive conditions on **p** for convergence
- Numerical: Monte Carlo simulation across the simplex
- Empirical: Calibrate to real data, infer population proportions

## Architecture

```
src/              # Python analysis, calibration, visualization
rust_core/        # Rust simulation engine (via PyO3)
notebooks/        # Exploratory analysis and figures
papers/           # PDFs of key references
data/             # Market data for calibration
docs/             # Paper drafts and notes
tests/            # Unit tests for both Python and Rust
```

## Tech Stack
- **Simulation**: Rust (via PyO3 → Python bindings)
- **Analysis**: Python (NumPy, SciPy, pandas)
- **Visualization**: matplotlib, plotly
- **Calibration**: scipy.optimize, optuna
- **Data**: yfinance, FRED

## Reading List

### Primary (must-read)
1. Bouchaud (2021) — "The Inelastic Market Hypothesis: A Microstructural Interpretation" [arXiv:2108.00242]
2. Kurth, Majewski & Bouchaud (2026) — "Revisiting the excess volatility puzzle through the lens of the Chiarella model" [PLoS ONE]
3. Maitrier, Loeper & Bouchaud (2025) — "The Subtle Interplay between Square-root Impact, Order Imbalance & Volatility II: An Artificial Market Generator" [arXiv:2509.05065]
4. Bouchaud (2024) — "The self-organized criticality paradigm in economics & finance" [SSRN]
5. Bouchaud, Bonart, Donier & Gould (2018) — "Trades, Quotes and Prices: Financial Markets Under the Microscope" [Cambridge UP]

### Secondary (important context)
6. Bouchaud et al. (2020) — "Co-impact: Crowding effects in institutional trading activity" [Quantitative Finance]
7. Bouchaud et al. (2020) — "The multivariate Kyle model: More is different" [SIAM]
8. Bouchaud (2021) — "Radical Complexity" [Entropy]
9. Chiarella (1992) — "The dynamics of speculative behaviour" [Annals of Operations Research] (the original model)
10. Gabaix & Koijen (2021) — "In Search of the Origins of Financial Fluctuations: The Inelastic Markets Hypothesis"

### Behavioral finance foundations
11. Kahneman & Tversky (1979) — Prospect Theory
12. Cont & Bouchaud (2000) — "Herd Behavior and Aggregate Fluctuations in Financial Markets"
13. Hommes (2006) — "Heterogeneous Agent Models in Economics and Finance"

## What Makes This Novel (vs. Chiarella/Bouchaud)

1. **Behaviorally-grounded Type II agents**: Not just mechanical trend followers — incorporates prospect theory, sentiment contagion, and asymmetric response. This can generate phenomena (panic cascades, FOMO bubbles) that Chiarella's framework can't.
2. **Full simplex analysis**: Chiarella papers typically vary one parameter at a time. We map the entire population simplex to find phase boundaries.
3. **Population inference**: Can we infer **p** from market observables? If so, this is directly useful to trading firms — it tells you what regime the market is in.
4. **Rust simulation engine**: Enables large-scale Monte Carlo that would be impractical in pure Python.

## Why Firms Care

- **Jane Street / Jump / HRT / GTS**: Understanding market regimes and participant composition is core to their alpha. If you can infer population mix from tape data, that's actionable.
- **Interview signal**: Shows you think about markets as complex adaptive systems, not just statistical patterns. The Rust + Python stack shows engineering maturity. The Bouchaud literature connection shows you read serious research.
- **Paper potential**: Extends a 2026 Bouchaud paper with novel behavioral agents and full simplex analysis. Publishable in Quantitative Finance if calibration is strong.
