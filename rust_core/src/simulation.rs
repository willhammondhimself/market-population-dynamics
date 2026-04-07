use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::agents::prospect_value;
use crate::types::{SimParams, SimResult};

/// Core simulation loop — matches Python's _simulate_python exactly.
pub fn simulate_market(
    p1: f64,
    p2: f64,
    p3: f64,
    params: &SimParams,
    seed: Option<u64>,
) -> SimResult {
    let n = params.n_steps();
    let dt = params.dt;
    let sqrt_dt = dt.sqrt();

    let mut rng: ChaCha8Rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_rng(rand::thread_rng()).unwrap(),
    };

    // Pre-allocate
    let len = n + 1;
    let mut price = vec![0.0f64; len];
    let mut fair_value = vec![0.0f64; len];
    let mut sentiment = vec![0.0f64; len];
    let mut momentum = vec![0.0f64; len];
    let mut d1 = vec![0.0f64; len];
    let mut d2 = vec![0.0f64; len];
    let mut d3 = vec![0.0f64; len];
    let mut returns = vec![0.0f64; len];

    // Initial conditions
    price[0] = params.p0;
    fair_value[0] = params.v_star;

    let decay = params.ewma_decay();

    // Pre-generate random draws (matching Python's randn * sqrt_dt pattern)
    let dw_f: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng) * sqrt_dt).collect();
    let dw_s: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng) * sqrt_dt).collect();
    let dw_n: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng) * sqrt_dt).collect();

    for t in 0..n {
        // Fair value: GBM with zero drift
        fair_value[t + 1] =
            fair_value[t] * (-0.5 * params.sigma_f * params.sigma_f * dt + params.sigma_f * dw_f[t]).exp();

        // Type I: mean-revert to fair value
        d1[t] = params.kappa * (fair_value[t] - price[t]) / price[t];

        // Type II: momentum + sentiment
        d2[t] = params.alpha * momentum[t] + params.beta * sentiment[t];

        // Type III: noise
        d3[t] = params.sigma_n * dw_n[t] / sqrt_dt;

        // Aggregate demand
        let d_total = p1 * d1[t] + p2 * d2[t] + p3 * d3[t];

        // Price update (log-normal)
        let dp = params.lam * d_total * dt;
        price[t + 1] = price[t] * (dp - 0.5 * (params.lam * d_total).powi(2) * dt * dt).exp();

        // Log return
        returns[t + 1] = (price[t + 1] / price[t]).ln();

        // Momentum (EWMA)
        momentum[t + 1] = decay * momentum[t] + (1.0 - decay) * returns[t + 1];

        // Sentiment with prospect theory
        let v_r = prospect_value(returns[t + 1], params.lam_pt, params.a_pt);
        let ds = -params.gamma * sentiment[t] * dt + params.delta * v_r * dt + params.sigma_s * dw_s[t];
        sentiment[t + 1] = sentiment[t] + ds;
    }

    // Compute mispricing: log(P/V)
    let mispricing: Vec<f64> = price
        .iter()
        .zip(fair_value.iter())
        .map(|(p, v)| (p / v).ln())
        .collect();

    // Time array
    let time: Vec<f64> = (0..len).map(|i| i as f64 * dt).collect();

    SimResult {
        time,
        price,
        fair_value,
        sentiment,
        momentum,
        demand_value: d1,
        demand_emotion: d2,
        demand_noise: d3,
        returns,
        mispricing,
    }
}

/// Box-Muller standard normal.
#[inline]
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    use std::f64::consts::TAU;
    let u1: f64 = rng.gen::<f64>();
    let u2: f64 = rng.gen::<f64>();
    (-2.0_f64 * u1.ln()).sqrt() * (TAU * u2).cos()
}
