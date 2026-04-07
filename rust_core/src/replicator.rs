use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::agents::prospect_value;
use crate::types::SimParams;

/// Parameters for two-timescale replicator dynamics.
pub struct ReplicatorParams {
    pub eta_fast: f64,
    pub eta_slow: f64,
    pub tau_short: f64,
    pub tau_long: f64,
    pub p_min: f64,
}

impl ReplicatorParams {
    pub fn short_decay(&self) -> f64 {
        (-1.0 / (self.tau_short * 252.0)).exp()
    }

    pub fn long_decay(&self) -> f64 {
        (-1.0 / (self.tau_long * 252.0)).exp()
    }
}

/// Result of endogenous simulation.
pub struct EndogenousResult {
    pub time: Vec<f64>,
    pub price: Vec<f64>,
    pub fair_value: Vec<f64>,
    pub sentiment: Vec<f64>,
    pub momentum: Vec<f64>,
    pub demand_value: Vec<f64>,
    pub demand_emotion: Vec<f64>,
    pub demand_noise: Vec<f64>,
    pub returns: Vec<f64>,
    pub mispricing: Vec<f64>,
    pub p1: Vec<f64>,
    pub p2: Vec<f64>,
    pub p3: Vec<f64>,
    pub pnl_short: Vec<[f64; 3]>,
    pub pnl_long: Vec<[f64; 3]>,
}

/// Box-Muller standard normal.
#[inline]
fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    use std::f64::consts::TAU;
    let u1: f64 = rng.gen::<f64>();
    let u2: f64 = rng.gen::<f64>();
    (-2.0_f64 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// Simulate market with endogenous population dynamics.
pub fn simulate_market_endogenous(
    p1_0: f64,
    p2_0: f64,
    p3_0: f64,
    sim_params: &SimParams,
    rep_params: &ReplicatorParams,
    seed: Option<u64>,
) -> EndogenousResult {
    let n = sim_params.n_steps();
    let dt = sim_params.dt;
    let sqrt_dt = dt.sqrt();
    let len = n + 1;

    let mut rng: ChaCha8Rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_rng(rand::thread_rng()).unwrap(),
    };

    let mut price = vec![0.0f64; len];
    let mut fair_value = vec![0.0f64; len];
    let mut sentiment = vec![0.0f64; len];
    let mut momentum = vec![0.0f64; len];
    let mut d1 = vec![0.0f64; len];
    let mut d2 = vec![0.0f64; len];
    let mut d3 = vec![0.0f64; len];
    let mut returns = vec![0.0f64; len];

    let mut pop_1 = vec![0.0f64; len];
    let mut pop_2 = vec![0.0f64; len];
    let mut pop_3 = vec![0.0f64; len];
    let mut pnl_short = vec![[0.0f64; 3]; len];
    let mut pnl_long = vec![[0.0f64; 3]; len];

    price[0] = sim_params.p0;
    fair_value[0] = sim_params.v_star;
    pop_1[0] = p1_0;
    pop_2[0] = p2_0;
    pop_3[0] = p3_0;

    let ewma_decay = sim_params.ewma_decay();
    let short_decay = rep_params.short_decay();
    let long_decay = rep_params.long_decay();

    // Pre-generate random draws
    let dw_f: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng) * sqrt_dt).collect();
    let dw_s: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng) * sqrt_dt).collect();
    let dw_n: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng) * sqrt_dt).collect();

    for t in 0..n {
        let p1 = pop_1[t];
        let p2 = pop_2[t];
        let p3 = pop_3[t];

        // Fair value: GBM
        fair_value[t + 1] = fair_value[t]
            * (-0.5 * sim_params.sigma_f * sim_params.sigma_f * dt
                + sim_params.sigma_f * dw_f[t])
                .exp();

        // Demands
        d1[t] = sim_params.kappa * (fair_value[t] - price[t]) / price[t];
        d2[t] = sim_params.alpha * momentum[t] + sim_params.beta * sentiment[t];
        d3[t] = sim_params.sigma_n * dw_n[t] / sqrt_dt;

        // Aggregate
        let d_total = p1 * d1[t] + p2 * d2[t] + p3 * d3[t];

        // Price
        let dp = sim_params.lam * d_total * dt;
        price[t + 1] =
            price[t] * (dp - 0.5 * (sim_params.lam * d_total).powi(2) * dt * dt).exp();

        // Return
        returns[t + 1] = (price[t + 1] / price[t]).ln();

        // Momentum
        momentum[t + 1] = ewma_decay * momentum[t] + (1.0 - ewma_decay) * returns[t + 1];

        // Sentiment
        let v_r = prospect_value(returns[t + 1], sim_params.lam_pt, sim_params.a_pt);
        let ds = -sim_params.gamma * sentiment[t] * dt
            + sim_params.delta * v_r * dt
            + sim_params.sigma_s * dw_s[t];
        sentiment[t + 1] = sentiment[t] + ds;

        // P&L for each type
        let ret = returns[t + 1];
        let pnl_1 = d1[t] * ret;
        let pnl_2 = d2[t] * ret;
        let pnl_3 = d3[t] * ret;

        // EWMA P&L
        for (i, pnl_inst) in [pnl_1, pnl_2, pnl_3].iter().enumerate() {
            pnl_short[t + 1][i] =
                short_decay * pnl_short[t][i] + (1.0 - short_decay) * pnl_inst;
            pnl_long[t + 1][i] =
                long_decay * pnl_long[t][i] + (1.0 - long_decay) * pnl_inst;
        }

        // Two-timescale replicator
        let p_vec = [p1, p2, p3];

        // Fast (short-window)
        let pi_bar_short: f64 = p_vec
            .iter()
            .zip(pnl_short[t + 1].iter())
            .map(|(p, pi)| p * pi)
            .sum();
        let dp_fast: [f64; 3] = std::array::from_fn(|i| {
            rep_params.eta_fast * p_vec[i] * (pnl_short[t + 1][i] - pi_bar_short)
        });

        // Slow (long-window)
        let pi_bar_long: f64 = p_vec
            .iter()
            .zip(pnl_long[t + 1].iter())
            .map(|(p, pi)| p * pi)
            .sum();
        let dp_slow: [f64; 3] = std::array::from_fn(|i| {
            rep_params.eta_slow * p_vec[i] * (pnl_long[t + 1][i] - pi_bar_long)
        });

        // Update population
        let mut p_new = [0.0f64; 3];
        for i in 0..3 {
            p_new[i] = (p_vec[i] + (dp_fast[i] + dp_slow[i]) * dt).max(rep_params.p_min);
        }

        // Normalize
        let sum: f64 = p_new.iter().sum();
        for i in 0..3 {
            p_new[i] /= sum;
        }

        pop_1[t + 1] = p_new[0];
        pop_2[t + 1] = p_new[1];
        pop_3[t + 1] = p_new[2];
    }

    let mispricing: Vec<f64> = price
        .iter()
        .zip(fair_value.iter())
        .map(|(p, v)| (p / v).ln())
        .collect();
    let time: Vec<f64> = (0..len).map(|i| i as f64 * dt).collect();

    EndogenousResult {
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
        p1: pop_1,
        p2: pop_2,
        p3: pop_3,
        pnl_short,
        pnl_long,
    }
}
