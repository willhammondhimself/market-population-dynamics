/// Simulation parameters matching Python's ModelParams.
#[derive(Clone, Debug)]
pub struct SimParams {
    pub t_total: f64,
    pub dt: f64,
    pub v_star: f64,
    pub p0: f64,
    pub lam: f64,
    pub sigma_f: f64,
    pub kappa: f64,
    pub alpha: f64,
    pub beta: f64,
    pub tau_m: f64,
    pub gamma: f64,
    pub delta: f64,
    pub sigma_s: f64,
    pub lam_pt: f64,
    pub a_pt: f64,
    pub sigma_n: f64,
}

impl SimParams {
    pub fn n_steps(&self) -> usize {
        (self.t_total / self.dt) as usize
    }

    pub fn ewma_decay(&self) -> f64 {
        (-self.dt / self.tau_m).exp()
    }
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            t_total: 10.0,
            dt: 1.0 / 252.0,
            v_star: 100.0,
            p0: 100.0,
            lam: 0.5,
            sigma_f: 0.01,
            kappa: 2.0,
            alpha: 5.0,
            beta: 3.0,
            tau_m: 20.0 / 252.0,
            gamma: 5.0,
            delta: 10.0,
            sigma_s: 0.5,
            lam_pt: 2.25,
            a_pt: 0.88,
            sigma_n: 1.0,
        }
    }
}

/// Result of a single simulation run.
pub struct SimResult {
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
}
