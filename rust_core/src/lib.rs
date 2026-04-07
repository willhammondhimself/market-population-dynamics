use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod agents;
mod replicator;
mod simulation;
mod types;

use replicator::ReplicatorParams;
use types::SimParams;

/// Python-callable simulation entry point.
///
/// Returns a dict with NumPy arrays matching the Python engine output.
#[pyfunction]
#[pyo3(signature = (
    p1, p2, p3,
    t_total, dt, v_star, p0,
    lam, sigma_f, kappa,
    alpha, beta, tau_m,
    gamma, delta, sigma_s,
    lam_pt, a_pt, sigma_n,
    seed
))]
fn simulate_market<'py>(
    py: Python<'py>,
    p1: f64,
    p2: f64,
    p3: f64,
    t_total: f64,
    dt: f64,
    v_star: f64,
    p0: f64,
    lam: f64,
    sigma_f: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
    tau_m: f64,
    gamma: f64,
    delta: f64,
    sigma_s: f64,
    lam_pt: f64,
    a_pt: f64,
    sigma_n: f64,
    seed: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let params = SimParams {
        t_total,
        dt,
        v_star,
        p0,
        lam,
        sigma_f,
        kappa,
        alpha,
        beta,
        tau_m,
        gamma,
        delta,
        sigma_s,
        lam_pt,
        a_pt,
        sigma_n,
    };

    let rust_seed = if seed >= 0 { Some(seed as u64) } else { None };
    let result = simulation::simulate_market(p1, p2, p3, &params, rust_seed);

    let dict = PyDict::new(py);
    dict.set_item("time", PyArray1::from_vec(py, result.time))?;
    dict.set_item("price", PyArray1::from_vec(py, result.price))?;
    dict.set_item("fair_value", PyArray1::from_vec(py, result.fair_value))?;
    dict.set_item("sentiment", PyArray1::from_vec(py, result.sentiment))?;
    dict.set_item("momentum", PyArray1::from_vec(py, result.momentum))?;
    dict.set_item("demand_value", PyArray1::from_vec(py, result.demand_value))?;
    dict.set_item("demand_emotion", PyArray1::from_vec(py, result.demand_emotion))?;
    dict.set_item("demand_noise", PyArray1::from_vec(py, result.demand_noise))?;
    dict.set_item("returns", PyArray1::from_vec(py, result.returns))?;
    dict.set_item("mispricing", PyArray1::from_vec(py, result.mispricing))?;

    Ok(dict)
}

/// Python-callable endogenous simulation entry point.
#[pyfunction]
#[pyo3(signature = (
    p1_0, p2_0, p3_0,
    t_total, dt, v_star, p0,
    lam, sigma_f, kappa,
    alpha, beta, tau_m,
    gamma, delta, sigma_s,
    lam_pt, a_pt, sigma_n,
    eta_fast, eta_slow, tau_short, tau_long, p_min,
    seed
))]
#[allow(clippy::too_many_arguments)]
fn simulate_market_endogenous_py<'py>(
    py: Python<'py>,
    p1_0: f64,
    p2_0: f64,
    p3_0: f64,
    t_total: f64,
    dt: f64,
    v_star: f64,
    p0: f64,
    lam: f64,
    sigma_f: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
    tau_m: f64,
    gamma: f64,
    delta: f64,
    sigma_s: f64,
    lam_pt: f64,
    a_pt: f64,
    sigma_n: f64,
    eta_fast: f64,
    eta_slow: f64,
    tau_short: f64,
    tau_long: f64,
    p_min: f64,
    seed: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let sim_params = SimParams {
        t_total, dt, v_star, p0, lam, sigma_f, kappa,
        alpha, beta, tau_m, gamma, delta, sigma_s,
        lam_pt, a_pt, sigma_n,
    };
    let rep_params = ReplicatorParams {
        eta_fast, eta_slow, tau_short, tau_long, p_min,
    };

    let rust_seed = if seed >= 0 { Some(seed as u64) } else { None };
    let result = replicator::simulate_market_endogenous(
        p1_0, p2_0, p3_0, &sim_params, &rep_params, rust_seed,
    );

    let dict = PyDict::new(py);
    dict.set_item("time", PyArray1::from_vec(py, result.time))?;
    dict.set_item("price", PyArray1::from_vec(py, result.price))?;
    dict.set_item("fair_value", PyArray1::from_vec(py, result.fair_value))?;
    dict.set_item("sentiment", PyArray1::from_vec(py, result.sentiment))?;
    dict.set_item("momentum", PyArray1::from_vec(py, result.momentum))?;
    dict.set_item("demand_value", PyArray1::from_vec(py, result.demand_value))?;
    dict.set_item("demand_emotion", PyArray1::from_vec(py, result.demand_emotion))?;
    dict.set_item("demand_noise", PyArray1::from_vec(py, result.demand_noise))?;
    dict.set_item("returns", PyArray1::from_vec(py, result.returns))?;
    dict.set_item("mispricing", PyArray1::from_vec(py, result.mispricing))?;
    dict.set_item("p1", PyArray1::from_vec(py, result.p1))?;
    dict.set_item("p2", PyArray1::from_vec(py, result.p2))?;
    dict.set_item("p3", PyArray1::from_vec(py, result.p3))?;

    Ok(dict)
}

#[pymodule]
fn market_pop_dynamics_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_market, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_market_endogenous_py, m)?)?;
    Ok(())
}
