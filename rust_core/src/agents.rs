/// Kahneman-Tversky prospect theory value function.
///
/// v(r) = r^a            if r >= 0
/// v(r) = -lam_pt * |r|^a  if r < 0
#[inline]
pub fn prospect_value(r: f64, lam_pt: f64, a: f64) -> f64 {
    if r >= 0.0 {
        r.powf(a)
    } else {
        -lam_pt * (-r).powf(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prospect_value_positive() {
        let v = prospect_value(0.05, 2.25, 0.88);
        assert!(v > 0.0);
    }

    #[test]
    fn test_prospect_value_negative() {
        let v = prospect_value(-0.05, 2.25, 0.88);
        assert!(v < 0.0);
    }

    #[test]
    fn test_loss_aversion() {
        let v_gain = prospect_value(0.05, 2.25, 0.88);
        let v_loss = prospect_value(-0.05, 2.25, 0.88);
        assert!(v_loss.abs() > v_gain.abs());
    }

    #[test]
    fn test_zero() {
        let v = prospect_value(0.0, 2.25, 0.88);
        assert_eq!(v, 0.0);
    }
}
