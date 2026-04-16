//! Unrefined q¹ adjoint projection (marginal check).
//!
//! Ports the inline logic from v0.5's `app/workers/weyl_worker.py:186-232`.
//! For a chosen cusp `cusp_idx`, the unrefined 3D index `I^{3D}(m=0, e)` is
//! evaluated at e ∈ {−2, −1, +1, +2} on that cusp (all other cusps at 0).
//! The SU(2) Haar × adjoint character projection reduces to the scalar
//!
//!   `proj = (c_{-1} + c_{+1} − c_{-2} − c_{+2}) / 2`,
//!
//! where `c_e` is the q¹ coefficient of `I^{3D}(m=0, e)`. An NC cycle is
//! marginal iff `proj ≥ 0`. `None` indicates missing or non-integer data.
//!
//! This drives kernel choice during Dehn filling: marginal → unrefined
//! kernel `K(P,Q)`; non-marginal → refined kernel `K^ref`.

use crate::index_unrefined::compute_unrefined_index;
use crate::summation::EnumerationState;

/// Result of the marginal check for one NC cycle at one cusp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarginalCheck {
    /// The scalar projection value. `None` if any required e-entry is
    /// missing or if the numerator was odd.
    pub unrefined_q1_proj: Option<i64>,
    /// `Some(true)` iff `proj ≥ 0`; `None` mirrors `unrefined_q1_proj`.
    pub is_marginal: Option<bool>,
    /// Per-e q¹ coefficients (only those that were computed).
    /// Key is `2·e` (doubled-e form used throughout the pipeline).
    pub c_e_x2: Vec<(i64, i64)>,
}

/// Compute the unrefined q¹ adjoint projection for cusp `cusp_idx`.
pub fn check_marginal(
    state: &EnumerationState,
    num_cusps: usize,
    cusp_idx: usize,
    q_order_half: i32,
) -> MarginalCheck {
    // e ∈ {−2, −1, +1, +2}, stored as 2·e = {−4, −2, +2, +4}.
    let needed_x2: [i64; 4] = [-4, -2, 2, 4];
    let mut c_e_x2: Vec<(i64, i64)> = Vec::with_capacity(4);
    let m_ext = vec![0i64; num_cusps];

    for &e_x2 in &needed_x2 {
        let mut e_ext_x2 = vec![0i64; num_cusps];
        e_ext_x2[cusp_idx] = e_x2;
        let res = compute_unrefined_index(state, &m_ext, &e_ext_x2, q_order_half);
        // q¹ = qq^2; account for min_power (which is in qq half-units).
        let idx = 2 - res.min_power;
        let c = if idx >= 0 && (idx as usize) < res.coeffs.len() {
            res.coeffs[idx as usize]
        } else {
            0
        };
        c_e_x2.push((e_x2, c));
    }

    // Unpack in fixed order for the projection.
    let get = |e_x2: i64| c_e_x2.iter().find(|&&(k, _)| k == e_x2).map(|&(_, v)| v);
    let (Some(cm1), Some(cp1), Some(cm2), Some(cp2)) =
        (get(-2), get(2), get(-4), get(4))
    else {
        return MarginalCheck {
            unrefined_q1_proj: None,
            is_marginal: None,
            c_e_x2,
        };
    };
    let num = cm1 + cp1 - cm2 - cp2;
    if num % 2 != 0 {
        return MarginalCheck {
            unrefined_q1_proj: None,
            is_marginal: None,
            c_e_x2,
        };
    }
    let proj = num / 2;
    MarginalCheck {
        unrefined_q1_proj: Some(proj),
        is_marginal: Some(proj >= 0),
        c_e_x2,
    }
}
