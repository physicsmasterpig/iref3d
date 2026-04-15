//! Unrefined 3D index `I(m_ext, e_ext)` — port of v0.5 `compute_index_3d_python`.
//!
//! η = 1 variant. Sums over internal-edge `e_int` via
//! [`crate::summation::enumerate_summation_terms`] and convolves the
//! tetrahedron-index series for each contributing summand.

use crate::kernel::tet_index_series;
use crate::poly::{convolve, QSeries};
use crate::summation::{enumerate_summation_terms, EnumerationState, SummationTerm};

/// Dense q^{1/2}-series result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Index3DResult {
    pub coeffs: Vec<i64>,
    pub min_power: i32,
    pub q_order_half: i32,
    pub n_terms: usize,
}

/// Compute `I(m_ext, e_ext)` entirely from state + external charges.
pub fn compute_unrefined_index(
    state: &EnumerationState,
    m_ext: &[i64],
    e_ext_x2: &[i64],
    q_order_half: i32,
) -> Index3DResult {
    let terms = enumerate_summation_terms(state, m_ext, e_ext_x2, q_order_half as i64);
    compute_from_terms(&terms, q_order_half)
}

/// Same as [`compute_unrefined_index`] but with pre-enumerated terms.
pub fn compute_from_terms(terms: &[SummationTerm], q_order_half: i32) -> Index3DResult {
    if terms.is_empty() {
        return Index3DResult {
            coeffs: vec![0; (q_order_half as usize) + 1],
            min_power: 0,
            q_order_half,
            n_terms: 0,
        };
    }

    let mut total: QSeries = QSeries::new();

    for term in terms {
        let phase_exp = term.phase_exp as i32;
        let budget = q_order_half - phase_exp;

        // Running product: start with the unit polynomial {0: 1}.
        let mut prod: QSeries = QSeries::new();
        prod.insert(0, 1);
        let mut prod_min_pow: i32 = 0;
        let mut killed = false;

        for &(ta, tb) in &term.tet_args {
            let cutoff = budget - prod_min_pow;
            if cutoff < 0 {
                killed = true;
                break;
            }
            let s = tet_index_series(ta as i32, tb as i32, cutoff);
            if s.is_empty() {
                killed = true;
                break;
            }
            prod = convolve(&prod, &s, budget);
            if prod.is_empty() {
                killed = true;
                break;
            }
            let s_min = *s.keys().min().unwrap();
            prod_min_pow += s_min;
        }

        if killed {
            continue;
        }

        let sign: i64 = if phase_exp.rem_euclid(2) == 0 { 1 } else { -1 };
        for (&pp, &c) in prod.iter() {
            let shifted = pp + phase_exp;
            if shifted < 0 || shifted > q_order_half {
                continue;
            }
            let acc = total.entry(shifted).or_insert(0);
            *acc += sign * c;
            if *acc == 0 {
                total.remove(&shifted);
            }
        }
    }

    if total.is_empty() {
        return Index3DResult {
            coeffs: vec![0; (q_order_half as usize) + 1],
            min_power: 0,
            q_order_half,
            n_terms: terms.len(),
        };
    }

    let min_power = *total.keys().min().unwrap();
    let max_power = q_order_half;
    let len = (max_power - min_power + 1) as usize;
    let mut coeffs = vec![0i64; len];
    for (&k, &v) in total.iter() {
        coeffs[(k - min_power) as usize] = v;
    }
    Index3DResult {
        coeffs,
        min_power,
        q_order_half,
        n_terms: terms.len(),
    }
}
