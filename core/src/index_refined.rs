//! Refined 3D index I^ref(q; η_0, …, η_{k-1}) — port of v0.5 `compute_refined_index`.
//!
//! Extends the unrefined index by attaching a fugacity exponent `η_a^{e_int[a]}`
//! for each hard internal edge (positions `0..num_hard` of `e_int`). Output is
//! a map from `(q_half_power, 2·η_0_exp, …, 2·η_{k-1}_exp)` → coefficient.

use hashbrown::HashMap;

use crate::kernel::tet_index_series;
use crate::poly::{convolve, QSeries};
use crate::summation::{enumerate_summation_terms, EnumerationState};

/// Key = `(q_half_power, 2·η_0_exp, …, 2·η_{k-1}_exp)`.
pub type RefinedKey = Vec<i64>;

/// Refined index result: `key → integer coefficient`, no zero coefficients.
pub type RefinedIndexResult = HashMap<RefinedKey, i64>;

/// Compute I^ref(q; η_0, …, η_{num_hard-1}) for one `(m_ext, e_ext)` point.
pub fn compute_refined_index(
    state: &EnumerationState,
    num_hard: usize,
    m_ext: &[i64],
    e_ext_x2: &[i64],
    q_order_half: i32,
) -> RefinedIndexResult {
    let terms = enumerate_summation_terms(state, m_ext, e_ext_x2, q_order_half as i64);
    let mut result: RefinedIndexResult = HashMap::new();
    if terms.is_empty() {
        return result;
    }

    for term in &terms {
        let phase_exp = term.phase_exp as i32;
        let budget = q_order_half - phase_exp;

        // Fugacity exponents (doubled): first num_hard entries of e_int_x2.
        let eta_x2: Vec<i64> = term.e_int_x2[..num_hard].to_vec();

        // Running product over tets.
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
            let mut key: RefinedKey = Vec::with_capacity(1 + num_hard);
            key.push(shifted as i64);
            key.extend_from_slice(&eta_x2);
            let acc = result.entry(key.clone()).or_insert(0);
            *acc += sign * c;
            if *acc == 0 {
                result.remove(&key);
            }
        }
    }

    result
}

/// Batch variant: reuse enumeration state across multiple evaluation points.
pub fn compute_refined_index_batch(
    state: &EnumerationState,
    num_hard: usize,
    entries: &[(Vec<i64>, Vec<i64>)],
    q_order_half: i32,
) -> Vec<RefinedIndexResult> {
    entries
        .iter()
        .map(|(m, e_x2)| compute_refined_index(state, num_hard, m, e_x2, q_order_half))
        .collect()
}

/// Sum over all fugacity monomials for each q-power (sets η_a = 1).
pub fn project_to_3d_index(refined: &RefinedIndexResult) -> HashMap<i32, i64> {
    let mut out: HashMap<i32, i64> = HashMap::new();
    for (key, &coeff) in refined.iter() {
        let q = key[0] as i32;
        let acc = out.entry(q).or_insert(0);
        *acc += coeff;
    }
    out.retain(|_, v| *v != 0);
    out
}
