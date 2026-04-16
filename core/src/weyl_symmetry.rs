//! Weyl-symmetry check + monomial stripping for refined-index entries.
//!
//! Ports v0.5's `check_weyl_symmetry` and `strip_weyl_monomial` from
//! `weyl_check.py`.

use hashbrown::HashMap;
use num_rational::Rational64;

use crate::ab_vectors::{ABVectors, Entry};
use crate::index_refined::RefinedIndexResult;

/// Per-sector shifted series keyed by `(m_ext, e_ext_x2)`.
pub type ShiftedLookup = HashMap<(Vec<i64>, Vec<i64>), RefinedIndexResult>;

/// Multiply each key's η-exponents by the Weyl shift for its sector.
fn shifted_series(
    result: &RefinedIndexResult,
    shift_x2: &[i64],
    num_hard: usize,
    q_cutoff: Option<i32>,
) -> RefinedIndexResult {
    let mut out: RefinedIndexResult = HashMap::new();
    for (k, &coeff) in result.iter() {
        if coeff == 0 {
            continue;
        }
        if let Some(cut) = q_cutoff {
            if k[0] > cut as i64 {
                continue;
            }
        }
        let mut new_key = Vec::with_capacity(1 + num_hard);
        new_key.push(k[0]);
        for j in 0..num_hard {
            new_key.push(k[1 + j] + shift_x2[j]);
        }
        let acc = out.entry(new_key.clone()).or_insert(0);
        *acc += coeff;
        if *acc == 0 {
            out.remove(&new_key);
        }
    }
    out
}

/// Check Weyl symmetry `f(m, e) = f(−m, −e)` across sector pairs.
///
/// Returns a map from `(m_ext, e_ext_x2)` → whether the sector and its negation
/// match as shifted series. Sectors whose partner is absent are marked `false`.
pub fn check_weyl_symmetry(
    entries: &[Entry],
    num_hard: usize,
    ab: &ABVectors,
    q_order_half: Option<i32>,
) -> HashMap<(Vec<i64>, Vec<i64>), bool> {
    let q_cutoff = q_order_half.map(|q| q - 2.max(q / 3));

    let mut shifted_lookup: ShiftedLookup = HashMap::new();
    for e in entries {
        let shift = ab.shift_x2(&e.m_ext, &e.e_ext_x2);
        let shifted = shifted_series(&e.result, &shift, num_hard, q_cutoff);
        shifted_lookup.insert((e.m_ext.clone(), e.e_ext_x2.clone()), shifted);
    }

    let mut results: HashMap<(Vec<i64>, Vec<i64>), bool> = HashMap::new();
    for ((m_key, e_key_x2), f_me) in &shifted_lookup {
        let neg_m: Vec<i64> = m_key.iter().map(|x| -x).collect();
        let neg_e: Vec<i64> = e_key_x2.iter().map(|x| -x).collect();
        let partner_key = (neg_m, neg_e);
        let ok = match shifted_lookup.get(&partner_key) {
            Some(f_neg) => f_me == f_neg,
            None => false,
        };
        results.insert((m_key.clone(), e_key_x2.clone()), ok);
    }
    results
}

/// Factor the Weyl η-monomial out of a single entry.
///
/// Returns `(centre, stripped)`:
///   - `centre[j] = -(a_j·e + b_j·m)` (η-exponent in the original index *I*).
///   - `stripped` is the Weyl-manifest series `f = η^{a·e + b·m} · I`.
pub fn strip_weyl_monomial(
    result: &RefinedIndexResult,
    m_ext: &[i64],
    e_ext_x2: &[i64],
    ab: &ABVectors,
    num_hard: usize,
) -> (Vec<Rational64>, RefinedIndexResult) {
    let shift = ab.shift_x2(m_ext, e_ext_x2);
    let centre: Vec<Rational64> = shift
        .iter()
        .map(|&s| Rational64::new(-s, 2))
        .collect();
    let stripped = shifted_series(result, &shift, num_hard, None);
    (centre, stripped)
}
