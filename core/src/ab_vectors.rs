//! Weyl (a, b) vector extraction from refined-index entries — v0.5 `weyl_check.py` port.
//!
//! This module implements the **scalar / single-cusp** algorithm that v0.5
//! calls `_compute_ab_vectors_scalar`. Multi-cusp (`cusp_columns`) support is
//! planned but not yet implemented.
//!
//! Exponents are stored as [`Rational64`] (so integer + half-integer fit
//! exactly). Inputs use `2·e_ext` as `i64` to match the rest of the pipeline.

use hashbrown::HashMap;
use num_rational::Rational64;

use crate::index_refined::RefinedIndexResult;

/// Weyl-symmetry vectors.
///
/// The physical multiplier is `η_j^{a_j·e + b_j·m}` (single-cusp convention).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ABVectors {
    pub a: Vec<Rational64>,
    pub b: Vec<Rational64>,
    pub num_hard: usize,
    pub warnings: Vec<String>,
}

impl ABVectors {
    /// Doubled Weyl shift per hard edge: `2·(a_j·Σe + b_j·Σm)`.
    ///
    /// `e_ext_x2` is `2·e_ext` (length = number of cusps).
    pub fn shift_x2(&self, m_ext: &[i64], e_ext_x2: &[i64]) -> Vec<i64> {
        let m_sum: i64 = m_ext.iter().sum();
        let e_sum_x2: i64 = e_ext_x2.iter().sum();
        // shift_x2 = 2·(a·e_sum + b·m_sum) = a·e_sum_x2 + 2·b·m_sum.
        let mut out = Vec::with_capacity(self.num_hard);
        for j in 0..self.num_hard {
            let term = self.a[j] * Rational64::from_integer(e_sum_x2)
                + self.b[j] * Rational64::from_integer(2 * m_sum);
            // v0.5 uses Python int() — trunc toward zero. Rational64 stores reduced
            // form with positive denom, so i64 division matches.
            out.push(*term.numer() / *term.denom());
        }
        out
    }

    pub fn a_is_integer(&self) -> Vec<bool> {
        self.a.iter().map(|v| *v.denom() == 1).collect()
    }
    pub fn b_is_half_integer(&self) -> Vec<bool> {
        self.b.iter().map(|v| *(v * Rational64::from_integer(2)).denom() == 1).collect()
    }
    pub fn is_valid(&self) -> bool {
        self.a_is_integer().iter().all(|&v| v)
            && self.b_is_half_integer().iter().all(|&v| v)
    }
    pub fn edge_compatible(&self) -> Vec<bool> {
        self.a_is_integer()
            .into_iter()
            .zip(self.b_is_half_integer())
            .map(|(aa, bb)| aa && bb)
            .collect()
    }
}

/// Coefficient-weighted centre of η-exponents at leading q-half-power.
///
/// Port of v0.5's `_eta_center_at_leading_q`.
pub fn eta_center_at_leading_q(
    result: &RefinedIndexResult,
    num_hard: usize,
) -> Option<Vec<Rational64>> {
    if result.is_empty() {
        return None;
    }
    let mut min_q = i64::MAX;
    for (k, &v) in result.iter() {
        if v != 0 && k[0] < min_q {
            min_q = k[0];
        }
    }
    if min_q == i64::MAX {
        return None;
    }
    let mut total_weight: i64 = 0;
    let mut weighted = vec![0i64; num_hard];
    for (k, &v) in result.iter() {
        if v == 0 || k[0] != min_q {
            continue;
        }
        total_weight += v;
        for j in 0..num_hard {
            weighted[j] += k[1 + j] * v;
        }
    }
    if total_weight == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(num_hard);
    for j in 0..num_hard {
        // weighted_sum / (2 * total_weight) → η stored doubled, so divide by 2·W.
        out.push(Rational64::new(weighted[j], 2 * total_weight));
    }
    Some(out)
}

/// Minimum η-exponents at leading q-half-power (port of
/// `extract_leading_eta_exponents`, component-wise minimum).
pub fn extract_leading_eta_exponents(
    result: &RefinedIndexResult,
    num_hard: usize,
) -> Option<Vec<Rational64>> {
    if result.is_empty() {
        return None;
    }
    let mut min_q = i64::MAX;
    for (k, &v) in result.iter() {
        if v != 0 && k[0] < min_q {
            min_q = k[0];
        }
    }
    if min_q == i64::MAX {
        return None;
    }
    let mut mins = vec![i64::MAX; num_hard];
    for (k, &v) in result.iter() {
        if v == 0 || k[0] != min_q {
            continue;
        }
        for j in 0..num_hard {
            if k[1 + j] < mins[j] {
                mins[j] = k[1 + j];
            }
        }
    }
    Some(mins.into_iter().map(|x| Rational64::new(x, 2)).collect())
}

/// One entry of the multi-point evaluation table.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Meridian charges (one per cusp).
    pub m_ext: Vec<i64>,
    /// `2·e_ext` (one per cusp).
    pub e_ext_x2: Vec<i64>,
    pub result: RefinedIndexResult,
}

/// Compute (a, b) from a table of refined-index evaluations.
///
/// Port of v0.5's `compute_ab_vectors` → `_compute_ab_vectors_scalar`
/// (single-cusp path, also used when all entries share cusp-0 isolation).
/// Returns `None` if neither a nor b can be determined.
pub fn compute_ab_vectors(entries: &[Entry], num_hard: usize) -> Option<ABVectors> {
    if num_hard == 0 {
        return Some(ABVectors {
            a: vec![],
            b: vec![],
            num_hard: 0,
            warnings: vec![],
        });
    }
    if entries.is_empty() {
        return None;
    }

    // Index by (m_ext, e_ext_x2) → centre.
    let mut indexed: HashMap<(Vec<i64>, Vec<i64>), Option<Vec<Rational64>>> = HashMap::new();
    for e in entries {
        let key = (e.m_ext.clone(), e.e_ext_x2.clone());
        let centre = eta_center_at_leading_q(&e.result, num_hard);
        indexed.insert(key, centre);
    }

    let get_c = |m: &[i64], e_x2: &[i64]| -> Option<Vec<Rational64>> {
        indexed.get(&(m.to_vec(), e_x2.to_vec())).cloned().flatten()
    };

    let mut b_estimates: Vec<Vec<Rational64>> = Vec::new();
    let mut a_estimates: Vec<Vec<Rational64>> = Vec::new();
    let mut seen_b: hashbrown::HashSet<(Vec<i64>, Vec<i64>)> = hashbrown::HashSet::new();
    let mut seen_a: hashbrown::HashSet<(Vec<i64>, Vec<i64>)> = hashbrown::HashSet::new();

    let keys: Vec<_> = indexed.keys().cloned().collect();
    for (m_key, e_key_x2) in &keys {
        let Some(c_pos) = get_c(m_key, e_key_x2) else { continue; };
        let m_sum: i64 = m_key.iter().sum();
        let e_sum_x2: i64 = e_key_x2.iter().sum();

        // meridian pair: all e == 0, sum(m) > 0
        if e_key_x2.iter().all(|&x| x == 0) && m_sum > 0 {
            let neg_m: Vec<i64> = m_key.iter().map(|x| -x).collect();
            let tag = (m_key.clone(), neg_m.clone());
            if !seen_b.contains(&tag) {
                if let Some(c_neg) = get_c(&neg_m, e_key_x2) {
                    seen_b.insert(tag);
                    let total_m: i64 = m_key.iter().map(|x| x.abs()).sum();
                    let mut b_vec = Vec::with_capacity(num_hard);
                    for j in 0..num_hard {
                        let v = -(c_pos[j] - c_neg[j])
                            / Rational64::from_integer(2 * total_m);
                        b_vec.push(v);
                    }
                    b_estimates.push(b_vec);
                }
            }
        }
        // longitude pair: all m == 0, sum(e) > 0
        if m_key.iter().all(|&x| x == 0) && e_sum_x2 > 0 {
            let neg_e: Vec<i64> = e_key_x2.iter().map(|x| -x).collect();
            let tag = (e_key_x2.clone(), neg_e.clone());
            if !seen_a.contains(&tag) {
                if let Some(c_neg) = get_c(m_key, &neg_e) {
                    seen_a.insert(tag);
                    // total_e = sum(|e_key|) = |sum(e_key)| because sum > 0.
                    // Using doubled form: divide by e_sum_x2.
                    let mut a_vec = Vec::with_capacity(num_hard);
                    for j in 0..num_hard {
                        // v0.5: -(c_pos - c_neg) / (2·total_e); total_e = |e_sum| = e_sum_x2/2
                        // ⇒ divide by e_sum_x2.
                        let diff = c_pos[j] - c_neg[j];
                        let v = -diff / Rational64::from_integer(e_sum_x2);
                        a_vec.push(v);
                    }
                    a_estimates.push(a_vec);
                }
            }
        }
    }

    // Fallback: individual m entries compared to the zero entry.
    if b_estimates.is_empty() {
        let r = keys[0].0.len();
        let zero_m = vec![0i64; r];
        let zero_e = vec![0i64; r];
        if let Some(c_zero) = get_c(&zero_m, &zero_e) {
            for (m_key, e_key_x2) in &keys {
                if m_key == &zero_m && e_key_x2 == &zero_e {
                    continue;
                }
                if !e_key_x2.iter().all(|&x| x == 0) {
                    continue;
                }
                let Some(c_pos) = get_c(m_key, e_key_x2) else { continue; };
                let total_m: i64 = m_key.iter().sum();
                if total_m == 0 {
                    continue;
                }
                let mut b_vec = Vec::with_capacity(num_hard);
                for j in 0..num_hard {
                    let v = -(c_pos[j] - c_zero[j])
                        / Rational64::from_integer(total_m);
                    b_vec.push(v);
                }
                b_estimates.push(b_vec);
            }
        }
    }

    let mut warnings: Vec<String> = Vec::new();
    let consensus = |ests: &[Vec<Rational64>], label: &str, warnings: &mut Vec<String>| {
        if ests.is_empty() {
            return None;
        }
        let reference = ests[0].clone();
        for est in &ests[1..] {
            for (j, (r_j, e_j)) in reference.iter().zip(est.iter()).enumerate() {
                if r_j != e_j {
                    warnings.push(format!(
                        "{label}[{j}]: inconsistent estimates {r_j} vs {e_j} (using first)"
                    ));
                }
            }
        }
        Some(reference)
    };

    let mut b_vec = consensus(&b_estimates, "b", &mut warnings);
    let mut a_vec = consensus(&a_estimates, "a", &mut warnings);

    if b_vec.is_none() && a_vec.is_none() {
        return None;
    }
    if b_vec.is_none() {
        warnings.push("b: no meridian pairs found; defaulting to 0".into());
        b_vec = Some(vec![Rational64::from_integer(0); num_hard]);
    }
    if a_vec.is_none() {
        warnings.push("a: no longitude pairs found; defaulting to 0".into());
        a_vec = Some(vec![Rational64::from_integer(0); num_hard]);
    }

    Some(ABVectors {
        a: a_vec.unwrap(),
        b: b_vec.unwrap(),
        num_hard,
        warnings,
    })
}
