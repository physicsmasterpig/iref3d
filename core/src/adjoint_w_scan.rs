//! Refined q¹ adjoint projection — w-vector scan (compute-filling phase).
//!
//! Port of v0.5's `check_adjoint_with_w_vector` and `scan_w_vectors`
//! (`weyl_check.py:1440-1660`). Projects the multi-η q¹ coefficient onto
//! a combined variable via `η_j ↦ η^{2·W_j}`, then applies the single
//! SU(2) Haar × adjoint projection on that combined variable.

use num_rational::Rational64;

use crate::ab_vectors::{ABVectors, Entry};
use crate::adjoint_eta0::AdjointResult;

/// One entry in the scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WScanEntry {
    pub w: Vec<i32>,
    pub a_eff: Rational64,
    pub b_eff: Rational64,
    pub a_eff_is_integer: bool,
    pub adjoint: Option<AdjointResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WScanResult {
    pub entries: Vec<WScanEntry>,
    pub passing: Vec<WScanEntry>,
}

fn extract_q1_projected(
    result: &crate::index_refined::RefinedIndexResult,
    num_hard: usize,
    w: &[i32],
    target_x2: i64,
) -> i64 {
    let mut coeff = 0i64;
    for (k, &v) in result.iter() {
        if v == 0 || k[0] != 2 {
            continue;
        }
        let mut combined: i64 = 0;
        for j in 0..num_hard {
            combined += w[j] as i64 * k[1 + j];
        }
        if combined == target_x2 {
            coeff += v;
        }
    }
    coeff
}

/// Adjoint projection for a single w-vector at cusp `cusp_idx`.
/// (Scalar-cusp convention: uses ab.a / ab.b as the per-cusp column.)
pub fn check_adjoint_with_w_vector(
    entries: &[Entry],
    num_hard: usize,
    ab: &ABVectors,
    w: &[i32],
    cusp_idx: usize,
) -> AdjointResult {
    let a_col = &ab.a;

    let mut c_e_x2: Vec<(i64, i64)> = Vec::new();
    fn lookup(v: &[(i64, i64)], ex2: i64) -> Option<usize> {
        v.iter().position(|&(k, _)| k == ex2)
    }

    for e in entries {
        if e.m_ext.iter().any(|&m| m != 0) {
            continue;
        }
        let mut skip = false;
        for (i, &ev) in e.e_ext_x2.iter().enumerate() {
            if i != cusp_idx && ev != 0 {
                skip = true;
                break;
            }
        }
        if skip {
            continue;
        }
        let e_val_x2 = e.e_ext_x2[cusp_idx];

        // target_raw = -2 * a_eff * e = - a_eff * e_x2.
        // a_eff = Σ w_j a_j; working entirely in Rational64.
        let mut a_eff = Rational64::from_integer(0);
        for j in 0..num_hard {
            a_eff += Rational64::from_integer(w[j] as i64) * a_col[j];
        }
        let target_raw = -a_eff * Rational64::from_integer(e_val_x2);
        if *target_raw.denom() != 1 {
            // Ensure the e-value is recorded even if target is non-integer.
            if lookup(&c_e_x2, e_val_x2).is_none() {
                c_e_x2.push((e_val_x2, 0));
            }
            continue;
        }
        let target_x2 = *target_raw.numer();
        let c = extract_q1_projected(&e.result, num_hard, w, target_x2);
        match lookup(&c_e_x2, e_val_x2) {
            Some(pos) => c_e_x2[pos].1 += c,
            None => c_e_x2.push((e_val_x2, c)),
        }
    }

    let needed: [i64; 4] = [-4, -2, 2, 4];
    let mut missing: Vec<i64> = Vec::new();
    for &n in &needed {
        if lookup(&c_e_x2, n).is_none() {
            missing.push(n);
        }
    }
    if !missing.is_empty() {
        return AdjointResult {
            projected_value: None,
            is_pass: false,
            c_e_x2,
            missing_e_x2: missing,
        };
    }
    let get = |k: i64| c_e_x2.iter().find(|&&(kk, _)| kk == k).unwrap().1;
    let num = get(-2) + get(2) - get(-4) - get(4);
    if num % 2 != 0 {
        return AdjointResult {
            projected_value: None,
            is_pass: false,
            c_e_x2,
            missing_e_x2: vec![],
        };
    }
    let proj = num / 2;
    AdjointResult {
        projected_value: Some(proj),
        is_pass: proj == -1,
        c_e_x2,
        missing_e_x2: vec![],
    }
}

/// Scan w-vectors with `|W_j| ≤ max_coeff`, canonicalising by sign.
pub fn scan_w_vectors(
    entries: &[Entry],
    num_hard: usize,
    ab: &ABVectors,
    cusp_idx: usize,
    max_coeff: i32,
    skip_incompatible: bool,
) -> WScanResult {
    if num_hard == 0 {
        return WScanResult {
            entries: vec![],
            passing: vec![],
        };
    }
    let a_col = &ab.a;
    let b_col = &ab.b;

    let mut all_entries: Vec<WScanEntry> = Vec::new();
    let mut passing: Vec<WScanEntry> = Vec::new();

    let span: i32 = 2 * max_coeff + 1;
    let total: u64 = (span as u64).pow(num_hard as u32);
    for idx in 0..total {
        let mut w = Vec::with_capacity(num_hard);
        let mut r = idx;
        for _ in 0..num_hard {
            let v = (r % span as u64) as i32 - max_coeff;
            w.push(v);
            r /= span as u64;
        }
        if w.iter().all(|&c| c == 0) {
            continue;
        }
        // v0.5 canonicalisation: first nonzero entry must be positive.
        let first_nz = *w.iter().find(|&&c| c != 0).unwrap();
        if first_nz < 0 {
            continue;
        }
        let mut a_eff = Rational64::from_integer(0);
        let mut b_eff = Rational64::from_integer(0);
        for j in 0..num_hard {
            a_eff += Rational64::from_integer(w[j] as i64) * a_col[j];
            b_eff += Rational64::from_integer(w[j] as i64) * b_col[j];
        }
        let a_int = *a_eff.denom() == 1;
        let adj = if skip_incompatible && !a_int {
            None
        } else {
            Some(check_adjoint_with_w_vector(
                entries, num_hard, ab, &w, cusp_idx,
            ))
        };
        let entry = WScanEntry {
            w,
            a_eff,
            b_eff,
            a_eff_is_integer: a_int,
            adjoint: adj.clone(),
        };
        if let Some(a) = &adj {
            if a.is_pass {
                passing.push(entry.clone());
            }
        }
        all_entries.push(entry);
    }
    WScanResult {
        entries: all_entries,
        passing,
    }
}
