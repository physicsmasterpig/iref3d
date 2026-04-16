//! Refined q¹-η⁰ adjoint projection (W_i compatibility).
//!
//! Ports v0.5's `check_adjoint_projection` and
//! `check_adjoint_projection_multi_cusp` (`weyl_check.py:1113-1434`).
//! Operates on refined-index entries after applying the Weyl shift
//! η^{a·e + b·m}; extracts the (q¹, all-η⁰) coefficient and integrates
//! against the SU(2) Haar × adjoint kernel.

use num_rational::Rational64;

use crate::ab_vectors::{ABVectors, Entry};

/// Single-cusp adjoint projection result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjointResult {
    pub projected_value: Option<i64>,
    pub is_pass: bool,
    /// `(2·e, coeff)` for each e that contributed (Weyl-shifted q¹-η⁰).
    pub c_e_x2: Vec<(i64, i64)>,
    /// 2·e values from {−4,−2,+2,+4} that were needed but not found.
    pub missing_e_x2: Vec<i64>,
}

/// Extract the (q¹, η⁰) coefficient from a Weyl-shifted result.
///
/// η⁰ after shifting by +`shift_x2` corresponds to raw-η = −`shift_x2`.
///
/// For **collapsed edges** (incompatible with Dehn filling, W_j turned off),
/// we sum over ALL η_j powers instead of extracting just η_j^0.
fn extract_q1_eta0_shifted(
    result: &crate::index_refined::RefinedIndexResult,
    num_hard: usize,
    shift_x2: &[i64],
    collapsed_edges: Option<&hashbrown::HashSet<usize>>,
) -> i64 {
    let empty_set = hashbrown::HashSet::new();
    let collapsed = collapsed_edges.unwrap_or(&empty_set);

    let mut coeff = 0i64;
    for (k, &v) in result.iter() {
        if v == 0 || k[0] != 2 {
            continue;
        }
        let mut ok = true;
        for h in 0..num_hard {
            if collapsed.contains(&h) {
                continue; // sum over all η_h values
            }
            if k[1 + h] != -shift_x2[h] {
                ok = false;
                break;
            }
        }
        if ok {
            coeff += v;
        }
    }
    coeff
}

/// Single-cusp adjoint projection — `check_adjoint_projection` port.
pub fn check_adjoint_projection(
    entries: &[Entry],
    num_hard: usize,
    ab: Option<&ABVectors>,
    cusp_idx: usize,
    collapsed_edges: Option<&hashbrown::HashSet<usize>>,
) -> AdjointResult {
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
        let shift_x2 = if let Some(ab) = ab {
            if num_hard > 0 {
                ab.shift_x2(&e.m_ext, &e.e_ext_x2)
            } else {
                vec![0; num_hard]
            }
        } else {
            vec![0; num_hard]
        };
        let c = extract_q1_eta0_shifted(&e.result, num_hard, &shift_x2, collapsed_edges);
        let key = e.e_ext_x2[cusp_idx];
        match lookup(&c_e_x2, key) {
            Some(pos) => c_e_x2[pos].1 += c,
            None => c_e_x2.push((key, c)),
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

/// Multi-cusp variant: per-cusp adjoint projection for d simultaneously
/// filled cusps. Port of `check_adjoint_projection_multi_cusp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiCuspAdjointResult {
    pub results: Vec<AdjointResult>,
    pub filled_cusp_indices: Vec<usize>,
}

impl MultiCuspAdjointResult {
    pub fn all_pass(&self) -> bool {
        self.results.iter().all(|r| r.is_pass)
    }
}

pub fn check_adjoint_projection_multi_cusp(
    entries: &[Entry],
    num_hard: usize,
    ab: Option<&ABVectors>,
    filled_cusp_indices: &[usize],
    collapsed_edges: Option<&hashbrown::HashSet<usize>>,
) -> MultiCuspAdjointResult {
    if filled_cusp_indices.is_empty() {
        return MultiCuspAdjointResult {
            results: vec![],
            filled_cusp_indices: vec![],
        };
    }
    let d = filled_cusp_indices.len();
    let filled_set: std::collections::BTreeSet<usize> =
        filled_cusp_indices.iter().copied().collect();

    // c_e keyed by (2·e_1, …, 2·e_d) in filled_cusp_indices order.
    let mut c_e: hashbrown::HashMap<Vec<i64>, i64> = hashbrown::HashMap::new();
    for e in entries {
        if e.m_ext.iter().any(|&m| m != 0) {
            continue;
        }
        let mut skip = false;
        for (i, &ev) in e.e_ext_x2.iter().enumerate() {
            if !filled_set.contains(&i) && ev != 0 {
                skip = true;
                break;
            }
        }
        if skip {
            continue;
        }
        let key: Vec<i64> = filled_cusp_indices.iter().map(|&ci| e.e_ext_x2[ci]).collect();
        let shift_x2 = if let Some(ab) = ab {
            if num_hard > 0 {
                ab.shift_x2(&e.m_ext, &e.e_ext_x2)
            } else {
                vec![0; num_hard]
            }
        } else {
            vec![0; num_hard]
        };
        let c = extract_q1_eta0_shifted(&e.result, num_hard, &shift_x2, collapsed_edges);
        *c_e.entry(key).or_insert(0) += c;
    }

    // Kernels (working in 2·e units). Return value is Rational.
    let k_target = |ex2: i64| -> Rational64 {
        match ex2.abs() {
            2 => Rational64::new(1, 2),
            4 => Rational64::new(-1, 2),
            _ => Rational64::from_integer(0),
        }
    };
    let k_other = |ex2: i64| -> Rational64 {
        if ex2 == 0 {
            Rational64::from_integer(1)
        } else if ex2.abs() == 2 {
            Rational64::new(-1, 2)
        } else {
            Rational64::from_integer(0)
        }
    };

    let e_target_x2: [i64; 4] = [-4, -2, 2, 4];
    let e_other_x2: [i64; 3] = [-2, 0, 2];

    let mut per_cusp: Vec<AdjointResult> = Vec::new();
    for target_idx in 0..d {
        let mut numerator = Rational64::from_integer(0);
        let mut incomplete = false;

        // Iterate all combinations of (e_target, e_others...).
        let other_count = d - 1;
        let combos: usize = 3usize.pow(other_count as u32);

        'outer: for &e_t in &e_target_x2 {
            for combo_idx in 0..combos {
                let mut e_o = Vec::with_capacity(other_count);
                let mut c = combo_idx;
                for _ in 0..other_count {
                    e_o.push(e_other_x2[c % 3]);
                    c /= 3;
                }
                let mut key = Vec::with_capacity(d);
                let mut o = 0;
                for j in 0..d {
                    if j == target_idx {
                        key.push(e_t);
                    } else {
                        key.push(e_o[o]);
                        o += 1;
                    }
                }
                let Some(&c_val) = c_e.get(&key) else {
                    incomplete = true;
                    break 'outer;
                };
                let mut k = k_target(e_t);
                for &ej in &e_o {
                    k *= k_other(ej);
                }
                numerator += k * Rational64::from_integer(c_val);
            }
        }

        if incomplete {
            per_cusp.push(AdjointResult {
                projected_value: None,
                is_pass: false,
                c_e_x2: vec![],
                missing_e_x2: e_target_x2.to_vec(),
            });
            continue;
        }
        if *numerator.denom() != 1 {
            per_cusp.push(AdjointResult {
                projected_value: None,
                is_pass: false,
                c_e_x2: vec![],
                missing_e_x2: vec![],
            });
            continue;
        }
        let proj = *numerator.numer();
        per_cusp.push(AdjointResult {
            projected_value: Some(proj),
            is_pass: proj == -1,
            c_e_x2: vec![],
            missing_e_x2: vec![],
        });
    }

    MultiCuspAdjointResult {
        results: per_cusp,
        filled_cusp_indices: filled_cusp_indices.to_vec(),
    }
}
