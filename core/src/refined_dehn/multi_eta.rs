//! MultiEtaSeries type and shared helpers for refined Dehn filling.
//!
//! A `MultiEtaSeries` is a sparse polynomial in `(qq, 2·η_0, …, 2·η_{k-1})`
//! with `Rational64` coefficients. It generalises `RefinedIndexResult`
//! (which uses `i64` coefficients) to support the half-integer factors
//! introduced by the Dehn filling kernel.

use num_rational::Rational64;

use crate::index_refined::RefinedIndexResult;

pub type MultiEtaSeries = hashbrown::HashMap<Vec<i64>, Rational64>;

pub fn multi_add(a: &MultiEtaSeries, b: &MultiEtaSeries) -> MultiEtaSeries {
    let zero = Rational64::from_integer(0);
    let mut out = a.clone();
    for (k, &v) in b {
        let e = out.entry(k.clone()).or_insert(zero);
        *e += v;
        if *e == zero {
            out.remove(k);
        }
    }
    out
}

pub fn multi_add_inplace(a: &mut MultiEtaSeries, b: &MultiEtaSeries) {
    let zero = Rational64::from_integer(0);
    for (k, &v) in b {
        let e = a.entry(k.clone()).or_insert(zero);
        *e += v;
        if *e == zero {
            a.remove(k);
        }
    }
}

/// Convert `RefinedIndexResult` (i64 coefficients) to `MultiEtaSeries`.
/// Optionally appends a cusp-η = 0 dimension.
pub fn refined_to_multi(
    refined: &RefinedIndexResult,
    append_cusp_eta: bool,
) -> MultiEtaSeries {
    let mut out = MultiEtaSeries::new();
    for (k, &c) in refined {
        if c == 0 {
            continue;
        }
        let new_key = if append_cusp_eta {
            let mut nk = k.clone();
            nk.push(0);
            nk
        } else {
            k.clone()
        };
        out.insert(new_key, Rational64::from_integer(c));
    }
    out
}

/// Project η_j to 1 for each j in `incompat_edges` (set dim 1+j to 0).
pub fn collapse_iref_edges(
    refined: &RefinedIndexResult,
    incompat_edges: &[usize],
) -> RefinedIndexResult {
    if incompat_edges.is_empty() || refined.is_empty() {
        return refined.clone();
    }
    let incompat_set: hashbrown::HashSet<usize> = incompat_edges.iter().copied().collect();
    let mut out = RefinedIndexResult::new();
    for (key, &coeff) in refined {
        let new_key: Vec<i64> = key
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i > 0 && incompat_set.contains(&(i - 1)) {
                    0
                } else {
                    v
                }
            })
            .collect();
        let e = out.entry(new_key).or_insert(0);
        *e += coeff;
        if *e == 0 {
            let k2 = key
                .iter()
                .enumerate()
                .map(|(i, &v)| if i > 0 && incompat_set.contains(&(i - 1)) { 0 } else { v })
                .collect::<Vec<_>>();
            out.remove(&k2);
        }
    }
    // Clean zeros
    out.retain(|_, v| *v != 0);
    out
}

/// Apply Weyl shift η^{a·e_I + b·m_I} to a refined index.
pub fn apply_weyl_shift(
    refined: &RefinedIndexResult,
    m_ext: &[i64],
    e_ext_x2: &[i64],
    weyl_a: &[Rational64],
    weyl_b: &[Rational64],
    num_hard: usize,
    cusp_idx: usize,
) -> RefinedIndexResult {
    let m_i = m_ext[cusp_idx];
    let e_i_x2 = e_ext_x2[cusp_idx];
    let mut shift_x2 = vec![0i64; num_hard];
    let mut all_zero = true;
    for j in 0..num_hard {
        let s = weyl_a[j] * Rational64::from_integer(e_i_x2)
            + weyl_b[j] * Rational64::from_integer(2 * m_i);
        let v = *s.numer() / *s.denom();
        shift_x2[j] = v;
        if v != 0 {
            all_zero = false;
        }
    }
    if all_zero {
        return refined.clone();
    }
    let mut out = RefinedIndexResult::new();
    for (key, &coeff) in refined {
        if coeff == 0 {
            continue;
        }
        let mut nk = Vec::with_capacity(key.len());
        nk.push(key[0]);
        for j in 0..num_hard {
            nk.push(key[1 + j] + shift_x2[j]);
        }
        *out.entry(nk).or_insert(0) += coeff;
    }
    out.retain(|_, v| *v != 0);
    out
}

/// Result of refined Dehn filling (all paths: ℓ=1, ℓ≥2, multi-cusp).
///
/// Key structure:
/// - ℓ=1: `(qq, 2·W_0, …, 2·W_{H-1})`
/// - ℓ≥2: `(qq, 2·W_0, …, 2·W_{H-1}, 2·V_0)`
/// - Multi-cusp: `(qq, 2·W_0, …, 2·W_{H-1}, 2·V_0, …, 2·V_{C-1})`
#[derive(Debug, Clone)]
pub struct FilledRefinedResult {
    pub p: i64,
    pub q: i64,
    pub cusp_idx: i64, // -1 for multi-cusp
    pub series: MultiEtaSeries,
    pub qq_order: i32,
    pub eta_order: i32,
    pub hj_ks: Vec<i64>,
    pub n_kernel_terms: usize,
    pub num_hard: usize,
    pub has_cusp_eta: bool,
    pub num_cusp_eta: usize,
}

impl FilledRefinedResult {
    /// Set η_j = 1 (W_j = 0) for hard-edge indices in `edges`.
    ///
    /// Zeros dimension 1+j in every key and sums colliding entries.
    pub fn collapse_eta_edges(&self, edges: &[usize]) -> Self {
        if edges.is_empty() {
            return self.clone();
        }
        let zero = Rational64::from_integer(0);
        let positions: Vec<usize> = edges.iter().map(|&j| 1 + j).collect();
        let mut new_series = MultiEtaSeries::new();
        for (key, &coeff) in &self.series {
            if coeff == zero {
                continue;
            }
            let mut nk = key.clone();
            for &pos in &positions {
                if pos < nk.len() {
                    nk[pos] = 0;
                }
            }
            let e = new_series.entry(nk).or_insert(zero);
            *e += coeff;
        }
        new_series.retain(|_, v| *v != zero);
        FilledRefinedResult {
            series: new_series,
            ..self.clone()
        }
    }

    /// Set η_{V_ci} = 1 for cusp-η indices in `cusp_indices`.
    ///
    /// Zeros dimension `1 + num_hard + ci` and sums colliding entries.
    pub fn collapse_cusp_etas(&self, cusp_indices: &[usize]) -> Self {
        if cusp_indices.is_empty() || !self.has_cusp_eta {
            return self.clone();
        }
        let zero = Rational64::from_integer(0);
        let positions: Vec<usize> = cusp_indices
            .iter()
            .filter(|&&ci| ci < self.num_cusp_eta)
            .map(|&ci| 1 + self.num_hard + ci)
            .collect();
        if positions.is_empty() {
            return self.clone();
        }
        let mut new_series = MultiEtaSeries::new();
        for (key, &coeff) in &self.series {
            if coeff == zero {
                continue;
            }
            let mut nk = key.clone();
            for &pos in &positions {
                if pos < nk.len() {
                    nk[pos] = 0;
                }
            }
            let e = new_series.entry(nk).or_insert(zero);
            *e += coeff;
        }
        new_series.retain(|_, v| *v != zero);
        FilledRefinedResult {
            series: new_series,
            ..self.clone()
        }
    }
}

/// Apply unrefined K(k,1; m, e) factor to a `MultiEtaSeries`.
/// Only the qq dimension (first element) is shifted; η dims are untouched.
pub fn apply_k1_factor_multi(
    series: &MultiEtaSeries,
    c: i64,
    phase: i64,
    multiplicity: i64,
    qq_order: i64,
) -> MultiEtaSeries {
    let sign = if phase % 2 == 0 {
        Rational64::from_integer(1)
    } else {
        Rational64::from_integer(-1)
    };
    let half = Rational64::new(1, 2);
    let mult = Rational64::from_integer(multiplicity);
    let zero = Rational64::from_integer(0);

    if c == 0 {
        let scalar = half * sign * mult;
        let mut out = MultiEtaSeries::new();
        for (key, &c_val) in series {
            let scaled = c_val * scalar;
            if scaled == zero {
                continue;
            }
            let qq_p = key[0];
            let rest = &key[1..];
            // +phase shift
            let new_qq_a = qq_p + phase;
            if new_qq_a <= qq_order {
                let mut nk = Vec::with_capacity(key.len());
                nk.push(new_qq_a);
                nk.extend_from_slice(rest);
                let e = out.entry(nk).or_insert(zero);
                *e += scaled;
            }
            // -phase shift
            let new_qq_b = qq_p - phase;
            if new_qq_b <= qq_order {
                let mut nk = Vec::with_capacity(key.len());
                nk.push(new_qq_b);
                nk.extend_from_slice(rest);
                let e = out.entry(nk).or_insert(zero);
                *e += scaled;
            }
        }
        out.retain(|_, v| *v != zero);
        out
    } else {
        let scalar = -half * sign * mult;
        if scalar == zero {
            return MultiEtaSeries::new();
        }
        series
            .iter()
            .filter_map(|(k, &v)| {
                let r = v * scalar;
                if r == zero { None } else { Some((k.clone(), r)) }
            })
            .collect()
    }
}
