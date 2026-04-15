//! Sparse q-series polynomials keyed by `i32` powers with `i64` coefficients.
//!
//! The series are stored as `hashbrown::HashMap<i32, i64>` with no zero
//! coefficients. All outputs from this module guarantee that invariant.

use hashbrown::HashMap;

/// Sparse q-series: `{power → coefficient}` with no zero coefficients.
pub type QSeries = HashMap<i32, i64>;

/// Multiply two sparse q-series keeping only terms with power ≤ `budget`.
///
/// Port of `poly_convolve` in v0.5's `_c_kernel/tet_index.c`.
/// The C version densifies to arrays of length `budget + 1`; we do the same
/// in Rust with `Vec<i64>` for cache-friendly O(n²) convolution.
pub fn convolve(a: &QSeries, b: &QSeries, budget: i32) -> QSeries {
    if budget < 0 {
        return QSeries::new();
    }
    let len = (budget as usize) + 1;
    let mut da = vec![0i64; len];
    let mut db = vec![0i64; len];
    for (&p, &c) in a {
        if p >= 0 && (p as usize) < len {
            da[p as usize] = c;
        }
    }
    for (&p, &c) in b {
        if p >= 0 && (p as usize) < len {
            db[p as usize] = c;
        }
    }
    let mut dc = vec![0i64; len];
    for i in 0..len {
        let ai = da[i];
        if ai == 0 {
            continue;
        }
        for j in 0..(len - i) {
            let bj = db[j];
            if bj == 0 {
                continue;
            }
            dc[i + j] += ai * bj;
        }
    }
    let mut out = QSeries::with_capacity(len);
    for (p, c) in dc.into_iter().enumerate() {
        if c != 0 {
            out.insert(p as i32, c);
        }
    }
    out
}

/// Convert a sorted `[[power, coeff], …]` list (as stored in goldens) into a
/// `QSeries`. Skips any zero-coefficient pairs defensively.
pub fn from_pairs(pairs: &[[i64; 2]]) -> QSeries {
    let mut out = QSeries::with_capacity(pairs.len());
    for &[p, c] in pairs {
        if c != 0 {
            out.insert(p as i32, c);
        }
    }
    out
}

/// Stable serialization: `QSeries` → sorted `Vec<[i64;2]>`.
pub fn to_sorted_pairs(q: &QSeries) -> Vec<[i64; 2]> {
    let mut pairs: Vec<[i64; 2]> = q.iter().map(|(&k, &v)| [k as i64, v]).collect();
    pairs.sort_by_key(|p| p[0]);
    pairs
}
