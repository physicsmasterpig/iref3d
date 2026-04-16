//! Tetrahedron index kernel — port of v0.5's `_c_kernel/tet_index.c`.
//!
//! Public functions:
//!   - [`tet_degree_x2`] — 2·δ(m,e), integer arithmetic.
//!   - [`tet_index_series`] — full MIt(m,e) sparse q^½ series (memoized).
//!
//! Results must be bit-identical to the Python/C reference.

use std::sync::RwLock;

use crate::poly::QSeries;

/// In-memory cache for `tet_index_series` results.
///
/// Key = `(m, e, qq_order)`, value = sparse QSeries.
/// Matches v0.5's module-level `_tet_cache` dict.
static TET_CACHE: RwLock<Option<hashbrown::HashMap<(i32, i32, i32), QSeries>>> =
    RwLock::new(None);

/// Clear the tet cache (e.g. when switching manifolds with different parameters).
pub fn clear_tet_cache() {
    let mut cache = TET_CACHE.write().unwrap();
    *cache = None;
}

/// `2 · δ(m, e)` — doubled tetrahedron degree as a plain integer.
///
/// Port of `c_tet_degree_x2` in `tet_index.c`.
pub fn tet_degree_x2(m: i32, e: i32) -> i32 {
    let pos_m = m.max(0);
    let pos_me = (m + e).max(0);
    let pos_nm = (-m).max(0);
    let pos_e = e.max(0);
    let pos_ne = (-e).max(0);
    let pos_nem = (-e - m).max(0);

    let half_sum = pos_m * pos_me + pos_nm * pos_e + pos_ne * pos_nem;
    let mx = 0.max(m).max(-e);
    half_sum + mx
}

/// Extend `inv_fact` up to and including index `up_to`.
///
/// `inv_fact[k]` is the dense truncation (length `inner_order + 1`) of
/// `1 / prod_{j=1..=k} (1 - qq^{2j})`.
fn extend_inv_fact(inv_fact: &mut Vec<Vec<i64>>, up_to: usize, inner_order: usize) {
    let poly_len = inner_order + 1;
    while inv_fact.len() <= up_to {
        let k = inv_fact.len();
        let mut new_poly = vec![0i64; poly_len];
        if k == 0 {
            new_poly[0] = 1;
        } else {
            let prev = &inv_fact[k - 1];
            let step = 2 * k;
            for p in 0..poly_len {
                let cp = prev[p];
                if cp == 0 {
                    continue;
                }
                let mut q = p;
                while q < poly_len {
                    new_poly[q] += cp;
                    q += step;
                }
            }
        }
        inv_fact.push(new_poly);
    }
}

/// Raw I_t(mm, ee) series up to `inner_order` (dense `Vec<i64>`).
///
/// Port of `it_direct` in `tet_index.c`. Caller owns the result.
fn it_direct(mm: i32, ee: i32, inner_order: usize) -> Vec<i64> {
    let poly_len = inner_order + 1;
    let mut result = vec![0i64; poly_len];

    let n_min: i64 = (-(ee as i64)).max(0);

    let mut inv_fact: Vec<Vec<i64>> = Vec::new();

    let mut n = n_min;
    loop {
        // exp_qq = n(n+1) - (2n+ee) * mm
        let exp_qq: i64 = n * (n + 1) - (2 * n + ee as i64) * mm as i64;
        if exp_qq > inner_order as i64 {
            break;
        }

        // Ensure inv_fact[n] and inv_fact[n+ee] are available (for non-negative).
        let need_a = n;
        let need_b = n + ee as i64;
        let mut need = need_a.max(need_b);
        if need < 0 {
            need = 0;
        }
        extend_inv_fact(&mut inv_fact, need as usize, inner_order);

        let d1 = &inv_fact[n as usize].clone();
        let ne = n + ee as i64;

        // For ne >= 0 use inv_fact[ne]; for ne < 0 use the unit polynomial {0: 1}.
        let d2_owned: Vec<i64>;
        let d2: &[i64];
        let d2_len: usize;
        if ne >= 0 {
            d2 = &inv_fact[ne as usize];
            d2_len = poly_len;
        } else {
            d2_owned = vec![1i64];
            d2 = &d2_owned;
            d2_len = 1;
        }

        let sign: i64 = if n % 2 == 0 { 1 } else { -1 };

        // Convolve d1 * d2, shift by exp_qq, accumulate into result.
        for p1 in 0..poly_len {
            let c1 = d1[p1];
            if c1 == 0 {
                continue;
            }
            let budget = inner_order as i64 - exp_qq - p1 as i64;
            if budget < 0 {
                continue;
            }
            let p2_max = (budget as usize).min(d2_len - 1);
            for p2 in 0..=p2_max {
                let c2 = d2[p2];
                if c2 == 0 {
                    continue;
                }
                let total = exp_qq as usize + p1 + p2;
                result[total] += sign * c1 * c2;
            }
        }

        n += 1;
    }

    result
}

/// Full MIt(m, e) sparse q^½ series up to `qq_order`.
///
/// Port of `py_tet_index_series` in `tet_index.c`. Returns the coefficient map
/// `{power → coeff}` with all zeros removed.
///
/// Results are memoized in a global cache keyed by `(m, e, qq_order)`.
pub fn tet_index_series(m: i32, e: i32, qq_order: i32) -> QSeries {
    if qq_order < 0 {
        return QSeries::new();
    }

    // Check cache (read lock)
    let key = (m, e, qq_order);
    {
        let cache = TET_CACHE.read().unwrap();
        if let Some(map) = cache.as_ref() {
            if let Some(cached) = map.get(&key) {
                return cached.clone();
            }
        }
    }

    // Compute
    let result = tet_index_series_uncached(m, e, qq_order);

    // Store in cache (write lock)
    {
        let mut cache = TET_CACHE.write().unwrap();
        let map = cache.get_or_insert_with(hashbrown::HashMap::new);
        map.insert(key, result.clone());
    }

    result
}

/// Uncached implementation of `tet_index_series`.
fn tet_index_series_uncached(m: i32, e: i32, qq_order: i32) -> QSeries {
    let (raw, raw_len, shift, sign_m) = if m + e >= 0 {
        let inner_order = (qq_order - m).max(0) as usize;
        let raw = it_direct(-m - e, m, inner_order);
        let sign_m: i64 = if m.rem_euclid(2) == 0 { 1 } else { -1 };
        (raw, inner_order + 1, m, sign_m)
    } else {
        let inner_order = qq_order as usize;
        let raw = it_direct(m, e, inner_order);
        (raw, inner_order + 1, 0, 1)
    };

    let mut out = QSeries::with_capacity(raw_len);
    for p in 0..raw_len {
        let rp = raw[p];
        if rp == 0 {
            continue;
        }
        let new_pwr = p as i32 + shift;
        if new_pwr < 0 || new_pwr > qq_order {
            continue;
        }
        let mut coeff = sign_m * rp;
        if coeff == 0 {
            continue;
        }
        if let Some(existing) = out.get(&new_pwr) {
            coeff += *existing;
        }
        if coeff != 0 {
            out.insert(new_pwr, coeff);
        } else {
            out.remove(&new_pwr);
        }
    }
    out
}
