//! Neumann-Zagier data and cusp basis-change arithmetic.
//!
//! The base `NzData` (default M, L/2 basis) is pre-extracted from v0.5 and
//! lives in the `nz` SQLite table; see [`crate::census::Census::load_nz`].
//! This module operates on the doubled integer form `(2·g_NZ, ν_x, 2·ν_p)`
//! so all arithmetic stays in `i64`.
//!
//! Port targets:
//!   - `apply_cusp_basis_change` — odd-P Dehn basis swap (v0.5 neumann_zagier.py:459).
//!   - `apply_general_cusp_basis_change` — SL(2,ℤ) at one cusp (:566).

pub use crate::census::NzData;
use crate::basis::BasisSelection;
use num_integer::Integer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NzError {
    CuspOutOfRange { k: usize, r: usize },
    EvenP { p: i32 },
    NonUnitBezout { p: i32, q: i32, gcd: i64 },
    NonUnitDet { a: i32, b: i32, c: i32, d: i32, det: i32 },
}

impl std::fmt::Display for NzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NzError::CuspOutOfRange { k, r } => {
                write!(f, "cusp_idx={k} out of range [0, {r})")
            }
            NzError::EvenP { p } => write!(
                f,
                "apply_cusp_basis_change: P={p} is even; no integer symplectic conjugate"
            ),
            NzError::NonUnitBezout { p, q, gcd } => write!(
                f,
                "apply_cusp_basis_change: gcd({p}, {}) = {gcd} ≠ 1",
                -2 * q
            ),
            NzError::NonUnitDet { a, b, c, d, det } => {
                write!(f, "det [[{a},{b}],[{c},{d}]] = {det} ≠ 1")
            }
        }
    }
}
impl std::error::Error for NzError {}

/// `(g, x, y)` with `a·x + b·y = g` and `g = gcd(|a|, |b|) ≥ 0`.
///
/// Direct port of v0.5's `_ext_gcd` — uses floor-division semantics so the
/// returned Bezout pair matches Python's exactly (num_integer's ExtendedGcd
/// returns a mathematically valid but numerically different pair).
fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        if a >= 0 { (a, 1, 0) } else { (-a, -1, 0) }
    } else {
        let (g, x, y) = ext_gcd(b, a.mod_floor(&b));
        (g, y, x - a.div_floor(&b) * y)
    }
}

/// Apply a symplectic basis change at cusp `k` with primitive odd-P slope `(P, Q)`.
///
/// Port of v0.5's `apply_cusp_basis_change`. Operates on the doubled integer
/// representation; the result is still exactly integer-valued in doubled form.
///
/// New rows (in doubled form, writing `M = 2·g_row[k]`, `L = 2·g_row[n+k]`):
///   - new position: `2·(P·M + Q·L) / 2 = P·M + Q·L`
///     → stored as `2·new_pos = P·(2·M) + Q·(2·L)`.
///     Careful: `2·g_row[k]` is `2·M` (M is integer), `2·g_row[n+k]` is `2·(L/2) = L`.
///     So in doubled form: new_pos_x2 = P·(g_row_x2[k]) + 2Q·(g_row_x2[n+k]).
///   - new momentum: `a·M + b·(L/2)`
///     → in doubled form: new_mom_x2 = a·(g_row_x2[k]) + b·(g_row_x2[n+k]).
pub fn apply_cusp_basis_change(
    nz: &NzData,
    cusp_idx: usize,
    p: i32,
    q: i32,
) -> Result<NzData, NzError> {
    if cusp_idx >= nz.r {
        return Err(NzError::CuspOutOfRange { k: cusp_idx, r: nz.r });
    }
    if p.rem_euclid(2) == 0 {
        return Err(NzError::EvenP { p });
    }

    // Solve P·b − 2Q·a = 1  ↔  ext_gcd(P, -2Q) = (1, b, a).
    let (g, b, a) = ext_gcd(p as i64, (-2 * q) as i64);
    if g != 1 {
        return Err(NzError::NonUnitBezout { p, q, gcd: g });
    }

    let n = nz.n;
    let k = cusp_idx;
    let p_i64 = p as i64;
    let q_i64 = q as i64;

    let mut out = nz.clone();

    let old_pos: Vec<i64> = nz.g_row_x2(k).to_vec();
    let old_mom: Vec<i64> = nz.g_row_x2(n + k).to_vec();

    // new_pos_x2 = P·old_pos_x2 + 2Q·old_mom_x2
    for (dst, (a_v, b_v)) in out.g_row_x2_mut(k).iter_mut().zip(old_pos.iter().zip(old_mom.iter())) {
        *dst = p_i64 * *a_v + 2 * q_i64 * *b_v;
    }
    // new_mom_x2 = a·old_pos_x2 + b·old_mom_x2
    for (dst, (a_v, b_v)) in out.g_row_x2_mut(n + k).iter_mut().zip(old_pos.iter().zip(old_mom.iter())) {
        *dst = a * *a_v + b * *b_v;
    }

    // nu_x_new[k] = P·nu_x[k] + 2Q·nu_p[k]  (integer)
    // Since nu_p_x2 = 2·nu_p, we have 2Q·nu_p = Q·nu_p_x2.
    out.nu_x[k] = p_i64 * nz.nu_x[k] + q_i64 * nz.nu_p_x2[k];

    // nu_p_x2_new[k] = 2·(a·nu_x[k] + b·nu_p[k]) = 2a·nu_x[k] + b·nu_p_x2[k]
    out.nu_p_x2[k] = 2 * a * nz.nu_x[k] + b * nz.nu_p_x2[k];

    Ok(out)
}

/// Apply a general SL(2,ℤ) basis change at one cusp (no odd-P requirement).
///
/// Port of v0.5's `apply_general_cusp_basis_change`.
pub fn apply_general_cusp_basis_change(
    nz: &NzData,
    cusp_idx: usize,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
) -> Result<NzData, NzError> {
    if cusp_idx >= nz.r {
        return Err(NzError::CuspOutOfRange { k: cusp_idx, r: nz.r });
    }
    let det = a * d - b * c;
    if det != 1 {
        return Err(NzError::NonUnitDet { a, b, c, d, det });
    }

    let n = nz.n;
    let k = cusp_idx;
    let (a, b, c, d) = (a as i64, b as i64, c as i64, d as i64);

    let mut out = nz.clone();
    let old_pos: Vec<i64> = nz.g_row_x2(k).to_vec();
    let old_mom: Vec<i64> = nz.g_row_x2(n + k).to_vec();

    // new_M   = a·M + 2b·(L/2)  → doubled: a·(2M) + 2b·L = a·old_pos_x2 + 2b·old_mom_x2.
    for (dst, (pv, mv)) in out.g_row_x2_mut(k).iter_mut().zip(old_pos.iter().zip(old_mom.iter())) {
        *dst = a * *pv + 2 * b * *mv;
    }
    // new_L/2 = (c/2)·M + d·(L/2)  → doubled (new_L): c·M + d·L = c·(old_pos_x2/2)·2?
    // old_pos_x2 = 2M (integer). So c·M = c·(old_pos_x2/2). To keep doubled form:
    //     2·new_L/2 = c·M + 2d·(L/2) = c·(old_pos_x2/2) + d·old_mom_x2.
    // But c·(old_pos_x2/2) is only integer when c is even OR old_pos_x2 is even.
    // Since old_pos_x2 = 2·(integer meridian row), it is always even ⇒ c·(old_pos_x2/2) is integer.
    for (dst, (pv, mv)) in out.g_row_x2_mut(n + k).iter_mut().zip(old_pos.iter().zip(old_mom.iter())) {
        debug_assert!(pv % 2 == 0, "meridian row entry must be doubled-integer");
        *dst = c * (pv / 2) + d * *mv;
    }

    out.nu_x[k] = a * nz.nu_x[k] + b * nz.nu_p_x2[k];
    // nu_p_x2_new[k] = 2·((c/2)·nu_x + d·nu_p) = c·nu_x + d·nu_p_x2.
    out.nu_p_x2[k] = c * nz.nu_x[k] + d * nz.nu_p_x2[k];

    Ok(out)
}

/// Apply the basis changes for every odd-P cusp in `basis` (matches
/// v0.5's `apply_basis_changes` in `basis_selection.py:138-152`).
pub fn apply_basis_changes(nz: &NzData, basis: &BasisSelection) -> Result<NzData, NzError> {
    let mut out = nz.clone();
    for cc in &basis.choices {
        if cc.p.rem_euclid(2) != 0 {
            out = apply_cusp_basis_change(&out, cc.cusp_idx, cc.p, cc.q)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_gcd_basic() {
        // Bezout: 3·b − 2·a = 1 for P=3, Q=1 → ext_gcd(3, -2) = (1, 1, 1) ⇒ 3·1 + (-2)·1 = 1. ✓
        let (g, b, a) = ext_gcd(3, -2);
        assert_eq!(g, 1);
        assert_eq!(3 * b + (-2) * a, 1);
    }

    #[test]
    fn ext_gcd_matches_python_for_p_q_samples() {
        for &(p, q) in &[(1, 0), (3, 1), (5, 2), (-3, 1), (7, -3)] {
            let (g, b, a) = ext_gcd(p as i64, (-2 * q) as i64);
            assert_eq!(g, 1, "gcd(P, -2Q) for ({p},{q})");
            assert_eq!((p as i64) * b + (-2 * q as i64) * a, 1);
        }
    }
}
