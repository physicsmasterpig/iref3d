//! Gluing-equation reduction: eliminates `Z_i' = 1 - Z_i - Z_i''`.
//!
//! Port of v0.5's `core/gluing_equations.py`. The column-pivot indices from
//! scipy's QR are read from the census DB (see [`crate::census`]), not
//! recomputed, so the reduction here is pure integer arithmetic.
//!
//! Interleaved ordering: reduced coefficient vectors are length `2n` with
//! entries `(Z_1, Z_1'', Z_2, Z_2'', …, Z_n, Z_n'')`.

use crate::census::ManifoldData;

/// Reduced gluing equations for a manifold.
#[derive(Debug, Clone)]
pub struct ReducedGluing {
    pub n: usize,
    pub r: usize,
    /// `(n, 2n)` row-major — reduced edge coefficients.
    pub edge_coeffs: Vec<i64>,
    /// `(n,)` — edge constants (sum of `Z_i'` coefficients replaced by 1).
    pub edge_consts: Vec<i64>,
    /// `(2r, 2n)` row-major — reduced cusp coefficients (meridian, longitude pairs).
    pub cusp_coeffs: Vec<i64>,
    /// `(2r,)` — cusp constants.
    pub cusp_consts: Vec<i64>,
    /// Length `n - r`, sorted — scipy column-pivot indices identifying
    /// a linearly independent subset of edge rows.
    pub pivots: Vec<usize>,
}

impl ReducedGluing {
    #[inline]
    pub fn edge_row(&self, i: usize) -> &[i64] {
        &self.edge_coeffs[i * 2 * self.n..(i + 1) * 2 * self.n]
    }
    #[inline]
    pub fn cusp_row(&self, i: usize) -> &[i64] {
        &self.cusp_coeffs[i * 2 * self.n..(i + 1) * 2 * self.n]
    }
    /// Reduced meridian for cusp `k`, length `2n`.
    #[inline]
    pub fn meridian_coeffs(&self, k: usize) -> &[i64] {
        self.cusp_row(2 * k)
    }
    /// Reduced longitude for cusp `k`, length `2n`.
    #[inline]
    pub fn longitude_coeffs(&self, k: usize) -> &[i64] {
        self.cusp_row(2 * k + 1)
    }
    /// Symplectic pairing `[a, b] = a · Ω · b` with interleaved Ω
    /// (Ω_{2i,2i+1} = +1, Ω_{2i+1,2i} = −1).
    pub fn commutator(&self, a: &[i64], b: &[i64]) -> i64 {
        assert_eq!(a.len(), 2 * self.n);
        assert_eq!(b.len(), 2 * self.n);
        let mut acc: i64 = 0;
        for i in 0..self.n {
            acc += a[2 * i] * b[2 * i + 1];
            acc -= a[2 * i + 1] * b[2 * i];
        }
        acc
    }
}

/// Substitute `Z_i' = 1 - Z_i - Z_i''` into one row of the raw `3n` gluing matrix.
///
/// Given row coefficients `(f_i, g_i, h_i)` for `(Z_i, Z_i', Z_i'')`, returns
/// `(const, 2n_coeffs)` where `const = Σ g_i` and the reduced coefficients are
/// `(f_i - g_i, h_i - g_i)` in interleaved order.
fn reduce_row(row_3n: &[i32], n: usize) -> (i64, Vec<i64>) {
    debug_assert_eq!(row_3n.len(), 3 * n);
    let mut out = vec![0i64; 2 * n];
    let mut c: i64 = 0;
    for i in 0..n {
        let f = row_3n[3 * i] as i64;
        let g = row_3n[3 * i + 1] as i64;
        let h = row_3n[3 * i + 2] as i64;
        c += g;
        out[2 * i] = f - g;
        out[2 * i + 1] = h - g;
    }
    (c, out)
}

/// Reduce all gluing equations (edge + cusp rows) for a manifold.
///
/// Port of `reduce_gluing_equations` in v0.5. The `pivots` vector is taken
/// verbatim from the census DB (scipy's QR column-pivot result).
pub fn reduce_gluing_equations(md: &ManifoldData) -> ReducedGluing {
    let n = md.n;
    let r = md.r;

    let mut edge_coeffs = Vec::with_capacity(n * 2 * n);
    let mut edge_consts = Vec::with_capacity(n);
    for i in 0..n {
        let (c, v) = reduce_row(md.row(i), n);
        edge_consts.push(c);
        edge_coeffs.extend_from_slice(&v);
    }

    let mut cusp_coeffs = Vec::with_capacity(2 * r * 2 * n);
    let mut cusp_consts = Vec::with_capacity(2 * r);
    for j in 0..(2 * r) {
        let (c, v) = reduce_row(md.row(n + j), n);
        cusp_consts.push(c);
        cusp_coeffs.extend_from_slice(&v);
    }

    let pivots: Vec<usize> = md.pivots.iter().map(|&p| p as usize).collect();
    debug_assert_eq!(pivots.len(), n.saturating_sub(r));

    ReducedGluing {
        n,
        r,
        edge_coeffs,
        edge_consts,
        cusp_coeffs,
        cusp_consts,
        pivots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md(n: usize, r: usize, gluing: Vec<i32>, pivots: Vec<i32>) -> ManifoldData {
        ManifoldData {
            census: "test".into(),
            name: "test".into(),
            n,
            r,
            gluing,
            pivots,
        }
    }

    #[test]
    fn reduce_row_simple() {
        // n=1: row [1, 2, 3] → const = 2, coeffs = (1-2, 3-2) = (-1, 1).
        let (c, v) = reduce_row(&[1, 2, 3], 1);
        assert_eq!(c, 2);
        assert_eq!(v, vec![-1, 1]);
    }

    #[test]
    fn commutator_symplectic_basis() {
        // n=2: e_0 = (1,0,0,0), e_1 = (0,1,0,0) → Ω_{0,1}=+1.
        let rg = ReducedGluing {
            n: 2,
            r: 1,
            edge_coeffs: vec![0; 8],
            edge_consts: vec![0; 2],
            cusp_coeffs: vec![0; 8],
            cusp_consts: vec![0; 2],
            pivots: vec![0],
        };
        assert_eq!(rg.commutator(&[1, 0, 0, 0], &[0, 1, 0, 0]), 1);
        assert_eq!(rg.commutator(&[0, 1, 0, 0], &[1, 0, 0, 0]), -1);
        assert_eq!(rg.commutator(&[0, 0, 1, 0], &[0, 0, 0, 1]), 1);
    }
}
