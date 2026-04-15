//! Degree bounds used by summation enumeration and filling feasibility.
//!
//! Exact (possibly half-integer) companion to [`crate::kernel::tet_degree_x2`].

use num_rational::Ratio;

/// `δ(m, e)` — leading q^{1/2}-power of the tetrahedron index I_Δ(m, e).
///
/// Returns an exact `Ratio<i32>` (integer or half-integer). Equivalent to
/// `Ratio::new(tet_degree_x2(m, e), 2)`.
///
/// Port of `tet_degree` in v0.5's `core/index_3d.py`.
pub fn tet_degree(m: i32, e: i32) -> Ratio<i32> {
    let x2 = crate::kernel::tet_degree_x2(m, e);
    Ratio::new(x2, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Ratio;

    #[test]
    fn symmetry_m_e_swap() {
        // tet_degree(m, e) == tet_degree(-e, -m) per v0.5 docstring.
        for m in -5..=5 {
            for e in -5..=5 {
                assert_eq!(tet_degree(m, e), tet_degree(-e, -m));
            }
        }
    }

    #[test]
    fn zero_at_origin() {
        assert_eq!(tet_degree(0, 0), Ratio::new(0, 1));
    }

    #[test]
    fn always_nonnegative() {
        for m in -10..=10 {
            for e in -10..=10 {
                assert!(tet_degree(m, e) >= Ratio::new(0, 1));
            }
        }
    }
}
