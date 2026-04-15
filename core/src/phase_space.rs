//! Phase-space basis: re-exports [`crate::census::PhaseSpaceData`] and builds
//! the `(n - r)` edge basis `[independent_easy…, hard_padding…]`.
//!
//! v0.5's `find_easy_edges` (pattern enumeration + LAPACK) is run ahead of
//! time by `data/extract_census.py --phase-space`; this module just reads the
//! pre-computed BLOB and assembles the basis.

pub use crate::census::PhaseSpaceData;

/// The ordered `(n - r)` edge basis used downstream for NZ / index computation.
///
/// Rows come from `easy_edges[easy_indep[k]]` first, then `hard_padding` rows.
/// Each row is a raw `3n` edge equation (the SnaPy convention).
pub fn basis_edges(ps: &PhaseSpaceData) -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = Vec::with_capacity(ps.easy_indep.len() + ps.num_hard());
    for &idx in &ps.easy_indep {
        out.push(ps.easy_row(idx as usize).to_vec());
    }
    for i in 0..ps.num_hard() {
        out.push(ps.hard_row(i).to_vec());
    }
    out
}
