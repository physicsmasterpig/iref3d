//! iref3d-core — Rust core for the refined 3D index (I^ref).
//!
//! Module layout mirrors the computational domains identified in the plan
//! (`/Users/pmp/.claude/plans/harmonic-hatching-puffin.md`). Modules are
//! brought online one at a time; the declarations below will grow as each
//! module lands.

pub mod census;
pub mod poly;
pub mod kernel;
pub mod degree_bounds;
pub mod gluing;
pub mod phase_space;
pub mod basis;
pub mod nz;
pub mod summation;
pub mod index_unrefined;
pub mod index_refined;
pub mod ab_vectors;
pub mod weyl_symmetry;
pub mod adjoint_unrefined;

// Upcoming modules (stubbed in the plan, not yet implemented):
//
// pub mod adjoint_unrefined;
// pub mod adjoint_eta0;
// pub mod adjoint_w_scan;
// pub mod dehn;
// pub mod refined_dehn;
// pub mod cache;

#[cfg(feature = "python")]
pub mod ffi;
