//! NC-cycle compatibility check — basis-change + Weyl recheck + marginal test.
//!
//! Ports v0.5's `NcCompatWorker` and `MultiCuspNcCompatWorker`
//! (`app/workers/weyl_worker.py:76-451`).
//!
//! For each non-closable cycle (P, Q) at a cusp, performs an SL(2,ℤ) basis
//! change so that the NC slope becomes the new meridian, then recomputes
//! refined-index entries and runs the full Weyl check pipeline:
//!   - `compute_ab_vectors` for the (a, b) Weyl vectors,
//!   - `check_adjoint_projection` for the refined q¹-η⁰ projection (W_i compat),
//!   - `check_marginal` for the unrefined q¹ projection (kernel choice).

use hashbrown::{HashMap, HashSet};

use crate::ab_vectors::{compute_ab_vectors, ABVectors, Entry};
use crate::adjoint_eta0::{
    check_adjoint_projection, check_adjoint_projection_multi_cusp, AdjointResult,
    MultiCuspAdjointResult,
};
use crate::adjoint_unrefined::{check_marginal, MarginalCheck};
use crate::dehn::kernel_terms::find_rs;
use crate::index_refined::compute_refined_index;
use crate::nz::{apply_general_cusp_basis_change, NzData};
use crate::summation::EnumerationState;
use crate::weyl_symmetry::check_weyl_symmetry;

/// Result of a single-cusp NC compatibility check.
#[derive(Debug, Clone)]
pub struct NcCompatResult {
    pub p: i64,
    pub q: i64,
    pub cusp_idx: usize,
    /// Weyl (a, b) vectors in the NC basis.  `None` if not computable.
    pub ab: Option<ABVectors>,
    /// Filling-compatible (a, b) with incompat edges zeroed.
    pub ab_compat: Option<ABVectors>,
    /// Indices of hard edges incompatible with Dehn filling.
    pub collapsed_edges: Vec<usize>,
    /// Refined adjoint projection result.
    pub adjoint: Option<AdjointResult>,
    /// Per-entry Weyl-symmetry pass/fail map.
    pub weyl_symmetric: HashMap<(Vec<i64>, Vec<i64>), bool>,
    pub all_weyl_symmetric: bool,
    /// Unrefined marginal check (drives kernel choice).
    pub marginal: MarginalCheck,
}

/// Result of a multi-cusp joint NC compatibility check.
#[derive(Debug, Clone)]
pub struct MultiCuspNcCompatResult {
    pub cusp_specs: Vec<(usize, i64, i64)>, // (cusp_idx, P, Q)
    pub ab: Option<ABVectors>,
    pub ab_compat: Option<ABVectors>,
    pub collapsed_edges: Vec<usize>,
    pub multi_cusp_adjoint: Option<MultiCuspAdjointResult>,
    pub weyl_symmetric: HashMap<(Vec<i64>, Vec<i64>), bool>,
    pub all_weyl_symmetric: bool,
}

/// Probe grid for extracting (a, b) vectors in the NC basis.
///
/// Returns `(m_ext, e_ext_x2)` pairs for the dedicated probe points:
///   - a-probe: m=0, e ∈ {−2,−1,−½,+½,+1,+2} at cusp
///   - b-probe: e=0, m ∈ {−2,−1,+1,+2} at cusp
fn ab_probe_grid(n_cusps: usize, cusp_idx: usize) -> Vec<(Vec<i64>, Vec<i64>)> {
    let mut points = Vec::new();
    // a-probe: m=0, e varies at target cusp (×2 encoding)
    for &e_x2 in &[-4i64, -2, -1, 1, 2, 4] {
        let m_ext = vec![0i64; n_cusps];
        let mut e_ext_x2 = vec![0i64; n_cusps];
        e_ext_x2[cusp_idx] = e_x2;
        points.push((m_ext, e_ext_x2));
    }
    // b-probe: e=0, m varies at target cusp
    for &m_val in &[-2i64, -1, 1, 2] {
        let mut m_ext = vec![0i64; n_cusps];
        m_ext[cusp_idx] = m_val;
        let e_ext_x2 = vec![0i64; n_cusps];
        points.push((m_ext, e_ext_x2));
    }
    points
}

/// Adjoint probe grid for multi-cusp check: all combinations of e values
/// across filled cusps at m=0.
fn adjoint_probe_grid_multi(
    n_cusps: usize,
    filled_cusp_indices: &[usize],
) -> Vec<(Vec<i64>, Vec<i64>)> {
    let d = filled_cusp_indices.len();
    let e_target_x2: [i64; 4] = [-4, -2, 2, 4];
    let e_other_x2: [i64; 3] = [-2, 0, 2];

    let mut points = Vec::new();
    for target_idx in 0..d {
        for &e_t in &e_target_x2 {
            let other_count = d - 1;
            let combos: usize = 3usize.pow(other_count as u32);
            for combo_idx in 0..combos {
                let mut e_o = Vec::with_capacity(other_count);
                let mut c = combo_idx;
                for _ in 0..other_count {
                    e_o.push(e_other_x2[c % 3]);
                    c /= 3;
                }
                let m_ext = vec![0i64; n_cusps];
                let mut e_ext_x2 = vec![0i64; n_cusps];
                let mut o = 0;
                for (j, &ci) in filled_cusp_indices.iter().enumerate() {
                    if j == target_idx {
                        e_ext_x2[ci] = e_t;
                    } else {
                        e_ext_x2[ci] = e_o[o];
                        o += 1;
                    }
                }
                points.push((m_ext, e_ext_x2));
            }
        }
    }
    points
}

/// Build Entry list from refined index computations in the NC basis.
///
/// Includes entries even when the refined index is empty — this is important
/// so that `check_adjoint_projection` sees all probed e values and doesn't
/// report them as missing.
fn build_entries(
    state: &EnumerationState,
    num_hard: usize,
    probe_points: &[(Vec<i64>, Vec<i64>)],
    q_order_half: i32,
) -> Vec<Entry> {
    let mut seen: HashSet<(Vec<i64>, Vec<i64>)> = HashSet::new();
    let mut entries = Vec::new();
    for (m_ext, e_ext_x2) in probe_points {
        let key = (m_ext.clone(), e_ext_x2.clone());
        if !seen.insert(key) {
            continue;
        }
        let result = compute_refined_index(state, num_hard, m_ext, e_ext_x2, q_order_half);
        entries.push(Entry {
            m_ext: m_ext.clone(),
            e_ext_x2: e_ext_x2.clone(),
            result,
        });
    }
    entries
}

/// Single-cusp NC compatibility check.
///
/// Applies SL(2,ℤ) basis change at `cusp_idx` so slope (P, Q) becomes
/// the new meridian, then runs the full Weyl check pipeline.
pub fn check_nc_compat(
    nz: &NzData,
    p: i64,
    q: i64,
    cusp_idx: usize,
    num_hard: usize,
    q_order_half: i32,
) -> NcCompatResult {
    // Basis change: (P,Q) → new meridian, (−R,−S) → new longitude
    let (r_val, s_val) = find_rs(p, q);
    let nz_nc = apply_general_cusp_basis_change(
        nz,
        cusp_idx,
        p as i32,
        q as i32,
        -(r_val as i32),
        -(s_val as i32),
    )
    .expect("basis change failed for NC cycle");

    let state_nc = EnumerationState::build(&nz_nc);
    let n_cusps = nz.r;

    // Build probe grid and compute entries
    let probe = ab_probe_grid(n_cusps, cusp_idx);
    let entries = build_entries(&state_nc, num_hard, &probe, q_order_half);

    if entries.is_empty() {
        return NcCompatResult {
            p,
            q,
            cusp_idx,
            ab: None,
            ab_compat: None,
            collapsed_edges: vec![],
            adjoint: None,
            weyl_symmetric: HashMap::new(),
            all_weyl_symmetric: false,
            marginal: check_marginal(&state_nc, n_cusps, cusp_idx, q_order_half),
        };
    }

    // Compute (a, b) vectors
    let ab = compute_ab_vectors(&entries, num_hard);
    let ab_compat = ab.as_ref().map(|a| a.make_filling_compatible());
    let collapsed_edges = ab.as_ref().map(|a| a.incompat_edges()).unwrap_or_default();
    let collapsed_set: HashSet<usize> = collapsed_edges.iter().copied().collect();

    // Weyl symmetry check
    let weyl_symmetric = if let Some(ref ab_v) = ab {
        check_weyl_symmetry(&entries, num_hard, ab_v, Some(q_order_half))
    } else {
        entries
            .iter()
            .map(|e| ((e.m_ext.clone(), e.e_ext_x2.clone()), false))
            .collect()
    };
    let all_weyl_symmetric = !weyl_symmetric.is_empty() && weyl_symmetric.values().all(|&v| v);

    // Adjoint projection (refined q¹-η⁰)
    let adjoint = ab_compat.as_ref().map(|ab_c| {
        let ce = if collapsed_edges.is_empty() {
            None
        } else {
            Some(&collapsed_set)
        };
        check_adjoint_projection(&entries, num_hard, Some(ab_c), cusp_idx, ce)
    });

    // Marginal check (unrefined q¹)
    let marginal = check_marginal(&state_nc, n_cusps, cusp_idx, q_order_half);

    NcCompatResult {
        p,
        q,
        cusp_idx,
        ab,
        ab_compat,
        collapsed_edges,
        adjoint,
        weyl_symmetric,
        all_weyl_symmetric,
        marginal,
    }
}

/// Multi-cusp joint NC compatibility check.
///
/// Applies basis changes for all filled cusps, then runs the joint
/// adjoint projection check across all d filled-cusp fugacities.
pub fn check_nc_compat_multi_cusp(
    nz: &NzData,
    cusp_specs: &[(usize, i64, i64)], // (cusp_idx, P, Q)
    num_hard: usize,
    q_order_half: i32,
) -> MultiCuspNcCompatResult {
    let n_cusps = nz.r;

    // Apply basis changes for all filled cusps
    let mut nz_nc = nz.clone();
    for &(cusp_idx, p, q) in cusp_specs {
        let (r_val, s_val) = find_rs(p, q);
        nz_nc = apply_general_cusp_basis_change(
            &nz_nc,
            cusp_idx,
            p as i32,
            q as i32,
            -(r_val as i32),
            -(s_val as i32),
        )
        .expect("basis change failed for multi-cusp NC cycle");
    }

    let state_nc = EnumerationState::build(&nz_nc);
    let filled_cusp_indices: Vec<usize> = cusp_specs.iter().map(|&(ci, _, _)| ci).collect();

    // Build probe grid: (a,b) probes for each filled cusp + adjoint combos
    let mut probe_points = Vec::new();
    for &ci in &filled_cusp_indices {
        probe_points.extend(ab_probe_grid(n_cusps, ci));
    }
    probe_points.extend(adjoint_probe_grid_multi(n_cusps, &filled_cusp_indices));

    let entries = build_entries(&state_nc, num_hard, &probe_points, q_order_half);

    if entries.is_empty() {
        return MultiCuspNcCompatResult {
            cusp_specs: cusp_specs.to_vec(),
            ab: None,
            ab_compat: None,
            collapsed_edges: vec![],
            multi_cusp_adjoint: None,
            weyl_symmetric: HashMap::new(),
            all_weyl_symmetric: false,
        };
    }

    // Compute (a, b) vectors
    let ab = compute_ab_vectors(&entries, num_hard);
    let ab_compat = ab.as_ref().map(|a| a.make_filling_compatible());
    let collapsed_edges = ab.as_ref().map(|a| a.incompat_edges()).unwrap_or_default();
    let collapsed_set: HashSet<usize> = collapsed_edges.iter().copied().collect();

    // Weyl symmetry check
    let weyl_symmetric = if let Some(ref ab_v) = ab {
        check_weyl_symmetry(&entries, num_hard, ab_v, Some(q_order_half))
    } else {
        entries
            .iter()
            .map(|e| ((e.m_ext.clone(), e.e_ext_x2.clone()), false))
            .collect()
    };
    let all_weyl_symmetric = !weyl_symmetric.is_empty() && weyl_symmetric.values().all(|&v| v);

    // Multi-cusp adjoint projection
    let multi_cusp_adjoint = ab_compat.as_ref().map(|ab_c| {
        let ce = if collapsed_edges.is_empty() {
            None
        } else {
            Some(&collapsed_set)
        };
        check_adjoint_projection_multi_cusp(
            &entries,
            num_hard,
            Some(ab_c),
            &filled_cusp_indices,
            ce,
        )
    });

    MultiCuspNcCompatResult {
        cusp_specs: cusp_specs.to_vec(),
        ab,
        ab_compat,
        collapsed_edges,
        multi_cusp_adjoint,
        weyl_symmetric,
        all_weyl_symmetric,
    }
}
