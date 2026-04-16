//! ℓ=1 path: apply unrefined kernel K(P,Q) to the refined index I^ref.
//!
//! Ports v0.5's `compute_unrefined_kernel_refined_index`.

use num_rational::Rational64;

use crate::dehn::kernel_terms::{enumerate_kernel_terms, find_rs};
use crate::index_refined::compute_refined_index;
use crate::refined_dehn::multi_eta::{
    apply_k1_factor_multi, apply_weyl_shift, collapse_iref_edges, multi_add_inplace,
    refined_to_multi, MultiEtaSeries,
};
use crate::summation::EnumerationState;

#[derive(Debug, Clone)]
pub struct FilledRefinedResult {
    pub p: i64,
    pub q: i64,
    pub cusp_idx: usize,
    pub series: MultiEtaSeries,
    pub qq_order: i32,
    pub n_kernel_terms: usize,
    pub num_hard: usize,
    pub has_cusp_eta: bool,
}

fn make_ext(
    cusp_idx: usize,
    r: usize,
    m_i: i64,
    e_i_x2: i64,
    m_other: &[i64],
    e_other_x2: &[i64],
) -> (Vec<i64>, Vec<i64>) {
    let mut m_ext = Vec::with_capacity(r);
    let mut e_ext = Vec::with_capacity(r);
    let mut o = 0;
    for k in 0..r {
        if k == cusp_idx {
            m_ext.push(m_i);
            e_ext.push(e_i_x2);
        } else {
            m_ext.push(m_other[o]);
            e_ext.push(e_other_x2[o]);
            o += 1;
        }
    }
    (m_ext, e_ext)
}

/// Apply unrefined kernel K(P,Q) to I^ref, preserving hard-edge η.
pub fn compute_unrefined_kernel_refined_index(
    state: &EnumerationState,
    num_hard: usize,
    cusp_idx: usize,
    p: i64,
    q: i64,
    m_other: &[i64],
    e_other_x2: &[i64],
    q_order_half: i32,
    incompat_edges: &[usize],
    weyl_a: Option<&[Rational64]>,
    weyl_b: Option<&[Rational64]>,
) -> FilledRefinedResult {
    let r = state.r;
    let (r_val, s_val) = find_rs(p, q);

    // Zero Weyl shifts for incompat edges
    let (wa, wb) = match (weyl_a, weyl_b) {
        (Some(a), Some(b)) => {
            let mut a = a.to_vec();
            let mut b = b.to_vec();
            for &j in incompat_edges {
                if j < a.len() {
                    a[j] = Rational64::from_integer(0);
                }
                if j < b.len() {
                    b[j] = Rational64::from_integer(0);
                }
            }
            (Some(a), Some(b))
        }
        _ => (None, None),
    };

    let kernel_terms = enumerate_kernel_terms(
        state,
        p,
        q,
        r_val,
        s_val,
        cusp_idx,
        m_other,
        e_other_x2,
        q_order_half,
    );

    let mut total = MultiEtaSeries::new();
    let mut n_terms = 0;

    for kt in &kernel_terms {
        let (m_ext, e_ext) = make_ext(cusp_idx, r, kt.m, kt.e_x2, m_other, e_other_x2);
        let extra_q = if kt.c == 0 { kt.phase.abs() as i32 } else { 0 };
        let mut refined = compute_refined_index(
            state,
            num_hard,
            &m_ext,
            &e_ext,
            q_order_half + extra_q,
        );
        if refined.is_empty() {
            continue;
        }
        if !incompat_edges.is_empty() {
            refined = collapse_iref_edges(&refined, incompat_edges);
            if refined.is_empty() {
                continue;
            }
        }
        if let (Some(wa), Some(wb)) = (&wa, &wb) {
            refined = apply_weyl_shift(
                &refined, &m_ext, &e_ext, wa, wb, num_hard, cusp_idx,
            );
        }
        n_terms += 1;
        let multi = refined_to_multi(&refined, false);
        let contrib = apply_k1_factor_multi(
            &multi,
            kt.c,
            kt.phase,
            kt.multiplicity,
            q_order_half as i64,
        );
        multi_add_inplace(&mut total, &contrib);
    }

    FilledRefinedResult {
        p,
        q,
        cusp_idx,
        series: total,
        qq_order: q_order_half,
        n_kernel_terms: n_terms,
        num_hard,
        has_cusp_eta: false,
    }
}
