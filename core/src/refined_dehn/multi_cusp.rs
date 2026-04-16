//! Multi-cusp sequential refined Dehn filling.
//!
//! Ports v0.5's `compute_multi_cusp_filled_refined_index`,
//! `_apply_filling_kernel_to_intermediate`, and `_batched_first_filling`.
//!
//! Algorithm: fill cusps one at a time.  After filling cusp j, the
//! intermediate result maps `(m, e_x2)` of the next cusp to a
//! `MultiEtaSeries`.  The next filling applies its kernel to that
//! intermediate.

use hashbrown::HashMap;
use num_rational::Rational64;

use crate::index_refined::compute_refined_index;
use crate::refined_dehn::hj_cf::hj_continued_fraction;
use crate::refined_dehn::is_chain::{
    enumerate_is_full, enumerate_slope1_all, is_kernel_x2, QEtaSeries,
};
use crate::refined_dehn::multi_eta::{
    apply_k1_factor_multi, multi_add_inplace, FilledRefinedResult, MultiEtaSeries,
};
use crate::summation::EnumerationState;

// ── Types ──

/// Fraction-valued IS chain state: `(m, e_x2) → MultiEtaSeries`.
type FracISState = HashMap<(i64, i64), MultiEtaSeries>;

/// Intermediate from a previous filling: `(m_next, e_next_x2) → MultiEtaSeries`.
type Intermediate = HashMap<(i64, i64), MultiEtaSeries>;

/// Specification for filling one cusp in a multi-cusp filling.
#[derive(Debug, Clone)]
pub struct MultiCuspFillSpec {
    pub cusp_idx: usize,
    pub p: i64,
    pub q: i64,
    pub weyl_a: Option<Vec<Rational64>>,
    pub weyl_b: Option<Vec<Rational64>>,
    pub incompat_edges: Vec<usize>,
}

// ── Slope enumeration ──

/// Enumerate K(k,1) terms with c ∈ {0, 2} only.
///
/// Returns `(m, e_x2, c, phase)` with deduplication.
/// c=-2 is NOT emitted — it's handled by multiplicity=2 on c=2 terms.
/// This matches v0.5's `_enumerate_slope1_terms`.
fn enumerate_slope1_terms(k: i64, t_range: i64) -> Vec<(i64, i64, i64, i64)> {
    use crate::dehn::kernel_terms::particular_solution;
    let mut terms = Vec::new();
    let mut seen = hashbrown::HashSet::new();

    for &c in &[0i64, 2] {
        let (m_c, e_c_x2) = particular_solution(k, 1, c);
        let phase_c0 = m_c; // R=1, S=0 for Q=1

        let signs: &[i64] = if c == 0 { &[1] } else { &[1, -1] };
        for &sign in signs {
            for t_abs in 0..=t_range {
                if sign == -1 && t_abs == 0 {
                    continue;
                }
                let t = sign * t_abs;
                let m_t = m_c + t;
                let e_t_x2 = e_c_x2 - k * t;
                let phase_t = phase_c0 + t;

                let key = (m_t, e_t_x2);
                if seen.insert(key) {
                    terms.push((m_t, e_t_x2, c, phase_t));
                }
            }
        }
    }
    terms
}

// ── Fraction-valued IS chain helpers ──

/// Convolve a QEtaSeries (i64, IS kernel ×2) with a MultiEtaSeries (Rational64).
///
/// The IS kernel's η maps to the LAST dimension of the multi-key.
fn multi_convolve_is_frac(
    is_series: &QEtaSeries,
    multi_series: &MultiEtaSeries,
    qq_order: Option<i32>,
) -> MultiEtaSeries {
    let zero = Rational64::from_integer(0);
    let mut result = MultiEtaSeries::new();
    for (&(qq_is, eta_is), &c_is) in is_series {
        let c_is_r = Rational64::from_integer(c_is);
        for (multi_key, &c_multi) in multi_series {
            let new_qq = multi_key[0] + qq_is as i64;
            if let Some(qo) = qq_order {
                if new_qq > qo as i64 {
                    continue;
                }
            }
            let len = multi_key.len();
            let mut new_key = Vec::with_capacity(len);
            new_key.push(new_qq);
            new_key.extend_from_slice(&multi_key[1..len - 1]);
            new_key.push(multi_key[len - 1] + eta_is as i64);

            let val = c_is_r * c_multi;
            if val != zero {
                let e = result.entry(new_key).or_insert(zero);
                *e += val;
            }
        }
    }
    result.retain(|_, v| *v != zero);
    result
}

/// Single IS step on Fraction-valued state (multi-cusp intermediate path).
///
/// The ×2 from `is_kernel_x2` is absorbed as a ×½ factor after convolution.
fn apply_is_step_frac(
    state: &FracISState,
    k_current: i64,
    k_next: i64,
    qq_order: i32,
    eta_order: i32,
    m1_range: i64,
    is_last_step: bool,
) -> FracISState {
    let zero = Rational64::from_integer(0);
    let half = Rational64::new(1, 2);
    let mut new_state = FracISState::new();

    if is_last_step {
        let m1_terms = enumerate_slope1_all(k_next, m1_range);
        let m1_even: Vec<_> = m1_terms.iter().filter(|e| e.0 % 2 == 0).cloned().collect();
        let m1_odd: Vec<_> = m1_terms.iter().filter(|e| e.0 % 2 != 0).cloned().collect();

        for (&(m, e_x2), src_series) in state {
            if src_series.is_empty() {
                continue;
            }
            let e_in_x2 = -(e_x2 + k_current * m);
            let compatible = if e_in_x2 % 2 == 0 { &m1_even } else { &m1_odd };

            for &(m1, e1_x2, _, _) in compatible {
                let is_val = is_kernel_x2(m, e_in_x2, m1, e1_x2, qq_order, eta_order);
                if is_val.is_empty() {
                    continue;
                }
                let product = multi_convolve_is_frac(&is_val, src_series, Some(qq_order));
                if product.is_empty() {
                    continue;
                }
                let entry = new_state.entry((m1, e1_x2)).or_insert_with(MultiEtaSeries::new);
                multi_add_inplace(entry, &product);
            }
        }
    } else {
        let e1_range = qq_order as i64 + m1_range / 2;
        let full_terms = enumerate_is_full(m1_range, e1_range);

        let mut even_eint: Vec<(i64, i64)> = Vec::new();
        let mut even_ehalf: Vec<(i64, i64)> = Vec::new();
        let mut odd_eint: Vec<(i64, i64)> = Vec::new();
        let mut odd_ehalf: Vec<(i64, i64)> = Vec::new();
        for &(m1, e1_x2) in &full_terms {
            match (m1 % 2 == 0, e1_x2 % 2 == 0) {
                (true, true) => even_eint.push((m1, e1_x2)),
                (true, false) => even_ehalf.push((m1, e1_x2)),
                (false, true) => odd_eint.push((m1, e1_x2)),
                (false, false) => odd_ehalf.push((m1, e1_x2)),
            }
        }

        for (&(m, e_x2), src_series) in state {
            if src_series.is_empty() {
                continue;
            }
            let e_in_x2 = -(e_x2 + k_current * m);
            let m_is_even = m % 2 == 0;

            let compatible: &[(i64, i64)] = match (e_in_x2 % 2 == 0, m_is_even) {
                (true, true) => &even_eint,
                (true, false) => &even_ehalf,
                (false, true) => &odd_eint,
                (false, false) => &odd_ehalf,
            };

            for &(m1, e1_x2) in compatible {
                let is_val = is_kernel_x2(m, e_in_x2, m1, e1_x2, qq_order, eta_order);
                if is_val.is_empty() {
                    continue;
                }
                let product = multi_convolve_is_frac(&is_val, src_series, Some(qq_order));
                if product.is_empty() {
                    continue;
                }
                let entry = new_state.entry((m1, e1_x2)).or_insert_with(MultiEtaSeries::new);
                multi_add_inplace(entry, &product);
            }
        }
    }

    // Absorb the ×2 from is_kernel_x2 as ×½
    for series in new_state.values_mut() {
        for v in series.values_mut() {
            *v *= half;
        }
        series.retain(|_, v| *v != zero);
    }

    new_state
}

// ── Intermediate filling ──

/// Apply the refined Dehn filling kernel to precomputed intermediate series.
///
/// This is the multi-cusp analogue of `compute_filled_refined_index_chain`.
/// Instead of computing I^ref from NZ data, it looks up results from
/// `intermediate`, which maps `(m, e_x2)` charges to `MultiEtaSeries`.
fn apply_filling_kernel_to_intermediate(
    intermediate: &Intermediate,
    p: i64,
    q: i64,
    qq_order: i32,
    num_hard: usize,
    num_cusp_eta_in: usize,
) -> FilledRefinedResult {
    let eta_order = qq_order;
    let m1_range = 2 * qq_order as i64;

    let hj_ks = hj_continued_fraction(p, q);
    let ell = hj_ks.len();

    // ── ℓ=1: direct K(k1,1) application ──
    if ell == 1 {
        let k1 = hj_ks[0];
        let slope1_terms = enumerate_slope1_terms(k1, m1_range);

        let mut total = MultiEtaSeries::new();
        let mut n_terms = 0usize;

        for &(m_t, e_t_x2, c_val, phase_t) in &slope1_terms {
            let multi = match intermediate.get(&(m_t, e_t_x2)) {
                Some(s) if !s.is_empty() => s,
                _ => continue,
            };
            n_terms += 1;

            // Multiplicity: c=2 → 2 (absorbs c=-2), c=0 with t≠0 → 2 (antipodal)
            let mult = if c_val == 2 || (c_val == 0 && m_t != 0) { 2 } else { 1 };
            let contrib = apply_k1_factor_multi(multi, c_val, phase_t, mult, qq_order as i64);
            multi_add_inplace(&mut total, &contrib);
        }

        // ℓ=1 does NOT add a new cusp η
        return FilledRefinedResult {
            p,
            q,
            cusp_idx: -1,
            series: total,
            qq_order,
            eta_order: 0,
            hj_ks,
            n_kernel_terms: n_terms,
            num_hard,
            has_cusp_eta: num_cusp_eta_in > 0,
            num_cusp_eta: num_cusp_eta_in,
        };
    }

    // ── ℓ≥2: IS convolution chain ──
    let is_buffer = qq_order + 4;
    let qq_internal = qq_order + is_buffer;
    let m1_range_internal = m1_range.max(2 * qq_internal as i64);

    let m_scan = 2 * qq_internal as i64;
    let e_scan = qq_internal as i64;

    // Build state from intermediate — extend each series with cusp_eta=0
    let zero = Rational64::from_integer(0);
    let mut state = FracISState::new();
    let mut n_grid_terms = 0usize;

    for m_i in -m_scan..=m_scan {
        for e_half in (-2 * e_scan)..=(2 * e_scan) {
            let multi = match intermediate.get(&(m_i, e_half)) {
                Some(s) if !s.is_empty() => s,
                _ => continue,
            };
            n_grid_terms += 1;

            let mut extended = MultiEtaSeries::new();
            for (k, &v) in multi {
                if v == zero {
                    continue;
                }
                let mut nk = k.clone();
                nk.push(0);
                extended.insert(nk, v);
            }
            state.insert((m_i, e_half), extended);
        }
    }

    // IS convolution steps
    for step_i in 0..(ell - 1) {
        let k_current = hj_ks[step_i];
        let k_next = hj_ks[step_i + 1];
        state = apply_is_step_frac(
            &state,
            k_current,
            k_next,
            qq_internal,
            eta_order,
            m1_range_internal,
            step_i == ell - 2,
        );
    }

    // Final K(k_ℓ, 1) application
    let k_final = hj_ks[ell - 1];
    let final_terms = enumerate_slope1_all(k_final, m1_range_internal);
    let mut final_lookup: HashMap<(i64, i64), (i64, i64, i64)> = HashMap::new();
    for &(m1, e1_x2, c, phase) in &final_terms {
        final_lookup.entry((m1, e1_x2)).or_insert((c, phase, 1));
    }

    let mut total = MultiEtaSeries::new();
    for (&(m1, e1_x2), src_series) in &state {
        if src_series.is_empty() {
            continue;
        }
        if let Some(&(c_final, phase_final, mult_final)) = final_lookup.get(&(m1, e1_x2)) {
            let contrib = apply_k1_factor_multi(
                src_series,
                c_final,
                phase_final,
                mult_final,
                qq_internal as i64,
            );
            multi_add_inplace(&mut total, &contrib);
        }
    }

    // Diamond truncation — generalized for multiple cusp η's
    let num_cusp_eta_out = num_cusp_eta_in + 1;
    let cusp_start = 1 + num_hard;

    total.retain(|k, v| {
        if *v == zero {
            return false;
        }
        let cusp_eta_sum: i64 = (0..num_cusp_eta_out)
            .filter_map(|i| {
                let pos = cusp_start + i;
                if pos < k.len() {
                    Some(k[pos].abs())
                } else {
                    None
                }
            })
            .sum();
        k[0] + cusp_eta_sum <= qq_order as i64
    });

    FilledRefinedResult {
        p,
        q,
        cusp_idx: -1,
        series: total,
        qq_order,
        eta_order,
        hj_ks,
        n_kernel_terms: n_grid_terms,
        num_hard,
        has_cusp_eta: true,
        num_cusp_eta: num_cusp_eta_out,
    }
}

// ── Batched first filling ──

fn make_ext_multi(
    r: usize,
    cusp_idx: usize,
    m_fill: i64,
    e_fill_x2: i64,
    next_cusp_idx: usize,
    m_next: i64,
    e_next_x2: i64,
) -> (Vec<i64>, Vec<i64>) {
    let mut m_ext = vec![0i64; r];
    let mut e_ext = vec![0i64; r];
    m_ext[cusp_idx] = m_fill;
    e_ext[cusp_idx] = e_fill_x2;
    m_ext[next_cusp_idx] = m_next;
    e_ext[next_cusp_idx] = e_next_x2;
    (m_ext, e_ext)
}

/// Batched first filling: compute the first filling for all spectator charges.
fn batched_first_filling(
    state: &EnumerationState,
    first: &MultiCuspFillSpec,
    next_cusp_idx: usize,
    needed_me: &hashbrown::HashSet<(i64, i64)>,
    qq_order: i32,
    num_hard: usize,
) -> (Intermediate, usize) {
    let r = state.r;
    let cusp_idx = first.cusp_idx;

    let hj_ks = hj_continued_fraction(first.p, first.q);
    let ell = hj_ks.len();

    // Zero Weyl shifts for incompat edges
    let (wa, wb) = if let (Some(a), Some(b)) = (&first.weyl_a, &first.weyl_b) {
        let mut a = a.clone();
        let mut b = b.clone();
        let zero_r = Rational64::from_integer(0);
        for &j in &first.incompat_edges {
            if j < a.len() {
                a[j] = zero_r;
            }
            if j < b.len() {
                b[j] = zero_r;
            }
        }
        (Some(a), Some(b))
    } else {
        (None, None)
    };

    // For ℓ≥2, probe to discover active spectators
    let active_me: Vec<(i64, i64)> = if ell >= 2 {
        let is_buffer = qq_order + 4;
        let qq_internal = qq_order + is_buffer;
        let m_scan = 2 * qq_internal as i64;
        let e_scan = qq_internal as i64;

        let mut probed: hashbrown::HashSet<(i64, i64)> = hashbrown::HashSet::new();
        for m_next in -m_scan..=m_scan {
            for e_half_next in (-2 * e_scan)..=(2 * e_scan) {
                let (m_ext, e_ext) = make_ext_multi(
                    r, cusp_idx, 0, 0, next_cusp_idx, m_next, e_half_next,
                );
                let refined =
                    compute_refined_index(state, num_hard, &m_ext, &e_ext, qq_internal);
                if !refined.is_empty() {
                    probed.insert((m_next, e_half_next));
                }
            }
        }
        let active: hashbrown::HashSet<(i64, i64)> =
            needed_me.intersection(&probed).copied().collect();
        if active.is_empty() {
            probed.into_iter().collect()
        } else {
            active.into_iter().collect()
        }
    } else {
        needed_me.iter().copied().collect()
    };

    let mut intermediate = Intermediate::new();

    for &(m_next, e_next_x2) in &active_me {
        // Build m_other/e_other for this spectator
        let mut m_other = Vec::with_capacity(r - 1);
        let mut e_other_x2 = Vec::with_capacity(r - 1);
        for j in 0..r {
            if j == cusp_idx {
                continue;
            }
            if j == next_cusp_idx {
                m_other.push(m_next);
                e_other_x2.push(e_next_x2);
            } else {
                m_other.push(0);
                e_other_x2.push(0);
            }
        }

        // Delegate to single-cusp filling
        let result = if ell >= 2 {
            crate::refined_dehn::is_chain::compute_filled_refined_index_chain(
                state, num_hard, cusp_idx, first.p, first.q,
                &m_other, &e_other_x2, qq_order,
                &first.incompat_edges,
                wa.as_deref(),
                wb.as_deref(),
            )
        } else {
            crate::refined_dehn::unrefined_kernel_path::compute_unrefined_kernel_refined_index(
                state, num_hard, cusp_idx, first.p, first.q,
                &m_other, &e_other_x2, qq_order,
                &first.incompat_edges,
                wa.as_deref(),
                wb.as_deref(),
            )
        };

        if !result.series.is_empty() {
            let series = if !first.incompat_edges.is_empty() {
                let collapsed = result.collapse_eta_edges(&first.incompat_edges);
                collapsed.series
            } else {
                result.series
            };
            if !series.is_empty() {
                intermediate.insert((m_next, e_next_x2), series);
            }
        }
    }

    let num_cusp_eta = if ell >= 2 { 1 } else { 0 };
    (intermediate, num_cusp_eta)
}

/// Determine which (m, e_x2) charge pairs the next filling needs.
fn needed_spectator_charges(
    p: i64,
    q: i64,
    qq_order: i32,
) -> hashbrown::HashSet<(i64, i64)> {
    let hj_ks = hj_continued_fraction(p, q);
    let ell = hj_ks.len();

    if ell == 1 {
        let k1 = hj_ks[0];
        let m1_range = 2 * qq_order as i64;
        enumerate_slope1_all(k1, m1_range)
            .into_iter()
            .map(|(m, e_x2, _, _)| (m, e_x2))
            .collect()
    } else {
        let is_buffer = qq_order + 4;
        let qq_internal = qq_order + is_buffer;
        let m_scan = 2 * qq_internal as i64;
        let e_scan = qq_internal as i64;
        let mut result = hashbrown::HashSet::new();
        for m in -m_scan..=m_scan {
            for e_half in (-2 * e_scan)..=(2 * e_scan) {
                result.insert((m, e_half));
            }
        }
        result
    }
}

// ── Public API ──

/// Sequentially fill multiple cusps of a manifold.
///
/// Fills cusps one at a time.  After filling cusp j, the intermediate
/// result maps `(m, e_x2)` of the next cusp to `MultiEtaSeries`.
/// The next filling applies its kernel to that intermediate.
///
/// Currently supports `n_fills ≤ 2`.  For `n_fills == 1` delegates
/// directly to the single-cusp path.
pub fn compute_multi_cusp_filled_refined_index(
    state: &EnumerationState,
    num_hard: usize,
    fill_specs: &[MultiCuspFillSpec],
    qq_order: i32,
) -> FilledRefinedResult {
    let r = state.r;
    let n_fills = fill_specs.len();
    assert!(n_fills >= 1, "at least one fill spec required");

    // Single-cusp case: delegate directly
    if n_fills == 1 {
        let spec = &fill_specs[0];
        let m_other = vec![0i64; r - 1];
        let e_other_x2 = vec![0i64; r - 1];

        let hj_ks = hj_continued_fraction(spec.p, spec.q);
        let ell = hj_ks.len();

        let result = if ell >= 2 {
            crate::refined_dehn::is_chain::compute_filled_refined_index_chain(
                state, num_hard, spec.cusp_idx, spec.p, spec.q,
                &m_other, &e_other_x2, qq_order,
                &spec.incompat_edges,
                spec.weyl_a.as_deref(),
                spec.weyl_b.as_deref(),
            )
        } else {
            crate::refined_dehn::unrefined_kernel_path::compute_unrefined_kernel_refined_index(
                state, num_hard, spec.cusp_idx, spec.p, spec.q,
                &m_other, &e_other_x2, qq_order,
                &spec.incompat_edges,
                spec.weyl_a.as_deref(),
                spec.weyl_b.as_deref(),
            )
        };
        return if !spec.incompat_edges.is_empty() {
            result.collapse_eta_edges(&spec.incompat_edges)
        } else {
            result
        };
    }

    // Multi-cusp (n_fills == 2): sequential filling
    assert!(
        n_fills == 2,
        "n_fills > 2 not yet supported (v0.5 intermediate restructuring needed)"
    );

    let first = &fill_specs[0];
    let second = &fill_specs[1];
    let next_cusp = second.cusp_idx;

    // Step 1: determine what charges the second filling needs
    let needed_me = needed_spectator_charges(second.p, second.q, qq_order);

    // Step 2: batched first filling
    let (intermediate, num_cusp_eta_accum) =
        batched_first_filling(state, first, next_cusp, &needed_me, qq_order, num_hard);

    // Step 3: apply second filling kernel to intermediate
    let result = apply_filling_kernel_to_intermediate(
        &intermediate,
        second.p,
        second.q,
        qq_order,
        num_hard,
        num_cusp_eta_accum,
    );

    if !second.incompat_edges.is_empty() {
        result.collapse_eta_edges(&second.incompat_edges)
    } else {
        result
    }
}
