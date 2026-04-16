//! ℓ≥2 path: IS-kernel chain for refined Dehn filling.
//!
//! Ports v0.5's `_etilde_is`, `_is_kernel`, `_apply_is_step`, and
//! `compute_filled_refined_index` (ℓ≥2 branch).

use hashbrown::HashMap;
use num_rational::Rational64;

use crate::dehn::kernel_terms::particular_solution;
use crate::index_refined::{compute_refined_index, RefinedIndexResult};
use crate::kernel::tet_index_series;
use crate::refined_dehn::hj_cf::hj_continued_fraction;
use crate::refined_dehn::multi_eta::{
    apply_weyl_shift, collapse_iref_edges, FilledRefinedResult, MultiEtaSeries,
};
use crate::summation::EnumerationState;

/// IS kernel series: `(qq_power, eta_x2) → coefficient` (×2 scaled integers).
pub(crate) type QEtaSeries = HashMap<(i32, i32), i64>;

/// Integer-valued multi-eta series for the IS chain hot path.
type IntMultiSeries = HashMap<Vec<i64>, i64>;

/// IS chain state: `(m, e_x2) → IntMultiSeries`.
type ISState = HashMap<(i64, i64), IntMultiSeries>;

// ── Dense convolution helper ──

fn dense_convolve(a: &[i64], b: &[i64], budget: usize) -> Vec<i64> {
    let out_len = budget + 1;
    let mut out = vec![0i64; out_len];
    let a_len = a.len().min(out_len);
    let b_len = b.len().min(out_len);
    for i in 0..a_len {
        let ai = a[i];
        if ai == 0 {
            continue;
        }
        let jmax = (out_len - i).min(b_len);
        for j in 0..jmax {
            let bj = b[j];
            if bj != 0 {
                out[i + j] += ai * bj;
            }
        }
    }
    out
}

fn qseries_to_dense(qs: &crate::poly::QSeries, len: usize) -> Vec<i64> {
    let mut dense = vec![0i64; len];
    for (&p, &c) in qs {
        if p >= 0 && (p as usize) < len {
            dense[p as usize] = c;
        }
    }
    dense
}

fn is_all_zero(v: &[i64]) -> bool {
    v.iter().all(|&x| x == 0)
}

// ── Core IS kernel ──

/// Compute ẽI_S(m1, e1, m2, e2; η) — the 4-tetrahedron IS kernel.
///
/// All e arguments are in ×2 representation (2*actual_e).
/// Returns `(qq_power, eta_x2) → i64` where eta_x2 = 2*eta_exponent.
fn etilde_is(
    m1: i64,
    e1_x2: i64,
    m2: i64,
    e2_x2: i64,
    qq_order: i32,
    eta_order: i32,
) -> QEtaSeries {
    // Integrality check A: m_a1 = -e1 - m2/2 = -(e1_x2 + m2)/2
    if (e1_x2 + m2) % 2 != 0 {
        return QEtaSeries::new();
    }
    // Integrality check B: m_a3 = -e2 - m1/2 = -(e2_x2 + m1)/2
    if (e2_x2 + m1) % 2 != 0 {
        return QEtaSeries::new();
    }

    let m_a1 = -(e1_x2 + m2) / 2;
    let m_a2 = -m_a1;
    let m_a3 = -(e2_x2 + m1) / 2;
    let m_a4 = -m_a3;

    // Base e-arguments for tind3 and tind4
    let e3_base_x2 = e2_x2 + m1;
    let e4_base_x2 = e1_x2 - m2;
    if e3_base_x2 % 2 != 0 || e4_base_x2 % 2 != 0 {
        return QEtaSeries::new();
    }
    let e3_base = e3_base_x2 / 2;
    let e4_base = e4_base_x2 / 2;

    // Phase constant B = e1 + e2 + m1/2 - m2/2 = (e1_x2 + e2_x2 + m1 - m2)/2
    let b_x2 = e1_x2 + e2_x2 + m1 - m2;
    if b_x2 % 2 != 0 {
        return QEtaSeries::new();
    }
    let big_b = b_x2 / 2;

    // e-var parity
    let e_var_parity = ((m1 + m2) % 2 + 2) % 2; // ensure non-negative

    // Base e-arguments for tind1/tind2 after factoring out n_eta
    let e_arg1_base_x2 = e1_x2 + m1 - e_var_parity;
    let e_arg2_base_x2 = e2_x2 - m2 - e_var_parity;
    if e_arg1_base_x2 % 2 != 0 || e_arg2_base_x2 % 2 != 0 {
        return QEtaSeries::new();
    }
    let e_arg1_base = e_arg1_base_x2 / 2;
    let e_arg2_base = e_arg2_base_x2 / 2;

    let t_range = qq_order as i64 + big_b.abs() + 10;
    let qo = qq_order as usize;

    let mut result: QEtaSeries = QEtaSeries::new();

    for t in -t_range..=t_range {
        let e3 = (e3_base + t) as i32;
        let e4 = (e4_base + t) as i32;

        let s3 = tet_index_series(m_a3 as i32, e3, qq_order);
        if s3.is_empty() {
            continue;
        }
        let s4 = tet_index_series(m_a4 as i32, e4, qq_order);
        if s4.is_empty() {
            continue;
        }

        let d3 = qseries_to_dense(&s3, qo + 1);
        let d4 = qseries_to_dense(&s4, qo + 1);
        let a34 = dense_convolve(&d3, &d4, qo);
        if is_all_zero(&a34) {
            continue;
        }

        // Cache s12 by u = t - n_eta
        let mut s12_cache: HashMap<i64, Option<Vec<i64>>> = HashMap::new();

        for n_eta in -(eta_order as i64)..=(eta_order as i64) {
            let u = t - n_eta;

            let a12 = s12_cache.entry(u).or_insert_with(|| {
                let e_a1 = (u + e_arg1_base) as i32;
                let e_a2 = (u + e_arg2_base) as i32;
                let s1 = tet_index_series(m_a1 as i32, e_a1, qq_order);
                if s1.is_empty() {
                    return None;
                }
                let s2 = tet_index_series(m_a2 as i32, e_a2, qq_order);
                if s2.is_empty() {
                    return None;
                }
                let d1 = qseries_to_dense(&s1, qo + 1);
                let d2 = qseries_to_dense(&s2, qo + 1);
                let conv = dense_convolve(&d1, &d2, qo);
                if is_all_zero(&conv) {
                    None
                } else {
                    Some(conv)
                }
            });

            let a12 = match a12 {
                Some(v) => v,
                None => continue,
            };

            let e_var = 2 * n_eta + e_var_parity;
            let big_x = -e_var + big_b + 2 * t;

            // Convolve s12 * s34 — full product (both inputs ≤ qo+1 long,
            // so full convolution extends to 2*qo).  v0.5 uses np.convolve
            // (untruncated) then clips via the X-shift bounds below.
            let conv = dense_convolve(a12, &a34, 2 * qo);
            let sign: i64 = if big_x % 2 == 0 { 1 } else { -1 };

            let eta_x2_val = (2 * n_eta + e_var_parity) as i32;

            let src_lo = 0i64.max(-big_x) as usize;
            let src_hi = conv.len().min((qq_order as i64 + 1 - big_x).max(0) as usize);
            for src in src_lo..src_hi {
                let c = conv[src];
                if c == 0 {
                    continue;
                }
                let dst = src as i64 + big_x;
                if dst < 0 || dst > qq_order as i64 {
                    continue;
                }
                let key = (dst as i32, eta_x2_val);
                let e = result.entry(key).or_insert(0);
                if sign == 1 {
                    *e += c;
                } else {
                    *e -= c;
                }
            }
        }
    }

    result.retain(|_, v| *v != 0);
    result
}

/// Compute 2 × I_S(m1, e1, m2, e2; η) — the symplectic IS kernel, ×2 scaled.
///
/// All e arguments in ×2 representation.
pub(crate) fn is_kernel_x2(
    m1: i64,
    e1_x2: i64,
    m2: i64,
    e2_x2: i64,
    qq_order: i32,
    eta_order: i32,
) -> QEtaSeries {
    let ei_center = etilde_is(m1, e1_x2, m2, e2_x2, qq_order, eta_order);
    let ei_minus = etilde_is(m1, e1_x2 - 2, m2, e2_x2, qq_order, eta_order);
    let ei_plus = etilde_is(m1, e1_x2 + 2, m2, e2_x2, qq_order, eta_order);

    if ei_center.is_empty() && ei_minus.is_empty() && ei_plus.is_empty() {
        return QEtaSeries::new();
    }

    let sign_m1: i64 = if m1 % 2 == 0 { 1 } else { -1 };

    let mut result = QEtaSeries::new();

    // Term A+B: (-1)^m1 * (qq^m1 + qq^{-m1}) * etilde(e1)
    for (&(qq_p, eta), &c) in &ei_center {
        let scaled = c * sign_m1;
        if scaled == 0 {
            continue;
        }
        for shift in [m1 as i32, -(m1 as i32)] {
            let new_qq = qq_p + shift;
            if new_qq >= 0 && new_qq <= qq_order {
                let e = result.entry((new_qq, eta)).or_insert(0);
                *e += scaled;
            }
        }
    }

    // Terms C+D: -(-1)^m1 * etilde(e1±1)
    let neg_sign = -sign_m1;
    for src in [&ei_minus, &ei_plus] {
        for (&(qq_p, eta), &c) in src {
            let scaled = c * neg_sign;
            if scaled == 0 || qq_p < 0 || qq_p > qq_order {
                continue;
            }
            let e = result.entry((qq_p, eta)).or_insert(0);
            *e += scaled;
        }
    }

    result.retain(|_, v| *v != 0);
    result
}

// ── Enumeration ──

/// Enumerate ALL (m, e_x2, c, phase) for K(k, 1; m, e) — no symmetry shortcuts.
pub(crate) fn enumerate_slope1_all(k: i64, t_range: i64) -> Vec<(i64, i64, i64, i64)> {
    let mut terms = Vec::new();
    let mut seen = hashbrown::HashSet::new();

    for &c in &[0i64, 2, -2] {
        let (m_c, e_c_x2) = particular_solution(k, 1, c);
        let phase_c0 = m_c; // R=1, S=0 for Q=1

        for t in -t_range..=t_range {
            let m_t = m_c + t; // Q=1
            let e_t_x2 = e_c_x2 - k * t;
            let phase_t = phase_c0 + t; // = m_t

            let key = (m_t, e_t_x2);
            if seen.insert(key) {
                terms.push((m_t, e_t_x2, c, phase_t));
            }
        }
    }
    terms
}

/// Enumerate the full (½)ℤ² lattice for intermediate IS steps.
pub(crate) fn enumerate_is_full(m1_range: i64, e1_range: i64) -> Vec<(i64, i64)> {
    let mut terms = Vec::new();
    for m1 in -m1_range..=m1_range {
        for f1 in (-2 * e1_range)..=(2 * e1_range) {
            terms.push((m1, f1));
        }
    }
    terms
}

// ── Convolution ──

/// Convolve a QEtaSeries (IS kernel) with an IntMultiSeries.
///
/// The IS kernel's η maps to the LAST dimension of the multi-key.
fn multi_convolve_is_int(
    is_series: &QEtaSeries,
    multi_series: &IntMultiSeries,
    qq_order: Option<i32>,
) -> IntMultiSeries {
    let mut result = IntMultiSeries::new();
    for (&(qq_is, eta_is), &c_is) in is_series {
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

            let val = c_is * c_multi;
            let e = result.entry(new_key).or_insert(0);
            *e += val;
        }
    }
    result.retain(|_, v| *v != 0);
    result
}

/// Add two IntMultiSeries.
fn int_multi_add(a: &mut IntMultiSeries, b: &IntMultiSeries) {
    for (k, &v) in b {
        let e = a.entry(k.clone()).or_insert(0);
        *e += v;
        if *e == 0 {
            a.remove(k);
        }
    }
}

/// Apply K(k,1) factor to IntMultiSeries (int_mode: ×2 absorbed, no ½).
fn apply_k1_factor_int(
    series: &IntMultiSeries,
    c: i64,
    phase: i64,
    multiplicity: i64,
    qq_order: i32,
) -> IntMultiSeries {
    let sign: i64 = if phase % 2 == 0 { 1 } else { -1 };
    let mult = multiplicity;

    if c == 0 {
        let scalar = sign * mult;
        let mut result = IntMultiSeries::new();
        for (key, &c_val) in series {
            let scaled = c_val * scalar;
            if scaled == 0 {
                continue;
            }
            let qq_p = key[0];
            let rest = &key[1..];
            for new_qq in [qq_p + phase, qq_p - phase] {
                if new_qq <= qq_order as i64 {
                    let mut nk = Vec::with_capacity(key.len());
                    nk.push(new_qq);
                    nk.extend_from_slice(rest);
                    let e = result.entry(nk).or_insert(0);
                    *e += scaled;
                }
            }
        }
        result.retain(|_, v| *v != 0);
        result
    } else {
        let scalar = -sign * mult;
        if scalar == 0 {
            return IntMultiSeries::new();
        }
        series
            .iter()
            .filter_map(|(k, &v)| {
                let r = v * scalar;
                if r == 0 {
                    None
                } else {
                    Some((k.clone(), r))
                }
            })
            .collect()
    }
}

// ── Single IS step ──

fn apply_is_step(
    state: &ISState,
    k_current: i64,
    k_next: i64,
    qq_order: i32,
    eta_order: i32,
    m1_range: i64,
    is_last_step: bool,
) -> ISState {
    let mut new_state = ISState::new();

    if is_last_step {
        // Last step: restrict to K(k_next, 1) support
        let m1_terms = enumerate_slope1_all(k_next, m1_range);
        let m1_even: Vec<_> = m1_terms.iter().filter(|e| e.0 % 2 == 0).cloned().collect();
        let m1_odd: Vec<_> = m1_terms.iter().filter(|e| e.0 % 2 != 0).cloned().collect();

        for (&(m, e_x2), src_series) in state {
            if src_series.is_empty() {
                continue;
            }

            // e_in_x2 = 2*(-e - k*m/2) = -e_x2 - k*m = -(e_x2 + k*m)
            let e_in_x2 = -(e_x2 + k_current * m);
            let p = e_in_x2; // p = 2*e_in

            let compatible = if p % 2 == 0 { &m1_even } else { &m1_odd };

            for &(m1, e1_x2, _, _) in compatible {
                let is_val = is_kernel_x2(m, e_in_x2, m1, e1_x2, qq_order, eta_order);
                if is_val.is_empty() {
                    continue;
                }
                let product = multi_convolve_is_int(&is_val, src_series, Some(qq_order));
                if product.is_empty() {
                    continue;
                }
                let key = (m1, e1_x2);
                let entry = new_state.entry(key).or_insert_with(IntMultiSeries::new);
                int_multi_add(entry, &product);
            }
        }
    } else {
        // Intermediate step: full (½)ℤ² lattice with 4-way parity filter
        let e1_range = qq_order as i64 + m1_range / 2;
        let full_terms = enumerate_is_full(m1_range, e1_range);

        // 4-way partition: (m1_even/odd) × (e1_int/half)
        let mut even_eint: Vec<(i64, i64)> = Vec::new();
        let mut even_ehalf: Vec<(i64, i64)> = Vec::new();
        let mut odd_eint: Vec<(i64, i64)> = Vec::new();
        let mut odd_ehalf: Vec<(i64, i64)> = Vec::new();
        for &(m1, e1_x2) in &full_terms {
            let is_m1_even = m1 % 2 == 0;
            let is_e1_int = e1_x2 % 2 == 0;
            match (is_m1_even, is_e1_int) {
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
            let p = e_in_x2;
            let m_is_even = m % 2 == 0;

            let compatible: &[(i64, i64)] = match (p % 2 == 0, m_is_even) {
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
                let product = multi_convolve_is_int(&is_val, src_series, Some(qq_order));
                if product.is_empty() {
                    continue;
                }
                let key = (m1, e1_x2);
                let entry = new_state.entry(key).or_insert_with(IntMultiSeries::new);
                int_multi_add(entry, &product);
            }
        }
    }

    new_state
}

// ── Helpers ──

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

fn refined_to_int_multi(refined: &RefinedIndexResult, append_cusp_eta: bool) -> IntMultiSeries {
    let mut out = IntMultiSeries::new();
    for (k, &c) in refined {
        if c == 0 {
            continue;
        }
        let new_key = if append_cusp_eta {
            let mut nk = k.clone();
            nk.push(0);
            nk
        } else {
            k.clone()
        };
        out.insert(new_key, c);
    }
    out
}

// ── Public API ──

/// Compute refined Dehn-filled index for ℓ≥2 slopes via IS chain.
///
/// This is the direct-computation path (no pre-computed kernel cache).
pub fn compute_filled_refined_index_chain(
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
    let hj_ks = hj_continued_fraction(p, q);
    let ell = hj_ks.len();
    assert!(ell >= 2);

    let qq_order = q_order_half;
    let eta_order = qq_order;
    let is_buffer = qq_order + 4;
    let qq_internal = qq_order + is_buffer;
    let m1_range = (2 * qq_internal) as i64;

    // Zero Weyl shifts for incompat edges
    let (wa, wb) = match (weyl_a, weyl_b) {
        (Some(a), Some(b)) => {
            let mut a = a.to_vec();
            let mut b = b.to_vec();
            let zero = Rational64::from_integer(0);
            for &j in incompat_edges {
                if j < a.len() {
                    a[j] = zero;
                }
                if j < b.len() {
                    b[j] = zero;
                }
            }
            (Some(a), Some(b))
        }
        _ => (None, None),
    };

    // Grid scan for non-zero I^ref
    let m_scan = 2 * qq_internal as i64;
    let e_scan = qq_internal as i64;

    let mut is_state = ISState::new();
    let mut n_grid_terms: usize = 0;

    for m_i in -m_scan..=m_scan {
        for e_half in (-2 * e_scan)..=(2 * e_scan) {
            let e_i_x2 = e_half;
            let (m_ext, e_ext) = make_ext(cusp_idx, r, m_i, e_i_x2, m_other, e_other_x2);

            let refined = compute_refined_index(state, num_hard, &m_ext, &e_ext, qq_internal);
            if refined.is_empty() {
                continue;
            }

            let refined = if !incompat_edges.is_empty() {
                let r2 = collapse_iref_edges(&refined, incompat_edges);
                if r2.is_empty() {
                    continue;
                }
                r2
            } else {
                refined
            };

            let refined = if let (Some(wa), Some(wb)) = (&wa, &wb) {
                apply_weyl_shift(&refined, &m_ext, &e_ext, wa, wb, num_hard, cusp_idx)
            } else {
                refined
            };

            n_grid_terms += 1;

            let multi = refined_to_int_multi(&refined, true);
            let entry = is_state.entry((m_i, e_i_x2)).or_insert_with(IntMultiSeries::new);
            int_multi_add(entry, &multi);
        }
    }

    // Apply ℓ-1 IS convolution steps
    for step_i in 0..(ell - 1) {
        let k_current = hj_ks[step_i];
        let k_next = hj_ks[step_i + 1];
        is_state = apply_is_step(
            &is_state,
            k_current,
            k_next,
            qq_internal,
            eta_order,
            m1_range,
            step_i == ell - 2,
        );
    }

    // Apply final K(k_ℓ, 1) factor
    let k_final = hj_ks[ell - 1];
    let final_terms = enumerate_slope1_all(k_final, m1_range);
    let mut final_lookup: HashMap<(i64, i64), (i64, i64, i64)> = HashMap::new();
    for &(m1, e1_x2, c, phase) in &final_terms {
        final_lookup.entry((m1, e1_x2)).or_insert((c, phase, 1));
    }

    let mut total = IntMultiSeries::new();
    for (&(m1, e1_x2), src_series) in &is_state {
        if src_series.is_empty() {
            continue;
        }
        if let Some(&(c_final, phase_final, mult_final)) = final_lookup.get(&(m1, e1_x2)) {
            let contribution =
                apply_k1_factor_int(src_series, c_final, phase_final, mult_final, qq_internal);
            int_multi_add(&mut total, &contribution);
        }
    }

    // Convert ×2^ℓ scaled ints back to Rational64 + diamond truncation
    let lcd = 1i64 << ell;
    let mut out_series = MultiEtaSeries::new();
    let zero = Rational64::from_integer(0);
    for (k, &v) in &total {
        if v == 0 {
            continue;
        }
        let qq_p = k[0];
        let cusp_eta = k[k.len() - 1];
        if qq_p + cusp_eta.abs() <= qq_order as i64 {
            let frac_v = Rational64::new(v, lcd);
            if frac_v != zero {
                out_series.insert(k.clone(), frac_v);
            }
        }
    }

    FilledRefinedResult {
        p,
        q,
        cusp_idx: cusp_idx as i64,
        series: out_series,
        qq_order,
        eta_order,
        hj_ks,
        n_kernel_terms: n_grid_terms,
        num_hard,
        has_cusp_eta: true,
        num_cusp_eta: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_kernel_x2_at_origin() {
        // v0.5: _is_kernel(0, Fraction(0), 0, Fraction(0), 16, 6) ×2 scaled
        let result = is_kernel_x2(0, 0, 0, 0, 16, 6);
        // Expected from v0.5:
        //   eta=-6: [(6, 2)]
        //   eta=-4: [(4, 2)]
        //   eta=-2: [(2, 2), (6, -6)]
        //   eta=0: [(0, 2), (2, -4), (4, -4), (6, 8)]
        //   eta=2: [(2, 2), (6, -6)]
        //   eta=4: [(4, 2)]
        //   eta=6: [(6, 2)]
        assert_eq!(result.get(&(6, -6)), Some(&2), "eta=-6, qq=6");
        assert_eq!(result.get(&(4, -4)), Some(&2), "eta=-4, qq=4");
        assert_eq!(result.get(&(0, 0)), Some(&2), "eta=0, qq=0");
        assert_eq!(result.get(&(6, 6)), Some(&2), "eta=6, qq=6");

        // Print full result for debugging
        let mut items: Vec<_> = result.iter().collect();
        items.sort();
        for (&(qq, eta), &v) in &items {
            if qq <= 6 {
                eprintln!("  (qq={}, eta={}) = {}", qq, eta, v);
            }
        }
    }

    #[test]
    fn etilde_is_at_origin() {
        let result = etilde_is(0, 0, 0, 0, 16, 6);
        // v0.5: etilde_is(0, 0, 0, 0, 6, 6) had (0,0)=1, etc.
        // At qq=16,eta=6: should produce non-zero entries
        let mut items: Vec<_> = result.iter().collect();
        items.sort();
        eprintln!("etilde_is(0,0,0,0,16,6): {} entries", items.len());
        for (&(qq, eta), &v) in &items {
            if qq <= 6 {
                eprintln!("  (qq={}, eta={}) = {}", qq, eta, v);
            }
        }
    }
}
