//! Enumeration of summation terms for the 3D index (v0.5 `index_3d.py:850`).
//!
//! Public API:
//!   - [`g_nz_inv_scaled`] — symplectic inverse of `g_NZ` with integer scale.
//!   - [`EnumerationState::build`] — precompute per-NZ state.
//!   - [`enumerate_summation_terms`] — enumerate `(e_int, phase_exp, tet_args)`.

use crate::census::NzData;
use crate::kernel::tet_degree_x2;

/// Compute `(S, S · g_NZ⁻¹)` as `(i64, Vec<i64>)` of shape `(2n, 2n)` row-major.
///
/// Port of v0.5's [`NeumannZagierData.g_NZ_inv_scaled`] using the symplectic
/// identity `g⁻¹ = [[Dᵀ, -Bᵀ], [-Cᵀ, Aᵀ]]`.
pub fn g_nz_inv_scaled(nz: &NzData) -> (i64, Vec<i64>) {
    let n = nz.n;
    let size = 2 * n;
    let g2 = &nz.g_nz_x2;
    let idx = |i: usize, j: usize| i * size + j;

    let mut twice_inv = vec![0i64; size * size];
    for i in 0..n {
        for j in 0..n {
            twice_inv[idx(i, j)] = g2[idx(n + j, n + i)];
            twice_inv[idx(i, n + j)] = -g2[idx(j, n + i)];
            twice_inv[idx(n + i, j)] = -g2[idx(n + j, i)];
            twice_inv[idx(n + i, n + j)] = g2[idx(j, i)];
        }
    }

    let any_odd = twice_inv.iter().any(|&v| v & 1 != 0);
    if any_odd {
        (2, twice_inv)
    } else {
        (1, twice_inv.into_iter().map(|v| v / 2).collect())
    }
}

/// Pre-computed state for [`enumerate_summation_terms`].
#[derive(Debug, Clone)]
pub struct EnumerationState {
    pub n: usize,
    pub r: usize,
    pub n_int: usize,
    /// LCD of `g_NZ⁻¹` entries — usually 1 or 2.
    pub s: i64,
    /// `S · g_NZ⁻¹`, shape `(2n, 2n)` row-major.
    pub g_inv_xs: Vec<i64>,
    /// Integer-valued internal-edge columns: `g_NZ⁻¹[:, n+r..2n]`,
    /// shape `(2n, n_int)` row-major. `g_inv_xs[:, n+r..2n] / S`.
    pub int_cols: Vec<i64>,
    /// `ν_x[r..n]` (easy-edge shift), length `n_int`.
    pub nu_x_int: Vec<i64>,
    /// Full `ν_x`, length `n`.
    pub nu_x_full: Vec<i64>,
    /// `2·ν_p`, length `n` (exact integer).
    pub nu_p_x2: Vec<i64>,
    /// `2^{n_int}` patterns, each of length `n_int`.
    pub patterns: Vec<Vec<u8>>,
    /// Per-pattern `g_inv_xs @ kappa_delta_x2`; each length `2n`. Scale `2S`.
    pub delta_contrib_x2s: Vec<Vec<i64>>,
    /// Per-pattern `-delta · nu_x_int` (equals `2·phase_delta`).
    pub delta_phase_x2: Vec<i64>,
    /// `g_inv_xs[:, 0..r]` row-major `(2n, r)`.
    pub cusp_m_cols_xs: Vec<i64>,
    /// `g_inv_xs[:, n..n+r]` row-major `(2n, r)`.
    pub cusp_e_cols_xs: Vec<i64>,
}

impl EnumerationState {
    pub fn build(nz: &NzData) -> Self {
        let n = nz.n;
        let r = nz.r;
        let n_int = n - r;
        let size = 2 * n;

        let (s, g_inv_xs) = g_nz_inv_scaled(nz);

        // Internal-edge columns, post-divide by S (must be exact).
        let mut int_cols = vec![0i64; size * n_int];
        for i in 0..size {
            for j in 0..n_int {
                let v = g_inv_xs[i * size + (n + r + j)];
                debug_assert!(v % s == 0, "internal-edge column not divisible by S");
                int_cols[i * n_int + j] = v / s;
            }
        }

        let nu_x_int = nz.nu_x[r..n].to_vec();
        let nu_x_full = nz.nu_x.clone();
        let nu_p_x2 = nz.nu_p_x2.clone();

        // All 2^{n_int} patterns.
        let num_patterns = 1usize << n_int;
        let mut patterns = Vec::with_capacity(num_patterns);
        for bits in 0..num_patterns {
            let mut p = vec![0u8; n_int];
            for j in 0..n_int {
                p[j] = ((bits >> j) & 1) as u8;
            }
            patterns.push(p);
        }

        // Per-pattern pre-computation.
        let mut delta_contrib_x2s: Vec<Vec<i64>> = Vec::with_capacity(num_patterns);
        let mut delta_phase_x2: Vec<i64> = Vec::with_capacity(num_patterns);
        for delta in &patterns {
            // kappa_delta_x2 nonzero only at positions n+r..2n (= delta).
            // contrib[i] = sum_j g_inv_xs[i, n+r+j] * delta[j]
            let mut contrib = vec![0i64; size];
            for i in 0..size {
                let mut s_i = 0i64;
                for j in 0..n_int {
                    let d = delta[j] as i64;
                    if d != 0 {
                        s_i += g_inv_xs[i * size + (n + r + j)] * d;
                    }
                }
                contrib[i] = s_i;
            }
            delta_contrib_x2s.push(contrib);

            let mut phase = 0i64;
            for j in 0..n_int {
                phase -= (delta[j] as i64) * nu_x_int[j];
            }
            delta_phase_x2.push(phase);
        }

        // cusp_m_cols_xs: g_inv_xs[:, 0..r], row-major (size, r)
        // cusp_e_cols_xs: g_inv_xs[:, n..n+r]
        let mut cusp_m_cols_xs = vec![0i64; size * r];
        let mut cusp_e_cols_xs = vec![0i64; size * r];
        for i in 0..size {
            for j in 0..r {
                cusp_m_cols_xs[i * r + j] = g_inv_xs[i * size + j];
                cusp_e_cols_xs[i * r + j] = g_inv_xs[i * size + (n + j)];
            }
        }

        EnumerationState {
            n, r, n_int,
            s,
            g_inv_xs,
            int_cols,
            nu_x_int,
            nu_x_full,
            nu_p_x2,
            patterns,
            delta_contrib_x2s,
            delta_phase_x2,
            cusp_m_cols_xs,
            cusp_e_cols_xs,
        }
    }
}

/// Fast O(1) pre-filter: does I(m_ext, e_ext) have *any* valid half-integer
/// pattern? Returns `false` if all patterns fail integrality, meaning the
/// index is structurally zero for these charges.
pub fn has_valid_summation_terms(
    state: &EnumerationState,
    m_ext: &[i64],
    e_ext_x2: &[i64],
) -> bool {
    let size = 2 * state.n;
    let s2 = 2 * state.s;
    // me_contrib[i] = cusp_m_cols_xs[i,:] · (2·m) + cusp_e_cols_xs[i,:] · e_x2
    let mut me_contrib = vec![0i64; size];
    for i in 0..size {
        let mut v = 0i64;
        for j in 0..state.r {
            v += state.cusp_m_cols_xs[i * state.r + j] * (2 * m_ext[j]);
            v += state.cusp_e_cols_xs[i * state.r + j] * e_ext_x2[j];
        }
        me_contrib[i] = v;
    }
    for pat_idx in 0..state.patterns.len() {
        let dc = &state.delta_contrib_x2s[pat_idx];
        let all_int = (0..size).all(|i| (me_contrib[i] + dc[i]) % s2 == 0);
        if all_int {
            return true;
        }
    }
    false
}

/// A single summation term produced by [`enumerate_summation_terms`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummationTerm {
    /// `2·e_int` as integer (handles half-integer deltas). Length `n_int`.
    pub e_int_x2: Vec<i64>,
    /// Integer exponent of `(−q^{1/2})`.
    pub phase_exp: i64,
    /// `(m_a, e_a)` for a = 0..n. Integer.
    pub tet_args: Vec<(i64, i64)>,
    /// `2 · Σ_a tet_degree(m_a, e_a)`.
    pub min_degree_x2: i64,
}

/// Enumerate all contributing `(e_int, phase_exp, tet_args)` triples for `I(m_ext, e_ext)`.
///
/// `e_ext_x2` is `2·e_ext` (so half-integers from Dehn filling fit as integers).
/// `q_order_half` is the cutoff in `q^{1/2}` (v0.5's `q_order_half`).
pub fn enumerate_summation_terms(
    state: &EnumerationState,
    m_ext: &[i64],
    e_ext_x2: &[i64],
    q_order_half: i64,
) -> Vec<SummationTerm> {
    assert_eq!(m_ext.len(), state.r);
    assert_eq!(e_ext_x2.len(), state.r);

    let n = state.n;
    let r = state.r;
    let n_int = state.n_int;
    let size = 2 * n;
    let s2 = 2 * state.s;

    // me_contrib_x2s[i] = sum_k cusp_m_cols_xs[i,k]*(2*m_ext[k]) + cusp_e_cols_xs[i,k]*e_ext_x2[k]
    let mut me_contrib_x2s = vec![0i64; size];
    for i in 0..size {
        let mut acc = 0i64;
        for k in 0..r {
            acc += state.cusp_m_cols_xs[i * r + k] * (2 * m_ext[k]);
            acc += state.cusp_e_cols_xs[i * r + k] * e_ext_x2[k];
        }
        me_contrib_x2s[i] = acc;
    }

    // phase_me_x2 = m_ext · nu_p_x2[:r] - e_ext_x2 · nu_x_full[:r]
    let mut phase_me_x2 = 0i64;
    for k in 0..r {
        phase_me_x2 += m_ext[k] * state.nu_p_x2[k];
        phase_me_x2 -= e_ext_x2[k] * state.nu_x_full[k];
    }

    let mut terms: Vec<SummationTerm> = Vec::new();

    for (pat_idx, delta) in state.patterns.iter().enumerate() {
        let mut base_args_x2s = vec![0i64; size];
        for i in 0..size {
            base_args_x2s[i] = me_contrib_x2s[i] + state.delta_contrib_x2s[pat_idx][i];
        }
        // Integrality check.
        if base_args_x2s.iter().any(|v| v.rem_euclid(s2) != 0) {
            continue;
        }
        let base_args: Vec<i64> = base_args_x2s.iter().map(|v| v / s2).collect();

        let phase_base_x2 = phase_me_x2 + state.delta_phase_x2[pat_idx];
        if phase_base_x2 & 1 != 0 {
            // Half-integer phase ⇒ q^{1/4} sector, skip.
            continue;
        }
        let phase_base = phase_base_x2 / 2;

        let candidates = exact_e0_candidates(
            &base_args,
            &state.int_cols,
            &state.nu_x_int,
            phase_base_x2,
            q_order_half,
            n,
            n_int,
        );

        for e0 in candidates {
            // args[i] = base_args[i] + Σ_j int_cols[i,j] * e0[j]
            let mut args = base_args.clone();
            for i in 0..size {
                let mut acc = 0i64;
                for j in 0..n_int {
                    acc += state.int_cols[i * n_int + j] * e0[j];
                }
                args[i] += acc;
            }

            let mut min_deg_x2 = 0i64;
            for a in 0..n {
                min_deg_x2 += tet_degree_x2(args[a] as i32, args[n + a] as i32) as i64;
            }
            let mut phase_exp = phase_base;
            for j in 0..n_int {
                phase_exp -= state.nu_x_int[j] * e0[j];
            }
            if min_deg_x2 + 2 * phase_exp > 2 * q_order_half {
                continue;
            }
            let tet_args: Vec<(i64, i64)> = (0..n).map(|a| (args[a], args[n + a])).collect();

            // Build e_int_x2 = 2·e0 + delta
            let mut e_int_x2 = vec![0i64; n_int];
            for j in 0..n_int {
                e_int_x2[j] = 2 * e0[j] + delta[j] as i64;
            }

            terms.push(SummationTerm {
                e_int_x2,
                phase_exp,
                tet_args,
                min_degree_x2: min_deg_x2,
            });
        }
    }

    terms
}

// ---------------------------------------------------------------------------
// Private: exact e0 candidate enumeration.
// ---------------------------------------------------------------------------

/// Evaluate `2·F(e0)` where F is the effective degree. Integer arithmetic.
#[inline]
fn f_x2(
    base_args: &[i64],
    int_cols: &[i64],
    nu_x_int: &[i64],
    phase_base_x2: i64,
    e0: &[i64],
    n: usize,
    n_int: usize,
) -> i64 {
    // args = base_args + int_cols @ e0
    let mut deg_x2 = 0i64;
    for a in 0..n {
        let mut m_a = base_args[a];
        let mut e_a = base_args[n + a];
        for j in 0..n_int {
            m_a += int_cols[a * n_int + j] * e0[j];
            e_a += int_cols[(n + a) * n_int + j] * e0[j];
        }
        deg_x2 += tet_degree_x2(m_a as i32, e_a as i32) as i64;
    }
    let mut shift = 0i64;
    for j in 0..n_int {
        shift += nu_x_int[j] * e0[j];
    }
    deg_x2 + phase_base_x2 - 2 * shift
}

/// Port of v0.5's `_axis_scan_bound`.
fn axis_scan_bound(mut eval: impl FnMut(i64) -> i64, q_bound_x2: i64) -> i64 {
    let mut max_abs = 0i64;
    let hard_max = 4 * q_bound_x2.max(0) + 50;

    for sign in [1i64, -1] {
        let mut prev_val = eval(0);
        let mut consec_over = 0i64;
        let mut ever_hit = false;
        for t in 1..=hard_max {
            let val = eval(sign * t);
            if val <= q_bound_x2 {
                if t > max_abs {
                    max_abs = t;
                }
                consec_over = 0;
                ever_hit = true;
            } else {
                consec_over += 1;
                if ever_hit {
                    if val >= prev_val && consec_over >= 2 {
                        break;
                    }
                } else {
                    if val > prev_val {
                        break;
                    }
                    let step = prev_val - val;
                    if step > 0 {
                        let steps_to_q = (val - q_bound_x2 + step - 1) / step;
                        if steps_to_q > hard_max - t {
                            break;
                        }
                    }
                }
            }
            prev_val = val;
        }
    }
    max_abs
}

/// Port of v0.5's `_proj_min_fixed`: min over `e0` with `e0[fixed_j] = tj`.
fn proj_min_fixed(
    base_args: &[i64],
    int_cols: &[i64],
    nu_x_int: &[i64],
    phase_base_x2: i64,
    fixed_j: usize,
    tj: i64,
    num_easy: usize,
    q_bound_x2: i64,
    n: usize,
) -> i64 {
    let free_dims: Vec<usize> = (0..num_easy).filter(|&k| k != fixed_j).collect();
    let d = free_dims.len();

    let mut e0_start = vec![0i64; num_easy];
    e0_start[fixed_j] = tj;

    let eval_full = |e0: &[i64]| -> i64 {
        f_x2(base_args, int_cols, nu_x_int, phase_base_x2, e0, n, num_easy)
    };

    if d == 0 {
        return eval_full(&e0_start);
    }

    // Direction set.
    let free_dir_tuples: Vec<Vec<i64>> = if d <= 4 {
        let total = 3i64.pow(d as u32) as usize;
        let mut out = Vec::with_capacity(total - 1);
        for idx in 0..total {
            let mut v = vec![0i64; d];
            let mut x = idx;
            let mut any = false;
            for i in 0..d {
                let digit = (x % 3) as i64 - 1;
                v[i] = digit;
                if digit != 0 {
                    any = true;
                }
                x /= 3;
            }
            if any {
                out.push(v);
            }
        }
        out
    } else {
        let mut out = Vec::new();
        for i in 0..d {
            for &s in &[-1i64, 1] {
                let mut v = vec![0i64; d];
                v[i] = s;
                out.push(v);
            }
        }
        for i in 0..d {
            for j in (i + 1)..d {
                for &si in &[-1i64, 1] {
                    for &sj in &[-1i64, 1] {
                        let mut v = vec![0i64; d];
                        v[i] = si;
                        v[j] = sj;
                        out.push(v);
                    }
                }
            }
        }
        out
    };

    let dir_vecs: Vec<Vec<i64>> = free_dir_tuples.iter().map(|fd| {
        let mut v = vec![0i64; num_easy];
        for (i, &k) in free_dims.iter().enumerate() {
            v[k] = fd[i];
        }
        v
    }).collect();

    let scan_max = 4 * q_bound_x2.max(0) + 50;
    let mut best_val = eval_full(&e0_start);
    let mut best_e0 = e0_start.clone();
    let max_outer = 2 * d as i64 + 4;

    for _ in 0..max_outer {
        let mut changed = false;
        for v_full in &dir_vecs {
            let start_e0 = best_e0.clone();
            let mut prev = best_val;
            let mut consec = 0i64;
            for s in 1..=scan_max {
                let trial: Vec<i64> = start_e0.iter().zip(v_full.iter())
                    .map(|(a, b)| a + s * b).collect();
                let val = eval_full(&trial);
                if val < best_val {
                    best_val = val;
                    best_e0 = trial.clone();
                    changed = true;
                    consec = 0;
                } else if val > best_val && val >= prev {
                    consec += 1;
                    if consec >= 2 {
                        break;
                    }
                } else {
                    consec = 0;
                }
                prev = val;
            }
        }
        if !changed {
            break;
        }
    }
    best_val
}

/// Port of v0.5's `_exact_e0_candidates`.
fn exact_e0_candidates(
    base_args: &[i64],
    int_cols: &[i64],
    nu_x_int: &[i64],
    phase_base_x2: i64,
    q_bound: i64,
    n: usize,
    num_easy: usize,
) -> Vec<Vec<i64>> {
    if num_easy == 0 {
        return vec![vec![]];
    }
    let q_bound_x2 = 2 * q_bound;

    // Step 1: per-axis R[j].
    let mut r = vec![0i64; num_easy];
    for j in 0..num_easy {
        let eval_g = |t: i64| -> i64 {
            proj_min_fixed(
                base_args, int_cols, nu_x_int, phase_base_x2,
                j, t, num_easy, q_bound_x2, n,
            )
        };
        r[j] = axis_scan_bound(eval_g, q_bound_x2);
    }

    // Clamp total box volume.
    const MAX_BOX: i64 = 50_000_000;
    let mut box_size: i64 = 1;
    for &rj in &r {
        box_size = box_size.saturating_mul(2 * rj + 1);
    }
    if box_size > MAX_BOX {
        let cap = ((MAX_BOX as f64).powf(1.0 / num_easy as f64) as i64 - 1) / 2;
        let cap = cap.max(1);
        for rj in r.iter_mut() {
            if *rj > cap {
                *rj = cap;
            }
        }
    }

    // Step 2: enumerate bounding box, filter by F ≤ q_bound_x2.
    let sizes: Vec<i64> = r.iter().map(|&rj| 2 * rj + 1).collect();
    let mut total: i64 = 1;
    for &s in &sizes {
        total = total.saturating_mul(s);
    }

    let mut out: Vec<Vec<i64>> = Vec::new();
    let mut e0 = vec![0i64; num_easy];
    for flat in 0..total {
        let mut rem = flat;
        for j in (0..num_easy).rev() {
            let sj = sizes[j];
            let idx = rem % sj;
            rem /= sj;
            e0[j] = idx - r[j];
        }
        let val = f_x2(base_args, int_cols, nu_x_int, phase_base_x2, &e0, n, num_easy);
        if val <= q_bound_x2 {
            out.push(e0.clone());
        }
    }
    out
}
