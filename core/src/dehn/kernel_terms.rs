//! Dehn filling kernel term enumeration.
//!
//! Ports v0.5's `find_rs`, `_particular_solution`, `KernelTerm`, and
//! `enumerate_kernel_terms` from `dehn_filling.py`.

use num_integer::Integer;

use crate::summation::{enumerate_summation_terms, has_valid_summation_terms, EnumerationState};

/// One (m, e) summand in the Dehn filling kernel for slope P/Q.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelTerm {
    pub m: i64,
    /// `2·e` (half-integer-safe).
    pub e_x2: i64,
    /// ∈ {0, 2}: P·m + 2Q·e mod {−2, 0, 2}.
    pub c: i64,
    /// R·m + 2S·e (integer).
    pub phase: i64,
    /// 1 or 2 (antipodal symmetry).
    pub multiplicity: i64,
}

/// Extended GCD: returns `(g, x, y)` with `a·x + b·y = g = gcd(|a|, |b|)`.
fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (a, 1, 0);
    }
    let (g, x1, y1) = ext_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

/// Find `(R, S)` with `R·Q − P·S = 1`.
pub fn find_rs(p: i64, q: i64) -> (i64, i64) {
    assert_eq!(p.abs().gcd(&q.abs()), 1, "slope must be primitive");
    let (_, mut x, mut y) = ext_gcd(q.abs(), p.abs());
    if q < 0 {
        x = -x;
    }
    if p < 0 {
        y = -y;
    }
    (x, -y) // (R, S)
}

/// Particular solution `(m0, 2·e0)` to `P·m0 + 2Q·e0 = c`.
fn particular_solution(p: i64, q: i64, c: i64) -> (i64, i64) {
    let (_, x, y) = ext_gcd(p, q);
    // P·(c·x) + Q·(c·y) = c  ⇒  m0 = c·x, f0 = c·y, e0 = f0/2.
    // Store 2·e0 = c·y to stay integer.
    (c * x, c * y)
}

/// Build full `(m_ext, e_ext_x2)` with cusp `cusp_idx` set to `(m_i, e_i_x2)`.
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

/// Enumerate kernel terms for slope P/Q at cusp `cusp_idx`.
///
/// Returns deduplicated `KernelTerm` list with multiplicity for antipodal symmetry.
pub fn enumerate_kernel_terms(
    state: &EnumerationState,
    p: i64,
    q: i64,
    r_val: i64,
    s_val: i64,
    cusp_idx: usize,
    m_other: &[i64],
    e_other_x2: &[i64],
    q_order_half: i32,
) -> Vec<KernelTerm> {
    let r = state.r;
    let consec_empty_stop: usize = 2;

    let mut seen = hashbrown::HashSet::new();
    let mut result: Vec<KernelTerm> = Vec::new();

    for &c in &[0i64, 2] {
        let (m_c, e_c_x2) = particular_solution(p, q, c);
        // phase_c0 = R·m_c + 2S·e_c = R·m_c + S·(2·e_c)
        let phase_c0 = r_val * m_c + s_val * e_c_x2;

        let signs: &[i64] = if c == 0 { &[1] } else { &[1, -1] };
        for &sign in signs {
            let mut consec_degree_miss: usize = 0;
            let mut consec_integ_miss: usize = 0;
            let integ_miss_cap = q_order_half as usize;
            let mut t_abs: i64 = 0;
            let t_limit = 4 * q_order_half as i64 + 50;

            while t_abs <= t_limit {
                let t = sign * t_abs;
                if sign == -1 && t_abs == 0 {
                    t_abs += 1;
                    continue;
                }

                let m_t = m_c + q * t;
                let e_t_x2 = e_c_x2 - p * t;
                let phase_t = phase_c0 + t;

                let adjusted_q = q_order_half as i64 + if c == 0 { phase_t.abs() } else { 0 };

                let (m_ext, e_ext) =
                    make_ext(cusp_idx, r, m_t, e_t_x2, m_other, e_other_x2);

                if !has_valid_summation_terms(state, &m_ext, &e_ext) {
                    consec_integ_miss += 1;
                    if consec_integ_miss >= integ_miss_cap {
                        break;
                    }
                    t_abs += 1;
                    continue;
                }

                let terms = enumerate_summation_terms(state, &m_ext, &e_ext, adjusted_q);
                if terms.is_empty() {
                    consec_degree_miss += 1;
                    consec_integ_miss = 0;
                    if consec_degree_miss >= consec_empty_stop {
                        break;
                    }
                    t_abs += 1;
                    continue;
                }

                let min_deg = terms.iter().map(|t| t.min_degree_x2).min().unwrap();
                if min_deg <= 2 * adjusted_q {
                    let key = (m_t, e_t_x2);
                    if !seen.contains(&key) {
                        seen.insert(key);
                        let mult = if c == 2 || (c == 0 && t_abs > 0) {
                            2
                        } else {
                            1
                        };
                        result.push(KernelTerm {
                            m: m_t,
                            e_x2: e_t_x2,
                            c,
                            phase: phase_t,
                            multiplicity: mult,
                        });
                    }
                    consec_degree_miss = 0;
                    consec_integ_miss = 0;
                } else {
                    consec_degree_miss += 1;
                    consec_integ_miss = 0;
                    if consec_degree_miss >= consec_empty_stop {
                        break;
                    }
                }
                t_abs += 1;
            }
        }
    }
    result
}
