//! Unrefined Dehn-filled 3D index.
//!
//! Ports v0.5's `compute_filled_index` from `dehn_filling.py`.

use num_rational::Rational64;

use crate::dehn::kernel_terms::{enumerate_kernel_terms, find_rs, KernelTerm};
use crate::index_unrefined::compute_unrefined_index;
use crate::summation::EnumerationState;

/// q^{1/2}-series as sparse map: key `k` → coefficient of `q^{k/2}`.
pub type QSeries = hashbrown::HashMap<i64, Rational64>;

fn qseries_from_unrefined(
    coeffs: &[i64],
    min_power: i64,
) -> QSeries {
    let mut s = QSeries::new();
    for (i, &c) in coeffs.iter().enumerate() {
        if c != 0 {
            s.insert(min_power + i as i64, Rational64::from_integer(c));
        }
    }
    s
}

fn qseries_shift(s: &QSeries, shift: i64) -> QSeries {
    s.iter().map(|(&k, &v)| (k + shift, v)).collect()
}

fn qseries_scale(s: &QSeries, scalar: Rational64) -> QSeries {
    if scalar == Rational64::from_integer(0) {
        return QSeries::new();
    }
    s.iter()
        .filter_map(|(&k, &v)| {
            let r = v * scalar;
            if r == Rational64::from_integer(0) {
                None
            } else {
                Some((k, r))
            }
        })
        .collect()
}

fn qseries_add(a: &QSeries, b: &QSeries) -> QSeries {
    let mut out = a.clone();
    let zero = Rational64::from_integer(0);
    for (&k, &v) in b {
        let e = out.entry(k).or_insert(zero);
        *e += v;
        if *e == zero {
            out.remove(&k);
        }
    }
    out
}

fn qseries_truncate(s: &QSeries, q_order_half: i64) -> QSeries {
    s.iter()
        .filter(|(&k, _)| k <= q_order_half)
        .map(|(&k, &v)| (k, v))
        .collect()
}

fn apply_kernel(term: &KernelTerm, index_series: &QSeries, q_order_half: Option<i64>) -> QSeries {
    let sign = if term.phase % 2 == 0 {
        Rational64::from_integer(1)
    } else {
        Rational64::from_integer(-1)
    };
    let half = Rational64::new(1, 2);
    if term.c == 0 {
        let b = qseries_shift(index_series, -term.phase);
        if q_order_half.is_none() || term.phase.abs() <= q_order_half.unwrap() {
            let a = qseries_shift(index_series, term.phase);
            qseries_scale(&qseries_add(&a, &b), half * sign)
        } else {
            qseries_scale(&b, half * sign)
        }
    } else {
        qseries_scale(index_series, -half * sign)
    }
}

#[derive(Debug, Clone)]
pub struct FilledIndexResult {
    pub p: i64,
    pub q: i64,
    pub cusp_idx: usize,
    pub series: QSeries,
    pub q_order_half: i32,
    pub n_kernel_terms: usize,
}

impl FilledIndexResult {
    pub fn is_zero(&self) -> bool {
        self.series.is_empty()
    }

    pub fn is_stably_zero(&self, buffer: Option<i32>) -> bool {
        let buf = buffer.unwrap_or_else(|| {
            std::cmp::min(
                std::cmp::max(5, self.q_order_half / 2),
                self.q_order_half - 1,
            )
        });
        let cutoff = (self.q_order_half - buf) as i64;
        !self
            .series
            .iter()
            .any(|(&k, v)| k <= cutoff && *v != Rational64::from_integer(0))
    }
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

/// Compute the Dehn-filled unrefined 3D index I_{P/Q} at cusp `cusp_idx`.
pub fn compute_filled_index(
    state: &EnumerationState,
    cusp_idx: usize,
    p: i64,
    q: i64,
    m_other: &[i64],
    e_other_x2: &[i64],
    q_order_half: i32,
) -> FilledIndexResult {
    let r = state.r;
    let (r_val, s_val) = find_rs(p, q);

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

    let mut total: QSeries = QSeries::new();
    for kt in &kernel_terms {
        let (m_ext, e_ext) = make_ext(cusp_idx, r, kt.m, kt.e_x2, m_other, e_other_x2);
        let index_q_order = q_order_half as i64 + if kt.c == 0 { kt.phase.abs() } else { 0 };
        let res = compute_unrefined_index(state, &m_ext, &e_ext, index_q_order as i32);
        let idx_series = qseries_from_unrefined(&res.coeffs, res.min_power as i64);
        let mut contrib = apply_kernel(kt, &idx_series, Some(q_order_half as i64));
        if kt.multiplicity != 1 {
            contrib = qseries_scale(&contrib, Rational64::from_integer(kt.multiplicity));
        }
        total = qseries_add(&total, &contrib);
    }
    total = qseries_truncate(&total, q_order_half as i64);

    FilledIndexResult {
        p,
        q,
        cusp_idx,
        series: total,
        q_order_half,
        n_kernel_terms: kernel_terms.len(),
    }
}
