//! Non-closable cycle search.
//!
//! Ports v0.5's `_candidate_slopes` and `find_non_closable_cycles`
//! from `dehn_filling.py`.

use num_integer::Integer;

use crate::dehn::unrefined_fill::{compute_filled_index, FilledIndexResult};
use crate::summation::EnumerationState;

/// A non-closable cycle at a given cusp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonClosableCycle {
    pub cusp_idx: usize,
    pub p: i64,
    pub q: i64,
}

/// Per-cusp NC search result.
#[derive(Debug, Clone)]
pub struct NcSearchResult {
    pub cusp_idx: usize,
    pub cycles: Vec<NonClosableCycle>,
    pub slopes_tested: Vec<(i64, i64)>,
    pub series_data: hashbrown::HashMap<(i64, i64), FilledIndexResult>,
}

/// All primitive slopes `(P, Q)` in `p_range × q_range` with `gcd(|P|,|Q|)=1`.
pub fn candidate_slopes(
    p_range: std::ops::Range<i64>,
    q_range: std::ops::Range<i64>,
    canonical_only: bool,
) -> Vec<(i64, i64)> {
    let mut slopes = Vec::new();
    let mut seen = hashbrown::HashSet::new();
    for p in p_range {
        for q in q_range.clone() {
            if p == 0 && q == 0 {
                continue;
            }
            if p.abs().gcd(&q.abs()) != 1 {
                continue;
            }
            if q == 0 && p < 0 {
                continue;
            }
            if canonical_only && q < 0 {
                continue;
            }
            let key = (p, q);
            if !seen.contains(&key) {
                seen.insert(key);
                slopes.push(key);
            }
        }
    }
    slopes
}

/// Search for non-closable cycles at `cusp_idx`.
pub fn find_non_closable_cycles(
    state: &EnumerationState,
    cusp_idx: usize,
    p_range: std::ops::Range<i64>,
    q_range: std::ops::Range<i64>,
    m_other: &[i64],
    e_other_x2: &[i64],
    q_order_half: i32,
    use_symmetry: bool,
) -> NcSearchResult {
    let all_slopes = candidate_slopes(p_range.clone(), q_range.clone(), false);
    let compute_slopes = candidate_slopes(p_range, q_range, use_symmetry);
    let all_set: hashbrown::HashSet<(i64, i64)> = all_slopes.iter().copied().collect();

    let mut result = NcSearchResult {
        cusp_idx,
        cycles: Vec::new(),
        slopes_tested: all_slopes,
        series_data: hashbrown::HashMap::new(),
    };

    let mut computed: hashbrown::HashSet<(i64, i64)> = hashbrown::HashSet::new();
    for &(p, q) in &compute_slopes {
        let filled = compute_filled_index(state, cusp_idx, p, q, m_other, e_other_x2, q_order_half);
        let nc = filled.is_stably_zero(None);
        computed.insert((p, q));
        if nc {
            result
                .cycles
                .push(NonClosableCycle { cusp_idx, p, q });
        }
        result.series_data.insert((p, q), filled);

        if use_symmetry {
            let neg = (-p, -q);
            if all_set.contains(&neg) && !computed.contains(&neg) {
                computed.insert(neg);
                if nc {
                    result.cycles.push(NonClosableCycle {
                        cusp_idx,
                        p: -p,
                        q: -q,
                    });
                }
            }
        }
    }
    result
}
