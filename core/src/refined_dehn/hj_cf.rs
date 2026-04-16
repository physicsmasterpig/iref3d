//! Hirzebruch-Jung continued fraction for P/Q.
//!
//! Ports v0.5's `hj_continued_fraction`, `_hj_cf_ceil`, `_hj_cf_round`.

use num_rational::Rational64;

/// Classical HJ-CF using ceiling (k_i ≥ 2 except terminal).
fn hj_cf_ceil(p: i64, q: i64) -> Vec<i64> {
    let mut x = Rational64::new(p, q);
    let mut ks = Vec::new();
    loop {
        let k = ceil_rational(x);
        ks.push(k);
        let rem = Rational64::from_integer(k) - x;
        if rem == Rational64::from_integer(0) {
            break;
        }
        x = Rational64::new(1, 1) / rem;
    }
    ks
}

/// HJ-CF using nearest-integer rounding (shorter chains, O(log Q)).
fn hj_cf_round(p: i64, q: i64) -> Vec<i64> {
    let mut x = Rational64::new(p, q);
    let mut ks = Vec::new();
    let half = Rational64::new(1, 2);
    let zero = Rational64::from_integer(0);
    loop {
        let k = if x >= zero {
            floor_rational(x + half)
        } else {
            -floor_rational(-x + half)
        };
        ks.push(k);
        let rem = Rational64::from_integer(k) - x;
        if rem == zero {
            break;
        }
        x = Rational64::new(1, 1) / rem;
    }
    ks
}

fn ceil_rational(r: Rational64) -> i64 {
    let (n, d) = (*r.numer(), *r.denom());
    if d == 1 {
        return n;
    }
    if n >= 0 {
        (n + d - 1) / d
    } else {
        n / d
    }
}

fn floor_rational(r: Rational64) -> i64 {
    let (n, d) = (*r.numer(), *r.denom());
    if d == 1 {
        return n;
    }
    if n >= 0 {
        n / d
    } else {
        (n - d + 1) / d
    }
}

/// Hirzebruch-Jung continued fraction for P/Q (shortest form).
///
/// Returns `[k_1, …, k_ℓ]` such that
///     `P/Q = k_1 − 1/(k_2 − 1/(… − 1/k_ℓ))`
pub fn hj_continued_fraction(mut p: i64, mut q: i64) -> Vec<i64> {
    if q == 0 {
        assert!(p.abs() == 1, "Q=0 but |P| != 1");
        return vec![0, 0];
    }
    if q < 0 {
        p = -p;
        q = -q;
    }
    if q == 1 {
        return vec![p];
    }

    // Try length-2: P/Q = k1 - 1/k2.
    let abs_q = q.abs();
    let mut best: Option<(i64, Vec<i64>)> = None;
    for i in 1..=abs_q {
        if abs_q % i != 0 {
            continue;
        }
        for &d in &[i, -i] {
            if (p + d) % q == 0 {
                let k1 = (p + d) / q;
                let k2 = q / d;
                let cost = k1.abs() + k2.abs();
                if best.is_none() || cost < best.as_ref().unwrap().0 {
                    best = Some((cost, vec![k1, k2]));
                }
            }
        }
    }
    if let Some((_, ks)) = best {
        return ks;
    }

    let ks_ceil = hj_cf_ceil(p, q);
    let ks_round = hj_cf_round(p, q);
    if ks_round.len() < ks_ceil.len() {
        ks_round
    } else {
        ks_ceil
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(hj_continued_fraction(1, 3), vec![0, -3]);
        assert_eq!(hj_continued_fraction(1, 4), vec![0, -4]);
        assert_eq!(hj_continued_fraction(4, 3), vec![1, -3]);
        assert_eq!(hj_continued_fraction(5, 2), vec![2, -2]);
        assert_eq!(hj_continued_fraction(1, 1), vec![1]);
        assert_eq!(hj_continued_fraction(1, 0), vec![0, 0]);
        assert_eq!(hj_continued_fraction(-1, 0), vec![0, 0]);
    }

    #[test]
    fn reconstruction() {
        fn eval(ks: &[i64]) -> Rational64 {
            let mut x = Rational64::from_integer(*ks.last().unwrap());
            for &k in ks[..ks.len() - 1].iter().rev() {
                x = Rational64::from_integer(k) - Rational64::new(1, 1) / x;
            }
            x
        }
        for p in -10i64..=10 {
            for q in 1i64..=10 {
                if num_integer::Integer::gcd(&p.abs(), &q) != 1 {
                    continue;
                }
                let ks = hj_continued_fraction(p, q);
                let reconstructed = eval(&ks);
                assert_eq!(
                    reconstructed,
                    Rational64::new(p, q),
                    "p={p}, q={q}, ks={ks:?}"
                );
            }
        }
    }
}
