//! Cusp cycle choice and basis selection.
//!
//! Port of v0.5's `core/basis_selection.py`. `apply_basis_changes` is
//! implemented alongside `nz.rs` since it depends on `apply_cusp_basis_change`.

use num_integer::Integer;
use num_rational::Ratio;

/// A primitive slope `(P, Q)` on one cusp, with human-readable label.
///
/// `m = P`, `e = Q / 2` (half-integer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleChoice {
    pub cusp_idx: usize,
    pub p: i32,
    pub q: i32,
    pub label: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasisError {
    ZeroSlope,
    NonPrimitive { p: i32, q: i32, gcd: i32 },
    EmptyChoices,
    CuspIdxMismatch { position: usize, got: usize },
}

impl std::fmt::Display for BasisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BasisError::ZeroSlope => write!(f, "CycleChoice: (P, Q) = (0, 0) is not a valid slope"),
            BasisError::NonPrimitive { p, q, gcd } => {
                write!(f, "CycleChoice: (P={p}, Q={q}) is not primitive; gcd = {gcd}")
            }
            BasisError::EmptyChoices => write!(f, "BasisSelection: choices must be non-empty"),
            BasisError::CuspIdxMismatch { position, got } => write!(
                f,
                "BasisSelection: choices[{position}].cusp_idx = {got} ≠ {position}"
            ),
        }
    }
}
impl std::error::Error for BasisError {}

impl CycleChoice {
    pub fn new(cusp_idx: usize, p: i32, q: i32) -> Result<Self, BasisError> {
        Self::with_label(cusp_idx, p, q, None, false)
    }

    pub fn with_label(
        cusp_idx: usize,
        p: i32,
        q: i32,
        label: Option<String>,
        is_default: bool,
    ) -> Result<Self, BasisError> {
        if p == 0 && q == 0 {
            return Err(BasisError::ZeroSlope);
        }
        let g = p.abs().gcd(&q.abs());
        if g != 1 {
            return Err(BasisError::NonPrimitive { p, q, gcd: g });
        }
        let label = label.unwrap_or_else(|| {
            if p == 1 && q == 0 {
                "meridian M (1/0)".to_owned()
            } else if p == 0 && q == 1 {
                "longitude L (0/1)".to_owned()
            } else {
                format!("slope {p}/{q}")
            }
        });
        Ok(Self { cusp_idx, p, q, label, is_default })
    }

    /// `m = P`.
    #[inline]
    pub fn m(&self) -> i32 { self.p }

    /// `e = Q / 2` as an exact rational.
    #[inline]
    pub fn e(&self) -> Ratio<i32> { Ratio::new(self.q, 2) }

    #[inline]
    pub fn slope_str(&self) -> String {
        format!("{}/{}", self.p, self.q)
    }
}

/// A full per-cusp basis selection (one `CycleChoice` per cusp, in order).
#[derive(Debug, Clone)]
pub struct BasisSelection {
    pub choices: Vec<CycleChoice>,
}

impl BasisSelection {
    pub fn new(choices: Vec<CycleChoice>) -> Result<Self, BasisError> {
        if choices.is_empty() {
            return Err(BasisError::EmptyChoices);
        }
        for (i, cc) in choices.iter().enumerate() {
            if cc.cusp_idx != i {
                return Err(BasisError::CuspIdxMismatch { position: i, got: cc.cusp_idx });
            }
        }
        Ok(Self { choices })
    }

    #[inline]
    pub fn r(&self) -> usize { self.choices.len() }

    pub fn m_ext(&self) -> Vec<i32> { self.choices.iter().map(|c| c.m()).collect() }

    pub fn e_ext(&self) -> Vec<Ratio<i32>> { self.choices.iter().map(|c| c.e()).collect() }
}

pub fn default_meridian_choice(cusp_idx: usize) -> CycleChoice {
    CycleChoice::with_label(
        cusp_idx,
        1,
        0,
        Some("meridian M (1/0)".to_owned()),
        true,
    )
    .expect("meridian (1,0) is primitive")
}

pub fn default_longitude_choice(cusp_idx: usize) -> CycleChoice {
    CycleChoice::with_label(
        cusp_idx,
        0,
        1,
        Some("longitude L (0/1)".to_owned()),
        true,
    )
    .expect("longitude (0,1) is primitive")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_check() {
        assert!(CycleChoice::new(0, 2, 4).is_err());
        assert_eq!(CycleChoice::new(0, 0, 0).unwrap_err(), BasisError::ZeroSlope);
        assert!(CycleChoice::new(0, 3, 5).is_ok());
    }

    #[test]
    fn m_and_e() {
        let cc = CycleChoice::new(0, 3, 4).unwrap();
        assert_eq!(cc.m(), 3);
        assert_eq!(cc.e(), Ratio::new(4, 2));
    }

    #[test]
    fn default_labels() {
        assert_eq!(default_meridian_choice(2).label, "meridian M (1/0)");
        assert_eq!(default_longitude_choice(0).label, "longitude L (0/1)");
    }

    #[test]
    fn basis_selection_indices_must_match() {
        let mut a = default_meridian_choice(0);
        let mut b = default_meridian_choice(1);
        assert!(BasisSelection::new(vec![a.clone(), b.clone()]).is_ok());
        a.cusp_idx = 1;
        b.cusp_idx = 0;
        assert!(BasisSelection::new(vec![a, b]).is_err());
    }
}
