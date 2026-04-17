//! Census manifold lookup from `data/census.db`.
//!
//! Replaces the three SnapPy calls in v0.5's `core/manifold.py`:
//!   - `snappy.Manifold(name)` / `num_tetrahedra()` / `num_cusps()`
//!   - `M.gluing_equations()` → `(n + 2r, 3n)` integer matrix
//!
//! The DB is produced by `data/extract_census.py`. Schema:
//!
//! ```sql
//! CREATE TABLE manifolds (
//!   census TEXT, name TEXT, n INTEGER, r INTEGER, gluing BLOB, pivots BLOB,
//!   PRIMARY KEY (census, name)
//! );
//! ```
//!
//! `gluing` is a raw little-endian i32 array of length `(n + 2r) * 3n`, row-major.
//! `pivots` is a raw little-endian i32 array of length `n - r`: sorted column-pivot
//! indices from scipy.linalg.qr on the reduced edge-coefficient matrix, identifying
//! an (n-r)-element linearly-independent subset of the n reduced edge rows.

use rusqlite::{Connection, OptionalExtension};
use std::path::Path;

#[derive(Debug)]
pub enum CensusError {
    Sqlite(rusqlite::Error),
    NotFound(String),
    BlobLength { got: usize, expected: usize, n: i64, r: i64 },
    BlobAlignment(usize),
    PivotsLength { got: usize, expected: usize, n: i64, r: i64 },
    PivotsAlignment(usize),
    Ambiguous(String),
}

impl From<rusqlite::Error> for CensusError {
    fn from(e: rusqlite::Error) -> Self { CensusError::Sqlite(e) }
}

/// A manifold loaded from the census DB.
///
/// The `gluing` matrix is row-major with shape `(n + 2r, 3n)`. Rows 0..n are
/// edge equations, then interleaved meridian/longitude rows per cusp.
#[derive(Debug, Clone)]
pub struct ManifoldData {
    pub census: String,
    pub name: String,
    pub n: usize,
    pub r: usize,
    pub gluing: Vec<i32>,
    /// Sorted column-pivot indices from scipy's QR — length `n - r`.
    /// Identifies the linearly-independent subset of reduced edge rows.
    pub pivots: Vec<i32>,
}

/// Phase-space basis from v0.5's `find_easy_edges` — pre-extracted in SQLite.
#[derive(Debug, Clone)]
pub struct PhaseSpaceData {
    pub n: usize,
    /// `num_easy × 3n` row-major: all discovered easy edges.
    pub easy_edges: Vec<i32>,
    /// Indices into `easy_edges` rows — length `num_independent_easy`, sorted.
    pub easy_indep: Vec<i32>,
    /// `num_hard × 3n` row-major: raw SnaPy edge rows used as hard padding.
    pub hard_padding: Vec<i32>,
}

/// Base Neumann-Zagier data, pre-extracted in SQLite.
///
/// `g_nz_x2` stores `2 · g_NZ` as `i64` entries so the half-integer longitude/2
/// rows are still integer. Same for `nu_p_x2 = 2 · ν_p`. `nu_x` is already an
/// integer vector.
#[derive(Debug, Clone)]
pub struct NzData {
    pub n: usize,
    pub r: usize,
    pub num_hard: usize,
    pub num_easy: usize,
    /// `(2n × 2n)` row-major `i64`: `2 · g_NZ`.
    pub g_nz_x2: Vec<i64>,
    /// Length `n` — `ν_x`.
    pub nu_x: Vec<i64>,
    /// Length `n` — `2 · ν_p`.
    pub nu_p_x2: Vec<i64>,
}

impl NzData {
    #[inline]
    pub fn g_row_x2(&self, i: usize) -> &[i64] {
        let w = 2 * self.n;
        &self.g_nz_x2[i * w..(i + 1) * w]
    }
    #[inline]
    pub fn g_row_x2_mut(&mut self, i: usize) -> &mut [i64] {
        let w = 2 * self.n;
        &mut self.g_nz_x2[i * w..(i + 1) * w]
    }
}

impl PhaseSpaceData {
    #[inline]
    pub fn num_easy(&self) -> usize {
        if self.n == 0 { 0 } else { self.easy_edges.len() / (3 * self.n) }
    }
    #[inline]
    pub fn num_hard(&self) -> usize {
        if self.n == 0 { 0 } else { self.hard_padding.len() / (3 * self.n) }
    }
    #[inline]
    pub fn easy_row(&self, i: usize) -> &[i32] {
        let w = 3 * self.n;
        &self.easy_edges[i * w..(i + 1) * w]
    }
    #[inline]
    pub fn hard_row(&self, i: usize) -> &[i32] {
        let w = 3 * self.n;
        &self.hard_padding[i * w..(i + 1) * w]
    }
}

impl ManifoldData {
    #[inline]
    pub fn rows(&self) -> usize {
        self.n + 2 * self.r
    }
    #[inline]
    pub fn cols(&self) -> usize {
        3 * self.n
    }
    /// Row `i` of the gluing matrix (panics if out of bounds).
    pub fn row(&self, i: usize) -> &[i32] {
        let cols = self.cols();
        &self.gluing[i * cols..(i + 1) * cols]
    }
    /// (meridian, longitude) rows for cusp `k`, shape `3n` each.
    pub fn cusp_equations(&self, k: usize) -> (&[i32], &[i32]) {
        let m = self.n + 2 * k;
        (self.row(m), self.row(m + 1))
    }
}

/// Thin wrapper around a SQLite connection to the census DB.
pub struct Census {
    conn: Connection,
}

impl Census {
    /// Open the census DB (read-only).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CensusError> {
        let conn = Connection::open_with_flags(
            path.as_ref(),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        Ok(Self { conn })
    }

    /// Total number of manifolds in the DB (all censuses).
    pub fn count(&self) -> Result<i64, CensusError> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM manifolds", [], |row| row.get(0))?;
        Ok(n)
    }

    /// Look up a manifold by bare name. If the name is unique across censuses,
    /// returns it. If it appears in multiple censuses, returns `Ambiguous`.
    /// If it does not exist, returns `NotFound`.
    pub fn load(&self, name: &str) -> Result<ManifoldData, CensusError> {
        // Try direct name lookup first.
        let mut stmt = self
            .conn
            .prepare("SELECT census, n, r, gluing, pivots FROM manifolds WHERE name = ?1")?;
        let mut rows = stmt.query([name])?;
        let first = rows.next()?;
        if let Some(row) = first {
            let census: String = row.get(0)?;
            let n: i64 = row.get(1)?;
            let r: i64 = row.get(2)?;
            let blob: Vec<u8> = row.get(3)?;
            let pivots_blob: Vec<u8> = row.get(4)?;
            if rows.next()?.is_some() {
                return Err(CensusError::Ambiguous(name.to_owned()));
            }
            return Self::decode(census, name.to_owned(), n, r, blob, pivots_blob);
        }

        // Try alias resolution (knot notation like 4_1, 5^2_1, etc.).
        if let Some((census_name, canon_name)) = self.resolve_alias(name)? {
            return self.load_by_census_name(&census_name, &canon_name);
        }

        Err(CensusError::NotFound(name.to_owned()))
    }

    /// Resolve an alias to (census, name). Returns None if no alias found.
    fn resolve_alias(&self, alias: &str) -> Result<Option<(String, String)>, CensusError> {
        let result: Option<(String, String)> = self
            .conn
            .query_row(
                "SELECT census, name FROM aliases WHERE alias = ?1",
                [alias],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        Ok(result)
    }

    /// Try all aliases for `input_name` until one has both manifold + NZ data.
    /// Returns None if no alias resolves to a manifold with NZ data.
    pub fn resolve_alias_with_nz(
        &self,
        input_name: &str,
    ) -> Result<Option<(ManifoldData, NzData)>, CensusError> {
        // Get all aliases for this name (there may be multiple census entries)
        let mut stmt = self.conn.prepare(
            "SELECT census, name FROM aliases WHERE alias = ?1"
        )?;
        let alias_rows: Vec<(String, String)> = stmt
            .query_map([input_name], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (census, name) in &alias_rows {
            if let Ok(nz) = self.load_nz(census, name) {
                let md = self.load_by_census_name(census, name)?;
                return Ok(Some((md, nz)));
            }
        }
        // Also try: input_name itself might be a census name in another census
        let mut stmt2 = self.conn.prepare(
            "SELECT census, name FROM manifolds WHERE name = ?1"
        )?;
        let direct_rows: Vec<(String, String)> = stmt2
            .query_map([input_name], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (census, name) in &direct_rows {
            if let Ok(nz) = self.load_nz(census, name) {
                let md = self.load_by_census_name(census, name)?;
                return Ok(Some((md, nz)));
            }
        }
        Ok(None)
    }

    /// Load by explicit (census, name) pair.
    fn load_by_census_name(&self, census: &str, name: &str) -> Result<ManifoldData, CensusError> {
        let row: Option<(i64, i64, Vec<u8>, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT n, r, gluing, pivots FROM manifolds WHERE census = ?1 AND name = ?2",
                [census, name],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let Some((n, r, blob, pivots_blob)) = row else {
            return Err(CensusError::NotFound(format!("{census}/{name}")));
        };
        Self::decode(census.to_owned(), name.to_owned(), n, r, blob, pivots_blob)
    }

    /// Fetch the base Neumann-Zagier data for `(census, name)`.
    pub fn load_nz(&self, census: &str, name: &str) -> Result<NzData, CensusError> {
        let row: Option<(i64, i64, i64, i64, Vec<u8>, Vec<u8>, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT n, r, num_hard, num_easy, g_nz_x2, nu_x, nu_p_x2 \
                 FROM nz WHERE census = ?1 AND name = ?2",
                [census, name],
                |row| Ok((
                    row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                    row.get(4)?, row.get(5)?, row.get(6)?,
                )),
            )
            .optional()?;
        let Some((n, r, num_hard, num_easy, g_blob, nu_x_blob, nu_p_blob)) = row else {
            return Err(CensusError::NotFound(format!("nz/{census}/{name}")));
        };
        Ok(NzData {
            n: n as usize,
            r: r as usize,
            num_hard: num_hard as usize,
            num_easy: num_easy as usize,
            g_nz_x2: decode_i64_blob(&g_blob)?,
            nu_x: decode_i64_blob(&nu_x_blob)?,
            nu_p_x2: decode_i64_blob(&nu_p_blob)?,
        })
    }

    /// Fetch the phase-space basis for `(census, name)` from the `phase_space`
    /// table. Returns `NotFound` if the table row is missing.
    pub fn load_phase_space(
        &self,
        census: &str,
        name: &str,
        n: usize,
    ) -> Result<PhaseSpaceData, CensusError> {
        let row: Option<(Vec<u8>, Vec<u8>, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT easy_edges, easy_indep, hard_pad FROM phase_space \
                 WHERE census = ?1 AND name = ?2",
                [census, name],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((ee, ei, hp)) = row else {
            return Err(CensusError::NotFound(format!("phase_space/{census}/{name}")));
        };
        Ok(PhaseSpaceData {
            n,
            easy_edges: decode_i32_blob(&ee)?,
            easy_indep: decode_i32_blob(&ei)?,
            hard_padding: decode_i32_blob(&hp)?,
        })
    }

    /// Look up by (census, name) — always unambiguous.
    pub fn load_in(
        &self,
        census: &str,
        name: &str,
    ) -> Result<ManifoldData, CensusError> {
        let row: Option<(i64, i64, Vec<u8>, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT n, r, gluing, pivots FROM manifolds WHERE census = ?1 AND name = ?2",
                [census, name],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let Some((n, r, blob, pivots_blob)) = row else {
            return Err(CensusError::NotFound(format!("{census}/{name}")));
        };
        Self::decode(census.to_owned(), name.to_owned(), n, r, blob, pivots_blob)
    }

    fn decode(
        census: String,
        name: String,
        n: i64,
        r: i64,
        blob: Vec<u8>,
        pivots_blob: Vec<u8>,
    ) -> Result<ManifoldData, CensusError> {
        if blob.len() % 4 != 0 {
            return Err(CensusError::BlobAlignment(blob.len()));
        }
        let expected_ints = ((n + 2 * r) * 3 * n) as usize;
        let got_ints = blob.len() / 4;
        if got_ints != expected_ints {
            return Err(CensusError::BlobLength {
                got: got_ints,
                expected: expected_ints,
                n,
                r,
            });
        }
        let mut gluing = Vec::with_capacity(got_ints);
        for chunk in blob.chunks_exact(4) {
            gluing.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        if pivots_blob.len() % 4 != 0 {
            return Err(CensusError::PivotsAlignment(pivots_blob.len()));
        }
        let expected_pivots = (n - r) as usize;
        let got_pivots = pivots_blob.len() / 4;
        if got_pivots != expected_pivots {
            return Err(CensusError::PivotsLength {
                got: got_pivots,
                expected: expected_pivots,
                n,
                r,
            });
        }
        let mut pivots = Vec::with_capacity(got_pivots);
        for chunk in pivots_blob.chunks_exact(4) {
            pivots.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Ok(ManifoldData {
            census,
            name,
            n: n as usize,
            r: r as usize,
            gluing,
            pivots,
        })
    }
}

fn decode_i64_blob(bytes: &[u8]) -> Result<Vec<i64>, CensusError> {
    if bytes.len() % 8 != 0 {
        return Err(CensusError::BlobAlignment(bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut a = [0u8; 8];
        a.copy_from_slice(chunk);
        out.push(i64::from_le_bytes(a));
    }
    Ok(out)
}

fn decode_i32_blob(bytes: &[u8]) -> Result<Vec<i32>, CensusError> {
    if bytes.len() % 4 != 0 {
        return Err(CensusError::BlobAlignment(bytes.len()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

impl std::fmt::Display for CensusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CensusError::Sqlite(e) => write!(f, "sqlite: {e}"),
            CensusError::NotFound(s) => write!(f, "manifold not found: {s}"),
            CensusError::BlobLength { got, expected, n, r } => write!(
                f,
                "gluing blob length {got} != expected {expected} for n={n} r={r}"
            ),
            CensusError::BlobAlignment(b) => {
                write!(f, "gluing blob byte-length {b} is not a multiple of 4")
            }
            CensusError::PivotsLength { got, expected, n, r } => write!(
                f,
                "pivots blob length {got} != expected {expected} for n={n} r={r}"
            ),
            CensusError::PivotsAlignment(b) => {
                write!(f, "pivots blob byte-length {b} is not a multiple of 4")
            }
            CensusError::Ambiguous(s) => {
                write!(f, "ambiguous name {s:?}: found in multiple censuses")
            }
        }
    }
}
impl std::error::Error for CensusError {}
