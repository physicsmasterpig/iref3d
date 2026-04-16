//! On-disk I^ref cache (per-manifold, content-hash keyed).
//!
//! Stores `RefinedIndexResult` entries keyed by a content hash of the
//! NZ data plus the query parameters `(m_ext, e_ext_x2, qq_order)`.
//!
//! Content hash ensures auto-invalidation when NZ data changes (e.g.
//! different triangulation or gluing).
//!
//! SQLite DB at `cache_dir()/iref_cache.db`.

use hashbrown::HashMap;
use rusqlite::{params, Connection, OptionalExtension};

use super::cache_dir;
use crate::index_refined::RefinedIndexResult;

fn open_db() -> rusqlite::Result<Connection> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("iref_cache.db");
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS iref_entries (
            nz_hash   TEXT NOT NULL,
            m_ext     TEXT NOT NULL,
            e_ext_x2  TEXT NOT NULL,
            qq_order  INTEGER NOT NULL,
            data      BLOB NOT NULL,
            created   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (nz_hash, m_ext, e_ext_x2, qq_order)
        );
        CREATE INDEX IF NOT EXISTS idx_iref_hash ON iref_entries(nz_hash);",
    )?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    Ok(conn)
}

/// Compute a 16-char hex content hash from NZ matrix data.
///
/// Mirrors v0.5's `_nz_hash`: SHA-256 of `g_NZ || nu_x || nu_p` bytes,
/// truncated to 16 hex chars.
pub fn nz_hash(g_nz: &[i32], nu_x: &[i32], nu_p: &[i32]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Use a fast hash — we don't need cryptographic strength here,
    // just collision resistance for ~thousands of manifolds.
    let mut h = DefaultHasher::new();
    g_nz.hash(&mut h);
    nu_x.hash(&mut h);
    nu_p.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn key_str(vals: &[i64]) -> String {
    serde_json::to_string(vals).expect("serialize key")
}

/// Save an I^ref result.
pub fn save(
    hash: &str,
    m_ext: &[i64],
    e_ext_x2: &[i64],
    qq_order: i32,
    result: &RefinedIndexResult,
) -> rusqlite::Result<()> {
    let conn = open_db()?;
    let m_key = key_str(m_ext);
    let e_key = key_str(e_ext_x2);
    // Serialize RefinedIndexResult: HashMap<Vec<i64>, i64> → JSON array of [key, value]
    let entries: Vec<(Vec<i64>, i64)> = result.iter().map(|(k, &v)| (k.clone(), v)).collect();
    let blob = serde_json::to_vec(&entries).expect("serialize iref result");
    conn.execute(
        "INSERT OR REPLACE INTO iref_entries (nz_hash, m_ext, e_ext_x2, qq_order, data)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![hash, m_key, e_key, qq_order, blob],
    )?;
    Ok(())
}

/// Load a cached I^ref result.
pub fn load(
    hash: &str,
    m_ext: &[i64],
    e_ext_x2: &[i64],
    qq_order: i32,
) -> rusqlite::Result<Option<RefinedIndexResult>> {
    let conn = open_db()?;
    let m_key = key_str(m_ext);
    let e_key = key_str(e_ext_x2);
    let row = conn
        .query_row(
            "SELECT data FROM iref_entries
             WHERE nz_hash=?1 AND m_ext=?2 AND e_ext_x2=?3 AND qq_order=?4",
            params![hash, m_key, e_key, qq_order],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()?;

    match row {
        Some(data) => {
            let entries: Vec<(Vec<i64>, i64)> =
                serde_json::from_slice(&data).expect("deserialize iref result");
            let mut result = HashMap::with_capacity(entries.len());
            for (k, v) in entries {
                result.insert(k, v);
            }
            Ok(Some(result))
        }
        None => Ok(None),
    }
}

/// Save multiple I^ref results in a single transaction.
pub fn save_batch(
    hash: &str,
    entries: &[(Vec<i64>, Vec<i64>, i32, RefinedIndexResult)],
) -> rusqlite::Result<()> {
    let mut conn = open_db()?;
    let tx = conn.transaction()?;
    for (m_ext, e_ext_x2, qq_order, result) in entries {
        let m_key = key_str(m_ext);
        let e_key = key_str(e_ext_x2);
        let items: Vec<(Vec<i64>, i64)> = result.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let blob = serde_json::to_vec(&items).expect("serialize iref result");
        tx.execute(
            "INSERT OR REPLACE INTO iref_entries (nz_hash, m_ext, e_ext_x2, qq_order, data)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![hash, m_key, e_key, *qq_order, blob],
        )?;
    }
    tx.commit()?;
    Ok(())
}

/// Count cached entries for a given manifold hash.
pub fn count(hash: &str) -> rusqlite::Result<usize> {
    let conn = open_db()?;
    conn.query_row(
        "SELECT COUNT(*) FROM iref_entries WHERE nz_hash=?1",
        params![hash],
        |row| row.get::<_, usize>(0),
    )
}

/// Delete all entries for a given manifold hash.
pub fn delete_manifold(hash: &str) -> rusqlite::Result<usize> {
    let conn = open_db()?;
    let n = conn.execute("DELETE FROM iref_entries WHERE nz_hash=?1", params![hash])?;
    Ok(n)
}

/// Delete all I^ref entries.
pub fn clear() -> rusqlite::Result<usize> {
    let conn = open_db()?;
    let n = conn.execute("DELETE FROM iref_entries", [])?;
    Ok(n)
}

/// List all manifold hashes in the cache.
pub fn list_manifolds() -> rusqlite::Result<Vec<(String, usize)>> {
    let conn = open_db()?;
    let mut stmt = conn.prepare(
        "SELECT nz_hash, COUNT(*) FROM iref_entries GROUP BY nz_hash ORDER BY nz_hash",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
    })?;
    rows.collect()
}
