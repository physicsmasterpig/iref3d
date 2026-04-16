//! On-disk kernel cache (manifold-independent).
//!
//! Stores pre-computed `Vec<KernelTerm>` plus `(R, S)` per slope `(P, Q)`
//! at a given `qq_order`.  These are reusable across all manifolds.
//!
//! SQLite DB at `cache_dir()/kernel_cache.db`.

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::cache_dir;
use crate::dehn::kernel_terms::KernelTerm;

/// Serializable kernel entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelCacheEntry {
    pub p: i64,
    pub q: i64,
    pub qq_order: i32,
    pub r_val: i64,
    pub s_val: i64,
    pub terms: Vec<KernelTerm>,
}

fn open_db() -> rusqlite::Result<Connection> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join("kernel_cache.db");
    let conn = Connection::open(path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS kernels (
            p         INTEGER NOT NULL,
            q         INTEGER NOT NULL,
            qq_order  INTEGER NOT NULL,
            r_val     INTEGER NOT NULL,
            s_val     INTEGER NOT NULL,
            data      BLOB NOT NULL,
            created   TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (p, q, qq_order)
        );",
    )?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    Ok(conn)
}

/// Save kernel terms for slope P/Q at `qq_order`.
pub fn save(entry: &KernelCacheEntry) -> rusqlite::Result<()> {
    let conn = open_db()?;
    let blob = serde_json::to_vec(&entry.terms).expect("serialize kernel terms");
    conn.execute(
        "INSERT OR REPLACE INTO kernels (p, q, qq_order, r_val, s_val, data)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![entry.p, entry.q, entry.qq_order, entry.r_val, entry.s_val, blob],
    )?;
    Ok(())
}

/// Load kernel terms for slope P/Q at `qq_order`, if cached.
pub fn load(p: i64, q: i64, qq_order: i32) -> rusqlite::Result<Option<KernelCacheEntry>> {
    let conn = open_db()?;
    let row = conn
        .query_row(
            "SELECT r_val, s_val, data FROM kernels WHERE p=?1 AND q=?2 AND qq_order=?3",
            params![p, q, qq_order],
            |row| {
                let r_val: i64 = row.get(0)?;
                let s_val: i64 = row.get(1)?;
                let data: Vec<u8> = row.get(2)?;
                Ok((r_val, s_val, data))
            },
        )
        .optional()?;

    match row {
        Some((r_val, s_val, data)) => {
            let terms: Vec<KernelTerm> =
                serde_json::from_slice(&data).expect("deserialize kernel terms");
            Ok(Some(KernelCacheEntry {
                p,
                q,
                qq_order,
                r_val,
                s_val,
                terms,
            }))
        }
        None => Ok(None),
    }
}

/// List all cached slopes at a given `qq_order`.
pub fn list_slopes(qq_order: i32) -> rusqlite::Result<Vec<(i64, i64)>> {
    let conn = open_db()?;
    let mut stmt = conn.prepare("SELECT p, q FROM kernels WHERE qq_order=?1 ORDER BY p, q")?;
    let rows = stmt.query_map(params![qq_order], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
    })?;
    rows.collect()
}

/// List all cached (P, Q, qq_order) triples.
pub fn list_all() -> rusqlite::Result<Vec<(i64, i64, i32)>> {
    let conn = open_db()?;
    let mut stmt = conn.prepare("SELECT p, q, qq_order FROM kernels ORDER BY qq_order, p, q")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i32>(2)?,
        ))
    })?;
    rows.collect()
}

/// Delete a specific kernel entry.
pub fn delete(p: i64, q: i64, qq_order: i32) -> rusqlite::Result<bool> {
    let conn = open_db()?;
    let n = conn.execute(
        "DELETE FROM kernels WHERE p=?1 AND q=?2 AND qq_order=?3",
        params![p, q, qq_order],
    )?;
    Ok(n > 0)
}

/// Delete all kernel entries.
pub fn clear() -> rusqlite::Result<usize> {
    let conn = open_db()?;
    let n = conn.execute("DELETE FROM kernels", [])?;
    Ok(n)
}
