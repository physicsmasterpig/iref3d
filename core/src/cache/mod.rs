//! Disk caching for expensive computations.
//!
//! Two cache tiers:
//! - **Kernel cache**: per-slope Dehn filling kernel K^ref(P/Q).
//!   Manifold-independent, reusable across all manifolds.
//! - **I^ref cache**: per-manifold refined index grid.
//!   Manifold-dependent, content-hash keyed for auto-invalidation.
//!
//! Cache directory follows platform conventions:
//! - macOS: `~/Library/Caches/iref3d/`
//! - Linux: `$XDG_CACHE_HOME/iref3d/` or `~/.cache/iref3d/`
//! - Override: `IREF3D_CACHE_DIR` environment variable.

pub mod kernel_disk;
pub mod iref_disk;

use std::path::PathBuf;

/// Resolve the base cache directory.
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("IREF3D_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if cfg!(target_os = "macos") {
        dirs_cache().join("iref3d")
    } else {
        dirs_cache().join("iref3d")
    }
}

fn dirs_cache() -> PathBuf {
    if cfg!(target_os = "macos") {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join("Library").join("Caches")
    } else {
        // XDG_CACHE_HOME or ~/.cache
        std::env::var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
                PathBuf::from(home).join(".cache")
            })
    }
}
