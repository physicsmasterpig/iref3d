//! C-ABI FFI for the Swift GUI.
//!
//! Each function takes/returns JSON as C strings. The Swift side calls these
//! via a bridging header. All computation happens in Rust; Swift just
//! serialises parameters and deserialises results.
//!
//! Memory: returned `*mut c_char` strings are allocated by Rust. The caller
//! must free them with `iref3d_free_string`.

use std::ffi::{c_char, CStr, CString};
use std::panic;
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json;

use crate::census::Census;
use crate::nz::NzData;
use crate::summation::EnumerationState;

// ── Global state: loaded manifold ──

struct LoadedManifold {
    name: String,
    nz: NzData,
    state: EnumerationState,
}

static CURRENT: Mutex<Option<LoadedManifold>> = Mutex::new(None);

/// Lock CURRENT, recovering from poison (prior panic).
fn lock_current() -> std::sync::MutexGuard<'static, Option<LoadedManifold>> {
    CURRENT.lock().unwrap_or_else(|e| e.into_inner())
}

// ── Helpers ──

fn json_ok(value: &impl Serialize) -> *mut c_char {
    let json = serde_json::to_string(value).unwrap_or_else(|e| {
        format!(r#"{{"error":"serialize: {}"}}"#, e)
    });
    CString::new(json).unwrap().into_raw()
}

fn json_err(msg: &str) -> *mut c_char {
    let json = format!(r#"{{"error":"{}"}}"#, msg.replace('"', "'"));
    CString::new(json).unwrap().into_raw()
}

fn parse_json_arg<'a>(ptr: *const c_char) -> Result<&'a str, *mut c_char> {
    if ptr.is_null() {
        return Err(json_err("null argument"));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| json_err("invalid UTF-8"))
}

// ── Public C-ABI functions ──

/// Free a string returned by any iref3d_* function.
#[unsafe(no_mangle)]
pub extern "C" fn iref3d_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)); }
    }
}

/// Load a manifold from the census database.
///
/// Input JSON: `{"db_path": "...", "name": "m006"}`
/// Returns JSON: `{"name", "n", "r", "hard", "easy", "nz_latex", "nu_x", "nu_p"}`
#[unsafe(no_mangle)]
pub extern "C" fn iref3d_load_manifold(json_in: *const c_char) -> *mut c_char {
    let s = match parse_json_arg(json_in) {
        Ok(s) => s,
        Err(e) => return e,
    };

    #[derive(Deserialize)]
    struct Args {
        db_path: String,
        name: String,
    }
    let args: Args = match serde_json::from_str(s) {
        Ok(a) => a,
        Err(e) => return json_err(&format!("parse: {e}")),
    };

    let census = match Census::open(Path::new(&args.db_path)) {
        Ok(c) => c,
        Err(e) => return json_err(&format!("census open: {e}")),
    };
    let md = match census.load(&args.name) {
        Ok(m) => m,
        Err(e) => return json_err(&format!("load '{}': {e}", args.name)),
    };
    // Try NZ data for the direct match; if missing, try alias resolution to
    // find a census entry that *does* have NZ data (e.g. "4_1" lives in
    // link_exteriors but NZ data only exists for orientable_cusped/m004).
    let (md, nz) = match census.load_nz(&md.census, &md.name) {
        Ok(n) => (md, n),
        Err(_) => {
            // Try alias → canonical name with NZ data
            match census.resolve_alias_with_nz(&args.name) {
                Ok(Some((alt_md, alt_nz))) => (alt_md, alt_nz),
                _ => match census.load_nz(&md.census, &md.name) {
                    Ok(n) => (md, n),
                    Err(e) => return json_err(&format!("load_nz '{}': {e}", args.name)),
                },
            }
        }
    };

    #[derive(Serialize)]
    struct Out {
        name: String,
        n: usize,
        r: usize,
        hard: usize,
        easy: usize,
        nz_g_x2: Vec<i64>,
        nu_x: Vec<i64>,
        nu_p_x2: Vec<i64>,
    }

    let out = Out {
        name: args.name.clone(),
        n: nz.n,
        r: nz.r,
        hard: nz.num_hard,
        easy: nz.num_easy,
        nz_g_x2: nz.g_nz_x2.clone(),
        nu_x: nz.nu_x.clone(),
        nu_p_x2: nz.nu_p_x2.clone(),
    };

    let state = EnumerationState::build(&nz);
    *lock_current() = Some(LoadedManifold {
        name: md.name,
        nz,
        state,
    });

    json_ok(&out)
}

/// Compute refined index for a grid of (m, e) values.
///
/// Input JSON: `{"m_lo", "m_hi", "e_lo_x2", "e_hi_x2", "e_step_x2", "qq_order"}`
/// Returns JSON: `{"entries": [{"m_ext", "e_ext_x2", "series": {key: val}}], "r"}`
#[unsafe(no_mangle)]
pub extern "C" fn iref3d_compute_index(json_in: *const c_char) -> *mut c_char {
    let s = match parse_json_arg(json_in) {
        Ok(s) => s,
        Err(e) => return e,
    };

    #[derive(Deserialize)]
    struct Args {
        m_lo: i64,
        m_hi: i64,
        e_lo_x2: i64,
        e_hi_x2: i64,
        e_step_x2: i64,
        qq_order: i32,
    }
    let args: Args = match serde_json::from_str(s) {
        Ok(a) => a,
        Err(e) => return json_err(&format!("parse: {e}")),
    };

    let guard = lock_current();
    let loaded = match guard.as_ref() {
        Some(l) => l,
        None => return json_err("no manifold loaded"),
    };

    let n_cusps = loaded.nz.r;
    let num_hard = loaded.nz.num_hard;

    #[derive(Serialize)]
    struct EntryOut {
        m_ext: Vec<i64>,
        e_ext_x2: Vec<i64>,
        /// Keys are JSON-encoded Vec<i64> (e.g. "[0,2]"), values are coefficients.
        series: std::collections::BTreeMap<String, i64>,
    }

    let mut entries = Vec::new();
    for m in args.m_lo..=args.m_hi {
        let mut e_x2 = args.e_lo_x2;
        while e_x2 <= args.e_hi_x2 {
            let mut m_ext = vec![0i64; n_cusps];
            let mut e_ext_x2 = vec![0i64; n_cusps];
            m_ext[0] = m;
            e_ext_x2[0] = e_x2;

            let result = crate::index_refined::compute_refined_index(
                &loaded.state, num_hard, &m_ext, &e_ext_x2, args.qq_order,
            );

            let series: std::collections::BTreeMap<String, i64> = result
                .into_iter()
                .filter(|(_, v)| *v != 0)
                .map(|(k, v)| (serde_json::to_string(&k).unwrap(), v))
                .collect();

            entries.push(EntryOut { m_ext, e_ext_x2, series });
            e_x2 += args.e_step_x2;
        }
    }

    #[derive(Serialize)]
    struct Out {
        entries: Vec<EntryOut>,
        r: usize,
    }

    json_ok(&Out { entries, r: n_cusps })
}

/// Find non-closable cycles for one cusp.
///
/// Input JSON: `{"cusp_idx", "p_min", "p_max", "q_min", "q_max", "qq_order"}`
/// Returns JSON: `{"cycles": [{"p", "q"}], "slopes_tested"}`
#[unsafe(no_mangle)]
pub extern "C" fn iref3d_find_nc_cycles(json_in: *const c_char) -> *mut c_char {
    let s = match parse_json_arg(json_in) {
        Ok(s) => s,
        Err(e) => return e,
    };

    #[derive(Deserialize)]
    struct Args {
        cusp_idx: usize,
        p_min: i64,
        p_max: i64,
        q_min: i64,
        q_max: i64,
        qq_order: i32,
    }
    let args: Args = match serde_json::from_str(s) {
        Ok(a) => a,
        Err(e) => return json_err(&format!("parse: {e}")),
    };

    let result = panic::catch_unwind(|| {
        let guard = lock_current();
        let loaded = match guard.as_ref() {
            Some(l) => l,
            None => return Err("no manifold loaded".to_string()),
        };

        let n_cusps = loaded.nz.r;
        let m_other = vec![0i64; n_cusps - 1];
        let e_other_x2 = vec![0i64; n_cusps - 1];

        let nc = crate::dehn::nc_search::find_non_closable_cycles(
            &loaded.state,
            args.cusp_idx,
            args.p_min..args.p_max,
            args.q_min..args.q_max,
            &m_other,
            &e_other_x2,
            args.qq_order,
            true,
        );
        Ok(nc)
    });

    let nc = match result {
        Ok(Ok(nc)) => nc,
        Ok(Err(e)) => return json_err(&e),
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            return json_err(&format!("panic in nc_search: {msg}"));
        }
    };

    #[derive(Serialize)]
    struct CycleOut { p: i64, q: i64 }
    #[derive(Serialize)]
    struct Out { cycles: Vec<CycleOut>, slopes_tested: usize }

    let out = Out {
        cycles: nc.cycles.iter().map(|c| CycleOut { p: c.p, q: c.q }).collect(),
        slopes_tested: nc.slopes_tested.len(),
    };
    json_ok(&out)
}

/// NC compatibility check for a single cycle.
///
/// Input JSON: `{"cusp_idx", "p", "q", "qq_order"}`
/// Returns JSON with Weyl (a,b), symmetry, adjoint, marginal info.
#[unsafe(no_mangle)]
pub extern "C" fn iref3d_nc_compat(json_in: *const c_char) -> *mut c_char {
    let s = match parse_json_arg(json_in) {
        Ok(s) => s,
        Err(e) => return e,
    };

    #[derive(Deserialize)]
    struct Args {
        cusp_idx: usize,
        p: i64,
        q: i64,
        qq_order: i32,
    }
    let args: Args = match serde_json::from_str(s) {
        Ok(a) => a,
        Err(e) => return json_err(&format!("parse: {e}")),
    };

    let result = panic::catch_unwind(|| {
        let guard = lock_current();
        let loaded = match guard.as_ref() {
            Some(l) => l,
            None => return Err("no manifold loaded".to_string()),
        };

        let num_hard = loaded.nz.num_hard;
        let r = crate::refined_dehn::nc_compat::check_nc_compat(
            &loaded.nz, args.p, args.q, args.cusp_idx, num_hard, args.qq_order,
        );
        Ok(r)
    });

    let compat = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return json_err(&e),
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            return json_err(&format!("panic in nc_compat: {msg}"));
        }
    };

    #[derive(Serialize)]
    struct Out {
        ab_valid: bool,
        collapsed_edges: Vec<usize>,
        all_weyl_symmetric: bool,
        adjoint_pass: Option<bool>,
        adjoint_value: Option<i64>,
        is_marginal: bool,
        unrefined_q1_proj: i64,
    }

    let out = Out {
        ab_valid: compat.ab.as_ref().map(|a| a.is_valid()).unwrap_or(false),
        collapsed_edges: compat.collapsed_edges,
        all_weyl_symmetric: compat.all_weyl_symmetric,
        adjoint_pass: compat.adjoint.as_ref().map(|a| a.is_pass),
        adjoint_value: compat.adjoint.as_ref().and_then(|a| a.projected_value),
        is_marginal: compat.marginal.is_marginal.unwrap_or(false),
        unrefined_q1_proj: compat.marginal.unrefined_q1_proj.unwrap_or(0),
    };
    json_ok(&out)
}

/// Compute refined filled index (single cusp, auto-selects path).
///
/// Input JSON: `{"cusp_idx", "p", "q", "qq_order", "incompat_edges": []}`
/// Returns JSON with series, HJ-CF, etc.
#[unsafe(no_mangle)]
pub extern "C" fn iref3d_filled_refined_index(json_in: *const c_char) -> *mut c_char {
    let s = match parse_json_arg(json_in) {
        Ok(s) => s,
        Err(e) => return e,
    };

    #[derive(Deserialize)]
    struct Args {
        cusp_idx: usize,
        p: i64,
        q: i64,
        qq_order: i32,
        #[serde(default)]
        incompat_edges: Vec<usize>,
    }
    let args: Args = match serde_json::from_str(s) {
        Ok(a) => a,
        Err(e) => return json_err(&format!("parse: {e}")),
    };

    let result = panic::catch_unwind(|| {
        let guard = lock_current();
        let loaded = match guard.as_ref() {
            Some(l) => l,
            None => return Err("no manifold loaded".to_string()),
        };

        let n_cusps = loaded.nz.r;
        let num_hard = loaded.nz.num_hard;
        let m_other = vec![0i64; n_cusps - 1];
        let e_other_x2 = vec![0i64; n_cusps - 1];

        let hj_ks = crate::refined_dehn::hj_cf::hj_continued_fraction(args.p, args.q);
        let r = if hj_ks.len() >= 2 {
            crate::refined_dehn::is_chain::compute_filled_refined_index_chain(
                &loaded.state, num_hard, args.cusp_idx, args.p, args.q,
                &m_other, &e_other_x2, args.qq_order,
                &args.incompat_edges, None, None,
            )
        } else {
            crate::refined_dehn::unrefined_kernel_path::compute_unrefined_kernel_refined_index(
                &loaded.state, num_hard, args.cusp_idx, args.p, args.q,
                &m_other, &e_other_x2, args.qq_order,
                &args.incompat_edges, None, None,
            )
        };
        Ok(r)
    });

    let filled = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return json_err(&e),
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            return json_err(&format!("panic in filled_refined_index: {msg}"));
        }
    };

    #[derive(Serialize)]
    struct Out {
        series: std::collections::BTreeMap<String, (i64, i64)>,
        hj_ks: Vec<i64>,
        has_cusp_eta: bool,
        qq_order: i32,
        n_kernel_terms: usize,
    }

    let series: std::collections::BTreeMap<String, (i64, i64)> = filled.series
        .into_iter()
        .filter(|(_, v)| *v.numer() != 0)
        .map(|(k, v)| (serde_json::to_string(&k).unwrap(), (*v.numer(), *v.denom())))
        .collect();

    let out = Out {
        series,
        hj_ks: filled.hj_ks,
        has_cusp_eta: filled.has_cusp_eta,
        qq_order: filled.qq_order,
        n_kernel_terms: filled.n_kernel_terms,
    };
    json_ok(&out)
}
