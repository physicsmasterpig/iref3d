// Drive the Swift-facing FFI directly for debugging.
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn call(name: &str, json: &str, f: unsafe extern "C" fn(*const c_char) -> *mut c_char) {
    let c = CString::new(json).unwrap();
    let out_ptr = unsafe { f(c.as_ptr()) };
    let s = unsafe { CStr::from_ptr(out_ptr).to_string_lossy().into_owned() };
    unsafe { iref3d_core::swift_ffi::iref3d_free_string(out_ptr) };
    println!("=== {name} ===\n{}\n", s);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let db = args.get(1).cloned()
        .unwrap_or("/Users/pmp/Documents/Research/iref3d/data/census.db".into());
    let name = args.get(2).cloned().unwrap_or("m003".into());
    let p: i64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let q: i64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let qq: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);

    call("load_manifold",
        &format!(r#"{{"db_path":"{db}","name":"{name}"}}"#),
        iref3d_core::swift_ffi::iref3d_load_manifold);

    call("filled_refined_index",
        &format!(r#"{{"cusp_idx":0,"p":{p},"q":{q},"qq_order":{qq},"incompat_edges":[]}}"#),
        iref3d_core::swift_ffi::iref3d_filled_refined_index);
}
