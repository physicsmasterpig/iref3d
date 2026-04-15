//! Integration test: NZ BLOB decode + apply_cusp_basis_change vs v0.5 goldens.

use iref3d_core::{census::Census, nz};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize, Debug)]
struct NzDump {
    n: usize,
    r: usize,
    num_hard: usize,
    num_easy: usize,
    g_nz_x2: Vec<i64>,
    nu_x: Vec<i64>,
    nu_p_x2: Vec<i64>,
}

#[derive(Deserialize)]
struct BasisChange {
    cusp_idx: usize,
    #[serde(rename = "P")]
    p: i32,
    #[serde(rename = "Q")]
    q: i32,
    result: NzDump,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    base: NzDump,
    basis_changes: Vec<BasisChange>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

fn assert_nz_eq(got: &iref3d_core::census::NzData, want: &NzDump, ctx: &str) {
    assert_eq!(got.n, want.n, "{ctx}: n");
    assert_eq!(got.r, want.r, "{ctx}: r");
    assert_eq!(got.num_hard, want.num_hard, "{ctx}: num_hard");
    assert_eq!(got.num_easy, want.num_easy, "{ctx}: num_easy");
    assert_eq!(got.g_nz_x2, want.g_nz_x2, "{ctx}: g_nz_x2");
    assert_eq!(got.nu_x, want.nu_x, "{ctx}: nu_x");
    assert_eq!(got.nu_p_x2, want.nu_p_x2, "{ctx}: nu_p_x2");
}

#[test]
fn nz_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens_path = root
        .join("tests")
        .join("goldens")
        .join("nz")
        .join("basis_changes.json");

    let goldens: Goldens = serde_json::from_str(
        &std::fs::read_to_string(&goldens_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", goldens_path.display())),
    )
    .expect("parse goldens");

    let census = Census::open(&db).unwrap();

    for case in &goldens.cases {
        let md = census.load(&case.name).unwrap();
        let base = census
            .load_nz(&md.census, &md.name)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_nz_eq(&base, &case.base, &format!("{}: base", case.name));

        for bc in &case.basis_changes {
            let got = nz::apply_cusp_basis_change(&base, bc.cusp_idx, bc.p, bc.q)
                .unwrap_or_else(|e| {
                    panic!(
                        "{}: apply_cusp_basis_change(k={}, P={}, Q={}): {e}",
                        case.name, bc.cusp_idx, bc.p, bc.q
                    )
                });
            let ctx = format!(
                "{}: basis_change(k={}, P={}, Q={})",
                case.name, bc.cusp_idx, bc.p, bc.q
            );
            assert_nz_eq(&got, &bc.result, &ctx);
        }
    }
}
