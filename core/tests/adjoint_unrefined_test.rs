//! Integration test: adjoint_unrefined matches v0.5 marginal check.

use iref3d_core::{
    adjoint_unrefined::check_marginal, census::Census, summation::EnumerationState,
};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct PerCusp {
    cusp_idx: usize,
    c_e_x2: Vec<(i64, i64)>,
    unrefined_q1_proj: Option<i64>,
    is_marginal: Option<bool>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    num_cusps: usize,
    qq_order: i32,
    per_cusp: Vec<PerCusp>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn adjoint_unrefined_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let path = root
        .join("tests")
        .join("goldens")
        .join("adjoint_unrefined")
        .join("results.json");
    let g: Goldens =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");

    let census = Census::open(&db).unwrap();
    for case in &g.cases {
        let md = match census.load(&case.name) {
            Ok(md) => md,
            Err(_) => continue,
        };
        let base = census.load_nz(&md.census, &md.name).unwrap();
        let state = EnumerationState::build(&base);

        for pc in &case.per_cusp {
            let got = check_marginal(&state, case.num_cusps, pc.cusp_idx, case.qq_order);
            let mut got_ce: Vec<(i64, i64)> = got.c_e_x2.clone();
            got_ce.sort();
            let mut want_ce = pc.c_e_x2.clone();
            want_ce.sort();
            assert_eq!(got_ce, want_ce, "{}: cusp {} c_e", case.name, pc.cusp_idx);
            assert_eq!(
                got.unrefined_q1_proj, pc.unrefined_q1_proj,
                "{}: cusp {} proj",
                case.name, pc.cusp_idx
            );
            assert_eq!(
                got.is_marginal, pc.is_marginal,
                "{}: cusp {} marginal",
                case.name, pc.cusp_idx
            );
        }
    }
}
