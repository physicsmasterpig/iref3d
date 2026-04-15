//! Integration test: compute_unrefined_index matches v0.5 compute_index_3d_python.

use iref3d_core::{
    census::Census,
    index_unrefined::compute_unrefined_index,
    summation::EnumerationState,
};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Query {
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    qq_order: i32,
    coeffs: Vec<i64>,
    min_power: i32,
    n_terms: usize,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn compute_unrefined_index_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens = root
        .join("tests")
        .join("goldens")
        .join("index_unrefined")
        .join("results.json");
    let g: Goldens = serde_json::from_str(
        &std::fs::read_to_string(&goldens).expect("read goldens"),
    )
    .expect("parse goldens");

    let census = Census::open(&db).unwrap();

    for case in &g.cases {
        let md = census.load(&case.name).unwrap();
        let base = census.load_nz(&md.census, &md.name).unwrap();
        let state = EnumerationState::build(&base);

        for q in &case.queries {
            let res = compute_unrefined_index(&state, &q.m_ext, &q.e_ext_x2, q.qq_order);
            let ctx = format!(
                "{}: m={:?} e_x2={:?} qq={}",
                case.name, q.m_ext, q.e_ext_x2, q.qq_order
            );
            assert_eq!(res.n_terms, q.n_terms, "{ctx}: n_terms");
            assert_eq!(res.min_power, q.min_power, "{ctx}: min_power");
            assert_eq!(res.coeffs, q.coeffs, "{ctx}: coeffs");
        }
    }
}
