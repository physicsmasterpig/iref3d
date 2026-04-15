//! Integration test: compute_refined_index matches v0.5.

use iref3d_core::{
    census::Census,
    index_refined::compute_refined_index,
    summation::EnumerationState,
};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Item {
    key: Vec<i64>,
    coeff: i64,
}

#[derive(Deserialize)]
struct Query {
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    qq_order: i32,
    num_hard: usize,
    items: Vec<Item>,
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
fn compute_refined_index_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens = root
        .join("tests")
        .join("goldens")
        .join("index_refined")
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
            assert_eq!(q.num_hard, base.num_hard, "{}: num_hard", case.name);
            let got =
                compute_refined_index(&state, q.num_hard, &q.m_ext, &q.e_ext_x2, q.qq_order);
            let ctx = format!(
                "{}: m={:?} e_x2={:?} qq={}",
                case.name, q.m_ext, q.e_ext_x2, q.qq_order
            );
            assert_eq!(got.len(), q.items.len(), "{ctx}: num items");
            for item in &q.items {
                let got_v = got.get(&item.key).copied().unwrap_or_else(|| {
                    panic!("{ctx}: missing key {:?}", item.key)
                });
                assert_eq!(got_v, item.coeff, "{ctx}: key {:?}", item.key);
            }
        }
    }
}
