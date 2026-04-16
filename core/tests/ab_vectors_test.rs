//! Integration test: compute_ab_vectors matches v0.5 (scalar / single-cusp).

use iref3d_core::ab_vectors::{compute_ab_vectors, Entry};
use num_rational::Rational64;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Item {
    key: Vec<i64>,
    coeff: i64,
}

#[derive(Deserialize)]
struct EntryJson {
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    items: Vec<Item>,
}

#[derive(Deserialize)]
struct Ab {
    a_num: Vec<i64>,
    a_den: Vec<i64>,
    b_num: Vec<i64>,
    b_den: Vec<i64>,
    num_hard: usize,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    num_hard: usize,
    entries: Vec<EntryJson>,
    ab: Option<Ab>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn compute_ab_vectors_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root
        .join("tests")
        .join("goldens")
        .join("ab_vectors")
        .join("results.json");
    let g: Goldens =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read goldens"))
            .expect("parse");

    for case in &g.cases {
        let entries: Vec<Entry> = case
            .entries
            .iter()
            .map(|e| {
                let mut result = hashbrown::HashMap::new();
                for it in &e.items {
                    result.insert(it.key.clone(), it.coeff);
                }
                Entry {
                    m_ext: e.m_ext.clone(),
                    e_ext_x2: e.e_ext_x2.clone(),
                    result,
                }
            })
            .collect();
        let got = compute_ab_vectors(&entries, case.num_hard);
        match (&got, &case.ab) {
            (None, None) => {}
            (Some(g), Some(want)) => {
                assert_eq!(g.num_hard, want.num_hard, "{}: num_hard", case.name);
                for j in 0..g.num_hard {
                    let ea = Rational64::new(want.a_num[j], want.a_den[j]);
                    let eb = Rational64::new(want.b_num[j], want.b_den[j]);
                    assert_eq!(g.a[j], ea, "{}: a[{}]", case.name, j);
                    assert_eq!(g.b[j], eb, "{}: b[{}]", case.name, j);
                }
            }
            _ => panic!("{}: ab presence mismatch", case.name),
        }
    }
}
