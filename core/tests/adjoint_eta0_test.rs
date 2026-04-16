//! Integration test: adjoint_eta0 matches v0.5 check_adjoint_projection.

use iref3d_core::ab_vectors::{ABVectors, Entry};
use iref3d_core::adjoint_eta0::check_adjoint_projection;
use num_rational::Rational64;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Item {
    key: Vec<i64>,
    coeff: i64,
}

#[derive(Deserialize)]
struct AbJson {
    a_num: Vec<i64>,
    a_den: Vec<i64>,
    b_num: Vec<i64>,
    b_den: Vec<i64>,
}

#[derive(Deserialize)]
struct EntryJson {
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    items: Vec<Item>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    num_hard: usize,
    cusp_idx: usize,
    ab: AbJson,
    entries: Vec<EntryJson>,
    c_e_x2: Vec<[i64; 2]>,
    projected_value: Option<i64>,
    is_pass: bool,
    missing_e_x2: Vec<i64>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn adjoint_eta0_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let gpath = root
        .join("tests")
        .join("goldens")
        .join("adjoint_eta0")
        .join("results.json");

    let g: Goldens = serde_json::from_str(&std::fs::read_to_string(&gpath).unwrap()).unwrap();

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
        let ab = ABVectors {
            a: (0..case.num_hard)
                .map(|j| Rational64::new(case.ab.a_num[j], case.ab.a_den[j]))
                .collect(),
            b: (0..case.num_hard)
                .map(|j| Rational64::new(case.ab.b_num[j], case.ab.b_den[j]))
                .collect(),
            num_hard: case.num_hard,
            warnings: vec![],
        };
        let got = check_adjoint_projection(&entries, case.num_hard, Some(&ab), case.cusp_idx);
        let mut got_ce = got.c_e_x2.clone();
        got_ce.sort();
        let want_ce: Vec<(i64, i64)> = case.c_e_x2.iter().map(|p| (p[0], p[1])).collect();
        assert_eq!(got_ce, want_ce, "{}: c_e", case.name);
        assert_eq!(got.projected_value, case.projected_value, "{}: proj", case.name);
        assert_eq!(got.is_pass, case.is_pass, "{}: pass", case.name);
        let mut got_missing = got.missing_e_x2.clone();
        got_missing.sort();
        let mut want_missing = case.missing_e_x2.clone();
        want_missing.sort();
        assert_eq!(got_missing, want_missing, "{}: missing", case.name);
    }
}
