//! Integration test: adjoint_w_scan matches v0.5 scan_w_vectors.

use iref3d_core::ab_vectors::{ABVectors, Entry};
use iref3d_core::adjoint_w_scan::scan_w_vectors;
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
struct AbJson {
    a_num: Vec<i64>,
    a_den: Vec<i64>,
    b_num: Vec<i64>,
    b_den: Vec<i64>,
}

#[derive(Deserialize)]
struct AdjJson {
    projected_value: Option<i64>,
    is_pass: bool,
    c_e_x2: Vec<[i64; 2]>,
    missing_e_x2: Vec<i64>,
}

#[derive(Deserialize)]
struct ScanEntry {
    w: Vec<i32>,
    a_eff_num: i64,
    a_eff_den: i64,
    b_eff_num: i64,
    b_eff_den: i64,
    a_eff_is_integer: bool,
    adjoint: Option<AdjJson>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    num_hard: usize,
    cusp_idx: usize,
    max_coeff: i32,
    ab: AbJson,
    entries_ref: Vec<EntryJson>,
    scan: Vec<ScanEntry>,
    passing_count: usize,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn adjoint_w_scan_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root
        .join("tests")
        .join("goldens")
        .join("adjoint_w_scan")
        .join("results.json");
    let g: Goldens = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    for case in &g.cases {
        let entries: Vec<Entry> = case
            .entries_ref
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
        let got = scan_w_vectors(
            &entries,
            case.num_hard,
            &ab,
            case.cusp_idx,
            case.max_coeff,
            false,
        );
        assert_eq!(
            got.entries.len(),
            case.scan.len(),
            "{}: scan len",
            case.name
        );
        assert_eq!(
            got.passing.len(),
            case.passing_count,
            "{}: passing count",
            case.name
        );
        for (gi, wi) in got.entries.iter().zip(case.scan.iter()) {
            assert_eq!(gi.w, wi.w, "{}: w", case.name);
            assert_eq!(
                gi.a_eff,
                Rational64::new(wi.a_eff_num, wi.a_eff_den),
                "{}: a_eff at w={:?}",
                case.name,
                wi.w
            );
            assert_eq!(
                gi.b_eff,
                Rational64::new(wi.b_eff_num, wi.b_eff_den),
                "{}: b_eff at w={:?}",
                case.name,
                wi.w
            );
            assert_eq!(
                gi.a_eff_is_integer, wi.a_eff_is_integer,
                "{}: a_int at w={:?}",
                case.name, wi.w
            );
            match (&gi.adjoint, &wi.adjoint) {
                (None, None) => {}
                (Some(ga), Some(wa)) => {
                    assert_eq!(
                        ga.projected_value, wa.projected_value,
                        "{}: proj at w={:?}",
                        case.name, wi.w
                    );
                    assert_eq!(ga.is_pass, wa.is_pass);
                    let mut got_ce = ga.c_e_x2.clone();
                    got_ce.sort();
                    let want_ce: Vec<(i64, i64)> =
                        wa.c_e_x2.iter().map(|p| (p[0], p[1])).collect();
                    assert_eq!(got_ce, want_ce, "{}: c_e at w={:?}", case.name, wi.w);
                }
                _ => panic!("{}: adjoint presence mismatch at w={:?}", case.name, wi.w),
            }
        }
    }
}
