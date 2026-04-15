//! Integration test: g_NZ_inv_scaled matches v0.5 after optional basis change.

use iref3d_core::{
    census::Census,
    nz,
    summation::{enumerate_summation_terms, g_nz_inv_scaled, EnumerationState, SummationTerm},
};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct InvDump {
    #[serde(rename = "S")]
    s: i64,
    matrix: Vec<i64>,
}

#[derive(Deserialize)]
struct BasisChange {
    cusp_idx: usize,
    #[serde(rename = "P")]
    p: i32,
    #[serde(rename = "Q")]
    q: i32,
    result: InvDump,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    base: InvDump,
    basis_changes: Vec<BasisChange>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

fn check(nz: &iref3d_core::census::NzData, want: &InvDump, ctx: &str) {
    let (s, m) = g_nz_inv_scaled(nz);
    assert_eq!(s, want.s, "{ctx}: S");
    assert_eq!(m, want.matrix, "{ctx}: matrix");
}

#[test]
fn g_nz_inv_scaled_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens = root
        .join("tests")
        .join("goldens")
        .join("summation")
        .join("g_nz_inv_scaled.json");

    let g: Goldens = serde_json::from_str(
        &std::fs::read_to_string(&goldens)
            .unwrap_or_else(|e| panic!("read {}: {e}", goldens.display())),
    )
    .expect("parse goldens");

    let census = Census::open(&db).unwrap();

    for case in &g.cases {
        let md = census.load(&case.name).unwrap();
        let base = census.load_nz(&md.census, &md.name).unwrap();
        check(&base, &case.base, &format!("{}: base", case.name));

        for bc in &case.basis_changes {
            let n = nz::apply_cusp_basis_change(&base, bc.cusp_idx, bc.p, bc.q).unwrap();
            let ctx = format!(
                "{}: basis_change(k={}, P={}, Q={})",
                case.name, bc.cusp_idx, bc.p, bc.q
            );
            check(&n, &bc.result, &ctx);
        }
    }
}

#[derive(Deserialize)]
struct TermDump {
    e_int_x2: Vec<i64>,
    phase_exp: i64,
    tet_args: Vec<[i64; 2]>,
    min_degree_x2: i64,
}

#[derive(Deserialize)]
struct Query {
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    qq_order: i64,
    terms: Vec<TermDump>,
}

#[derive(Deserialize)]
struct TermsCase {
    name: String,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct TermsGoldens {
    cases: Vec<TermsCase>,
}

fn term_key(t: &SummationTerm) -> (Vec<i64>, i64, Vec<(i64, i64)>) {
    (t.e_int_x2.clone(), t.phase_exp, t.tet_args.clone())
}

#[test]
fn enumerate_summation_terms_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens = root
        .join("tests")
        .join("goldens")
        .join("summation")
        .join("terms.json");
    let g: TermsGoldens = serde_json::from_str(
        &std::fs::read_to_string(&goldens).expect("read terms goldens"),
    )
    .expect("parse terms goldens");

    let census = Census::open(&db).unwrap();

    for case in &g.cases {
        let md = census.load(&case.name).unwrap();
        let base = census.load_nz(&md.census, &md.name).unwrap();
        let state = EnumerationState::build(&base);

        for q in &case.queries {
            let mut got = enumerate_summation_terms(&state, &q.m_ext, &q.e_ext_x2, q.qq_order);
            got.sort_by(|a, b| term_key(a).cmp(&term_key(b)));

            let ctx = format!(
                "{}: m={:?} e_x2={:?} qq={}",
                case.name, q.m_ext, q.e_ext_x2, q.qq_order
            );
            assert_eq!(got.len(), q.terms.len(), "{ctx}: num terms");
            for (i, (gt, wt)) in got.iter().zip(q.terms.iter()).enumerate() {
                assert_eq!(gt.e_int_x2, wt.e_int_x2, "{ctx}[{i}]: e_int_x2");
                assert_eq!(gt.phase_exp, wt.phase_exp, "{ctx}[{i}]: phase_exp");
                let want_args: Vec<(i64, i64)> =
                    wt.tet_args.iter().map(|p| (p[0], p[1])).collect();
                assert_eq!(gt.tet_args, want_args, "{ctx}[{i}]: tet_args");
                assert_eq!(gt.min_degree_x2, wt.min_degree_x2, "{ctx}[{i}]: min_deg_x2");
            }
        }
    }
}
