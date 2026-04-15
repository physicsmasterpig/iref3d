//! Integration test: census.rs vs goldens/census/manifolds.json.

use iref3d_core::census::Census;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    r: usize,
    gluing: Vec<i32>,
    pivots: Vec<i32>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points to `core/`.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn load_roundtrip_matches_v05_goldens() {
    let root = repo_root();
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens_path = root
        .join("tests")
        .join("goldens")
        .join("census")
        .join("manifolds.json");

    let goldens: Goldens = serde_json::from_str(
        &std::fs::read_to_string(&goldens_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", goldens_path.display())),
    )
    .expect("parse goldens");

    let census = Census::open(&db)
        .unwrap_or_else(|e| panic!("open {}: {e}", db.display()));

    assert_eq!(census.count().unwrap(), goldens.cases.len() as i64);

    for case in &goldens.cases {
        let m = census
            .load(&case.name)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(m.n, case.n, "{}: n mismatch", case.name);
        assert_eq!(m.r, case.r, "{}: r mismatch", case.name);
        assert_eq!(
            m.gluing, case.gluing,
            "{}: gluing bytes diverge from v0.5",
            case.name
        );
        assert_eq!(m.gluing.len(), (m.n + 2 * m.r) * 3 * m.n);
        assert_eq!(
            m.pivots, case.pivots,
            "{}: scipy pivot indices diverge from v0.5",
            case.name
        );
        assert_eq!(m.pivots.len(), m.n - m.r);
    }
}

#[test]
fn load_in_with_explicit_census() {
    let root = repo_root();
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let census = Census::open(&db).unwrap();
    let m = census.load_in("orientable_cusped", "m003").unwrap();
    assert_eq!(m.n, 2);
    assert_eq!(m.r, 1);
    assert_eq!(m.gluing.len(), (2 + 2) * 6);
}

#[test]
fn missing_name_is_notfound() {
    let root = repo_root();
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let census = Census::open(&db).unwrap();
    match census.load("does_not_exist_xyz") {
        Err(iref3d_core::census::CensusError::NotFound(_)) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}
