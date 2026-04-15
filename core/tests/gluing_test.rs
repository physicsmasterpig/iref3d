//! Integration test: gluing.rs vs goldens/gluing/reduced.json.

use iref3d_core::{census::Census, gluing::reduce_gluing_equations};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    r: usize,
    edge_coeffs: Vec<i64>,
    edge_consts: Vec<i64>,
    cusp_coeffs: Vec<i64>,
    cusp_consts: Vec<i64>,
    pivots: Vec<usize>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn reduce_gluing_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens_path = root
        .join("tests")
        .join("goldens")
        .join("gluing")
        .join("reduced.json");

    let goldens: Goldens = serde_json::from_str(
        &std::fs::read_to_string(&goldens_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", goldens_path.display())),
    )
    .expect("parse goldens");

    let census = Census::open(&db).unwrap();

    for case in &goldens.cases {
        let md = census
            .load(&case.name)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let rg = reduce_gluing_equations(&md);
        assert_eq!(rg.n, case.n, "{}: n", case.name);
        assert_eq!(rg.r, case.r, "{}: r", case.name);
        assert_eq!(
            rg.edge_coeffs, case.edge_coeffs,
            "{}: edge_coeffs diverge from v0.5",
            case.name
        );
        assert_eq!(rg.edge_consts, case.edge_consts, "{}: edge_consts", case.name);
        assert_eq!(rg.cusp_coeffs, case.cusp_coeffs, "{}: cusp_coeffs", case.name);
        assert_eq!(rg.cusp_consts, case.cusp_consts, "{}: cusp_consts", case.name);
        assert_eq!(rg.pivots, case.pivots, "{}: pivots", case.name);
    }
}
