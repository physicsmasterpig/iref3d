//! Integration test: phase_space BLOB decode vs goldens/phase_space/basis.json.

use iref3d_core::census::Census;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Case {
    name: String,
    n: usize,
    r: usize,
    easy_edges: Vec<Vec<i32>>,
    easy_indep: Vec<i32>,
    hard_padding: Vec<Vec<i32>>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn phase_space_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let goldens_path = root
        .join("tests")
        .join("goldens")
        .join("phase_space")
        .join("basis.json");

    let goldens: Goldens = serde_json::from_str(
        &std::fs::read_to_string(&goldens_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", goldens_path.display())),
    )
    .expect("parse goldens");

    let census = Census::open(&db).unwrap();

    for case in &goldens.cases {
        let md = census.load(&case.name).unwrap();
        let ps = census
            .load_phase_space(&md.census, &md.name, md.n)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(md.n, case.n);
        assert_eq!(md.r, case.r);
        assert_eq!(
            ps.num_easy(),
            case.easy_edges.len(),
            "{}: easy count",
            case.name
        );
        for (i, golden_row) in case.easy_edges.iter().enumerate() {
            assert_eq!(ps.easy_row(i), golden_row.as_slice(), "{}: easy row {i}", case.name);
        }
        assert_eq!(ps.easy_indep, case.easy_indep, "{}: indep", case.name);
        assert_eq!(
            ps.num_hard(),
            case.hard_padding.len(),
            "{}: hard count",
            case.name
        );
        for (i, golden_row) in case.hard_padding.iter().enumerate() {
            assert_eq!(ps.hard_row(i), golden_row.as_slice(), "{}: hard row {i}", case.name);
        }
    }
}
