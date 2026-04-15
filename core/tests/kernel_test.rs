//! Integration tests: kernel.rs + poly.rs vs v0.5 goldens.

use iref3d_core::{kernel, poly};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct TetSeriesCase {
    m: i32,
    e: i32,
    qq_order: i32,
    series: Vec<[i64; 2]>,
}

#[derive(Deserialize)]
struct TetSeriesFile {
    cases: Vec<TetSeriesCase>,
}

#[derive(Deserialize)]
struct DegCase {
    m: i32,
    e: i32,
    degree_x2: i32,
}

#[derive(Deserialize)]
struct DegFile {
    cases: Vec<DegCase>,
}

#[derive(Deserialize)]
struct ConvCase {
    lhs: Vec<[i64; 2]>,
    rhs: Vec<[i64; 2]>,
    budget: i32,
    result: Vec<[i64; 2]>,
}

#[derive(Deserialize)]
struct ConvFile {
    cases: Vec<ConvCase>,
}

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
        .join("kernel")
}

fn read_json<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = goldens_dir().join(name);
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&txt).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

#[test]
fn tet_degree_x2_matches_v05() {
    let file: DegFile = read_json("tet_degree_x2.json");
    for c in &file.cases {
        let got = kernel::tet_degree_x2(c.m, c.e);
        assert_eq!(got, c.degree_x2, "tet_degree_x2({}, {})", c.m, c.e);
    }
}

#[test]
fn tet_index_series_matches_v05() {
    let file: TetSeriesFile = read_json("tet_index_series.json");
    for c in &file.cases {
        let got = kernel::tet_index_series(c.m, c.e, c.qq_order);
        let got_pairs = poly::to_sorted_pairs(&got);
        assert_eq!(
            got_pairs, c.series,
            "tet_index_series({}, {}, {}) diverged",
            c.m, c.e, c.qq_order
        );
    }
}

#[test]
fn poly_convolve_matches_v05() {
    let file: ConvFile = read_json("poly_convolve.json");
    for (i, c) in file.cases.iter().enumerate() {
        let lhs = poly::from_pairs(&c.lhs);
        let rhs = poly::from_pairs(&c.rhs);
        let got = poly::convolve(&lhs, &rhs, c.budget);
        let got_pairs = poly::to_sorted_pairs(&got);
        assert_eq!(
            got_pairs, c.result,
            "poly_convolve case {i} budget={} diverged",
            c.budget
        );
    }
}
