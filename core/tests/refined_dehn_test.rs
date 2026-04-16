//! Integration test: refined_dehn (HJ-CF + unrefined kernel on I^ref).

use iref3d_core::{
    census::Census,
    refined_dehn::{
        hj_cf::hj_continued_fraction,
        unrefined_kernel_path::compute_unrefined_kernel_refined_index,
    },
    summation::EnumerationState,
};
use num_rational::Rational64;
use serde::Deserialize;
use std::path::PathBuf;

// ── HJ-CF ──

#[derive(Deserialize)]
struct HjCase {
    p: i64,
    q: i64,
    ks: Vec<i64>,
}

#[derive(Deserialize)]
struct HjGoldens {
    cases: Vec<HjCase>,
}

#[test]
fn hj_cf_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root
        .join("tests")
        .join("goldens")
        .join("refined_dehn")
        .join("hj_cf.json");
    let g: HjGoldens =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for c in &g.cases {
        let got = hj_continued_fraction(c.p, c.q);
        assert_eq!(got, c.ks, "p={} q={}", c.p, c.q);
    }
}

// ── Unrefined kernel on I^ref ──

#[derive(Deserialize)]
struct SeriesItem {
    key: Vec<i64>,
    coeff_num: i64,
    coeff_den: i64,
}

#[derive(Deserialize)]
struct FillJson {
    #[allow(non_snake_case)]
    P: i64,
    #[allow(non_snake_case)]
    Q: i64,
    n_kernel_terms: usize,
    has_cusp_eta: bool,
    series: Vec<SeriesItem>,
}

#[derive(Deserialize)]
struct FillCase {
    name: String,
    num_hard: usize,
    fills: Vec<FillJson>,
}

#[derive(Deserialize)]
struct FillGoldens {
    cases: Vec<FillCase>,
}

#[test]
fn unrefined_kernel_refined_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let path = root
        .join("tests")
        .join("goldens")
        .join("refined_dehn")
        .join("unrefined_kernel.json");
    let g: FillGoldens =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    let census = Census::open(&db).unwrap();

    for case in &g.cases {
        let md = census.load(&case.name).unwrap();
        let base = census.load_nz(&md.census, &md.name).unwrap();
        let state = EnumerationState::build(&base);

        for fill in &case.fills {
            let ctx = format!("{}: {}/{}", case.name, fill.P, fill.Q);
            let got = compute_unrefined_kernel_refined_index(
                &state,
                case.num_hard,
                0,
                fill.P,
                fill.Q,
                &[],
                &[],
                10,
                &[],
                None,
                None,
            );
            assert_eq!(got.has_cusp_eta, fill.has_cusp_eta, "{ctx}: has_cusp_eta");
            assert_eq!(
                got.n_kernel_terms, fill.n_kernel_terms,
                "{ctx}: n_kernel_terms"
            );

            // Compare series
            let zero = Rational64::from_integer(0);
            let mut got_items: Vec<(Vec<i64>, Rational64)> = got
                .series
                .into_iter()
                .filter(|(_, v)| *v != zero)
                .collect();
            got_items.sort_by(|a, b| a.0.cmp(&b.0));
            let want_items: Vec<(Vec<i64>, Rational64)> = fill
                .series
                .iter()
                .map(|s| {
                    (
                        s.key.clone(),
                        Rational64::new(s.coeff_num, s.coeff_den),
                    )
                })
                .filter(|(_, v)| *v != zero)
                .collect();
            assert_eq!(
                got_items.len(),
                want_items.len(),
                "{ctx}: series len (got {} want {})",
                got_items.len(),
                want_items.len()
            );
            for (g, w) in got_items.iter().zip(want_items.iter()) {
                assert_eq!(g.0, w.0, "{ctx}: key mismatch");
                assert_eq!(g.1, w.1, "{ctx}: coeff at {:?}", g.0);
            }
        }
    }
}
