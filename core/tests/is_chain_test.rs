//! Integration test: IS chain (ℓ≥2 refined Dehn filling).

use iref3d_core::{
    census::Census,
    refined_dehn::is_chain::compute_filled_refined_index_chain,
    summation::EnumerationState,
};
use num_rational::Rational64;
use serde::Deserialize;
use std::path::PathBuf;

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
    qq_order: i32,
    hj_ks: Vec<i64>,
    n_kernel_terms: usize,
    has_cusp_eta: bool,
    series: Vec<SeriesItem>,
}

#[derive(Deserialize)]
struct CaseJson {
    name: String,
    num_hard: usize,
    fills: Vec<FillJson>,
}

#[derive(Deserialize)]
struct GoldensJson {
    cases: Vec<CaseJson>,
}

#[test]
fn is_chain_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let path = root
        .join("tests")
        .join("goldens")
        .join("refined_dehn")
        .join("is_chain.json");
    let g: GoldensJson =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    let census = Census::open(&db).unwrap();

    for case in &g.cases {
        let md = census.load(&case.name).unwrap();
        let base = census.load_nz(&md.census, &md.name).unwrap();
        let state = EnumerationState::build(&base);

        for fill in &case.fills {
            let ctx = format!("{}: {}/{}", case.name, fill.P, fill.Q);
            println!("Testing {ctx} hj_ks={:?}", fill.hj_ks);

            let got = compute_filled_refined_index_chain(
                &state,
                case.num_hard,
                0,
                fill.P,
                fill.Q,
                &[],
                &[],
                fill.qq_order,
                &[],
                None,
                None,
            );

            assert_eq!(got.has_cusp_eta, fill.has_cusp_eta, "{ctx}: has_cusp_eta");
            assert_eq!(got.hj_ks, fill.hj_ks, "{ctx}: hj_ks");

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
