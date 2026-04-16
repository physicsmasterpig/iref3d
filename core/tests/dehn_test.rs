//! Integration test: dehn filling (kernel terms + filled index + NC search).

use iref3d_core::{
    census::Census,
    dehn::{
        kernel_terms::{enumerate_kernel_terms, find_rs},
        nc_search::find_non_closable_cycles,
        unrefined_fill::compute_filled_index,
    },
    summation::EnumerationState,
};
use num_rational::Rational64;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct KtJson {
    m: i64,
    e_x2: i64,
    c: i64,
    phase: i64,
    multiplicity: i64,
}

#[derive(Deserialize)]
struct FillJson {
    #[allow(non_snake_case)]
    P: i64,
    #[allow(non_snake_case)]
    Q: i64,
    #[allow(non_snake_case)]
    R: i64,
    #[allow(non_snake_case)]
    S: i64,
    kernel_terms: Vec<KtJson>,
    n_kernel_terms: usize,
    series: Vec<(i64, [i64; 2])>,
    is_stably_zero: bool,
}

#[derive(Deserialize)]
struct NcCycle {
    p: i64,
    q: i64,
}

#[derive(Deserialize)]
struct NcJson {
    cusp_idx: usize,
    cycles: Vec<NcCycle>,
    slopes_tested: Vec<[i64; 2]>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    fills: Vec<FillJson>,
    nc: Option<NcJson>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn dehn_filling_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let path = root
        .join("tests")
        .join("goldens")
        .join("dehn")
        .join("results.json");
    let g: Goldens =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");

    let census = Census::open(&db).unwrap();

    for case in &g.cases {
        let md = census.load(&case.name).unwrap();
        let base = census.load_nz(&md.census, &md.name).unwrap();
        let state = EnumerationState::build(&base);

        for fill in &case.fills {
            let ctx = format!("{}: {}/{}", case.name, fill.P, fill.Q);

            // Verify find_rs
            let (r, s) = find_rs(fill.P, fill.Q);
            assert_eq!(r * fill.Q - fill.P * s, 1, "{ctx}: R·Q - P·S = 1");

            // Kernel terms
            let kts = enumerate_kernel_terms(
                &state,
                fill.P,
                fill.Q,
                fill.R,
                fill.S,
                0,
                &[],
                &[],
                10,
            );
            assert_eq!(
                kts.len(),
                fill.kernel_terms.len(),
                "{ctx}: kernel_terms count"
            );
            // Sort both by (m, e_x2) for stable comparison
            let mut got_kts: Vec<_> = kts.iter().map(|k| (k.m, k.e_x2)).collect();
            got_kts.sort();
            let mut want_kts: Vec<_> = fill
                .kernel_terms
                .iter()
                .map(|k| (k.m, k.e_x2))
                .collect();
            want_kts.sort();
            assert_eq!(got_kts, want_kts, "{ctx}: kernel_terms (m, e_x2)");

            // Filled index
            let filled = compute_filled_index(&state, 0, fill.P, fill.Q, &[], &[], 10);
            assert_eq!(
                filled.n_kernel_terms, fill.n_kernel_terms,
                "{ctx}: n_kernel_terms"
            );
            assert_eq!(
                filled.is_stably_zero(None),
                fill.is_stably_zero,
                "{ctx}: is_stably_zero"
            );

            // Compare series (Rational coefficients)
            let mut got_series: Vec<(i64, Rational64)> = filled
                .series
                .iter()
                .filter(|(_, v)| **v != Rational64::from_integer(0))
                .map(|(&k, &v)| (k, v))
                .collect();
            got_series.sort_by_key(|&(k, _)| k);
            let want_series: Vec<(i64, Rational64)> = fill
                .series
                .iter()
                .map(|&(k, [n, d])| (k, Rational64::new(n, d)))
                .filter(|(_, v)| *v != Rational64::from_integer(0))
                .collect();
            assert_eq!(got_series, want_series, "{ctx}: series");
        }

        // NC search
        if let Some(nc) = &case.nc {
            let got = find_non_closable_cycles(
                &state,
                nc.cusp_idx,
                -2..3,
                0..3,
                &[],
                &[],
                10,
                true,
            );
            let mut got_nc: Vec<(i64, i64)> =
                got.cycles.iter().map(|c| (c.p, c.q)).collect();
            got_nc.sort();
            let mut want_nc: Vec<(i64, i64)> =
                nc.cycles.iter().map(|c| (c.p, c.q)).collect();
            want_nc.sort();
            assert_eq!(got_nc, want_nc, "{}: NC cycles", case.name);
        }
    }
}
