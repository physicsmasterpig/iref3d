//! Integration test: nc_compat matches v0.5 NcCompatWorker.

use iref3d_core::{
    census::Census,
    refined_dehn::nc_compat::check_nc_compat,
};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct AbJson {
    a_num: Vec<i64>,
    a_den: Vec<i64>,
    b_num: Vec<i64>,
    b_den: Vec<i64>,
}

#[derive(Deserialize)]
struct CaseJson {
    name: String,
    #[allow(non_snake_case)]
    P: i64,
    #[allow(non_snake_case)]
    Q: i64,
    #[allow(dead_code)]
    #[allow(non_snake_case)]
    R: i64,
    #[allow(dead_code)]
    #[allow(non_snake_case)]
    S: i64,
    cusp_idx: usize,
    num_hard: usize,
    qq_order: i32,
    ab_valid: bool,
    ab: Option<AbJson>,
    collapsed_edges: Vec<usize>,
    adjoint_pass: Option<bool>,
    adjoint_value: Option<i64>,
    c_e_x2_adj: Vec<[i64; 2]>,
    missing_adj: Vec<i64>,
    marginal_proj: Option<i64>,
    is_marginal: Option<bool>,
    #[allow(dead_code)]
    c_e_x2_marg: Vec<[i64; 2]>,
}

#[derive(Deserialize)]
struct GoldensJson {
    cases: Vec<CaseJson>,
}

#[test]
fn nc_compat_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let db = root.join("tests").join("fixtures").join("census_fixture.db");
    let path = root
        .join("tests")
        .join("goldens")
        .join("refined_dehn")
        .join("nc_compat.json");
    let g: GoldensJson =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    let census = Census::open(&db).unwrap();

    for case in &g.cases {
        let ctx = format!("{}: {}/{}", case.name, case.P, case.Q);
        println!("Testing {ctx}");

        let md = census.load(&case.name).unwrap();
        let base = census.load_nz(&md.census, &md.name).unwrap();

        let got = check_nc_compat(
            &base,
            case.P,
            case.Q,
            case.cusp_idx,
            case.num_hard,
            case.qq_order,
        );

        // Check ab vectors
        let got_ab_valid = got.ab.as_ref().map(|a| a.is_valid()).unwrap_or(false);
        assert_eq!(got_ab_valid, case.ab_valid, "{ctx}: ab_valid");

        if let (Some(ab_got), Some(ab_want)) = (&got.ab, &case.ab) {
            use num_rational::Rational64;
            for (j, ((&an, &ad), (gn, gd))) in ab_want
                .a_num
                .iter()
                .zip(ab_want.a_den.iter())
                .zip(ab_got.a.iter().map(|v| (*v.numer(), *v.denom())))
                .enumerate()
            {
                let want = Rational64::new(an, ad);
                let got_v = Rational64::new(gn, gd);
                assert_eq!(got_v, want, "{ctx}: a[{j}]");
            }
            for (j, ((&bn, &bd), (gn, gd))) in ab_want
                .b_num
                .iter()
                .zip(ab_want.b_den.iter())
                .zip(ab_got.b.iter().map(|v| (*v.numer(), *v.denom())))
                .enumerate()
            {
                let want = Rational64::new(bn, bd);
                let got_v = Rational64::new(gn, gd);
                assert_eq!(got_v, want, "{ctx}: b[{j}]");
            }
        }

        // Check collapsed edges
        assert_eq!(got.collapsed_edges, case.collapsed_edges, "{ctx}: collapsed_edges");

        // Check adjoint
        if let Some(ref adj) = got.adjoint {
            assert_eq!(
                adj.is_pass, case.adjoint_pass.unwrap_or(false),
                "{ctx}: adjoint_pass"
            );
            assert_eq!(
                adj.projected_value, case.adjoint_value,
                "{ctx}: adjoint_value"
            );
            let mut got_ce = adj.c_e_x2.clone();
            got_ce.sort();
            let want_ce: Vec<(i64, i64)> = case.c_e_x2_adj.iter().map(|p| (p[0], p[1])).collect();
            assert_eq!(got_ce, want_ce, "{ctx}: c_e_x2_adj");
            let mut got_miss = adj.missing_e_x2.clone();
            got_miss.sort();
            assert_eq!(got_miss, case.missing_adj, "{ctx}: missing_adj");
        } else {
            assert!(
                case.adjoint_pass.is_none(),
                "{ctx}: expected adjoint result but got None"
            );
        }

        // Check marginal
        assert_eq!(
            got.marginal.unrefined_q1_proj, case.marginal_proj,
            "{ctx}: marginal_proj"
        );
        assert_eq!(
            got.marginal.is_marginal, case.is_marginal,
            "{ctx}: is_marginal"
        );
    }
}
