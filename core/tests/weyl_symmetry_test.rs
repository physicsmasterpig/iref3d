//! Integration test: check_weyl_symmetry + strip_weyl_monomial match v0.5.

use iref3d_core::ab_vectors::{ABVectors, Entry};
use iref3d_core::index_refined::RefinedIndexResult;
use iref3d_core::weyl_symmetry::{check_weyl_symmetry, strip_weyl_monomial};
use num_rational::Rational64;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Item {
    key: Vec<i64>,
    coeff: i64,
}

#[derive(Deserialize)]
struct Ab {
    a_num: Vec<i64>,
    a_den: Vec<i64>,
    b_num: Vec<i64>,
    b_den: Vec<i64>,
}

#[derive(Deserialize)]
struct SymEntry {
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    ok: bool,
}

#[derive(Deserialize)]
struct StripEntry {
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    centre_num: Vec<i64>,
    centre_den: Vec<i64>,
    stripped: Vec<Item>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    num_hard: usize,
    qq_order: i32,
    ab: Ab,
    symmetry: Vec<SymEntry>,
    strip: Vec<StripEntry>,
}

#[derive(Deserialize)]
struct Goldens {
    cases: Vec<Case>,
}

#[test]
fn weyl_symmetry_matches_v05() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wpath = root
        .join("tests")
        .join("goldens")
        .join("weyl_symmetry")
        .join("results.json");
    let abpath = root
        .join("tests")
        .join("goldens")
        .join("ab_vectors")
        .join("results.json");

    // Load originals (m, e, result) from the ab_vectors goldens.
    #[derive(Deserialize)]
    struct EJ {
        m_ext: Vec<i64>,
        e_ext_x2: Vec<i64>,
        items: Vec<Item>,
    }
    #[derive(Deserialize)]
    struct AbCase {
        name: String,
        num_hard: usize,
        entries: Vec<EJ>,
    }
    #[derive(Deserialize)]
    struct AbG {
        cases: Vec<AbCase>,
    }
    let ab_g: AbG = serde_json::from_str(&std::fs::read_to_string(&abpath).unwrap()).unwrap();

    let g: Goldens =
        serde_json::from_str(&std::fs::read_to_string(&wpath).expect("read")).expect("parse");

    for case in &g.cases {
        // Build entries from ab_vectors goldens using the name.
        let ab_case = ab_g
            .cases
            .iter()
            .find(|c| c.name == case.name && c.num_hard == case.num_hard)
            .unwrap_or_else(|| panic!("no ab entries for {}", case.name));

        let entries: Vec<Entry> = ab_case
            .entries
            .iter()
            .map(|e| {
                let mut result: RefinedIndexResult = hashbrown::HashMap::new();
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

        let num_hard = case.num_hard;
        let ab = ABVectors {
            a: (0..num_hard)
                .map(|j| Rational64::new(case.ab.a_num[j], case.ab.a_den[j]))
                .collect(),
            b: (0..num_hard)
                .map(|j| Rational64::new(case.ab.b_num[j], case.ab.b_den[j]))
                .collect(),
            num_hard,
            warnings: vec![],
        };

        // Check symmetry.
        let sym = check_weyl_symmetry(&entries, num_hard, &ab, Some(case.qq_order));
        for s in &case.symmetry {
            let key = (s.m_ext.clone(), s.e_ext_x2.clone());
            let got = *sym.get(&key).unwrap_or(&false);
            assert_eq!(got, s.ok, "{}: symmetry at {:?}", case.name, key);
        }

        // Check strip.
        for s in &case.strip {
            let entry = entries
                .iter()
                .find(|e| e.m_ext == s.m_ext && e.e_ext_x2 == s.e_ext_x2)
                .expect("entry");
            let (centre, stripped) =
                strip_weyl_monomial(&entry.result, &s.m_ext, &s.e_ext_x2, &ab, num_hard);
            assert_eq!(centre.len(), num_hard);
            for j in 0..num_hard {
                let expected = Rational64::new(s.centre_num[j], s.centre_den[j]);
                assert_eq!(centre[j], expected, "{}: centre[{}]", case.name, j);
            }
            let stripped_nonzero: usize = stripped.values().filter(|&&v| v != 0).count();
            assert_eq!(
                stripped_nonzero,
                s.stripped.len(),
                "{}: strip size at m={:?} e={:?}",
                case.name,
                s.m_ext,
                s.e_ext_x2
            );
            for item in &s.stripped {
                let got = stripped.get(&item.key).copied().unwrap_or(0);
                assert_eq!(
                    got, item.coeff,
                    "{}: stripped[{:?}] at m={:?} e={:?}",
                    case.name, item.key, s.m_ext, s.e_ext_x2
                );
            }
        }
    }
}
