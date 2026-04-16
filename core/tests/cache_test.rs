use iref3d_core::cache::{iref_disk, kernel_disk};
use iref3d_core::dehn::kernel_terms::KernelTerm;

// Tests must run sequentially because they share the process-level IREF3D_CACHE_DIR env var.
// Use #[serial] pattern via a single test that runs both.

#[test]
fn cache_round_trips() {
    kernel_cache_round_trip();
    iref_cache_round_trip();
}

fn kernel_cache_round_trip() {
    let tmp = std::env::temp_dir().join("iref3d_test_cache_rt");
    let _ = std::fs::remove_dir_all(&tmp);
    unsafe { std::env::set_var("IREF3D_CACHE_DIR", &tmp) };

    let terms = vec![
        KernelTerm {
            m: 1,
            e_x2: -2,
            c: 0,
            phase: 3,
            multiplicity: 2,
        },
        KernelTerm {
            m: 0,
            e_x2: 0,
            c: 2,
            phase: 0,
            multiplicity: 1,
        },
    ];

    let entry = kernel_disk::KernelCacheEntry {
        p: 3,
        q: 5,
        qq_order: 10,
        r_val: 2,
        s_val: -1,
        terms: terms.clone(),
    };

    kernel_disk::save(&entry).unwrap();

    // Load it back.
    let loaded = kernel_disk::load(3, 5, 10).unwrap().expect("should find entry");
    assert_eq!(loaded.r_val, 2);
    assert_eq!(loaded.s_val, -1);
    assert_eq!(loaded.terms.len(), 2);
    assert_eq!(loaded.terms[0].m, 1);
    assert_eq!(loaded.terms[1].c, 2);

    // Miss returns None.
    assert!(kernel_disk::load(3, 5, 20).unwrap().is_none());

    // List slopes.
    let slopes = kernel_disk::list_slopes(10).unwrap();
    assert_eq!(slopes, vec![(3, 5)]);

    // Delete.
    assert!(kernel_disk::delete(3, 5, 10).unwrap());
    assert!(kernel_disk::load(3, 5, 10).unwrap().is_none());

    let _ = std::fs::remove_dir_all(&tmp);
}

fn iref_cache_round_trip() {
    let tmp = std::env::temp_dir().join("iref3d_test_cache_rt2");
    let _ = std::fs::remove_dir_all(&tmp);
    unsafe { std::env::set_var("IREF3D_CACHE_DIR", &tmp) };

    let hash = iref_disk::nz_hash(&[1, 2, 3], &[4, 5], &[6, 7]);
    assert_eq!(hash.len(), 16);

    let m_ext = vec![0i64, 1];
    let e_ext = vec![0i64, -2];

    let mut result = hashbrown::HashMap::new();
    result.insert(vec![0i64, 2], 5i64);
    result.insert(vec![1, -1], -3);

    iref_disk::save(&hash, &m_ext, &e_ext, 10, &result).unwrap();

    let loaded = iref_disk::load(&hash, &m_ext, &e_ext, 10)
        .unwrap()
        .expect("should find entry");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.get(&vec![0i64, 2]), Some(&5));
    assert_eq!(loaded.get(&vec![1i64, -1]), Some(&-3));

    // Miss.
    assert!(iref_disk::load(&hash, &m_ext, &e_ext, 20).unwrap().is_none());

    // Count.
    assert_eq!(iref_disk::count(&hash).unwrap(), 1);

    // Clear.
    assert_eq!(iref_disk::clear().unwrap(), 1);
    assert_eq!(iref_disk::count(&hash).unwrap(), 0);

    let _ = std::fs::remove_dir_all(&tmp);
}
