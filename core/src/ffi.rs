//! PyO3 bindings for the iref3d-core Rust library.
//!
//! Exposes the full pipeline as a Python extension module: census loading,
//! index computation, Weyl checks, Dehn filling, and NC compatibility.
//!
//! Build with: `cargo build --features python --release`
//!
//! Convention: external charges `e_ext` are always in ×2 (doubled) form
//! to avoid Fraction overhead. Python callers multiply by 2 before calling.

use hashbrown::HashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

type PyObject = Py<PyAny>;

use crate::ab_vectors::{compute_ab_vectors, Entry};
use crate::adjoint_eta0::{
    check_adjoint_projection, check_adjoint_projection_multi_cusp,
};
use crate::adjoint_unrefined::check_marginal;
use crate::census::Census;
use crate::dehn::kernel_terms::find_rs;
use crate::dehn::nc_search::find_non_closable_cycles;
use crate::dehn::unrefined_fill::compute_filled_index;
use crate::index_refined::{compute_refined_index, RefinedIndexResult};
use crate::index_unrefined::compute_unrefined_index;
use crate::nz::{apply_general_cusp_basis_change, NzData};
use crate::refined_dehn::is_chain::compute_filled_refined_index_chain;
use crate::refined_dehn::multi_cusp::{
    compute_multi_cusp_filled_refined_index, MultiCuspFillSpec,
};
use crate::refined_dehn::nc_compat::check_nc_compat;
use crate::refined_dehn::unrefined_kernel_path::compute_unrefined_kernel_refined_index;
use crate::summation::EnumerationState;
use crate::weyl_symmetry::check_weyl_symmetry;

// ── Wrapper types ──

/// Python-visible NZ data handle. Wraps the Rust NzData + precomputed state.
#[pyclass(name = "NzData", skip_from_py_object)]
#[derive(Clone)]
struct PyNzData {
    nz: NzData,
    state: EnumerationState,
}

#[pymethods]
impl PyNzData {
    #[getter]
    fn n(&self) -> usize {
        self.nz.n
    }
    #[getter]
    fn r(&self) -> usize {
        self.nz.r
    }
    #[getter]
    fn num_hard(&self) -> usize {
        self.nz.num_hard
    }
    #[getter]
    fn n_int(&self) -> usize {
        self.state.n_int
    }
}

// ── Helper conversions ──

fn entries_from_py(_py: Python<'_>, entries: &Bound<'_, PyList>) -> PyResult<(Vec<Entry>, usize)> {
    let mut out = Vec::new();
    let mut num_hard = 0usize;
    for item in entries.iter() {
        let tuple = item.cast::<PyTuple>()?;
        let m_ext: Vec<i64> = tuple.get_item(0)?.extract()?;
        let e_ext_x2: Vec<i64> = tuple.get_item(1)?.extract()?;
        let result_dict = tuple.get_item(2)?.cast::<PyDict>()?.clone();
        let mut result: RefinedIndexResult = HashMap::new();
        for (k, v) in result_dict.iter() {
            let key: Vec<i64> = k.extract()?;
            let val: i64 = v.extract()?;
            if val != 0 {
                if num_hard == 0 && key.len() > 1 {
                    num_hard = key.len() - 1;
                }
                result.insert(key, val);
            }
        }
        out.push(Entry {
            m_ext,
            e_ext_x2,
            result,
        });
    }
    Ok((out, num_hard))
}

fn refined_result_to_py(py: Python<'_>, result: &RefinedIndexResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (k, &v) in result {
        if v != 0 {
            let key = PyTuple::new(py, k.iter().copied())?;
            dict.set_item(key, v)?;
        }
    }
    Ok(dict.into_any().unbind())
}

// ── Module functions ──

/// Load a manifold from the census database and return NzData.
#[pyfunction]
fn load_nz(db_path: &str, name: &str) -> PyResult<PyNzData> {
    let census = Census::open(std::path::Path::new(db_path))
        .map_err(|e| PyValueError::new_err(format!("census open: {e}")))?;
    let md = census
        .load(name)
        .map_err(|e| PyValueError::new_err(format!("load {name}: {e}")))?;
    let nz = census
        .load_nz(&md.census, &md.name)
        .map_err(|e| PyValueError::new_err(format!("load_nz {name}: {e}")))?;
    let state = EnumerationState::build(&nz);
    Ok(PyNzData { nz, state })
}

/// Load NZ data with explicit census name.
#[pyfunction]
fn load_nz_with_census(db_path: &str, census_name: &str, name: &str) -> PyResult<PyNzData> {
    let census = Census::open(std::path::Path::new(db_path))
        .map_err(|e| PyValueError::new_err(format!("census open: {e}")))?;
    let nz = census
        .load_nz(census_name, name)
        .map_err(|e| PyValueError::new_err(format!("load_nz {name}: {e}")))?;
    let state = EnumerationState::build(&nz);
    Ok(PyNzData { nz, state })
}

/// Apply SL(2,Z) basis change at one cusp.
#[pyfunction]
fn basis_change(data: &PyNzData, cusp_idx: usize, a: i32, b: i32, c: i32, d: i32) -> PyResult<PyNzData> {
    let nz = apply_general_cusp_basis_change(&data.nz, cusp_idx, a, b, c, d)
        .map_err(|e| PyValueError::new_err(format!("basis_change: {e}")))?;
    let state = EnumerationState::build(&nz);
    Ok(PyNzData { nz, state })
}

/// Find R, S for slope (P, Q): P*S - Q*R = 1.
#[pyfunction]
fn py_find_rs(p: i64, q: i64) -> (i64, i64) {
    find_rs(p, q)
}

/// Compute refined index I^ref at one (m_ext, e_ext_x2) point.
#[pyfunction]
fn refined_index(
    py: Python<'_>,
    data: &PyNzData,
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    qq_order: i32,
) -> PyResult<PyObject> {
    let result = compute_refined_index(&data.state, data.nz.num_hard, &m_ext, &e_ext_x2, qq_order);
    refined_result_to_py(py, &result)
}

/// Compute unrefined 3D index at one (m_ext, e_ext_x2) point.
#[pyfunction]
fn unrefined_index(
    py: Python<'_>,
    data: &PyNzData,
    m_ext: Vec<i64>,
    e_ext_x2: Vec<i64>,
    qq_order: i32,
) -> PyResult<PyObject> {
    let result = compute_unrefined_index(&data.state, &m_ext, &e_ext_x2, qq_order);
    let dict = PyDict::new(py);
    dict.set_item("min_power", result.min_power)?;
    dict.set_item("q_order_half", result.q_order_half)?;
    dict.set_item("n_terms", result.n_terms)?;
    let coeffs: Vec<i64> = result.coeffs;
    dict.set_item("coeffs", coeffs)?;
    Ok(dict.into_any().unbind())
}

/// Compute (a, b) Weyl vectors from refined-index entries.
#[pyfunction]
fn ab_vectors(
    py: Python<'_>,
    entries: &Bound<'_, PyList>,
    num_hard: usize,
) -> PyResult<PyObject> {
    let (rust_entries, _) = entries_from_py(py, entries)?;
    match compute_ab_vectors(&rust_entries, num_hard) {
        Some(ab) => {
            let dict = PyDict::new(py);
            let a: Vec<(i64, i64)> = ab.a.iter().map(|v| (*v.numer(), *v.denom())).collect();
            let b: Vec<(i64, i64)> = ab.b.iter().map(|v| (*v.numer(), *v.denom())).collect();
            dict.set_item("a", a)?;
            dict.set_item("b", b)?;
            dict.set_item("num_hard", ab.num_hard)?;
            dict.set_item("is_valid", ab.is_valid())?;
            dict.set_item("edge_compatible", ab.edge_compatible())?;
            dict.set_item("incompat_edges", ab.incompat_edges())?;
            dict.set_item("warnings", ab.warnings.clone())?;
            Ok(dict.into_any().unbind())
        }
        None => Ok(py.None()),
    }
}

/// Run the full Weyl check pipeline (ab + symmetry + adjoint + marginal).
#[pyfunction]
#[pyo3(signature = (data, entries, num_hard, qq_order, cusp_idx=0, filled_cusp_indices=None))]
fn weyl_checks(
    py: Python<'_>,
    data: &PyNzData,
    entries: &Bound<'_, PyList>,
    num_hard: usize,
    qq_order: i32,
    cusp_idx: usize,
    filled_cusp_indices: Option<Vec<usize>>,
) -> PyResult<PyObject> {
    let (rust_entries, _) = entries_from_py(py, entries)?;

    let ab = compute_ab_vectors(&rust_entries, num_hard);

    let weyl_sym = if let Some(ref ab_v) = ab {
        check_weyl_symmetry(&rust_entries, num_hard, ab_v, Some(qq_order))
    } else {
        rust_entries
            .iter()
            .map(|e| ((e.m_ext.clone(), e.e_ext_x2.clone()), false))
            .collect()
    };
    let all_weyl = !weyl_sym.is_empty() && weyl_sym.values().all(|&v| v);

    let ab_compat = ab.as_ref().map(|a| a.make_filling_compatible());
    let collapsed: hashbrown::HashSet<usize> = ab
        .as_ref()
        .map(|a| a.incompat_edges().into_iter().collect())
        .unwrap_or_default();

    let dict = PyDict::new(py);
    dict.set_item("ab_valid", ab.as_ref().map(|a| a.is_valid()).unwrap_or(false))?;
    dict.set_item("all_weyl_symmetric", all_weyl)?;

    // Adjoint projection
    if let Some(ref fci) = filled_cusp_indices {
        if fci.len() >= 2 {
            if let Some(ref ab_c) = ab_compat {
                let ce = if collapsed.is_empty() { None } else { Some(&collapsed) };
                let mc = check_adjoint_projection_multi_cusp(
                    &rust_entries, num_hard, Some(ab_c), fci, ce,
                );
                let results_py = PyList::empty(py);
                for r in &mc.results {
                    let rd = PyDict::new(py);
                    rd.set_item("is_pass", r.is_pass)?;
                    rd.set_item("projected_value", r.projected_value)?;
                    results_py.append(rd)?;
                }
                dict.set_item("multi_cusp_adjoint", results_py)?;
                dict.set_item("adjoint_all_pass", mc.all_pass())?;
            }
        } else {
            let ci = fci.first().copied().unwrap_or(cusp_idx);
            if let Some(ref ab_c) = ab_compat {
                let ce = if collapsed.is_empty() { None } else { Some(&collapsed) };
                let adj = check_adjoint_projection(
                    &rust_entries, num_hard, Some(ab_c), ci, ce,
                );
                dict.set_item("adjoint_pass", adj.is_pass)?;
                dict.set_item("adjoint_value", adj.projected_value)?;
            }
        }
    } else {
        if let Some(ref ab_c) = ab_compat {
            let ce = if collapsed.is_empty() { None } else { Some(&collapsed) };
            let adj = check_adjoint_projection(
                &rust_entries, num_hard, Some(ab_c), cusp_idx, ce,
            );
            dict.set_item("adjoint_pass", adj.is_pass)?;
            dict.set_item("adjoint_value", adj.projected_value)?;
        }
    }

    // Marginal check
    let n_cusps = data.nz.r;
    let marginal = check_marginal(&data.state, n_cusps, cusp_idx, qq_order);
    dict.set_item("is_marginal", marginal.is_marginal)?;
    dict.set_item("unrefined_q1_proj", marginal.unrefined_q1_proj)?;

    Ok(dict.into_any().unbind())
}

/// Find non-closable cycles for one cusp.
#[pyfunction]
#[pyo3(signature = (data, cusp_idx, p_min, p_max, q_min, q_max, qq_order, use_symmetry=true))]
fn nc_search(
    py: Python<'_>,
    data: &PyNzData,
    cusp_idx: usize,
    p_min: i64,
    p_max: i64,
    q_min: i64,
    q_max: i64,
    qq_order: i32,
    use_symmetry: bool,
) -> PyResult<PyObject> {
    let n_cusps = data.nz.r;
    let m_other = vec![0i64; n_cusps - 1];
    let e_other_x2 = vec![0i64; n_cusps - 1];
    let result = find_non_closable_cycles(
        &data.state,
        cusp_idx,
        p_min..p_max,
        q_min..q_max,
        &m_other,
        &e_other_x2,
        qq_order,
        use_symmetry,
    );

    let dict = PyDict::new(py);
    let cycles = PyList::empty(py);
    for c in &result.cycles {
        let cd = PyDict::new(py);
        cd.set_item("p", c.p)?;
        cd.set_item("q", c.q)?;
        cycles.append(cd)?;
    }
    dict.set_item("cycles", cycles)?;
    dict.set_item("slopes_tested", result.slopes_tested.len())?;
    Ok(dict.into_any().unbind())
}

/// Compute filled (unrefined) index.
#[pyfunction]
#[pyo3(signature = (data, cusp_idx, p, q, qq_order, m_other=None, e_other_x2=None))]
fn filled_index(
    py: Python<'_>,
    data: &PyNzData,
    cusp_idx: usize,
    p: i64,
    q: i64,
    qq_order: i32,
    m_other: Option<Vec<i64>>,
    e_other_x2: Option<Vec<i64>>,
) -> PyResult<PyObject> {
    let n_cusps = data.nz.r;
    let m_o = m_other.unwrap_or_else(|| vec![0i64; n_cusps - 1]);
    let e_o = e_other_x2.unwrap_or_else(|| vec![0i64; n_cusps - 1]);
    let result = compute_filled_index(&data.state, cusp_idx, p, q, &m_o, &e_o, qq_order);
    let dict = PyDict::new(py);
    let series = PyDict::new(py);
    for (&k, v) in &result.series {
        if *v.numer() != 0 {
            series.set_item(k, (*v.numer(), *v.denom()))?;
        }
    }
    dict.set_item("series", series)?;
    dict.set_item("n_kernel_terms", result.n_kernel_terms)?;
    Ok(dict.into_any().unbind())
}

/// Compute refined filled index (single cusp, auto-selects ℓ=1 or ℓ≥2 path).
#[pyfunction]
#[pyo3(signature = (data, cusp_idx, p, q, qq_order, m_other=None, e_other_x2=None, incompat_edges=None))]
fn filled_refined_index(
    py: Python<'_>,
    data: &PyNzData,
    cusp_idx: usize,
    p: i64,
    q: i64,
    qq_order: i32,
    m_other: Option<Vec<i64>>,
    e_other_x2: Option<Vec<i64>>,
    incompat_edges: Option<Vec<usize>>,
) -> PyResult<PyObject> {
    let n_cusps = data.nz.r;
    let m_o = m_other.unwrap_or_else(|| vec![0i64; n_cusps - 1]);
    let e_o = e_other_x2.unwrap_or_else(|| vec![0i64; n_cusps - 1]);
    let ie = incompat_edges.unwrap_or_default();
    let num_hard = data.nz.num_hard;

    let hj_ks = crate::refined_dehn::hj_cf::hj_continued_fraction(p, q);
    let result = if hj_ks.len() >= 2 {
        compute_filled_refined_index_chain(
            &data.state, num_hard, cusp_idx, p, q,
            &m_o, &e_o, qq_order,
            &ie, None, None,
        )
    } else {
        compute_unrefined_kernel_refined_index(
            &data.state, num_hard, cusp_idx, p, q,
            &m_o, &e_o, qq_order,
            &ie, None, None,
        )
    };

    let dict = PyDict::new(py);
    let series = PyDict::new(py);
    for (k, v) in &result.series {
        if *v.numer() != 0 {
            let key = PyTuple::new(py, k.iter().copied())?;
            series.set_item(key, (*v.numer(), *v.denom()))?;
        }
    }
    dict.set_item("series", series)?;
    dict.set_item("hj_ks", &result.hj_ks)?;
    dict.set_item("has_cusp_eta", result.has_cusp_eta)?;
    dict.set_item("num_cusp_eta", result.num_cusp_eta)?;
    dict.set_item("qq_order", result.qq_order)?;
    dict.set_item("n_kernel_terms", result.n_kernel_terms)?;
    Ok(dict.into_any().unbind())
}

/// Multi-cusp sequential refined filling.
#[pyfunction]
fn multi_cusp_filled_refined_index(
    py: Python<'_>,
    data: &PyNzData,
    fill_specs: &Bound<'_, PyList>,
    qq_order: i32,
) -> PyResult<PyObject> {
    let num_hard = data.nz.num_hard;
    let mut specs = Vec::new();
    for item in fill_specs.iter() {
        let d = item.cast::<PyDict>()?;
        let cusp_idx: usize = d.get_item("cusp_idx")?.unwrap().extract()?;
        let p: i64 = d.get_item("p")?.unwrap().extract()?;
        let q: i64 = d.get_item("q")?.unwrap().extract()?;
        specs.push(MultiCuspFillSpec {
            cusp_idx,
            p,
            q,
            weyl_a: None,
            weyl_b: None,
            incompat_edges: vec![],
        });
    }

    let result = compute_multi_cusp_filled_refined_index(
        &data.state, num_hard, &specs, qq_order,
    );

    let dict = PyDict::new(py);
    let series = PyDict::new(py);
    for (k, v) in &result.series {
        if *v.numer() != 0 {
            let key = PyTuple::new(py, k.iter().copied())?;
            series.set_item(key, (*v.numer(), *v.denom()))?;
        }
    }
    dict.set_item("series", series)?;
    dict.set_item("hj_ks", &result.hj_ks)?;
    dict.set_item("has_cusp_eta", result.has_cusp_eta)?;
    dict.set_item("num_cusp_eta", result.num_cusp_eta)?;
    dict.set_item("qq_order", result.qq_order)?;
    Ok(dict.into_any().unbind())
}

/// Single-cusp NC compatibility check.
#[pyfunction]
fn nc_compat(
    py: Python<'_>,
    data: &PyNzData,
    p: i64,
    q: i64,
    cusp_idx: usize,
    qq_order: i32,
) -> PyResult<PyObject> {
    let num_hard = data.nz.num_hard;
    let result = check_nc_compat(&data.nz, p, q, cusp_idx, num_hard, qq_order);

    let dict = PyDict::new(py);
    dict.set_item("ab_valid", result.ab.as_ref().map(|a| a.is_valid()).unwrap_or(false))?;
    dict.set_item("collapsed_edges", result.collapsed_edges)?;
    dict.set_item("all_weyl_symmetric", result.all_weyl_symmetric)?;

    if let Some(ref adj) = result.adjoint {
        dict.set_item("adjoint_pass", adj.is_pass)?;
        dict.set_item("adjoint_value", adj.projected_value)?;
    } else {
        dict.set_item("adjoint_pass", py.None())?;
        dict.set_item("adjoint_value", py.None())?;
    }

    dict.set_item("is_marginal", result.marginal.is_marginal)?;
    dict.set_item("unrefined_q1_proj", result.marginal.unrefined_q1_proj)?;

    Ok(dict.into_any().unbind())
}

/// Clear the tetrahedron index cache.
#[pyfunction]
fn clear_cache() {
    crate::kernel::clear_tet_cache();
}

// ── Module registration ──

#[pymodule]
fn iref3d_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNzData>()?;
    m.add_function(wrap_pyfunction!(load_nz, m)?)?;
    m.add_function(wrap_pyfunction!(load_nz_with_census, m)?)?;
    m.add_function(wrap_pyfunction!(basis_change, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_rs, m)?)?;
    m.add_function(wrap_pyfunction!(refined_index, m)?)?;
    m.add_function(wrap_pyfunction!(unrefined_index, m)?)?;
    m.add_function(wrap_pyfunction!(ab_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(weyl_checks, m)?)?;
    m.add_function(wrap_pyfunction!(nc_search, m)?)?;
    m.add_function(wrap_pyfunction!(filled_index, m)?)?;
    m.add_function(wrap_pyfunction!(filled_refined_index, m)?)?;
    m.add_function(wrap_pyfunction!(multi_cusp_filled_refined_index, m)?)?;
    m.add_function(wrap_pyfunction!(nc_compat, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;
    Ok(())
}
