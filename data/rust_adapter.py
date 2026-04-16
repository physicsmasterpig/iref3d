"""Adapter: bridges v0.5's Python service layer to the Rust iref3d_core backend.

Drop this module into the v0.5 source tree and patch the services to import
from it.  Converts between v0.5 types (Fraction, NeumannZagierData) and
Rust types (×2 integers, PyNzData).

Usage:
    1. Build the Rust extension:
         cd core && cargo build --features python --release
    2. Ensure the .dylib/.so is importable (symlink or PYTHONPATH).
    3. In v0.5's compute_service.py / filling_service.py, replace core module
       imports with:
         from manifold_index.core._rust_adapter import RustBackend
         _backend = RustBackend(db_path="path/to/census_fixture.db")

Or use the patching approach (see install_adapter below).
"""
from __future__ import annotations

import sys
import os
from fractions import Fraction
from pathlib import Path
from typing import Any


def _ensure_rust_module():
    """Add the Rust cdylib to sys.path if not already importable."""
    try:
        import iref3d_core
        return iref3d_core
    except ImportError:
        pass

    # Try common build output locations.
    candidates = [
        Path(__file__).resolve().parent.parent / "core" / "target" / "release",
        Path(__file__).resolve().parent.parent / "core" / "target" / "debug",
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            try:
                import iref3d_core
                return iref3d_core
            except ImportError:
                sys.path.remove(str(p))
    raise ImportError(
        "Cannot import iref3d_core. Build with: "
        "cd core && cargo build --features python --release"
    )


_rust = None


def _get_rust():
    global _rust
    if _rust is None:
        _rust = _ensure_rust_module()
    return _rust


# ── Type conversion helpers ──

def _frac_to_x2(f: Fraction) -> int:
    """Convert a Fraction to ×2 integer: Fraction(1,2) → 1."""
    return int(f * 2)


def _e_ext_to_x2(e_ext: list) -> list[int]:
    """Convert v0.5's e_ext (list of Fraction) to ×2 integers."""
    return [_frac_to_x2(Fraction(e)) for e in e_ext]


def _nz_to_rust(nz_data: Any) -> Any:
    """Convert a v0.5 NeumannZagierData to a Rust PyNzData.

    v0.5 NeumannZagierData has:
      .n, .r, .num_hard
      .g_NZ         — numpy (2n × 2n) rational matrix
      .nu_x, .nu_p  — numpy n-vectors (rational)
    """
    rust = _get_rust()
    n = int(nz_data.n)
    r = int(nz_data.r)
    num_hard = int(nz_data.num_hard)
    num_easy = n - r - num_hard

    # g_NZ is rational — multiply by 2 to get integer g_nz_x2.
    g_nz_x2 = []
    for i in range(2 * n):
        for j in range(2 * n):
            val = nz_data.g_NZ[i, j]
            g_nz_x2.append(int(Fraction(val) * 2))

    nu_x = [int(nz_data.nu_x[i]) for i in range(n)]
    nu_p_x2 = [int(Fraction(nz_data.nu_p[i]) * 2) for i in range(n)]

    return rust.create_nz_data(n, r, num_hard, num_easy, g_nz_x2, nu_x, nu_p_x2)


def _refined_result_from_rust(rust_dict: dict) -> dict:
    """Convert Rust refined index result (dict with tuple keys) to v0.5 format.

    Both use dict[tuple[int,...], int], so no conversion needed.
    """
    return dict(rust_dict)


# ── Public API matching v0.5's service methods ──

def compute_refined_index(nz_data, m_ext, e_ext, q_order_half):
    """Drop-in for compute_service.compute_refined_index."""
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    e_x2 = _e_ext_to_x2(e_ext)
    result = rust.refined_index(data, list(m_ext), e_x2, q_order_half)
    return _refined_result_from_rust(result)


def compute_refined_index_batch(nz_data, queries, q_order_half):
    """Batch compute — queries is list of (m_ext, e_ext) pairs."""
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    results = []
    for m_ext, e_ext in queries:
        e_x2 = _e_ext_to_x2(e_ext)
        r = rust.refined_index(data, list(m_ext), e_x2, q_order_half)
        results.append(_refined_result_from_rust(r))
    return results


def compute_index_3d_python(nz_data, m_ext, e_ext, q_order_half):
    """Drop-in for core.index_3d.compute_index_3d_python (unrefined 3D index).

    Rust FFI returns {min_power, q_order_half, n_terms, coeffs} —
    convert to v0.5's sparse dict[int_power, int_coeff] format.
    """
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    e_x2 = _e_ext_to_x2(e_ext)
    result = rust.unrefined_index(data, list(m_ext), e_x2, q_order_half)
    # result is a dict with keys: min_power, q_order_half, n_terms, coeffs
    min_power = result["min_power"]
    coeffs = result["coeffs"]
    out = {}
    for i, c in enumerate(coeffs):
        if c != 0:
            out[min_power + i] = c
    return out


def find_non_closable_cycles(nz_data, cusp_idx, p_range, q_range, q_order_half,
                              use_symmetry=True, progress_fn=None):
    """Drop-in for dehn_filling.find_non_closable_cycles."""
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    p_min, p_max = min(p_range), max(p_range)
    q_min, q_max = min(q_range), max(q_range)
    result = rust.nc_search(
        data, cusp_idx, p_min, p_max + 1, q_min, q_max + 1,
        q_order_half, use_symmetry,
    )
    return result


def compute_filled_index(nz_data, cusp_idx, P, Q, m_other, e_other, q_order_half):
    """Drop-in for dehn_filling.compute_filled_index (unrefined filled).

    Rust FFI signature: filled_index(data, cusp_idx, p, q, qq_order, m_other, e_other_x2)
    """
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    m_other_list = list(m_other) if m_other else None
    e_other_x2 = _e_ext_to_x2(e_other) if e_other else None
    result = rust.filled_index(data, cusp_idx, P, Q, q_order_half, m_other_list, e_other_x2)
    return result


def compute_filled_refined_index(nz_data, cusp_idx, P, Q,
                                  m_other=None, e_other=None,
                                  q_order_half=10,
                                  weyl_a=None, weyl_b=None,
                                  incompat_edges=None,
                                  auto_precompute=True):
    """Drop-in for refined_dehn_filling.compute_filled_refined_index."""
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    m_other_list = list(m_other) if m_other else []
    e_other_x2 = _e_ext_to_x2(e_other) if e_other else []

    wa = [float(Fraction(a)) for a in weyl_a] if weyl_a else []
    wb = [float(Fraction(b)) for b in weyl_b] if weyl_b else []
    ie = list(incompat_edges) if incompat_edges else []

    result = rust.filled_refined_index(
        data, cusp_idx, P, Q, m_other_list, e_other_x2,
        q_order_half, wa, wb, ie,
    )
    return result


def find_rs(P, Q):
    """Drop-in for dehn_filling.find_rs."""
    rust = _get_rust()
    return rust.find_rs(P, Q)


def apply_general_cusp_basis_change(nz_data, cusp_idx, a, b, c, d):
    """Drop-in for neumann_zagier.apply_general_cusp_basis_change.

    Returns a Rust PyNzData (not a v0.5 NeumannZagierData), so downstream
    calls must also go through Rust.
    """
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    return rust.basis_change(data, cusp_idx, int(a), int(b), int(c), int(d))


def run_weyl_checks(entries, num_hard, cusp_idx=0, q_order_half=10,
                     filled_cusp_indices=None):
    """Drop-in for weyl_check.run_weyl_checks.

    Converts v0.5 entry format to Rust format.
    """
    rust = _get_rust()
    # entries: list of (m_ext, e_ext, RefinedIndexResult)
    rust_entries = []
    for m_ext, e_ext, result in entries:
        e_x2 = _e_ext_to_x2(e_ext)
        rust_entries.append((list(m_ext), e_x2, dict(result)))

    return rust.weyl_checks(
        rust_entries, num_hard, cusp_idx, q_order_half,
        filled_cusp_indices or [],
    )


def clear_tet_cache():
    """Drop-in for index_3d.clear_tet_cache / clear_enum_state_cache."""
    rust = _get_rust()
    rust.clear_cache()


def hj_continued_fraction(P, Q):
    """Expose HJ continued fraction (not in v0.5's core, but useful)."""
    # This isn't directly exposed via FFI yet — compute in Python.
    if Q == 0:
        return [P]
    # Standard HJ CF algorithm
    ks = []
    p, q = abs(P), abs(Q)
    while q > 0:
        k = -(-p // q)  # ceiling division
        ks.append(k)
        p, q = q, k * q - p
    if P < 0:
        ks = [-k for k in ks]
    return ks


# ── Convenience: install as drop-in ──

def install_adapter():
    """Monkey-patch v0.5's service imports to use the Rust backend.

    Call this once at startup before any computation:

        from rust_adapter import install_adapter
        install_adapter()

    After this, v0.5's ComputeService and FillingService will dispatch
    to Rust for all core computation.
    """
    import importlib

    # Patch core modules used by compute_service
    this = sys.modules[__name__]

    # Create fake modules
    class _FakeRefinedIndex:
        compute_refined_index = staticmethod(compute_refined_index)
        compute_refined_index_batch = staticmethod(compute_refined_index_batch)

    class _FakeIndex3D:
        compute_index_3d_python = staticmethod(compute_index_3d_python)

    class _FakeDehnFilling:
        find_non_closable_cycles = staticmethod(find_non_closable_cycles)
        compute_filled_index = staticmethod(compute_filled_index)
        find_rs = staticmethod(find_rs)

    class _FakeNZ:
        apply_general_cusp_basis_change = staticmethod(apply_general_cusp_basis_change)

    # Register in sys.modules so imports resolve to our fakes
    sys.modules["manifold_index.core.refined_index"] = _FakeRefinedIndex()
    sys.modules["manifold_index.core.index_3d"] = _FakeIndex3D()
    sys.modules["manifold_index.core.dehn_filling"] = _FakeDehnFilling()
    sys.modules["manifold_index.core.neumann_zagier"] = _FakeNZ()

    print("[rust_adapter] Installed Rust backend for v0.5 services")
