"""Adapter: bridges v0.5's Python service layer to the Rust iref3d_core backend.

Two integration modes:
  1. C extension drop-in (install_rust_backend.py) — replaces inner kernel only
  2. Service-level patching (patch_services()) — replaces full computation pipeline

Mode 2 gives 10–150× speedup vs mode 1's ~2×.
"""
from __future__ import annotations

import sys
import os
import weakref
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


# ── NZ data conversion with caching ──

# Cache: id(nz_data) → (weakref_or_hash, PyNzData)
# We use id() for fast lookup but verify the object is still alive.
_nz_cache: dict[int, tuple[Any, Any]] = {}
_NZ_CACHE_MAX = 32


def _frac_to_x2(f) -> int:
    return int(Fraction(f) * 2)


def _e_ext_to_x2(e_ext: list) -> list[int]:
    return [_frac_to_x2(e) for e in e_ext]


def _is_rust_nz(obj: Any) -> bool:
    """Check if obj is already a Rust PyNzData."""
    return type(obj).__name__ == "NzData" and hasattr(obj, "num_hard") and hasattr(obj, "n_int")


def _nz_to_rust(nz_data: Any) -> Any:
    """Convert a v0.5 NeumannZagierData to a Rust PyNzData.

    Caches by object identity for repeated calls with the same data.
    If nz_data is already a Rust PyNzData, returns it directly.
    """
    # Already Rust?
    if _is_rust_nz(nz_data):
        return nz_data

    # Check cache by id
    obj_id = id(nz_data)
    if obj_id in _nz_cache:
        _ref, rust_data = _nz_cache[obj_id]
        # Verify the object is still the same (id can be reused after GC)
        if _ref is nz_data:
            return rust_data

    rust = _get_rust()
    n = int(nz_data.n)
    r = int(nz_data.r)
    num_hard = int(nz_data.num_hard)
    num_easy = n - r - num_hard

    g_nz_x2 = []
    for i in range(2 * n):
        for j in range(2 * n):
            g_nz_x2.append(int(Fraction(nz_data.g_NZ[i, j]) * 2))

    nu_x = [int(nz_data.nu_x[i]) for i in range(n)]
    nu_p_x2 = [int(Fraction(nz_data.nu_p[i]) * 2) for i in range(n)]

    rust_data = rust.create_nz_data(n, r, num_hard, num_easy, g_nz_x2, nu_x, nu_p_x2)

    # Cache (evict oldest if full)
    if len(_nz_cache) >= _NZ_CACHE_MAX:
        oldest_key = next(iter(_nz_cache))
        del _nz_cache[oldest_key]
    _nz_cache[obj_id] = (nz_data, rust_data)

    return rust_data


# ── Public API matching v0.5's core module functions ──

def compute_refined_index(nz_data, m_ext, e_ext, q_order_half=10, **_kw):
    """Drop-in for refined_index.compute_refined_index.

    Returns dict[tuple[int,...], int] — same as v0.5.
    """
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    e_x2 = _e_ext_to_x2(e_ext)
    return dict(rust.refined_index(data, list(m_ext), e_x2, int(q_order_half)))


def compute_refined_index_batch(nz_data, queries, q_order_half):
    """Batch compute — queries is list of (m_ext, e_ext) pairs."""
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    results = []
    for m_ext, e_ext in queries:
        e_x2 = _e_ext_to_x2(e_ext)
        r = rust.refined_index(data, list(m_ext), e_x2, int(q_order_half))
        results.append(dict(r))
    return results


def compute_index_3d_python(nz_data, m_ext, e_ext, q_order_half=10, **_kw):
    """Drop-in for index_3d.compute_index_3d_python.

    v0.5 returns Index3DResult; we return a compatible wrapper.
    """
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    e_x2 = _e_ext_to_x2(e_ext)
    result = rust.unrefined_index(data, list(m_ext), e_x2, int(q_order_half))
    return _Index3DResultCompat(
        coeffs=list(result["coeffs"]),
        min_power=result["min_power"],
        q_order_half=result["q_order_half"],
        m_ext=list(m_ext),
        e_ext=list(e_ext),
        n_terms=result["n_terms"],
    )


class _Index3DResultCompat:
    """Minimal wrapper matching v0.5's Index3DResult interface."""
    def __init__(self, coeffs, min_power, q_order_half, m_ext, e_ext, n_terms):
        self.coeffs = coeffs
        self.min_power = min_power
        self.q_order_half = q_order_half
        self.m_ext = m_ext
        self.e_ext = e_ext
        self.n_terms = n_terms
        # Sparse dict for iteration
        self.series = {}
        for i, c in enumerate(coeffs):
            if c != 0:
                self.series[min_power + i] = c

    def as_polynomial_string(self, var="q"):
        parts = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            p = self.min_power + i
            if p == 0:
                parts.append(str(c))
            elif c == 1:
                parts.append(f"{var}^{p}" if p != 1 else var)
            elif c == -1:
                parts.append(f"-{var}^{p}" if p != 1 else f"-{var}")
            else:
                parts.append(f"{c}*{var}^{p}" if p != 1 else f"{c}*{var}")
        return " + ".join(parts).replace("+ -", "- ") if parts else "0"


def find_non_closable_cycles(nz_data, cusp_idx, p_range, q_range,
                              q_order_half, use_symmetry=True, progress_fn=None,
                              **_kw):
    """Drop-in for dehn_filling.find_non_closable_cycles.

    Returns a compatible NonClosableCycleResult wrapper.
    """
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    p_min, p_max = min(p_range), max(p_range)
    q_min, q_max = min(q_range), max(q_range)
    result = rust.nc_search(
        data, cusp_idx, p_min, p_max + 1, q_min, q_max + 1,
        int(q_order_half), use_symmetry,
    )
    return _NcResultCompat(result, cusp_idx)


class _NcCycleCompat:
    """Matches v0.5's NonClosableCycle interface."""
    def __init__(self, cusp_idx, P, Q):
        self.cusp_idx = cusp_idx
        self.P = P
        self.Q = Q

    def __repr__(self):
        return f"NonClosableCycle(cusp={self.cusp_idx}, P={self.P}, Q={self.Q})"


class _NcResultCompat:
    """Matches v0.5's NonClosableCycleResult interface."""
    def __init__(self, rust_dict, cusp_idx):
        self.cusp_idx = cusp_idx
        self.cycles = [
            _NcCycleCompat(cusp_idx, c["p"], c["q"])
            for c in rust_dict["cycles"]
        ]
        self.slopes_tested = rust_dict.get("slopes_tested", [])
        self.series_data = rust_dict.get("series_data", {})


def compute_filled_index(nz_data, cusp_idx, P, Q,
                          m_other=None, e_other=None,
                          q_order_half=10, **_kw):
    """Drop-in for dehn_filling.compute_filled_index (unrefined)."""
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    m_other_list = list(m_other) if m_other else None
    e_other_x2 = _e_ext_to_x2(e_other) if e_other else None
    return rust.filled_index(data, cusp_idx, P, Q, int(q_order_half),
                             m_other_list, e_other_x2)


def find_rs(P, Q):
    """Drop-in for dehn_filling.find_rs."""
    return _get_rust().py_find_rs(P, Q)


def apply_general_cusp_basis_change(nz_data, cusp_idx, a, b, c, d):
    """Drop-in for neumann_zagier.apply_general_cusp_basis_change.

    Returns Rust PyNzData. Downstream calls detect this via _is_rust_nz()
    and skip conversion.
    """
    rust = _get_rust()
    data = _nz_to_rust(nz_data)
    return rust.basis_change(data, cusp_idx, int(a), int(b), int(c), int(d))


def clear_tet_cache():
    """Drop-in for index_3d.clear_tet_cache."""
    _get_rust().clear_cache()
    _nz_cache.clear()


# ── Service-level patching ──

_patched = False
_originals = {}


def patch_services():
    """Monkey-patch v0.5's ComputeService and FillingService to use Rust.

    Call AFTER v0.5 modules are imported but BEFORE the GUI starts computation.
    Safe to call multiple times (idempotent).
    """
    global _patched
    if _patched:
        return

    # Import the actual v0.5 core modules and patch their functions in-place.
    # This way, any code that imported them (services, workers) sees the patches.
    import manifold_index.core.refined_index as _ri
    import manifold_index.core.index_3d as _i3d
    import manifold_index.core.dehn_filling as _df
    import manifold_index.core.neumann_zagier as _nz

    # Only patch functions whose output is consumed directly (dicts, results).
    # Do NOT patch apply_general_cusp_basis_change or find_rs — their output
    # feeds into un-patched Python code (refined_dehn_filling) that expects
    # v0.5 NeumannZagierData objects.
    _originals["ri.compute_refined_index"] = _ri.compute_refined_index
    _originals["i3d.compute_index_3d_python"] = _i3d.compute_index_3d_python
    _originals["df.find_non_closable_cycles"] = _df.find_non_closable_cycles

    _ri.compute_refined_index = compute_refined_index
    _i3d.compute_index_3d_python = compute_index_3d_python
    _df.find_non_closable_cycles = find_non_closable_cycles

    _patched = True

    print("[rust_adapter] Patched 3 core functions → Rust backend")
    print("[rust_adapter]   refined_index, index_3d, nc_search")


def unpatch_services():
    """Restore original v0.5 Python functions."""
    global _patched
    if not _patched:
        return

    import manifold_index.core.refined_index as _ri
    import manifold_index.core.index_3d as _i3d
    import manifold_index.core.dehn_filling as _df
    import manifold_index.core.neumann_zagier as _nz

    _ri.compute_refined_index = _originals["ri.compute_refined_index"]
    _i3d.compute_index_3d_python = _originals["i3d.compute_index_3d_python"]
    _df.find_non_closable_cycles = _originals["df.find_non_closable_cycles"]

    _patched = False
    print("[rust_adapter] Unpatched — restored Python backend")
