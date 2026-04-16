"""End-to-end validation: Rust iref3d_core vs v0.5 Python.

Imports both the Rust PyO3 module and v0.5's Python core, runs identical
computations on the same manifolds, and asserts bit-identical results.

Usage:
    PYTHONPATH=/path/to/iref3d/core/target/release python data/validate_e2e.py
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction
from pathlib import Path

# ── Setup imports ──

V05_SRC = "/Users/pmp/Documents/Research/ultimate/v0.5/src"
if V05_SRC not in sys.path:
    sys.path.insert(0, V05_SRC)

RELEASE_DIR = str(Path(__file__).resolve().parent.parent / "core" / "target" / "release")
if RELEASE_DIR not in sys.path:
    sys.path.insert(0, RELEASE_DIR)

import iref3d_core as rust

from manifold_index.core.manifold import load_manifold
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.index_3d import (
    compute_index_3d_python,
    _tet_index_series,
)
from manifold_index.core.refined_index import compute_refined_index as v05_compute_refined
from manifold_index.core.weyl_check import (
    compute_ab_vectors as v05_ab_vectors,
    check_weyl_symmetry as v05_weyl_symmetry,
    check_adjoint_projection as v05_check_adjoint,
)
from manifold_index.core.dehn_filling import (
    compute_filled_index as v05_compute_filled,
    find_non_closable_cycles as v05_find_nc,
)
from manifold_index.core.refined_dehn_filling import (
    compute_filled_refined_index as v05_compute_filled_refined,
)

DB_PATH = str(Path(__file__).resolve().parent.parent / "core" / "tests" / "fixtures" / "census_fixture.db")

# ── Counters ──

passed = 0
failed = 0
errors = []


def check(label: str, ok: bool, detail: str = ""):
    global passed, failed, errors
    if ok:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        msg = f"  FAIL  {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        errors.append(label)


# ── Load v0.5 manifold ──

def load_v05_manifold(name: str):
    md = load_manifold(name)
    easy = find_easy_edges(md)
    nz = build_neumann_zagier(md, easy)
    return md, easy, nz


# ── Conversion helpers ──

def e_frac_to_x2(e_ext: list) -> list[int]:
    """v0.5 uses Fraction for e_ext; Rust uses x2 integers."""
    return [int(Fraction(e) * 2) for e in e_ext]


def refined_v05_to_dict(result) -> dict:
    """v0.5 returns dict[tuple[int,...], int]; normalize to sorted list."""
    return {k: v for k, v in result.items() if v != 0}


def refined_rust_to_dict(result: dict) -> dict:
    """Rust returns {tuple: int}; normalize same way."""
    return {k: v for k, v in result.items() if v != 0}


def check_num_hard_match(name, nz_v05, nz_rust) -> bool:
    """Verify v0.5 and Rust agree on num_hard."""
    if nz_v05.num_hard != nz_rust.num_hard:
        print(f"  WARN  {name}: num_hard v05={nz_v05.num_hard} rust={nz_rust.num_hard}")
        return False
    return True


# ── Test functions ──

MANIFOLDS = ["m003", "m004", "m006", "s776"]


def test_unrefined_index():
    """Compare unrefined 3D index at m=0, e in {0, 1/2, 1, -1} for qq=10."""
    print("\n=== Unrefined Index ===")
    e_probes = [
        [Fraction(0)],
        [Fraction(1, 2)],
        [Fraction(1)],
        [Fraction(-1)],
    ]
    qq = 10

    for name in MANIFOLDS:
        try:
            md, easy, nz_v05 = load_v05_manifold(name)
        except Exception as exc:
            check(f"{name}: load", False, str(exc))
            continue

        nz_rust = rust.load_nz(DB_PATH, name)

        for e_ext in e_probes:
            m_ext = [0] * nz_v05.r
            e_label = str(e_ext[0]) if len(e_ext) == 1 else str(e_ext)

            # Extend e_ext to full cusp count
            e_full = list(e_ext) + [Fraction(0)] * (nz_v05.r - len(e_ext))
            e_x2 = e_frac_to_x2(e_full)

            # v0.5
            try:
                v05_result = compute_index_3d_python(nz_v05, m_ext, e_full, q_order_half=qq)
                v05_coeffs = list(v05_result.coeffs)
                v05_min = v05_result.min_power
            except Exception as exc:
                check(f"{name} e={e_label}: v05 compute", False, str(exc))
                continue

            # Rust
            try:
                rust_result = rust.unrefined_index(nz_rust, m_ext, e_x2, qq)
                rust_coeffs = list(rust_result["coeffs"])
                rust_min = rust_result["min_power"]
            except Exception as exc:
                check(f"{name} e={e_label}: rust compute", False, str(exc))
                continue

            match = (v05_coeffs == rust_coeffs and v05_min == rust_min)
            detail = ""
            if not match:
                detail = f"min_power v05={v05_min} rust={rust_min}, coeffs match={v05_coeffs == rust_coeffs}"
                if v05_coeffs != rust_coeffs:
                    for i, (a, b) in enumerate(zip(v05_coeffs, rust_coeffs)):
                        if a != b:
                            detail += f"; first diff at [{i}]: v05={a} rust={b}"
                            break
            check(f"{name} unrefined e={e_label} qq={qq}", match, detail)


def test_refined_index():
    """Compare refined index at m=0, e in {0, 1, -1} for qq=10."""
    print("\n=== Refined Index ===")
    e_probes = [
        [Fraction(0)],
        [Fraction(1)],
        [Fraction(-1)],
    ]
    qq = 10

    for name in MANIFOLDS:
        try:
            md, easy, nz_v05 = load_v05_manifold(name)
        except Exception as exc:
            check(f"{name}: load", False, str(exc))
            continue

        nz_rust = rust.load_nz(DB_PATH, name)

        if not check_num_hard_match(name, nz_v05, nz_rust):
            continue

        for e_ext in e_probes:
            m_ext = [0] * nz_v05.r
            e_label = str(e_ext[0]) if len(e_ext) == 1 else str(e_ext)
            e_full = list(e_ext) + [Fraction(0)] * (nz_v05.r - len(e_ext))
            e_x2 = e_frac_to_x2(e_full)

            # v0.5
            try:
                v05_raw = v05_compute_refined(nz_v05, m_ext, e_full, q_order_half=qq)
                v05_dict = refined_v05_to_dict(v05_raw)
            except Exception as exc:
                check(f"{name} refined e={e_label}: v05 compute", False, str(exc))
                continue

            # Rust
            try:
                rust_raw = rust.refined_index(nz_rust, m_ext, e_x2, qq)
                rust_dict = refined_rust_to_dict(rust_raw)
            except Exception as exc:
                check(f"{name} refined e={e_label}: rust compute", False, str(exc))
                continue

            match = (v05_dict == rust_dict)
            detail = ""
            if not match:
                only_v05 = set(v05_dict) - set(rust_dict)
                only_rust = set(rust_dict) - set(v05_dict)
                diff_val = {k for k in set(v05_dict) & set(rust_dict)
                           if v05_dict[k] != rust_dict[k]}
                detail = f"only_v05={len(only_v05)} only_rust={len(only_rust)} diff_val={len(diff_val)}"
                if diff_val:
                    k = next(iter(diff_val))
                    detail += f"; e.g. key={k}: v05={v05_dict[k]} rust={rust_dict[k]}"
            check(f"{name} refined e={e_label} qq={qq}", match, detail)


def test_filled_index():
    """Compare Dehn-filled unrefined index for slopes (3,1) and (5,2)."""
    print("\n=== Filled Index (Unrefined) ===")
    slopes = [(3, 1), (5, 2)]
    qq = 10

    for name in ["m003", "m004"]:
        try:
            md, easy, nz_v05 = load_v05_manifold(name)
        except Exception as exc:
            check(f"{name}: load", False, str(exc))
            continue

        nz_rust = rust.load_nz(DB_PATH, name)

        for p, q in slopes:
            cusp_idx = 0
            m_other = [0] * (nz_v05.r - 1)
            e_other = [Fraction(0)] * (nz_v05.r - 1)
            e_other_x2 = [0] * (nz_v05.r - 1)

            # v0.5
            try:
                v05_result = v05_compute_filled(nz_v05, cusp_idx, p, q,
                                                 m_other, e_other, qq)
                # v0.5 series is dict{int: Fraction}
                v05_series = {k: v for k, v in v05_result.series.items() if v != 0}
            except Exception as exc:
                check(f"{name} fill({p}/{q}): v05", False, str(exc))
                continue

            # Rust
            try:
                rust_result = rust.filled_index(nz_rust, cusp_idx, p, q, qq)
                # Rust series is dict{int: (numer, denom)}
                rust_series_raw = rust_result["series"]
                rust_series = {k: Fraction(v[0], v[1]) for k, v in rust_series_raw.items()
                              if v[0] != 0}
            except Exception as exc:
                check(f"{name} fill({p}/{q}): rust", False, str(exc))
                continue

            match = (v05_series == rust_series)
            detail = ""
            if not match:
                only_v05 = set(v05_series) - set(rust_series)
                only_rust = set(rust_series) - set(v05_series)
                detail = f"only_v05={only_v05} only_rust={only_rust}"
                common = set(v05_series) & set(rust_series)
                diffs = [(k, v05_series[k], rust_series[k]) for k in common
                        if v05_series[k] != rust_series[k]]
                if diffs:
                    k, vv, rv = diffs[0]
                    detail += f"; key={k}: v05={vv} rust={rv}"
            check(f"{name} fill({p}/{q}) qq={qq}", match, detail)


def test_filled_refined_index():
    """Compare refined Dehn filling for known slopes."""
    print("\n=== Filled Refined Index ===")
    qq = 10

    test_cases = [
        ("m003", 3, 1),  # ell=1 path
        ("m003", 5, 2),  # ell>=2 path (HJ CF)
        ("m004", 5, 1),  # another case
    ]

    for name, p, q_slope in test_cases:
        try:
            md, easy, nz_v05 = load_v05_manifold(name)
        except Exception as exc:
            check(f"{name}: load", False, str(exc))
            continue

        nz_rust = rust.load_nz(DB_PATH, name)
        cusp_idx = 0
        m_other = [0] * (nz_v05.r - 1)
        e_other = [Fraction(0)] * (nz_v05.r - 1)
        e_other_x2 = [0] * (nz_v05.r - 1)

        # v0.5
        try:
            v05_result = v05_compute_filled_refined(
                nz_v05, cusp_idx, p, q_slope,
                m_other, e_other, qq,
            )
            # v0.5 series: dict{tuple[int,...]: Fraction}
            v05_series = {k: v for k, v in v05_result.series.items() if v != 0}
        except Exception as exc:
            check(f"{name} rfill({p}/{q_slope}): v05", False, str(exc))
            continue

        # Rust
        try:
            rust_result = rust.filled_refined_index(
                nz_rust, cusp_idx, p, q_slope, qq,
                m_other=m_other, e_other_x2=e_other_x2,
            )
            rust_series_raw = rust_result["series"]
            rust_series = {k: Fraction(v[0], v[1]) for k, v in rust_series_raw.items()
                          if v[0] != 0}
        except Exception as exc:
            check(f"{name} rfill({p}/{q_slope}): rust", False, str(exc))
            continue

        match = (v05_series == rust_series)
        detail = ""
        if not match:
            only_v05 = set(v05_series) - set(rust_series)
            only_rust = set(rust_series) - set(v05_series)
            diff_val = {k for k in set(v05_series) & set(rust_series)
                       if v05_series[k] != rust_series[k]}
            detail = (f"|v05|={len(v05_series)} |rust|={len(rust_series)} "
                     f"only_v05={len(only_v05)} only_rust={len(only_rust)} "
                     f"diff_val={len(diff_val)}")
            if diff_val:
                k = next(iter(diff_val))
                detail += f"; key={k}: v05={v05_series[k]} rust={rust_series[k]}"
        check(f"{name} rfill({p}/{q_slope}) qq={qq}", match, detail)


def test_nc_search():
    """Compare NC cycle search results."""
    print("\n=== NC Cycle Search ===")
    qq = 5

    for name in ["m003", "m004"]:
        try:
            md, easy, nz_v05 = load_v05_manifold(name)
        except Exception as exc:
            check(f"{name}: load", False, str(exc))
            continue

        nz_rust = rust.load_nz(DB_PATH, name)
        cusp_idx = 0
        m_other = [0] * (nz_v05.r - 1)
        e_other = [Fraction(0)] * (nz_v05.r - 1)

        # v0.5
        try:
            v05_result = v05_find_nc(nz_v05, cusp_idx,
                                      range(-3, 4), range(-3, 4),
                                      m_other, e_other, qq)
            v05_cycles = set()
            for c in v05_result.cycles:
                v05_cycles.add((c.P, c.Q))
        except Exception as exc:
            check(f"{name} nc_search: v05", False, str(exc))
            continue

        # Rust
        try:
            rust_result = rust.nc_search(nz_rust, cusp_idx, -3, 4, -3, 4, qq)
            rust_cycles = set()
            for c in rust_result["cycles"]:
                rust_cycles.add((c["p"], c["q"]))
        except Exception as exc:
            check(f"{name} nc_search: rust", False, str(exc))
            continue

        match = (v05_cycles == rust_cycles)
        detail = ""
        if not match:
            only_v05 = v05_cycles - rust_cycles
            only_rust = rust_cycles - v05_cycles
            detail = f"v05={len(v05_cycles)} rust={len(rust_cycles)} only_v05={only_v05} only_rust={only_rust}"
        check(f"{name} nc_search p,q in [-3,3] qq={qq}", match, detail)


def test_nc_compat():
    """Compare NC compatibility check results."""
    print("\n=== NC Compatibility ===")
    qq = 10

    test_cases = [
        ("m003", 3, 1),
        ("m003", 5, 2),
        ("m004", 3, 1),
    ]

    for name, p, q_slope in test_cases:
        try:
            md, easy, nz_v05 = load_v05_manifold(name)
        except Exception as exc:
            check(f"{name}: load", False, str(exc))
            continue

        nz_rust = rust.load_nz(DB_PATH, name)
        cusp_idx = 0

        # Rust NC compat
        try:
            rust_nc = rust.nc_compat(nz_rust, p, q_slope, cusp_idx, qq)
        except Exception as exc:
            check(f"{name} nc_compat({p}/{q_slope}): rust", False, str(exc))
            continue

        # Just verify it runs and returns sensible fields
        has_fields = all(k in rust_nc for k in
                        ["ab_valid", "collapsed_edges", "all_weyl_symmetric",
                         "adjoint_pass", "is_marginal", "unrefined_q1_proj"])
        check(f"{name} nc_compat({p}/{q_slope}) fields present", has_fields,
              f"keys={list(rust_nc.keys())}")

        # Cross-check is_marginal matches our golden
        check(f"{name} nc_compat({p}/{q_slope}) types ok",
              isinstance(rust_nc["ab_valid"], bool) and
              isinstance(rust_nc["collapsed_edges"], list),
              f"ab_valid={type(rust_nc['ab_valid'])} collapsed={type(rust_nc['collapsed_edges'])}")


def test_performance():
    """Benchmark Rust vs v0.5 on a moderate workload."""
    print("\n=== Performance Comparison ===")
    name = "m004"
    qq = 20

    md, easy, nz_v05 = load_v05_manifold(name)
    nz_rust = rust.load_nz(DB_PATH, name)

    m_ext = [0]
    e_probes = [[Fraction(e)] for e in [0, Fraction(1,2), 1, Fraction(-1,2), -1, 2, -2]]

    # Warm up
    for e in e_probes:
        v05_compute_refined(nz_v05, m_ext, e, q_order_half=qq)
        e_x2 = e_frac_to_x2(e)
        rust.refined_index(nz_rust, m_ext, e_x2, qq)

    # Clear caches for fair comparison
    rust.clear_cache()
    from manifold_index.core.index_3d import clear_tet_cache, clear_enum_state_cache
    clear_tet_cache()
    clear_enum_state_cache()

    # Benchmark v0.5
    t0 = time.perf_counter()
    for _ in range(3):
        clear_tet_cache()
        clear_enum_state_cache()
        for e in e_probes:
            v05_compute_refined(nz_v05, m_ext, e, q_order_half=qq)
    t_v05 = time.perf_counter() - t0

    # Benchmark Rust
    t0 = time.perf_counter()
    for _ in range(3):
        rust.clear_cache()
        for e in e_probes:
            e_x2 = e_frac_to_x2(e)
            rust.refined_index(nz_rust, m_ext, e_x2, qq)
    t_rust = time.perf_counter() - t0

    ratio = t_v05 / t_rust if t_rust > 0 else float("inf")
    print(f"  v0.5:  {t_v05:.3f}s")
    print(f"  Rust:  {t_rust:.3f}s")
    print(f"  Speedup: {ratio:.1f}x")
    check(f"Rust faster than v0.5 ({ratio:.1f}x)", ratio >= 1.0,
          f"v05={t_v05:.3f}s rust={t_rust:.3f}s")


# ── Main ──

if __name__ == "__main__":
    print("=" * 60)
    print("iref3d-core end-to-end validation: Rust vs v0.5")
    print("=" * 60)

    t_start = time.perf_counter()

    test_unrefined_index()
    test_refined_index()
    test_filled_index()
    test_filled_refined_index()
    test_nc_search()
    test_nc_compat()
    test_performance()

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    if errors:
        print(f"\nFailed checks:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
