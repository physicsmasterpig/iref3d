"""Phase 3 validation: Rust adapter vs v0.5 Python, using v0.5's own data types.

This script:
1. Loads manifolds via v0.5's full pipeline (SnapPy → ManifoldData → NZ)
2. Calls both v0.5 Python and the Rust adapter with identical inputs
3. Compares outputs exactly

Manifolds tested: m003, m004, m006 (plan verification targets).
Operations: refined index, unrefined index, filled index at qq10/qq20.

Usage:
    python data/validate_phase3.py
"""
from __future__ import annotations

import sys
import os
import time
from fractions import Fraction
from pathlib import Path

# ── Setup paths ──
V05_SRC = os.environ.get(
    "IREF3D_V05_SRC",
    "/Users/pmp/Documents/Research/ultimate/v0.5/src",
)
if V05_SRC not in sys.path:
    sys.path.insert(0, V05_SRC)

# Rust dylib
RUST_LIB = str(Path(__file__).resolve().parent.parent / "core" / "target" / "release")
if RUST_LIB not in sys.path:
    sys.path.insert(0, RUST_LIB)

# Import adapter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rust_adapter

# Import v0.5
from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.refined_index import compute_refined_index as v05_refined_index
from manifold_index.core.index_3d import compute_index_3d_python as v05_unrefined_index
from manifold_index.core.dehn_filling import (
    compute_filled_index as v05_filled_index,
    find_non_closable_cycles as v05_nc_search,
)
from manifold_index.core.index_3d import clear_tet_cache
from manifold_index.core.refined_dehn_filling import clear_computation_caches

# ── Test infrastructure ──
pass_count = 0
fail_count = 0
total_count = 0


def check(name: str, rust_val, v05_val, tolerance=None):
    global pass_count, fail_count, total_count
    total_count += 1
    ok = (rust_val == v05_val) if tolerance is None else _approx_eq(rust_val, v05_val, tolerance)
    if ok:
        pass_count += 1
        print(f"  PASS  {name}")
    else:
        fail_count += 1
        print(f"  FAIL  {name}")
        print(f"    rust: {_trunc(rust_val)}")
        print(f"    v05:  {_trunc(v05_val)}")


def _trunc(val, maxlen=200):
    s = repr(val)
    return s if len(s) <= maxlen else s[:maxlen] + "..."


def _approx_eq(a, b, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_approx_eq(a[k], b[k], tol) for k in a)
    if isinstance(a, (int, float, Fraction)) and isinstance(b, (int, float, Fraction)):
        return abs(float(a) - float(b)) < tol
    return a == b


# ── Load manifolds ──
MANIFOLDS = ["m003", "m004", "m006"]


def load_v05(name):
    """Load via v0.5 pipeline."""
    md = load_manifold(name)
    easy = find_easy_edges(md)
    nz = build_neumann_zagier(md, easy)
    return nz


def main():
    print("=" * 60)
    print("Phase 3 validation: Rust adapter vs v0.5 Python")
    print("=" * 60)

    for name in MANIFOLDS:
        print(f"\n── {name} ──")
        nz = load_v05(name)
        print(f"  n={nz.n}, r={nz.r}, num_hard={nz.num_hard}")

        # Clear v0.5 caches between manifolds
        clear_tet_cache()
        clear_computation_caches()

        r = nz.r
        e_probes = [
            [Fraction(0)] * r,
            [Fraction(1, 2)] + [Fraction(0)] * (r - 1),
        ]

        for qq in [10, 20]:
            for e_ext in e_probes:
                m_ext = [0] * r
                e_label = ",".join(str(e) for e in e_ext)

                # ── Refined index ──
                t0 = time.perf_counter()
                v05_ref = v05_refined_index(nz, m_ext, e_ext, q_order_half=qq)
                t_v05 = time.perf_counter() - t0

                t0 = time.perf_counter()
                rust_ref = rust_adapter.compute_refined_index(nz, m_ext, e_ext, qq)
                t_rust = time.perf_counter() - t0

                check(
                    f"refined_index m={m_ext} e=[{e_label}] qq={qq}",
                    rust_ref, dict(v05_ref),
                )
                speedup = t_v05 / t_rust if t_rust > 0 else float("inf")
                print(f"         v05={t_v05:.3f}s  rust={t_rust:.3f}s  ({speedup:.1f}x)")

                # ── Unrefined index ──
                v05_unref = v05_unrefined_index(nz, m_ext, e_ext, q_order_half=qq)
                rust_unref = rust_adapter.compute_index_3d_python(nz, m_ext, e_ext, qq)

                # Both return sparse dict[power, coeff] — convert v0.5 result
                v05_unref_dict = {}
                for i, c in enumerate(v05_unref.coeffs):
                    if c != 0:
                        v05_unref_dict[v05_unref.min_power + i] = c

                check(
                    f"unrefined_index m={m_ext} e=[{e_label}] qq={qq}",
                    rust_unref, v05_unref_dict,
                )

    # ── Filled index (plan target: m006 at qq10, qq20) ──
    print(f"\n── m006 filled index ──")
    nz_m006 = load_v05("m006")
    clear_tet_cache()
    clear_computation_caches()

    for qq in [10, 20]:
        for P, Q in [(1, 0), (1, 1)]:
            # v0.5 filled index
            v05_filled = v05_filled_index(
                nz_m006, cusp_idx=0, P=P, Q=Q,
                m_other=[], e_other=[],
                q_order_half=qq,
            )
            # Convert v0.5 series to {power: Fraction}
            v05_series = {}
            for k, v in v05_filled.series.items():
                f = Fraction(v)
                if f != 0:
                    v05_series[k] = (f.numerator, f.denominator)

            # Rust returns {"series": {power: (numer, denom)}, ...}
            rust_result = rust_adapter.compute_filled_index(
                nz_m006, 0, P, Q, [], [], qq,
            )
            rust_series = {}
            for k, v in rust_result["series"].items():
                n, d = v
                if n != 0:
                    rust_series[k] = (n, d)

            check(
                f"filled_index P={P} Q={Q} qq={qq}",
                rust_series, v05_series,
            )

    print(f"\n{'=' * 60}")
    print(f"Results: {pass_count}/{total_count} passed, {fail_count} failed")
    print(f"{'=' * 60}")
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
