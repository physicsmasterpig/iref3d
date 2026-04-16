"""Phase 3 benchmark: Rust vs v0.5 Python on progressively larger workloads.

Tests:
  1. Refined index grid scan (m006, s776 at qq10/20/40)
  2. NC search on slope grids (7×7, 15×15)
  3. Filled refined index (ℓ=1 and ℓ≥2 slopes)

Usage:
    python3 data/benchmark.py
"""
from __future__ import annotations

import sys
import os
import time
from fractions import Fraction
from pathlib import Path

V05_SRC = os.environ.get(
    "IREF3D_V05_SRC",
    "/Users/pmp/Documents/Research/ultimate/v0.5/src",
)
if V05_SRC not in sys.path:
    sys.path.insert(0, V05_SRC)

RUST_LIB = str(Path(__file__).resolve().parent.parent / "core" / "target" / "release")
if RUST_LIB not in sys.path:
    sys.path.insert(0, RUST_LIB)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rust_adapter

from manifold_index.core.manifold import load_manifold
from manifold_index.core.phase_space import find_easy_edges
from manifold_index.core.neumann_zagier import build_neumann_zagier
from manifold_index.core.refined_index import (
    compute_refined_index as v05_refined_index,
)
from manifold_index.core.index_3d import (
    clear_tet_cache,
)
from manifold_index.core.refined_dehn_filling import (
    clear_computation_caches,
    clear_filling_caches,
)
from manifold_index.core.dehn_filling import (
    find_non_closable_cycles as v05_nc_search,
)


def load_v05(name):
    md = load_manifold(name)
    easy = find_easy_edges(md)
    nz = build_neumann_zagier(md, easy)
    return nz


def clear_v05():
    clear_tet_cache()
    clear_computation_caches()
    clear_filling_caches()


def fmt_time(t):
    if t < 0.001:
        return f"{t*1e6:.0f}μs"
    if t < 1:
        return f"{t*1000:.1f}ms"
    return f"{t:.2f}s"


def fmt_speedup(v05_t, rust_t):
    if rust_t < 1e-7:
        return "∞"
    return f"{v05_t / rust_t:.1f}x"


# ── Benchmark 1: Refined index grid scan ──

def bench_refined_grid(name, qq_orders, m_range, e_fracs):
    print(f"\n{'─' * 60}")
    print(f"Refined index grid scan: {name}")
    print(f"{'─' * 60}")

    nz = load_v05(name)
    r = nz.r
    queries = []
    for m0 in range(-m_range, m_range + 1):
        for e in e_fracs:
            m_ext = [m0] + [0] * (r - 1)
            e_ext = [e] + [Fraction(0)] * (r - 1)
            queries.append((m_ext, e_ext))

    for qq in qq_orders:
        n_queries = len(queries)
        print(f"\n  qq={qq}, {n_queries} grid points:")

        # v0.5
        clear_v05()
        t0 = time.perf_counter()
        for m_ext, e_ext in queries:
            v05_refined_index(nz, m_ext, e_ext, q_order_half=qq)
        t_v05 = time.perf_counter() - t0

        # Rust
        rust_adapter.clear_tet_cache()
        t0 = time.perf_counter()
        for m_ext, e_ext in queries:
            rust_adapter.compute_refined_index(nz, m_ext, e_ext, qq)
        t_rust = time.perf_counter() - t0

        print(f"    v0.5:  {fmt_time(t_v05)}  ({fmt_time(t_v05/n_queries)}/query)")
        print(f"    Rust:  {fmt_time(t_rust)}  ({fmt_time(t_rust/n_queries)}/query)")
        print(f"    Speedup: {fmt_speedup(t_v05, t_rust)}")


# ── Benchmark 2: NC search ──

def bench_nc_search(name, grids):
    print(f"\n{'─' * 60}")
    print(f"NC search: {name}")
    print(f"{'─' * 60}")

    nz = load_v05(name)
    qq = 10

    for p_range, q_range in grids:
        p_lo, p_hi = p_range
        q_lo, q_hi = q_range
        label = f"P∈[{p_lo},{p_hi}] Q∈[{q_lo},{q_hi}]"
        print(f"\n  {label}, qq={qq}:")

        # v0.5
        clear_v05()
        t0 = time.perf_counter()
        v05_result = v05_nc_search(
            nz, cusp_idx=0,
            p_range=range(p_lo, p_hi + 1),
            q_range=range(q_lo, q_hi + 1),
            q_order_half=qq,
        )
        t_v05 = time.perf_counter() - t0
        v05_nc = len(v05_result.cycles)

        # Rust
        rust_adapter.clear_tet_cache()
        t0 = time.perf_counter()
        rust_result = rust_adapter.find_non_closable_cycles(
            nz, 0, range(p_lo, p_hi + 1), range(q_lo, q_hi + 1), qq,
        )
        t_rust = time.perf_counter() - t0
        rust_nc = len(rust_result["cycles"])

        print(f"    v0.5:  {fmt_time(t_v05)}  ({v05_nc} NC cycles)")
        print(f"    Rust:  {fmt_time(t_rust)}  ({rust_nc} NC cycles)")
        print(f"    Speedup: {fmt_speedup(t_v05, t_rust)}")
        if v05_nc != rust_nc:
            print(f"    *** MISMATCH: v0.5={v05_nc}, Rust={rust_nc} ***")


# ── Benchmark 3: Filled index ──

def bench_filled(name, slopes, qq_orders):
    print(f"\n{'─' * 60}")
    print(f"Filled unrefined index: {name}")
    print(f"{'─' * 60}")

    nz = load_v05(name)
    from manifold_index.core.dehn_filling import compute_filled_index as v05_filled

    for qq in qq_orders:
        for P, Q in slopes:
            label = f"({P},{Q}) qq={qq}"
            print(f"\n  {label}:")

            clear_v05()
            t0 = time.perf_counter()
            v05_filled(nz, cusp_idx=0, P=P, Q=Q, m_other=[], e_other=[], q_order_half=qq)
            t_v05 = time.perf_counter() - t0

            rust_adapter.clear_tet_cache()
            t0 = time.perf_counter()
            rust_adapter.compute_filled_index(nz, 0, P, Q, [], [], qq)
            t_rust = time.perf_counter() - t0

            print(f"    v0.5:  {fmt_time(t_v05)}")
            print(f"    Rust:  {fmt_time(t_rust)}")
            print(f"    Speedup: {fmt_speedup(t_v05, t_rust)}")


def main():
    print("=" * 60)
    print("Phase 3 Benchmark: Rust vs v0.5 Python")
    print("=" * 60)

    # 1. Refined index grid
    e_fracs = [Fraction(0), Fraction(1, 2), Fraction(-1, 2), Fraction(1)]
    bench_refined_grid("m006", [10, 20, 40], m_range=3, e_fracs=e_fracs)
    bench_refined_grid("s776", [10, 20], m_range=2, e_fracs=e_fracs[:2])

    # 2. NC search
    bench_nc_search("m003", [
        ((-3, 3), (-3, 3)),
        ((-7, 7), (-7, 7)),
    ])
    bench_nc_search("m006", [
        ((-3, 3), (-3, 3)),
        ((-7, 7), (-7, 7)),
    ])

    # 3. Filled index
    bench_filled("m006", [(1, 0), (1, 1), (2, 1), (3, 1)], [10, 20])

    print(f"\n{'=' * 60}")
    print("Benchmark complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
