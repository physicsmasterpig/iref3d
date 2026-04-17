#!/usr/bin/env python3
"""Populate the NZ table for all censuses using multiprocessing.

Designed for M1 Max (10 cores, 64 GB RAM). Skips manifolds that already
have NZ data in the DB. Safe to interrupt and resume.

Usage:
    python3 data/extract_nz_parallel.py [--db data/census.db] [--workers 8]

Prerequisites:
    pip install snappy scipy numpy
    # v0.5 source tree must be at IREF3D_V05_SRC or default path below
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sqlite3
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

V05_SRC_CANDIDATES = [
    # Try env var first, then common locations
    "/Users/pmp/Documents/Research/ultimate/v0.5/src",
    "/Users/pmp/Documents/ultimate/refined-index/v0.5/src",
    os.path.expanduser("~/Documents/Research/ultimate/v0.5/src"),
    os.path.expanduser("~/Documents/ultimate/refined-index/v0.5/src"),
]


def _find_v05_src() -> str:
    env = os.environ.get("IREF3D_V05_SRC")
    if env and os.path.isdir(env):
        return env
    for p in V05_SRC_CANDIDATES:
        if os.path.isdir(os.path.join(p, "manifold_index")):
            return p
    raise RuntimeError(
        "Cannot find v0.5 source tree. Set IREF3D_V05_SRC env var to "
        "the directory containing manifold_index/ (e.g. .../v0.5/src)"
    )


def _init_worker():
    """Per-worker init: import snappy + v0.5 once per process."""
    global _snappy, _find_easy_edges, _build_neumann_zagier, _ManifoldData, _np
    import numpy as _np
    import snappy as _snappy

    v05_src = _find_v05_src()
    if v05_src not in sys.path:
        sys.path.insert(0, v05_src)

    from manifold_index.core.manifold import ManifoldData as _ManifoldData
    from manifold_index.core.phase_space import find_easy_edges as _find_easy_edges
    from manifold_index.core.neumann_zagier import (
        build_neumann_zagier as _build_neumann_zagier,
    )


def _process_one(args: tuple) -> tuple | None:
    """Process a single (census, name, n, r) → NZ row tuple, or None on failure."""
    census_tag, name, n, r = args
    try:
        M = _snappy.Manifold(name)
        raw = M.gluing_equations()
        flat = [int(x) for x in raw.list()]
        gm = _np.array(flat, dtype=int).reshape(n + 2 * r, 3 * n)
        mdata = _ManifoldData(
            name=name,
            num_tetrahedra=n,
            num_cusps=r,
            gluing_matrix=gm,
            raw=raw,
        )
        ps = _find_easy_edges(mdata)
        nz = _build_neumann_zagier(mdata, ps)

        nn = int(nz.n)
        # g_NZ (2n x 2n) -> x2 integers, row-major i64 LE
        g_flat = []
        for i in range(2 * nn):
            for j in range(2 * nn):
                g_flat.append(int(Fraction(nz.g_NZ[i, j]) * 2))
        g_blob = struct.pack(f"<{len(g_flat)}q", *g_flat)

        # nu_x (n,) -> i64 LE
        nux = [int(nz.nu_x[i]) for i in range(nn)]
        nux_blob = struct.pack(f"<{len(nux)}q", *nux)

        # nu_p (n,) -> x2 integers, i64 LE
        nup = [int(Fraction(nz.nu_p[i]) * 2) for i in range(nn)]
        nup_blob = struct.pack(f"<{len(nup)}q", *nup)

        return (
            census_tag, name, nn, int(nz.r),
            int(nz.num_hard), int(nz.num_easy),
            g_blob, nux_blob, nup_blob,
        )
    except Exception as exc:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path(__file__).parent / "census.db"))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--only", default=None,
                    help="restrict to one census tag (e.g. link_exteriors)")
    ap.add_argument("--batch", type=int, default=200,
                    help="commit batch size")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)

    # Get manifolds that DON'T yet have NZ data (safe to resume)
    if args.only:
        todo = conn.execute("""
            SELECT m.census, m.name, m.n, m.r
            FROM manifolds m
            LEFT JOIN nz ON nz.census = m.census AND nz.name = m.name
            WHERE nz.name IS NULL AND m.census = ?
            ORDER BY m.n ASC
        """, [args.only]).fetchall()
    else:
        todo = conn.execute("""
            SELECT m.census, m.name, m.n, m.r
            FROM manifolds m
            LEFT JOIN nz ON nz.census = m.census AND nz.name = m.name
            WHERE nz.name IS NULL
            ORDER BY m.n ASC
        """).fetchall()

    existing = conn.execute("SELECT COUNT(*) FROM nz").fetchone()[0]
    print(f"NZ extraction: {len(todo)} manifolds to process "
          f"({existing} already done), {args.workers} workers", flush=True)

    if not todo:
        print("Nothing to do.")
        return 0

    t0 = time.time()
    done = 0
    failed = 0
    batch = []

    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(_process_one, todo, chunksize=4):
            if result is not None:
                batch.append(result)
                done += 1
            else:
                failed += 1

            if len(batch) >= args.batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO nz VALUES (?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                conn.commit()
                batch.clear()

            total = done + failed
            if total % 500 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(todo) - total) / rate if rate > 0 else 0
                print(f"  {total}/{len(todo)} "
                      f"(ok={done} fail={failed} {rate:.0f}/s ETA {eta:.0f}s)",
                      flush=True)

    # Final flush
    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO nz VALUES (?,?,?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()

    dt = time.time() - t0
    total_nz = conn.execute("SELECT COUNT(*) FROM nz").fetchone()[0]
    conn.close()
    print(f"\nDone: {done} new NZ rows ({failed} failed) in {dt:.0f}s")
    print(f"Total NZ rows in DB: {total_nz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
