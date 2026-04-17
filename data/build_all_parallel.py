#!/usr/bin/env python3
"""Build census.db from scratch using multiprocessing.

Runs all three phases:
  1. Manifolds table  — gluing equations + pivots (SnapPy + scipy)
  2. Aliases table     — SnapPy identify()
  3. NZ table          — Neumann-Zagier data (v0.5 find_easy_edges + build_neumann_zagier)

Each phase skips rows already present, so it's safe to interrupt and resume.

Usage:
    python3 data/build_all_parallel.py [--workers 8] [--db data/census.db]
    python3 data/build_all_parallel.py --phase manifolds   # only step 1
    python3 data/build_all_parallel.py --phase aliases      # only step 2
    python3 data/build_all_parallel.py --phase nz           # only step 3
    python3 data/build_all_parallel.py --phase nz --only orientable_cusped

Prerequisites:
    pip install snappy scipy numpy
    v0.5 source tree auto-detected or set IREF3D_V05_SRC
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sqlite3
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

# ── Census definitions ──

CENSUSES = [
    ("orientable_cusped",    "OrientableCuspedCensus"),
    ("link_exteriors",       "LinkExteriors"),
    ("ht_link_exteriors",    "HTLinkExteriors"),
    ("nonorientable_cusped", "NonorientableCuspedCensus"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS manifolds (
    census  TEXT    NOT NULL,
    name    TEXT    NOT NULL,
    n       INTEGER NOT NULL,
    r       INTEGER NOT NULL,
    gluing  BLOB    NOT NULL,
    pivots  BLOB    NOT NULL,
    PRIMARY KEY (census, name)
);
CREATE INDEX IF NOT EXISTS idx_name ON manifolds(name);

CREATE TABLE IF NOT EXISTS nz (
    census   TEXT    NOT NULL,
    name     TEXT    NOT NULL,
    n        INTEGER NOT NULL,
    r        INTEGER NOT NULL,
    num_hard INTEGER NOT NULL,
    num_easy INTEGER NOT NULL,
    g_nz_x2  BLOB    NOT NULL,
    nu_x     BLOB    NOT NULL,
    nu_p_x2  BLOB    NOT NULL,
    PRIMARY KEY (census, name)
);

CREATE TABLE IF NOT EXISTS phase_space (
    census     TEXT    NOT NULL,
    name       TEXT    NOT NULL,
    num_easy   INTEGER NOT NULL,
    num_hard   INTEGER NOT NULL,
    easy_edges BLOB    NOT NULL,
    easy_indep BLOB    NOT NULL,
    hard_pad   BLOB    NOT NULL,
    PRIMARY KEY (census, name)
);

CREATE TABLE IF NOT EXISTS aliases (
    alias   TEXT    NOT NULL,
    census  TEXT    NOT NULL,
    name    TEXT    NOT NULL,
    PRIMARY KEY (alias)
);
CREATE INDEX IF NOT EXISTS idx_alias ON aliases(alias);
"""

# ── v0.5 source discovery ──

V05_SRC_CANDIDATES = [
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


# ── Phase 1: Manifolds ──

def _init_manifold_worker():
    global _snappy, _np, _qr
    import numpy as _np
    import snappy as _snappy
    from scipy.linalg import qr as _qr


def _reduce_edge_coeffs(flat, n):
    import numpy as np
    out = np.zeros((n, 2 * n), dtype=int)
    cols = 3 * n
    for row in range(n):
        base = row * cols
        for i in range(n):
            f = flat[base + 3 * i]
            g = flat[base + 3 * i + 1]
            h = flat[base + 3 * i + 2]
            out[row, 2 * i] = f - g
            out[row, 2 * i + 1] = h - g
    return out


def _extract_manifold(args):
    """(census_tag, cls_name, idx) -> (census, name, n, r, gluing_blob, pivots_blob) or None."""
    census_tag, cls_name, idx = args
    try:
        cls = getattr(_snappy, cls_name)
        M = cls()[idx]
        name = M.name()
        n = M.num_tetrahedra()
        r = M.num_cusps()
        raw = M.gluing_equations()
        flat = [int(x) for x in raw.list()]
        blob = struct.pack(f"<{len(flat)}i", *flat)

        edge_coeffs = _reduce_edge_coeffs(flat, n)
        expected_rank = n - r
        if expected_rank > 0:
            _, _, piv = _qr(edge_coeffs.astype(float).T, pivoting=True)
            pivots = sorted(piv[:expected_rank].tolist())
        else:
            pivots = []
        pivots_blob = struct.pack(f"<{len(pivots)}i", *pivots)

        return (census_tag, name, n, r, blob, pivots_blob)
    except Exception as exc:
        return None


def phase_manifolds(conn, workers, only=None):
    """Phase 1: populate manifolds table."""
    import snappy

    # Figure out which (census, index) pairs still need processing
    existing = set()
    for row in conn.execute("SELECT census, name FROM manifolds"):
        existing.add((row[0], row[1]))

    todo = []
    for census_tag, cls_name in CENSUSES:
        if only and census_tag != only:
            continue
        cls = getattr(snappy, cls_name)
        census_list = cls()
        total_in_census = len(census_list)
        # We need to check which indices are already done
        # First pass: get all names in this census to check
        names_in_db = {r[0] for r in conn.execute(
            "SELECT name FROM manifolds WHERE census = ?", [census_tag]
        )}
        for idx in range(total_in_census):
            # We can't easily check by index, so just add all and let INSERT OR IGNORE handle dupes
            todo.append((census_tag, cls_name, idx))
        skip = len(names_in_db)
        print(f"  [{census_tag}] {total_in_census} total, {skip} already in DB", flush=True)

    if not todo:
        print("  Nothing to do for manifolds.")
        return

    print(f"\n[Phase 1: Manifolds] {len(todo)} to process, {workers} workers", flush=True)
    t0 = time.time()
    done = 0
    failed = 0
    batch = []

    with mp.Pool(workers, initializer=_init_manifold_worker) as pool:
        for result in pool.imap_unordered(_extract_manifold, todo, chunksize=16):
            if result is not None:
                batch.append(result)
                done += 1
            else:
                failed += 1

            if len(batch) >= 500:
                conn.executemany(
                    "INSERT OR IGNORE INTO manifolds VALUES (?,?,?,?,?,?)",
                    batch,
                )
                conn.commit()
                batch.clear()

            total = done + failed
            if total % 2000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(todo) - total) / rate if rate > 0 else 0
                print(f"  {total}/{len(todo)} "
                      f"(ok={done} fail={failed} {rate:.0f}/s ETA {eta:.0f}s)",
                      flush=True)

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO manifolds VALUES (?,?,?,?,?,?)",
            batch,
        )
        conn.commit()

    dt = time.time() - t0
    total_m = conn.execute("SELECT COUNT(*) FROM manifolds").fetchone()[0]
    print(f"  Phase 1 done: {done} added ({failed} failed) in {dt:.0f}s. "
          f"Total manifolds: {total_m}", flush=True)


# ── Phase 2: Aliases ──

def _init_alias_worker():
    global _snappy_a, _re_a
    import snappy as _snappy_a
    import re as _re_a


def _extract_aliases(args):
    """(census_tag, name) -> list of (alias, census, name) or empty list."""
    census_tag, name = args
    try:
        M = _snappy_a.Manifold(name)
        ids = M.identify()
    except Exception:
        return []

    results = []
    for ident in ids:
        alias = _re_a.sub(r'\(.*$', '', str(ident)).strip()
        if alias and alias != name:
            results.append((alias, census_tag, name))
    return results


def phase_aliases(conn, workers):
    """Phase 2: populate aliases table."""
    # Get manifolds that don't have aliases yet — but since aliases map
    # TO manifolds (not FROM), we just process all manifolds and INSERT OR IGNORE.
    rows = conn.execute(
        "SELECT census, name FROM manifolds "
        "ORDER BY CASE census "
        "  WHEN 'orientable_cusped' THEN 0 "
        "  WHEN 'link_exteriors' THEN 1 "
        "  WHEN 'nonorientable_cusped' THEN 2 "
        "  ELSE 3 END, name"
    ).fetchall()

    existing = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    print(f"\n[Phase 2: Aliases] {len(rows)} manifolds to scan, "
          f"{existing} aliases already in DB, {workers} workers", flush=True)

    t0 = time.time()
    done = 0
    alias_count = 0
    batch = []

    with mp.Pool(workers, initializer=_init_alias_worker) as pool:
        for result in pool.imap_unordered(_extract_aliases, rows, chunksize=32):
            batch.extend(result)
            alias_count += len(result)
            done += 1

            if len(batch) >= 500:
                conn.executemany(
                    "INSERT OR IGNORE INTO aliases VALUES (?,?,?)",
                    batch,
                )
                conn.commit()
                batch.clear()

            if done % 5000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(rows) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(rows)} manifolds scanned, "
                      f"{alias_count} aliases ({rate:.0f}/s ETA {eta:.0f}s)",
                      flush=True)

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO aliases VALUES (?,?,?)",
            batch,
        )
        conn.commit()

    dt = time.time() - t0
    total_a = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    print(f"  Phase 2 done: {alias_count} aliases from {done} manifolds in {dt:.0f}s. "
          f"Total aliases: {total_a}", flush=True)


# ── Phase 3: NZ ──

def _init_nz_worker():
    global _snappy_nz, _np_nz, _fee, _bnz, _MD_nz
    import numpy as _np_nz
    import snappy as _snappy_nz

    v05_src = _find_v05_src()
    if v05_src not in sys.path:
        sys.path.insert(0, v05_src)

    from manifold_index.core.manifold import ManifoldData as _MD_nz
    from manifold_index.core.phase_space import find_easy_edges as _fee
    from manifold_index.core.neumann_zagier import build_neumann_zagier as _bnz


def _extract_nz(args):
    """(census, name, n, r) -> NZ row tuple or None."""
    census_tag, name, n, r = args
    try:
        M = _snappy_nz.Manifold(name)
        raw = M.gluing_equations()
        flat = [int(x) for x in raw.list()]
        gm = _np_nz.array(flat, dtype=int).reshape(n + 2 * r, 3 * n)
        mdata = _MD_nz(
            name=name,
            num_tetrahedra=n,
            num_cusps=r,
            gluing_matrix=gm,
            raw=raw,
        )
        ps = _fee(mdata)
        nz = _bnz(mdata, ps)

        nn = int(nz.n)
        g_flat = []
        for i in range(2 * nn):
            for j in range(2 * nn):
                g_flat.append(int(Fraction(nz.g_NZ[i, j]) * 2))
        g_blob = struct.pack(f"<{len(g_flat)}q", *g_flat)
        nux = [int(nz.nu_x[i]) for i in range(nn)]
        nux_blob = struct.pack(f"<{len(nux)}q", *nux)
        nup = [int(Fraction(nz.nu_p[i]) * 2) for i in range(nn)]
        nup_blob = struct.pack(f"<{len(nup)}q", *nup)

        return (
            census_tag, name, nn, int(nz.r),
            int(nz.num_hard), int(nz.num_easy),
            g_blob, nux_blob, nup_blob,
        )
    except Exception:
        return None


def phase_nz(conn, workers, only=None):
    """Phase 3: populate NZ table."""
    if only:
        todo = conn.execute("""
            SELECT m.census, m.name, m.n, m.r
            FROM manifolds m
            LEFT JOIN nz ON nz.census = m.census AND nz.name = m.name
            WHERE nz.name IS NULL AND m.census = ?
            ORDER BY m.n ASC
        """, [only]).fetchall()
    else:
        todo = conn.execute("""
            SELECT m.census, m.name, m.n, m.r
            FROM manifolds m
            LEFT JOIN nz ON nz.census = m.census AND nz.name = m.name
            WHERE nz.name IS NULL
            ORDER BY m.n ASC
        """).fetchall()

    existing = conn.execute("SELECT COUNT(*) FROM nz").fetchone()[0]
    print(f"\n[Phase 3: NZ] {len(todo)} manifolds to process "
          f"({existing} already done), {workers} workers", flush=True)

    if not todo:
        print("  Nothing to do.")
        return

    t0 = time.time()
    done = 0
    failed = 0
    batch = []

    with mp.Pool(workers, initializer=_init_nz_worker) as pool:
        for result in pool.imap_unordered(_extract_nz, todo, chunksize=4):
            if result is not None:
                batch.append(result)
                done += 1
            else:
                failed += 1

            if len(batch) >= 200:
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

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO nz VALUES (?,?,?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()

    dt = time.time() - t0
    total_nz = conn.execute("SELECT COUNT(*) FROM nz").fetchone()[0]
    print(f"  Phase 3 done: {done} added ({failed} failed) in {dt:.0f}s. "
          f"Total NZ rows: {total_nz}", flush=True)


# ── Main ──

def main():
    ap = argparse.ArgumentParser(
        description="Build census.db from scratch (parallel)."
    )
    ap.add_argument("--db", default=str(Path(__file__).parent / "census.db"))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--phase", choices=["manifolds", "aliases", "nz", "all"],
                    default="all",
                    help="run only one phase (default: all three)")
    ap.add_argument("--only", default=None,
                    help="restrict to one census tag (manifolds/nz phases)")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    print(f"Database: {db_path}")
    print(f"Workers:  {args.workers}")
    print(f"Phase:    {args.phase}")
    if args.only:
        print(f"Census:   {args.only}")

    t_total = time.time()

    if args.phase in ("all", "manifolds"):
        phase_manifolds(conn, args.workers, only=args.only)

    if args.phase in ("all", "aliases"):
        phase_aliases(conn, args.workers)

    if args.phase in ("all", "nz"):
        phase_nz(conn, args.workers, only=args.only)

    dt = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All done in {dt:.0f}s")
    for table in ["manifolds", "aliases", "nz"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n} rows")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
