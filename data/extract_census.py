"""Extract all cusped-census gluing matrices into data/census.db.

Run once with SnapPy installed. Output is checked into the repo (or regenerated
on demand) and consumed by the Rust core via rusqlite.

Schema:
    manifolds(census, name, n, r, gluing)   PRIMARY KEY (census, name)

where `gluing` is a raw little-endian i32 array of length (n + 2r) * 3n,
row-major. Shape is recovered from n, r.

`pivots` is a raw little-endian i32 array of length (n - r): the sorted
column-pivot indices from scipy.linalg.qr(reduced_edge_coeffs.T, pivoting=True),
identifying an (n-r)-element linearly-independent subset of the n reduced edge
rows. Mirrors v0.5's `_independent_row_indices`.

Censuses covered:
    orientable_cusped      snappy.OrientableCuspedCensus()
    link_exteriors         snappy.LinkExteriors()
    ht_link_exteriors      snappy.HTLinkExteriors()
    nonorientable_cusped   snappy.NonorientableCuspedCensus()

Usage:
    python data/extract_census.py [--db data/census.db] [--limit N]
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
import time
from pathlib import Path


CENSUSES = [
    ("orientable_cusped",   "OrientableCuspedCensus"),
    ("link_exteriors",      "LinkExteriors"),
    ("ht_link_exteriors",   "HTLinkExteriors"),
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

-- Base Neumann-Zagier data (from v0.5's build_neumann_zagier, default M/L basis).
-- g_nz_x2 : i64 LE, row-major (2n × 2n) of 2·g_NZ (always integer)
-- nu_x    : i64 LE, length n
-- nu_p_x2 : i64 LE, length n (2·nu_p, always integer)
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

-- Phase-space basis (from v0.5's find_easy_edges).
-- easy_edges : i32 LE concat of `num_easy` rows of length 3n
-- easy_indep : i32 LE list of indices into easy_edges rows
-- hard_pad   : i32 LE concat of hard rows of length 3n
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
"""


def _reduce_edge_coeffs(flat: list[int], n: int) -> "numpy.ndarray":
    """Port of v0.5's `_reduce_row` across the n edge rows only.

    Returns the n × 2n int array of reduced edge coefficients.
    """
    import numpy as np
    out = np.zeros((n, 2 * n), dtype=int)
    cols = 3 * n
    for row in range(n):
        base = row * cols
        for i in range(n):
            f = flat[base + 3 * i]
            g = flat[base + 3 * i + 1]
            h = flat[base + 3 * i + 2]
            out[row, 2 * i]     = f - g
            out[row, 2 * i + 1] = h - g
    return out


def _independent_row_indices(edge_coeffs, expected_rank: int) -> list[int]:
    """Sorted column-pivot indices from QR on edge_coeffs.T. Mirrors v0.5."""
    if expected_rank == 0:
        return []
    from scipy.linalg import qr
    _, _, piv = qr(edge_coeffs.astype(float).T, pivoting=True)
    return sorted(piv[:expected_rank].tolist())


def extract_one(M) -> tuple[int, int, bytes, bytes]:
    """Return (n, r, gluing_blob, pivots_blob) for a single SnapPy manifold."""
    n = M.num_tetrahedra()
    r = M.num_cusps()
    raw = M.gluing_equations()
    rows, cols = raw.shape
    expected_rows = n + 2 * r
    expected_cols = 3 * n
    if rows != expected_rows or cols != expected_cols:
        raise ValueError(
            f"unexpected gluing shape {(rows, cols)}, "
            f"expected ({expected_rows}, {expected_cols})"
        )

    flat = [int(x) for x in raw.list()]
    if len(flat) != rows * cols:
        raise ValueError(
            f"flat list length {len(flat)} != rows*cols {rows * cols}"
        )
    blob = struct.pack(f"<{len(flat)}i", *flat)

    edge_coeffs = _reduce_edge_coeffs(flat, n)
    pivots = _independent_row_indices(edge_coeffs, n - r)
    pivots_blob = struct.pack(f"<{len(pivots)}i", *pivots)
    return n, r, blob, pivots_blob


def iter_census(census_cls_name: str, limit: int | None):
    import snappy
    cls = getattr(snappy, census_cls_name)
    it = cls()
    for i, M in enumerate(it):
        if limit is not None and i >= limit:
            break
        yield M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        default=str(Path(__file__).parent / "census.db"),
        help="output SQLite path",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap per-census count (smoke test)",
    )
    ap.add_argument(
        "--only",
        choices=[c for c, _ in CENSUSES],
        default=None,
        help="extract only this census",
    )
    ap.add_argument(
        "--phase-space",
        action="store_true",
        help="also populate the phase_space table via v0.5's find_easy_edges "
             "(slow: 4^n pattern enumeration per manifold)",
    )
    ap.add_argument(
        "--names",
        type=str,
        default=None,
        help="comma-separated manifold names — restricts extraction to just these "
             "(useful for phase-space-only runs on fixtures)",
    )
    args = ap.parse_args()

    try:
        import snappy  # noqa: F401
    except ImportError:
        print("snappy is required; `pip install snappy`", file=sys.stderr)
        return 2

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    name_filter = None
    if args.names:
        name_filter = {s.strip() for s in args.names.split(",") if s.strip()}

    if args.phase_space:
        # Lazy import — pulls in scipy + the v0.5 src tree.
        import os
        v05_src = os.environ.get(
            "IREF3D_V05_SRC",
            "/Users/pmp/Documents/Research/ultimate/v0.5/src",
        )
        if v05_src not in sys.path:
            sys.path.insert(0, v05_src)
        from manifold_index.core.phase_space import find_easy_edges

    total = 0
    for census_tag, cls_name in CENSUSES:
        if args.only and census_tag != args.only:
            continue
        print(f"[{census_tag}] extracting via snappy.{cls_name}() ...", flush=True)
        t0 = time.time()
        count = 0
        rows = []
        ps_rows = []
        for M in iter_census(cls_name, args.limit):
            name = M.name()
            if name_filter is not None and name not in name_filter:
                continue
            try:
                n, r, blob, pivots_blob = extract_one(M)
            except Exception as exc:  # pragma: no cover
                print(f"  skip {name}: {exc}", file=sys.stderr)
                continue
            rows.append((census_tag, name, n, r, blob, pivots_blob))
            count += 1

            if args.phase_space:
                try:
                    ps = find_easy_edges(_load_manifold_data(M, n, r))
                except Exception as exc:
                    print(f"  phase_space skip {name}: {exc}", file=sys.stderr)
                else:
                    easy_flat = []
                    for row in ps.all_easy:
                        easy_flat.extend(int(x) for x in row.tolist())
                    easy_blob = struct.pack(f"<{len(easy_flat)}i", *easy_flat)
                    indep = [int(i) for i in ps.independent_easy_indices]
                    indep_blob = struct.pack(f"<{len(indep)}i", *indep)
                    hard_flat = []
                    for row in ps.hard_padding:
                        hard_flat.extend(int(x) for x in row.tolist())
                    hard_blob = struct.pack(f"<{len(hard_flat)}i", *hard_flat)
                    ps_rows.append((
                        census_tag, name,
                        len(ps.all_easy), len(ps.hard_padding),
                        easy_blob, indep_blob, hard_blob,
                    ))

            if len(rows) >= 500:
                conn.executemany(
                    "INSERT OR REPLACE INTO manifolds VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                if ps_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO phase_space VALUES (?,?,?,?,?,?,?)",
                        ps_rows,
                    )
                conn.commit()
                rows.clear()
                ps_rows.clear()
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO manifolds VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        if ps_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO phase_space VALUES (?,?,?,?,?,?,?)",
                ps_rows,
            )
        conn.commit()
        dt = time.time() - t0
        print(f"  {count} manifolds in {dt:.1f}s", flush=True)
        total += count

    print(f"done: {total} manifolds written to {db_path}")
    conn.close()
    return 0


def _load_manifold_data(M, n: int, r: int):
    """Build a v0.5 ManifoldData straight from a snappy Manifold (avoids re-load)."""
    import numpy as np
    from manifold_index.core.manifold import ManifoldData
    raw = M.gluing_equations()
    flat = [int(x) for x in raw.list()]
    gm = np.array(flat, dtype=int).reshape(n + 2 * r, 3 * n)
    return ManifoldData(
        name=M.name(),
        num_tetrahedra=n,
        num_cusps=r,
        gluing_matrix=gm,
        raw=raw,
    )


if __name__ == "__main__":
    raise SystemExit(main())
