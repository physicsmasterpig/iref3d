"""Build the aliases table in census.db from SnapPy's identify().

Much faster than running --aliases through extract_census.py because it
skips all gluing/NZ computation — just reads names from the DB and calls
identify() on each.

Usage:
    python data/build_aliases.py [--db data/census.db] [--limit N]
"""
import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path(__file__).parent / "census.db"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    try:
        import snappy
    except ImportError:
        print("snappy required: pip install snappy", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            alias   TEXT NOT NULL,
            census  TEXT NOT NULL,
            name    TEXT NOT NULL,
            PRIMARY KEY (alias)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alias ON aliases(alias)")

    # Get all manifold names — prioritise orientable_cusped (has knot names like 4_1)
    rows = conn.execute(
        "SELECT census, name FROM manifolds "
        "ORDER BY CASE census "
        "  WHEN 'orientable_cusped' THEN 0 "
        "  WHEN 'link_exteriors' THEN 1 "
        "  WHEN 'nonorientable_cusped' THEN 2 "
        "  ELSE 3 END, name"
    ).fetchall()
    if args.limit:
        rows = rows[:args.limit]

    print(f"Scanning {len(rows)} manifolds for aliases...", flush=True)
    t0 = time.time()
    alias_batch = []
    count = 0
    skipped = 0

    for census_tag, name in rows:
        try:
            M = snappy.Manifold(name)
            ids = M.identify()
        except Exception:
            skipped += 1
            continue

        for ident in ids:
            alias = re.sub(r'\(.*$', '', str(ident)).strip()
            if alias and alias != name:
                alias_batch.append((alias, census_tag, name))

        count += 1
        if count % 1000 == 0:
            elapsed = time.time() - t0
            rate = count / elapsed
            eta = (len(rows) - count) / rate if rate > 0 else 0
            print(f"  {count}/{len(rows)} ({rate:.0f}/s, ETA {eta:.0f}s) ...",
                  flush=True)

        if len(alias_batch) >= 500:
            conn.executemany(
                "INSERT OR IGNORE INTO aliases VALUES (?, ?, ?)",
                alias_batch,
            )
            conn.commit()
            alias_batch.clear()

    if alias_batch:
        conn.executemany(
            "INSERT OR IGNORE INTO aliases VALUES (?, ?, ?)",
            alias_batch,
        )
        conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
    dt = time.time() - t0
    print(f"Done: {total} aliases from {count} manifolds in {dt:.1f}s "
          f"({skipped} skipped)")
    conn.close()


if __name__ == "__main__":
    main()
