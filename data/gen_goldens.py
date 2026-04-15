"""Generate JSON golden fixtures from v0.5 for Rust test diffs.

Each fixture is one JSON file under core/tests/goldens/<module>/ and contains
the inputs plus the v0.5 output. Rust tests load the JSON, recompute with the
Rust implementation, and assert exact equality.

Fixtures are grouped by module matching the Rust module tree:
    kernel/        tet_index_series, tet_degree_x2, poly_convolve
    (more added as each module is ported)

Usage:
    python data/gen_goldens.py [--out core/tests/goldens]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path


V05_SRC = "/Users/pmp/Documents/Research/ultimate/v0.5/src"


def load_v05():
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from manifold_index.core.index_3d import (
        _tet_index_series,
        _tet_degree_x2,
    )
    try:
        from manifold_index.core._c_kernel._c_tet_index import (
            poly_convolve as _c_poly_convolve,
        )
    except ImportError:
        _c_poly_convolve = None
    return _tet_index_series, _tet_degree_x2, _c_poly_convolve


def load_v05_manifold():
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from manifold_index.core.manifold import load_manifold
    return load_manifold


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def gen_kernel(out_root: Path) -> int:
    """Kernel-module goldens: tet_index_series, tet_degree_x2, poly_convolve."""
    tet_index_series, tet_degree_x2, poly_convolve = load_v05()
    out = out_root / "kernel"

    # Grid: m, e ∈ [-4, 4], qq_order ∈ {4, 10, 20}.
    ms = list(range(-4, 5))
    es = list(range(-4, 5))
    qq_orders = [4, 10, 20]

    tet_series_cases = []
    for m, e, qq in product(ms, es, qq_orders):
        series = tet_index_series(m, e, qq)
        # serialize dict[int,int] as list of [k, v] pairs, sorted by k
        pairs = sorted(series.items())
        tet_series_cases.append(
            {"m": m, "e": e, "qq_order": qq, "series": pairs}
        )
    write_json(out / "tet_index_series.json", {"cases": tet_series_cases})

    deg_cases = []
    for m, e in product(ms, es):
        deg_cases.append({"m": m, "e": e, "degree_x2": tet_degree_x2(m, e)})
    write_json(out / "tet_degree_x2.json", {"cases": deg_cases})

    count = len(tet_series_cases) + len(deg_cases)

    if poly_convolve is not None:
        # Small convolution cases — use series from tet_index at (m,e,qq=8)
        conv_cases = []
        seed_inputs = [
            (0, 0, 8),
            (1, 0, 8),
            (0, 1, 8),
            (-1, 2, 8),
            (2, -1, 8),
        ]
        budgets = [4, 8, 16]
        for (m1, e1, qq), (m2, e2, _) in [
            (a, b) for a in seed_inputs for b in seed_inputs
        ][:20]:  # cap 20 pairs
            p1 = tet_index_series(m1, e1, qq)
            p2 = tet_index_series(m2, e2, qq)
            for budget in budgets:
                result = poly_convolve(p1, p2, budget)
                conv_cases.append(
                    {
                        "lhs": sorted(p1.items()),
                        "rhs": sorted(p2.items()),
                        "budget": budget,
                        "result": sorted(result.items()),
                    }
                )
        write_json(out / "poly_convolve.json", {"cases": conv_cases})
        count += len(conv_cases)

    return count


def gen_census(out_root: Path) -> int:
    """Census-module goldens: a handful of manifolds, full gluing matrices + pivots."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from manifold_index.core.gluing_equations import reduce_gluing_equations

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1", "m000"]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            rg = reduce_gluing_equations(md)
        except Exception as exc:  # pragma: no cover
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        cases.append({
            "name": name,
            "n": md.num_tetrahedra,
            "r": md.num_cusps,
            "gluing": md.gluing_matrix.astype(int).reshape(-1).tolist(),
            "pivots": [int(p) for p in rg.independent_edge_indices],
        })
    write_json(out_root / "census" / "manifolds.json", {"cases": cases})
    return len(cases)


def gen_gluing(out_root: Path) -> int:
    """Gluing-module goldens: reduced edge/cusp coeffs + consts per manifold."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from manifold_index.core.gluing_equations import reduce_gluing_equations

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1", "m000"]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            rg = reduce_gluing_equations(md)
        except Exception as exc:  # pragma: no cover
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        cases.append({
            "name": name,
            "n": rg.n,
            "r": rg.r,
            "edge_coeffs": rg.edge_coeffs.astype(int).reshape(-1).tolist(),
            "edge_consts": rg.edge_consts.astype(int).tolist(),
            "cusp_coeffs": rg.cusp_coeffs.astype(int).reshape(-1).tolist(),
            "cusp_consts": rg.cusp_consts.astype(int).tolist(),
            "pivots": [int(p) for p in rg.independent_edge_indices],
        })
    write_json(out_root / "gluing" / "reduced.json", {"cases": cases})
    return len(cases)


def gen_phase_space(out_root: Path) -> int:
    """phase_space-module goldens: easy_edges, indep indices, hard padding."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from manifold_index.core.phase_space import find_easy_edges

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1", "m000"]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
        except Exception as exc:  # pragma: no cover
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        cases.append({
            "name": name,
            "n": md.num_tetrahedra,
            "r": md.num_cusps,
            "easy_edges": [row.astype(int).tolist() for row in ps.all_easy],
            "easy_indep": [int(i) for i in ps.independent_easy_indices],
            "hard_padding": [row.astype(int).tolist() for row in ps.hard_padding],
        })
    write_json(out_root / "phase_space" / "basis.json", {"cases": cases})
    return len(cases)


def gen_nz(out_root: Path) -> int:
    """NZ-module goldens: base NZ + post-basis-change NZ for several slopes."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import (
        build_neumann_zagier, apply_cusp_basis_change,
    )

    def dump_nz(nz):
        # `2·g_NZ`, `ν_x`, `2·ν_p` as int arrays for exact Rust-side reconstruction.
        import numpy as np
        g2 = np.round(2.0 * nz.g_NZ).astype(int).reshape(-1).tolist()
        nu_x = nz.nu_x.astype(int).tolist()
        nu_p2 = np.round(2.0 * nz.nu_p).astype(int).tolist()
        return {
            "n": nz.n, "r": nz.r,
            "num_hard": nz.num_hard, "num_easy": nz.num_easy,
            "g_nz_x2": g2, "nu_x": nu_x, "nu_p_x2": nu_p2,
        }

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1"]
    slopes_to_try = [(1, 0), (3, 1), (5, 2), (-3, 1), (1, 2), (3, -2)]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz0 = build_neumann_zagier(md, ps)
        except Exception as exc:  # pragma: no cover
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        entry = {
            "name": name,
            "base": dump_nz(nz0),
            "basis_changes": [],
        }
        for (P, Q) in slopes_to_try:
            if P % 2 == 0:
                continue
            for k in range(nz0.r):
                try:
                    nz1 = apply_cusp_basis_change(nz0, k, P, Q)
                except Exception as exc:
                    continue
                entry["basis_changes"].append({
                    "cusp_idx": k, "P": P, "Q": Q, "result": dump_nz(nz1),
                })
        cases.append(entry)
    write_json(out_root / "nz" / "basis_changes.json", {"cases": cases})
    return len(cases)


def gen_summation(out_root: Path) -> int:
    """Summation-module goldens: g_NZ_inv_scaled (S, S·g_NZ⁻¹) per manifold,
    plus post-basis-change variants to exercise NzData after apply_cusp_basis_change."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import (
        build_neumann_zagier, apply_cusp_basis_change,
    )

    def dump_inv(nz):
        S, arr = nz.g_NZ_inv_scaled()
        return {"S": int(S), "matrix": [int(x) for x in arr.reshape(-1).tolist()]}

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1"]
    slopes = [(1, 0), (3, 1), (5, 2), (-3, 1)]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz0 = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        entry = {"name": name, "base": dump_inv(nz0), "basis_changes": []}
        for (P, Q) in slopes:
            if P % 2 == 0:
                continue
            for k in range(nz0.r):
                try:
                    nz1 = apply_cusp_basis_change(nz0, k, P, Q)
                except Exception:
                    continue
                entry["basis_changes"].append({
                    "cusp_idx": k, "P": P, "Q": Q, "result": dump_inv(nz1),
                })
        cases.append(entry)
    write_json(out_root / "summation" / "g_nz_inv_scaled.json", {"cases": cases})
    return len(cases)


def gen_summation_terms(out_root: Path) -> int:
    """End-to-end enumerate_summation_terms goldens.

    For each manifold, pick a few (m_ext, e_ext, qq_order) triples and dump the
    full sorted list of contributing terms.
    """
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.index_3d import enumerate_summation_terms

    def term_key(t):
        return (
            tuple(int(2 * x) for x in t["e_int"]),
            int(t["phase_exp"]),
            tuple((int(a), int(b)) for a, b in t["tet_args"]),
        )

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1"]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        entry = {"name": name, "queries": []}
        # A handful of (m_ext, e_ext, qq) triples.
        triples = []
        r = nz.r
        triples.append(([0] * r, [0] * r, 10))
        triples.append(([1] + [0] * (r - 1), [0] * r, 10))
        triples.append(([0] * r, [1] + [0] * (r - 1), 10))
        triples.append(([0] * r, [Fraction(1, 2)] + [0] * (r - 1), 8))
        for (m_ext, e_ext, qq) in triples:
            terms = enumerate_summation_terms(nz, m_ext, e_ext, qq)
            # Sort by (e_int_x2, phase_exp, tet_args) for stable comparison.
            terms_sorted = sorted(terms, key=term_key)
            dumped = []
            for t in terms_sorted:
                dumped.append({
                    "e_int_x2": [int(2 * x) for x in t["e_int"]],
                    "phase_exp": int(t["phase_exp"]),
                    "tet_args": [[int(a), int(b)] for a, b in t["tet_args"]],
                    "min_degree_x2": int(2 * t["min_degree"]),
                })
            entry["queries"].append({
                "m_ext": [int(x) for x in m_ext],
                "e_ext_x2": [int(Fraction(x) * 2) for x in e_ext],
                "qq_order": int(qq),
                "terms": dumped,
            })
        cases.append(entry)
    write_json(out_root / "summation" / "terms.json", {"cases": cases})
    return len(cases)


def gen_index_unrefined(out_root: Path) -> int:
    """Unrefined 3D index goldens: coeffs/min_power/n_terms per query."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.index_3d import compute_index_3d_python

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1"]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        r = nz.r
        triples = [
            ([0] * r, [0] * r, 10),
            ([1] + [0] * (r - 1), [0] * r, 10),
            ([0] * r, [1] + [0] * (r - 1), 10),
            ([0] * r, [Fraction(1, 2)] + [0] * (r - 1), 8),
            ([0] * r, [0] * r, 20),
        ]
        queries = []
        for (m_ext, e_ext, qq) in triples:
            res = compute_index_3d_python(nz, m_ext, e_ext, qq)
            queries.append({
                "m_ext": [int(x) for x in m_ext],
                "e_ext_x2": [int(Fraction(x) * 2) for x in e_ext],
                "qq_order": int(qq),
                "coeffs": [int(c) for c in res.coeffs],
                "min_power": int(res.min_power),
                "n_terms": int(res.n_terms),
            })
        cases.append({"name": name, "queries": queries})
    write_json(out_root / "index_unrefined" / "results.json", {"cases": cases})
    return len(cases)


def gen_index_refined(out_root: Path) -> int:
    """Refined 3D index goldens: full keyed results per query."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1"]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        r = nz.r
        triples = [
            ([0] * r, [0] * r, 10),
            ([1] + [0] * (r - 1), [0] * r, 10),
            ([0] * r, [1] + [0] * (r - 1), 10),
            ([0] * r, [Fraction(1, 2)] + [0] * (r - 1), 8),
        ]
        queries = []
        for (m_ext, e_ext, qq) in triples:
            res = compute_refined_index(nz, m_ext, e_ext, qq)
            # Serialize as sorted list of (key_tuple, coeff).
            items = sorted(res.items())
            queries.append({
                "m_ext": [int(x) for x in m_ext],
                "e_ext_x2": [int(Fraction(x) * 2) for x in e_ext],
                "qq_order": int(qq),
                "num_hard": int(nz.num_hard),
                "items": [
                    {"key": [int(x) for x in k], "coeff": int(v)}
                    for k, v in items
                ],
            })
        cases.append({"name": name, "queries": queries})
    write_json(out_root / "index_refined" / "results.json", {"cases": cases})
    return len(cases)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "core" / "tests" / "goldens"),
        help="output directory",
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    print("kernel …", flush=True)
    total += gen_kernel(out_root)
    print("census …", flush=True)
    total += gen_census(out_root)
    print("gluing …", flush=True)
    total += gen_gluing(out_root)
    print("phase_space …", flush=True)
    total += gen_phase_space(out_root)
    print("nz …", flush=True)
    total += gen_nz(out_root)
    print("summation …", flush=True)
    total += gen_summation(out_root)
    total += gen_summation_terms(out_root)
    print("index_unrefined …", flush=True)
    total += gen_index_unrefined(out_root)
    print("index_refined …", flush=True)
    total += gen_index_refined(out_root)
    # future modules: gen_nz, gen_summation, gen_index, gen_refined,
    # gen_weyl, gen_adjoint_*, gen_dehn, gen_refined_dehn.

    print(f"done: {total} cases written under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
