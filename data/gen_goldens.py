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


def gen_ab_vectors(out_root: Path) -> int:
    """ABVectors goldens: for single-cusp manifolds, build a grid of refined-index
    entries and dump compute_ab_vectors output."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index
    from manifold_index.core.weyl_check import compute_ab_vectors

    targets = ["m003", "m004", "m006"]
    qq = 10
    m_grid = [-2, -1, 0, 1, 2]
    e_grid = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]

    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if nz.r != 1:
            continue
        entries = []
        entries_dump = []
        for m in m_grid:
            for e in e_grid:
                res = compute_refined_index(nz, [m], [e], qq)
                entries.append(([m], [e], res))
                entries_dump.append({
                    "m_ext": [m],
                    "e_ext_x2": [int(e * 2)],
                    "items": sorted(
                        ({"key": [int(x) for x in k], "coeff": int(v)} for k, v in res.items()),
                        key=lambda d: d["key"],
                    ),
                })
        ab = compute_ab_vectors(entries, nz.num_hard)
        if ab is None:
            ab_dump = None
        else:
            ab_dump = {
                "a_num": [int(v.numerator) for v in ab.a],
                "a_den": [int(v.denominator) for v in ab.a],
                "b_num": [int(v.numerator) for v in ab.b],
                "b_den": [int(v.denominator) for v in ab.b],
                "num_hard": int(ab.num_hard),
            }
        cases.append({
            "name": name,
            "num_hard": int(nz.num_hard),
            "qq_order": qq,
            "entries": entries_dump,
            "ab": ab_dump,
        })
    write_json(out_root / "ab_vectors" / "results.json", {"cases": cases})
    return len(cases)


def gen_weyl_symmetry(out_root: Path) -> int:
    """Weyl-symmetry goldens: check_weyl_symmetry + strip_weyl_monomial per entry."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index
    from manifold_index.core.weyl_check import (
        compute_ab_vectors, check_weyl_symmetry, strip_weyl_monomial,
    )

    targets = ["m003", "m004", "m006"]
    qq = 10
    m_grid = [-2, -1, 0, 1, 2]
    e_grid = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]

    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if nz.r != 1:
            continue
        entries = []
        for m in m_grid:
            for e in e_grid:
                res = compute_refined_index(nz, [m], [e], qq)
                entries.append(([m], [e], res))
        ab = compute_ab_vectors(entries, nz.num_hard)
        if ab is None:
            continue
        sym = check_weyl_symmetry(entries, nz.num_hard, ab, q_order_half=qq)
        sym_dump = []
        strips_dump = []
        for (m_ext, e_ext, res) in entries:
            key = (tuple(m_ext), tuple(e_ext))
            ok = bool(sym.get(key, False))
            sym_dump.append({
                "m_ext": list(m_ext),
                "e_ext_x2": [int(Fraction(v) * 2) for v in e_ext],
                "ok": ok,
            })
            centre, stripped = strip_weyl_monomial(res, m_ext, e_ext, ab, nz.num_hard)
            strips_dump.append({
                "m_ext": list(m_ext),
                "e_ext_x2": [int(Fraction(v) * 2) for v in e_ext],
                "centre_num": [int(c.numerator) for c in centre],
                "centre_den": [int(c.denominator) for c in centre],
                "stripped": sorted(
                    ({"key": [int(x) for x in k], "coeff": int(v)} for k, v in stripped.items()),
                    key=lambda d: d["key"],
                ),
            })
        ab_dump = {
            "a_num": [int(v.numerator) for v in ab.a],
            "a_den": [int(v.denominator) for v in ab.a],
            "b_num": [int(v.numerator) for v in ab.b],
            "b_den": [int(v.denominator) for v in ab.b],
        }
        cases.append({
            "name": name,
            "num_hard": int(nz.num_hard),
            "qq_order": qq,
            "ab": ab_dump,
            "symmetry": sym_dump,
            "strip": strips_dump,
        })
    write_json(out_root / "weyl_symmetry" / "results.json", {"cases": cases})
    return len(cases)


def gen_adjoint_unrefined(out_root: Path) -> int:
    """Marginal-check goldens: unrefined q^1 projection at e ∈ {-2,-1,1,2}."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.index_3d import compute_index_3d_python

    targets = ["m003", "m004", "m006", "s776", "3_1", "K3a1"]
    qq = 10
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        n_cusps = nz.r
        entry = {"name": name, "num_cusps": n_cusps, "qq_order": qq, "per_cusp": []}
        for cusp_idx in range(n_cusps):
            c_e = {}
            for e_val in [Fraction(-2), Fraction(-1), Fraction(1), Fraction(2)]:
                m_ext_s = [0] * n_cusps
                e_ext_s = [Fraction(0)] * n_cusps
                e_ext_s[cusp_idx] = e_val
                r3d = compute_index_3d_python(nz, m_ext_s, e_ext_s, qq)
                idx = 2 - r3d.min_power
                q1 = r3d.coeffs[idx] if 0 <= idx < len(r3d.coeffs) else 0
                c_e[int(e_val * 2)] = int(q1)
            num = c_e[-2] + c_e[2] - c_e[-4] - c_e[4]
            if num % 2 != 0:
                proj, marginal = None, None
            else:
                proj = num // 2
                marginal = (proj >= 0)
            entry["per_cusp"].append({
                "cusp_idx": cusp_idx,
                "c_e_x2": sorted(c_e.items()),
                "unrefined_q1_proj": proj,
                "is_marginal": marginal,
            })
        cases.append(entry)
    write_json(out_root / "adjoint_unrefined" / "results.json", {"cases": cases})
    return len(cases)


def gen_adjoint_eta0(out_root: Path) -> int:
    """Refined q^1-η^0 adjoint projection goldens (single-cusp check_adjoint_projection)."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index
    from manifold_index.core.weyl_check import (
        compute_ab_vectors, check_adjoint_projection,
    )

    targets = ["m003", "m004", "m006"]
    qq = 10
    m_grid = [-2, -1, 0, 1, 2]
    e_grid = [Fraction(-2), Fraction(-1), Fraction(-1, 2), Fraction(0),
              Fraction(1, 2), Fraction(1), Fraction(2)]

    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if nz.r != 1:
            continue
        entries = []
        for m in m_grid:
            for e in e_grid:
                res = compute_refined_index(nz, [m], [e], qq)
                entries.append(([m], [e], res))
        ab = compute_ab_vectors(entries, nz.num_hard)
        if ab is None:
            continue
        res = check_adjoint_projection(entries, nz.num_hard, ab, 0)
        entries_dump = []
        for (m_ext, e_ext, r) in entries:
            entries_dump.append({
                "m_ext": list(m_ext),
                "e_ext_x2": [int(Fraction(v) * 2) for v in e_ext],
                "items": sorted(
                    ({"key": [int(x) for x in k], "coeff": int(v)} for k, v in r.items()),
                    key=lambda d: d["key"],
                ),
            })
        cases.append({
            "name": name,
            "entries": entries_dump,
            "num_hard": int(nz.num_hard),
            "qq_order": qq,
            "cusp_idx": 0,
            "ab": {
                "a_num": [int(v.numerator) for v in ab.a],
                "a_den": [int(v.denominator) for v in ab.a],
                "b_num": [int(v.numerator) for v in ab.b],
                "b_den": [int(v.denominator) for v in ab.b],
            },
            "c_e_x2": sorted(
                [[int(Fraction(k) * 2), int(v)] for k, v in res.c_e.items()]
            ),
            "projected_value": (None if res.projected_value is None
                                else int(res.projected_value)),
            "is_pass": bool(res.is_pass),
            "missing_e_x2": sorted(int(Fraction(e) * 2) for e in res.missing_e),
        })
    write_json(out_root / "adjoint_eta0" / "results.json", {"cases": cases})
    return len(cases)


def gen_adjoint_w_scan(out_root: Path) -> int:
    """W-vector scan goldens (single-cusp)."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_index import compute_refined_index
    from manifold_index.core.weyl_check import (
        compute_ab_vectors, scan_w_vectors,
    )

    targets = ["m003", "m004", "m006"]
    qq = 10
    m_grid = [-2, -1, 0, 1, 2]
    e_grid = [Fraction(-2), Fraction(-1), Fraction(-1, 2), Fraction(0),
              Fraction(1, 2), Fraction(1), Fraction(2)]
    max_coeff = 3

    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if nz.r != 1:
            continue
        entries = []
        for m in m_grid:
            for e in e_grid:
                res = compute_refined_index(nz, [m], [e], qq)
                entries.append(([m], [e], res))
        ab = compute_ab_vectors(entries, nz.num_hard)
        if ab is None:
            continue
        scan = scan_w_vectors(entries, nz.num_hard, ab, 0, max_coeff)
        entries_dump = []
        for ent in scan.entries:
            adj_dump = None
            if ent.adjoint is not None:
                adj_dump = {
                    "projected_value": (None if ent.adjoint.projected_value is None
                                        else int(ent.adjoint.projected_value)),
                    "is_pass": bool(ent.adjoint.is_pass),
                    "c_e_x2": sorted(
                        [[int(Fraction(k) * 2), int(v)] for k, v in ent.adjoint.c_e.items()]
                    ),
                    "missing_e_x2": sorted(int(Fraction(e) * 2) for e in ent.adjoint.missing_e),
                }
            entries_dump.append({
                "w": [int(x) for x in ent.w],
                "a_eff_num": int(ent.a_eff.numerator),
                "a_eff_den": int(ent.a_eff.denominator),
                "b_eff_num": int(ent.b_eff.numerator),
                "b_eff_den": int(ent.b_eff.denominator),
                "a_eff_is_integer": bool(ent.a_eff_is_integer),
                "adjoint": adj_dump,
            })
        cases.append({
            "name": name,
            "num_hard": int(nz.num_hard),
            "qq_order": qq,
            "cusp_idx": 0,
            "max_coeff": max_coeff,
            "ab": {
                "a_num": [int(v.numerator) for v in ab.a],
                "a_den": [int(v.denominator) for v in ab.a],
                "b_num": [int(v.numerator) for v in ab.b],
                "b_den": [int(v.denominator) for v in ab.b],
            },
            "entries_ref": [
                {
                    "m_ext": list(m_ext),
                    "e_ext_x2": [int(Fraction(v) * 2) for v in e_ext],
                    "items": sorted(
                        ({"key": [int(x) for x in k], "coeff": int(v)} for k, v in r.items()),
                        key=lambda d: d["key"],
                    ),
                }
                for (m_ext, e_ext, r) in entries
            ],
            "scan": entries_dump,
            "passing_count": int(len(scan.passing)),
        })
    write_json(out_root / "adjoint_w_scan" / "results.json", {"cases": cases})
    return len(cases)


def gen_dehn_filling(out_root: Path) -> int:
    """Dehn filling goldens: kernel terms + filled index + NC search."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.dehn_filling import (
        find_rs, enumerate_kernel_terms,
        compute_filled_index, find_non_closable_cycles,
    )

    targets = ["m003", "m004", "m006"]
    qq = 10
    slopes = [(1, 0), (3, 1), (5, 2), (-3, 1)]
    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if nz.r != 1:
            continue
        entry = {"name": name, "fills": [], "nc": None}
        for (P, Q) in slopes:
            from math import gcd
            if gcd(abs(P), abs(Q)) != 1:
                continue
            R, S = find_rs(P, Q)
            kts = enumerate_kernel_terms(
                P, Q, R, S, nz, 0, [], [Fraction(0)] * 0, qq)
            filled = compute_filled_index(nz, 0, P, Q, q_order_half=qq)
            series_dump = sorted(
                [[int(k), [int(v.numerator), int(v.denominator)]]
                 for k, v in filled.series.items()],
                key=lambda x: x[0],
            )
            kt_dump = [
                {"m": int(kt.m), "e_x2": int(Fraction(kt.e) * 2),
                 "c": int(kt.c), "phase": int(kt.phase),
                 "multiplicity": int(kt.multiplicity)}
                for kt in kts
            ]
            entry["fills"].append({
                "P": P, "Q": Q, "R": R, "S": S,
                "kernel_terms": kt_dump,
                "n_kernel_terms": int(filled.n_kernel_terms),
                "series": series_dump,
                "is_stably_zero": bool(filled.is_stably_zero()),
            })
        # Small NC search
        nc = find_non_closable_cycles(
            nz, 0, range(-2, 3), range(0, 3),
            q_order_half=qq, use_symmetry=True)
        nc_dump = {
            "cusp_idx": 0,
            "cycles": [{"p": int(c.P), "q": int(c.Q)} for c in nc.cycles],
            "slopes_tested": [[int(p), int(q)] for p, q in nc.slopes_tested],
        }
        entry["nc"] = nc_dump
        cases.append(entry)
    write_json(out_root / "dehn" / "results.json", {"cases": cases})
    return len(cases)


def gen_refined_dehn(out_root: Path) -> int:
    """Refined Dehn filling goldens: HJ-CF + unrefined-kernel-on-refined path."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_dehn_filling import (
        hj_continued_fraction,
        compute_unrefined_kernel_refined_index,
    )

    # HJ-CF goldens
    hj_cases = []
    for p in range(-10, 11):
        for q in range(1, 11):
            from math import gcd
            if gcd(abs(p), q) != 1:
                continue
            ks = hj_continued_fraction(p, q)
            hj_cases.append({"p": p, "q": q, "ks": ks})
    # Special cases
    hj_cases.append({"p": 1, "q": 0, "ks": hj_continued_fraction(1, 0)})
    hj_cases.append({"p": -1, "q": 0, "ks": hj_continued_fraction(-1, 0)})
    write_json(out_root / "refined_dehn" / "hj_cf.json", {"cases": hj_cases})

    # Unrefined kernel applied to I^ref
    targets = ["m003", "m004", "m006"]
    qq = 10
    slopes = [(1, 0), (3, 1), (5, 2), (-3, 1)]
    fill_cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if nz.r != 1:
            continue
        entry = {"name": name, "num_hard": int(nz.num_hard), "fills": []}
        for (P, Q) in slopes:
            from math import gcd as _gcd
            if _gcd(abs(P), abs(Q)) != 1:
                continue
            res = compute_unrefined_kernel_refined_index(
                nz, 0, P, Q, q_order_half=qq,
            )
            series_dump = sorted(
                [{"key": [int(x) for x in k],
                  "coeff_num": int(v.numerator), "coeff_den": int(v.denominator)}
                 for k, v in res.series.items()],
                key=lambda d: d["key"],
            )
            entry["fills"].append({
                "P": P, "Q": Q,
                "n_kernel_terms": int(res.n_kernel_terms),
                "has_cusp_eta": bool(res.has_cusp_eta),
                "series": series_dump,
            })
        fill_cases.append(entry)
    write_json(out_root / "refined_dehn" / "unrefined_kernel.json", {"cases": fill_cases})
    return len(hj_cases) + len(fill_cases)


def gen_is_chain(out_root: Path) -> int:
    """IS chain goldens: ℓ≥2 refined Dehn filling via compute_filled_refined_index."""
    load_manifold = load_v05_manifold()
    if V05_SRC not in sys.path:
        sys.path.insert(0, V05_SRC)
    from fractions import Fraction
    from manifold_index.core.phase_space import find_easy_edges
    from manifold_index.core.neumann_zagier import build_neumann_zagier
    from manifold_index.core.refined_dehn_filling import (
        compute_filled_refined_index,
    )

    targets = ["m003", "m004"]
    # ℓ≥2 slopes (|Q|>1) at small qq for feasibility
    slopes = [(5, 2), (7, 3), (-3, 2)]
    qq = 6

    cases = []
    for name in targets:
        try:
            md = load_manifold(name)
            ps = find_easy_edges(md)
            nz = build_neumann_zagier(md, ps)
        except Exception as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)
            continue
        if nz.r != 1:
            continue
        entry = {"name": name, "num_hard": int(nz.num_hard), "fills": []}
        for (P, Q) in slopes:
            from math import gcd as _gcd
            if _gcd(abs(P), abs(Q)) != 1:
                continue
            print(f"  IS chain: {name} P={P} Q={Q} qq={qq} ...", flush=True)
            res = compute_filled_refined_index(
                nz, 0, P, Q, q_order_half=qq, verbose=False,
            )
            series_dump = sorted(
                [{"key": [int(x) for x in k],
                  "coeff_num": int(v.numerator), "coeff_den": int(v.denominator)}
                 for k, v in res.series.items()],
                key=lambda d: d["key"],
            )
            entry["fills"].append({
                "P": P, "Q": Q,
                "qq_order": qq,
                "hj_ks": list(res.hj_ks),
                "n_kernel_terms": int(res.n_kernel_terms),
                "has_cusp_eta": bool(res.has_cusp_eta),
                "eta_order": int(res.eta_order),
                "series": series_dump,
            })
        cases.append(entry)
    write_json(out_root / "refined_dehn" / "is_chain.json", {"cases": cases})
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
    print("ab_vectors …", flush=True)
    total += gen_ab_vectors(out_root)
    print("weyl_symmetry …", flush=True)
    total += gen_weyl_symmetry(out_root)
    print("adjoint_unrefined …", flush=True)
    total += gen_adjoint_unrefined(out_root)
    print("adjoint_eta0 …", flush=True)
    total += gen_adjoint_eta0(out_root)
    print("adjoint_w_scan …", flush=True)
    total += gen_adjoint_w_scan(out_root)
    print("dehn …", flush=True)
    total += gen_dehn_filling(out_root)
    print("refined_dehn …", flush=True)
    total += gen_refined_dehn(out_root)
    print("is_chain …", flush=True)
    total += gen_is_chain(out_root)

    print(f"done: {total} cases written under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
