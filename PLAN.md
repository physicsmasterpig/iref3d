# IRef3D — Architecture Plan

## What this is

IRef3D computes the **refined 3D index** (I^ref) of hyperbolic 3-manifolds.
Name comes from the mathematical notation I^ref used in papers in this field.

The reference implementation is the Python v0.5 app at:
`/Users/pmp/Documents/Research/ultimate/v0.5/`

This repo is a ground-up rewrite with a Rust computation core.
Goals: **performance** and **future native GUI** (Swift/SwiftUI).

---

## Why Rust

- The v0.5 hot path already had a C extension (`_c_kernel/tet_index.c`) proving
  C-level speed is needed. Rust gives the same speed with memory safety and
  no PyArg boilerplate.
- Clean Swift FFI story: Rust compiles to a static library callable from Swift
  via a bridging header. Write the computation once, call it from both Python
  (now) and Swift (later).
- NC cycle search is combinatorial and parallelisable — Rust + Rayon gives
  significant gains over the Python outer loop.

---

## Why SnapPy is being eliminated

SnapPy (the hyperbolic geometry library) is the current Python dependency that
forces GPL licensing and requires Python at runtime. In v0.5 it is used for
exactly three calls:

```python
M = snappy.Manifold(name)   # load by name
M.num_tetrahedra()          # → int
M.num_cusps()               # → int
M.gluing_equations()        # → integer matrix, shape (n+2r, 3n)
```

The app only works with **census manifolds** — no arbitrary Dehn surgery input.
For census manifolds all gluing matrices can be pre-extracted once at build time
and stored in `data/census.db`. At runtime: zero SnapPy dependency, just a
SQLite lookup.

---

## Repo structure

```
iref3d/
  core/                    ← Rust library crate
    src/
      lib.rs               ← module declarations
      census.rs            ← manifold lookup from data/census.db
      nz.rs                ← Neumann-Zagier matrix construction
      index.rs             ← 3D index computation
      refined.rs           ← refined index (eta fugacity variables)
      filling.rs           ← Dehn filling
      weyl.rs              ← Weyl symmetry check
    Cargo.toml
  data/
    extract_census.py      ← run once with SnapPy, outputs census.db
    census.db              ← pre-extracted gluing matrices (build artifact)
  PLAN.md                  ← this file
  README.md
  .gitignore
```

---

## core/Cargo.toml dependencies

```toml
[package]
name = "iref3d-core"
version = "0.1.0"
edition = "2021"

[dependencies]
hashbrown = "0.14"         # fast HashMap — critical for sparse polynomial arithmetic
ndarray = "0.16"           # NZ matrix operations (numpy equivalent)
rusqlite = { version = "0.31", features = ["bundled"] }  # census.db lookup

[dependencies.pyo3]
version = "0.22"
optional = true

[features]
python = ["pyo3"]          # opt-in Python bindings, not built by default
```

---

## Module mapping — Rust to Python reference

Each Rust module ports exactly one Python file. Always check the Python
reference before writing Rust — results must be bit-identical.

| Rust module   | Python reference file                          | Notes |
|---------------|------------------------------------------------|-------|
| census.rs     | core/manifold.py                               | Replace SnapPy calls with census.db lookup |
| nz.rs         | core/neumann_zagier.py                         | Pure linear algebra, numpy → ndarray |
| index.rs      | core/index_3d.py + _c_kernel/tet_index.c       | C extension is the exact spec for Rust inner loop |
| refined.rs    | core/refined_index.py                          | Calls index inner loop with eta fugacity |
| filling.rs    | core/dehn_filling.py + refined_dehn_filling.py | |
| weyl.rs       | core/weyl_check.py                             | |

The C extension (`tet_index.c`) is ~400 lines and already isolated — it is the
most important reference. Port it first.

Key functions to port from `tet_index.c`:
- `tet_index_series(m, e, qq_order)` → `HashMap<i64, i64>`
- `tet_degree_x2(m, e)` → `i64`
- `poly_convolve(poly1, poly2, budget)` → `HashMap<i64, i64>`

---

## Performance bottlenecks in v0.5 (where Rust wins)

| Component | v0.5 status | Expected Rust gain |
|---|---|---|
| Per-tet index inner loop | Already C extension | None — already C speed |
| Sparse polynomial arithmetic | Python dict ops around C | Significant — tight Rust loop |
| Summation enumeration loop | Python iterator | Significant |
| NC cycle search | Python combinatorial | Large — parallelise with Rayon |
| NZ matrix algebra | numpy (already BLAS) | Minimal |

---

## Migration phases

### Phase 1 — Census extraction (do this first, days)
- Write `data/extract_census.py`
- Run with SnapPy, produce `data/census.db`
- Schema: `(name TEXT, n INT, r INT, gluing BLOB)`
- Verify: spot-check a few manifolds against v0.5 output

### Phase 2 — Rust core (weeks)
- Implement modules in order: census → nz → index → refined → filling → weyl
- Each module has unit tests checking against v0.5 Python output
- No Python bindings yet — pure Rust library

### Phase 3 — PyO3 bindings
- Add `python` feature flag
- Expose functions to Python, replace v0.5 C extension and outer loops
- Existing PySide6 GUI keeps working unchanged
- Benchmark against v0.5

### Phase 4 — Swift native GUI
- Rust core compiles to static library
- Swift app calls it via C FFI bridging header
- Python GUI maintained in parallel until feature parity

---

## Reference paths on this machine

| What | Path |
|---|---|
| v0.5 Python app | `/Users/pmp/Documents/Research/ultimate/v0.5/` |
| Python package root | `/Users/pmp/Documents/Research/ultimate/v0.5/src/manifold_index/` |
| C extension (key reference) | `/Users/pmp/Documents/Research/ultimate/v0.5/src/manifold_index/core/_c_kernel/tet_index.c` |
| NZ construction | `/Users/pmp/Documents/Research/ultimate/v0.5/src/manifold_index/core/neumann_zagier.py` |
| Index computation | `/Users/pmp/Documents/Research/ultimate/v0.5/src/manifold_index/core/index_3d.py` |
| Refined index | `/Users/pmp/Documents/Research/ultimate/v0.5/src/manifold_index/core/refined_index.py` |
| SnapPy manifold loader | `/Users/pmp/Documents/Research/ultimate/v0.5/src/manifold_index/core/manifold.py` |

---

## First session instructions

Read this file, then:
1. Write `data/extract_census.py`
2. Write `core/Cargo.toml`
3. Write `core/src/lib.rs` skeleton with module declarations
4. Start `core/src/index.rs` — port `tet_index.c` first
