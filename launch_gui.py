"""Launch v0.5 GUI with full Rust backend (10–150× speedup).

Patches v0.5's core computation functions in-place so the entire
computation pipeline runs in Rust, not just the inner kernel.

Usage:
    python3 launch_gui.py
"""
import multiprocessing
multiprocessing.freeze_support()

import sys
import os
from pathlib import Path

V05_SRC = os.environ.get(
    "IREF3D_V05_SRC",
    str(Path(__file__).resolve().parent.parent / "ultimate" / "v0.5" / "src"),
)
RUST_LIB = str(Path(__file__).resolve().parent / "core" / "target" / "release")
ADAPTER_DIR = str(Path(__file__).resolve().parent / "data")

for p in [V05_SRC, RUST_LIB, ADAPTER_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# SQLite thread-safety patch (from v0.5's launcher.py)
import sqlite3 as _sq
_sq_orig = _sq.connect
def _sq_nothreadcheck(*a, **kw):
    kw.setdefault("check_same_thread", False)
    return _sq_orig(*a, **kw)
_sq.connect = _sq_nothreadcheck

# ── Patch v0.5 core → Rust ──
try:
    import rust_adapter
    rust_adapter.patch_services()
except Exception as e:
    print(f"[launch_gui] Rust patching failed: {e}")
    print("[launch_gui] Falling back to pure Python")

# ── Launch GUI ──
from manifold_index.app import launch_gui
launch_gui()
