"""Launch v0.5 GUI with Rust backend.

The Rust backend is installed as a drop-in C extension replacement via
`data/install_rust_backend.py`. This launcher just sets up paths and
starts the GUI.

Usage:
    python3 launch_gui.py

To install/uninstall Rust backend:
    python3 data/install_rust_backend.py [--uninstall]
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
if V05_SRC not in sys.path:
    sys.path.insert(0, V05_SRC)

# SQLite thread-safety patch (from v0.5's launcher.py)
import sqlite3 as _sq
_sq_orig = _sq.connect
def _sq_nothreadcheck(*a, **kw):
    kw.setdefault("check_same_thread", False)
    return _sq_orig(*a, **kw)
_sq.connect = _sq_nothreadcheck

# Report backend status
try:
    from manifold_index.core.index_3d import _HAS_C_KERNEL
    from manifold_index.core.refined_index import _HAS_C_POLY
    if _HAS_C_KERNEL:
        print("[iref3d] Rust backend active (tet_index + poly_convolve)")
    else:
        print("[iref3d] C extension not loaded — using pure Python")
except Exception as e:
    print(f"[iref3d] Backend check failed: {e}")

from manifold_index.app import launch_gui
launch_gui()
