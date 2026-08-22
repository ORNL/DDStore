"""
Method 2 — extra member side (read-only, no MPI communicator needed).

Run after (or concurrently with) test_method2_core.py:

  python test/test_method2_extra.py [handshake_dir] [n_core]

Directory resolution priority (same as the C library):
  1. Command-line argument
  2. DDSTORE_HANDSHAKE_DIR environment variable
  3. ./ddstore_hs  (current working directory — must be on shared filesystem)

The handshake directory MUST be on a shared filesystem visible to all nodes.

Environment:
  DDSTORE_HANDSHAKE_DIR       — shared Lustre path for handshake files
  DDSTORE_N_CORE              — number of core ranks (default: 4)
  DDSTORE_HANDSHAKE_TIMEOUT_S — poll timeout in seconds (default: 300)
"""

import os
import sys
import numpy as np
import pyddstore as dds

# ---------------------------------------------------------------------------
# configuration — resolve directory with same priority as the C library
# ---------------------------------------------------------------------------

def _resolve_dir(arg):
    if arg:
        return arg
    env = os.environ.get("DDSTORE_HANDSHAKE_DIR", "")
    if env:
        return env
    return "./ddstore_hs"

hs_dir = _resolve_dir(sys.argv[1] if len(sys.argv) > 1 else "")
n_core = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("DDSTORE_N_CORE", "4"))

nrows = 8
ncols = 4

# ---------------------------------------------------------------------------
# create DDStore as extra member (no MPI comm)
# ---------------------------------------------------------------------------

print(f"[extra] joining store at {hs_dir}, n_core={n_core}", flush=True)
store = dds.PyDDStore(None, method=2, handshake_dir=hs_dir, n_core=n_core)

# discover variable "x" published by core members
store.join("x")
print("[extra] join('x') succeeded", flush=True)

# ---------------------------------------------------------------------------
# read every row from every core rank and verify
# ---------------------------------------------------------------------------

out = np.zeros((1, ncols), dtype=np.float32)
errors = []

for target_rank in range(n_core):
    for row in range(nrows):
        global_idx = target_rank * nrows + row
        store.get("x", out, start=global_idx)
        expected = float(target_rank + 1)
        if not np.all(out == expected):
            errors.append(
                f"  global_idx={global_idx} (rank={target_rank} row={row}): "
                f"expected {expected}, got {out[0]}"
            )

if errors:
    print("[extra] FAILED — mismatches:", flush=True)
    for e in errors:
        print(e, flush=True)
    raise AssertionError(f"{len(errors)} mismatches")

print(f"[extra] all {n_core * nrows} reads verified correctly", flush=True)

# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

store.free()

# signal core side that we are done
sentinel = os.path.join(hs_dir, "done_extra")
with open(sentinel, "w") as f:
    f.write("done\n")

print("[extra] sentinel written, exiting", flush=True)
