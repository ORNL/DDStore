"""
Method 2 — core member side.

Run together with test_method2_extra.py in two separate mpirun launches
that share a handshake directory on a shared filesystem (e.g. Lustre).

Usage (two terminals / two job steps):

  Terminal 1 (core):
    mpirun -n 4 python test/test_method2_core.py [handshake_dir] [n_core]

  Terminal 2 (extra):
    python test/test_method2_extra.py [handshake_dir] [n_core]

Directory resolution priority (same as the C library):
  1. Command-line argument
  2. DDSTORE_HANDSHAKE_DIR environment variable
  3. ./ddstore_hs  (current working directory — must be on shared filesystem)

The handshake directory MUST be on a shared filesystem visible to all nodes.

Environment:
  DDSTORE_HANDSHAKE_DIR       — shared Lustre path for handshake files
  DDSTORE_HANDSHAKE_TIMEOUT_S — poll timeout in seconds (default: 300)
"""

import os
import sys
import time
import numpy as np
from mpi4py import MPI
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
n_core = (
    int(sys.argv[2])
    if len(sys.argv) > 2
    else int(os.environ.get("DDSTORE_N_CORE", "4"))
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == n_core, f"Expected {n_core} core ranks, got {size}"

# ---------------------------------------------------------------------------
# prepare handshake directory (rank 0 creates it and removes stale files)
# ---------------------------------------------------------------------------

if rank == 0:
    os.makedirs(hs_dir, exist_ok=True)
    for fname in os.listdir(hs_dir):
        if fname.endswith(".bin"):
            os.remove(os.path.join(hs_dir, fname))
    print(f"[core] handshake_dir={hs_dir}", flush=True)
comm.Barrier()

# ---------------------------------------------------------------------------
# build per-rank data: each rank holds `nrows` rows of `ncols` float32s
# ---------------------------------------------------------------------------

nrows = 8
ncols = 4
data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)

# ---------------------------------------------------------------------------
# create DDStore (method=2, core member)
# ---------------------------------------------------------------------------

store = dds.PyDDStore(comm, method=2, handshake_dir=hs_dir)
store.add("x", data)

if rank == 0:
    print(f"[core] all {n_core} records written to {hs_dir}", flush=True)

# ---------------------------------------------------------------------------
# self-check: core members can also read from each other via method=2
# ---------------------------------------------------------------------------

out = np.zeros((1, ncols), dtype=np.float32)
for target_rank in range(n_core):
    global_idx = target_rank * nrows
    store.get("x", out, start=global_idx)
    expected = float(target_rank + 1)
    assert np.all(out == expected), (
        f"[core rank {rank}] get from target {target_rank}: "
        f"expected {expected}, got {out}"
    )

if rank == 0:
    print("[core] self-check passed", flush=True)

# ---------------------------------------------------------------------------
# wait for extra member to signal it is done
# ---------------------------------------------------------------------------

sentinel = os.path.join(hs_dir, "done_extra")
timeout_s = int(os.environ.get("DDSTORE_HANDSHAKE_TIMEOUT_S", "300"))
t0 = time.monotonic()

if rank == 0:
    print("[core] waiting for extra member sentinel ...", flush=True)
    while not os.path.exists(sentinel):
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError("timed out waiting for done_extra sentinel")
        time.sleep(0.2)
    print("[core] sentinel received, shutting down", flush=True)
comm.Barrier()

# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

store.free()

if rank == 0:
    for fname in os.listdir(hs_dir):
        if fname.endswith(".bin"):
            try:
                os.remove(os.path.join(hs_dir, fname))
            except OSError:
                pass
    try:
        os.remove(sentinel)
    except OSError:
        pass

print(f"[core rank {rank}] done", flush=True)
