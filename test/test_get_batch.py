"""
Batched get (PyDDStore.get_batch) tests — run with 2+ ranks, e.g.:
  mpirun -n 4 pytest test/test_get_batch.py -v

Each case runs with method 0 (MPI RMA) and, where a CXI device is present,
method 1 over libfabric: cxi by default, or the provider named by
DDSTORE_FABRIC if it is set (e.g. DDSTORE_FABRIC=hsn). Every rank fills its shard with values that encode the
global row id, so any misplaced or missing row is caught exactly.
"""

import glob
import os
import threading

import numpy as np
import pytest
from mpi4py import MPI

import pyddstore as dds

# A CXI device alone isn't enough (login nodes have one but can't open the
# fabric); also require running inside a Slurm job step.
HAVE_CXI = bool(glob.glob("/dev/cxi*")) and "SLURM_STEP_ID" in os.environ
FABRIC = os.environ.get("DDSTORE_FABRIC", "cxi")
METHODS = [
    0,
    pytest.param(1, marks=pytest.mark.skipif(not HAVE_CXI, reason="no CXI device")),
]

try:
    import torch

    HAVE_GPU = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    HAVE_GPU = False

NROWS, NCOLS = 16, 5


def all_passed(comm, local_ok):
    return comm.allreduce(int(local_ok), op=MPI.LAND)


def make_store(comm, method, monkeypatch, dtype=np.float32):
    if method != 0:
        monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    rank = comm.Get_rank()
    store = dds.PyDDStore(comm, method=method)
    # row r (global) holds r*100 + column, so every element is identifiable
    first = rank * NROWS
    data = (np.arange(first, first + NROWS)[:, None] * 100 + np.arange(NCOLS)).astype(
        dtype
    )
    store.add("x", data)
    store.epoch_begin()
    return store


def expected_rows(idx, dtype=np.float32):
    idx = np.asarray(idx)
    return (idx[:, None] * 100 + np.arange(NCOLS)).astype(dtype)


def finish(store):
    store.epoch_end()
    store.free()


@pytest.mark.parametrize("method", METHODS)
def test_batch_all_ranks_shuffled_with_repeats(comm, monkeypatch, method):
    size = comm.Get_size()
    if size < 2:
        pytest.skip("requires at least 2 ranks")
    store = make_store(comm, method, monkeypatch)
    rng = np.random.default_rng(comm.Get_rank())
    idx = rng.integers(0, NROWS * size, size=200)  # spans every rank, with repeats
    out = np.zeros((len(idx), NCOLS), dtype=np.float32)
    store.get_batch("x", out, idx)
    ok = np.array_equal(out, expected_rows(idx))
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)


@pytest.mark.parametrize("method", METHODS)
def test_batch_matches_per_row_get(comm, monkeypatch, method):
    size = comm.Get_size()
    store = make_store(comm, method, monkeypatch)
    idx = list(range(NROWS * size))[::-1]
    out = np.zeros((len(idx), NCOLS), dtype=np.float32)
    store.get_batch("x", out, idx)
    row = np.zeros((1, NCOLS), dtype=np.float32)
    ok = True
    for i, g in enumerate(idx):
        store.get("x", row, g)
        ok &= np.array_equal(row[0], out[i])
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)


@pytest.mark.parametrize("method", METHODS)
def test_batch_single_row(comm, monkeypatch, method):
    size = comm.Get_size()
    store = make_store(comm, method, monkeypatch)
    g = (comm.Get_rank() + 1) % size * NROWS + 3
    out = np.zeros((1, NCOLS), dtype=np.float32)
    store.get_batch("x", out, [g])
    ok = np.array_equal(out, expected_rows([g]))
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "dtype", [np.uint8, np.int32, np.float32, np.int64, np.float64]
)
def test_batch_dtypes(comm, monkeypatch, method, dtype):
    size = comm.Get_size()
    store = make_store(comm, method, monkeypatch, dtype=dtype)
    # row r holds r*100 + col: only rows 0..2 fit in uint8
    idx = [2, 0, 1] if dtype == np.uint8 else [NROWS * size - 1, 0, NROWS * size // 2]
    out = np.zeros((len(idx), NCOLS), dtype=dtype)
    store.get_batch("x", out, idx)
    ok = np.array_equal(out, expected_rows(idx, dtype))
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)


@pytest.mark.parametrize("method", METHODS)
def test_batch_errors_leave_store_usable(comm, monkeypatch, method):
    size = comm.Get_size()
    store = make_store(comm, method, monkeypatch)
    out = np.zeros((2, NCOLS), dtype=np.float32)
    with pytest.raises(IndexError):
        store.get_batch("x", out, [0, NROWS * size])  # second index out of range
    with pytest.raises(ValueError):
        store.get_batch("x", out, [0, 1, 2])  # 3 indices for 2 rows
    with pytest.raises(Exception):
        store.get_batch("x", out.astype(np.float64), [0, 1])  # wrong item size
    idx = [NROWS * size - 1, 0]
    store.get_batch("x", out, idx)
    ok = np.array_equal(out, expected_rows(idx))
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)


@pytest.mark.skipif(
    not (HAVE_CXI and HAVE_GPU and FABRIC == "cxi"),
    reason="requires the cxi provider and a GPU",
)
def test_batch_into_gpu_tensor(comm, monkeypatch):
    size = comm.Get_size()
    store = make_store(comm, 1, monkeypatch)
    rng = np.random.default_rng(100 + comm.Get_rank())
    ok = True
    for _ in range(20):
        idx = rng.integers(0, NROWS * size, size=64)
        out = torch.empty((len(idx), NCOLS), dtype=torch.float32, device="cuda")
        store.get_batch("x", out, idx)
        # compute-kernel read, like a training step would do
        diff = (out - torch.from_numpy(expected_rows(idx)).cuda()).abs().sum().item()
        ok &= diff == 0.0
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)


@pytest.mark.parametrize(
    "method",
    [pytest.param(1, marks=pytest.mark.skipif(not HAVE_CXI, reason="no CXI device"))],
)
def test_batch_concurrent_threads(comm, monkeypatch, method):
    """get_batch from several threads at once on one variable: the
    per-variable lock must keep each batch's rows and completions apart."""
    size = comm.Get_size()
    store = make_store(comm, method, monkeypatch)
    errors, bad = [], []

    def worker(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(50):
                idx = rng.integers(0, NROWS * size, size=32)
                out = np.zeros((len(idx), NCOLS), dtype=np.float32)
                store.get_batch("x", out, idx)
                if not np.array_equal(out, expected_rows(idx)):
                    bad.append(seed)
        except Exception as exc:  # noqa: BLE001 - surface any thread exception
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(1000 * comm.Get_rank() + t,))
        for t in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    comm.Barrier()
    finish(store)
    assert not errors, f"worker thread(s) raised: {errors}"
    assert all_passed(comm, not bad)


def _mr_miss(store):
    """recv registrations so far (None unless DDSTORE_PROFILE was set at start)."""
    if os.environ.get("DDSTORE_PROFILE", "0") in ("", "0"):
        return None
    return store.get_profile("x")["mr_miss"]


@pytest.mark.parametrize("method", METHODS)
def test_register_recv_pool(comm, monkeypatch, method):
    """Reads into registered buffers, from several threads, each with its own
    buffer: every row correct and (method 1) no registration per read."""
    size = comm.Get_size()
    store = make_store(comm, method, monkeypatch)
    nthreads, batch = 2, 16
    pools = [np.zeros((4 * batch, NCOLS), dtype=np.float32) for _ in range(nthreads)]
    for pool in pools:
        store.register_recv("x", pool)
    store.register_recv("x", pools[0])  # registering twice is a no-op
    miss0 = _mr_miss(store)
    errors, bad = [], []

    def worker(t):
        rng = np.random.default_rng(1000 * comm.Get_rank() + t)
        pool = pools[t]
        try:
            for it in range(40):
                k = it % 4
                out = pool[k * batch : (k + 1) * batch]  # a slice of the pool
                idx = rng.integers(0, NROWS * size, size=batch)
                store.get_batch("x", out, idx)
                if not np.array_equal(out, expected_rows(idx)):
                    bad.append((t, "batch"))
                one = pool[k * batch : k * batch + 1]
                store.get("x", one, int(idx[0]))
                if not np.array_equal(one, expected_rows(idx[:1])):
                    bad.append((t, "get"))
        except Exception as exc:  # noqa: BLE001 - surface any thread exception
            errors.append(exc)

    if method == 0:  # get_batch is collective there: one thread at a time
        for t in range(nthreads):
            worker(t)
    else:
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(nthreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    ok = not errors and not bad
    if method != 0 and miss0 is not None:
        ok &= _mr_miss(store) == miss0
    for pool in pools:
        store.unregister_recv("x", pool)
    if method != 0:
        with pytest.raises(ValueError):
            store.unregister_recv("x", pools[0])
    # unregistered buffers still work (through the one-slot cache)
    idx = np.arange(batch) % (NROWS * size)
    store.get_batch("x", pools[1][:batch], idx)
    ok &= np.array_equal(pools[1][:batch], expected_rows(idx))
    comm.Barrier()
    finish(store)
    assert not errors, f"worker thread(s) raised: {errors}"
    assert all_passed(comm, ok), bad


@pytest.mark.skipif(
    not (HAVE_CXI and HAVE_GPU and FABRIC == "cxi"),
    reason="requires the cxi provider and a GPU",
)
def test_register_recv_gpu(comm, monkeypatch):
    size = comm.Get_size()
    store = make_store(comm, 1, monkeypatch)
    pool = torch.empty((64, NCOLS), dtype=torch.float32, device="cuda")
    store.register_recv("x", pool)
    miss0 = _mr_miss(store)
    rng = np.random.default_rng(200 + comm.Get_rank())
    ok = True
    for it in range(20):
        k = it % 4
        out = pool[k * 16 : (k + 1) * 16]
        idx = rng.integers(0, NROWS * size, size=16)
        store.get_batch("x", out, idx)
        diff = (out - torch.from_numpy(expected_rows(idx)).cuda()).abs().sum().item()
        ok &= diff == 0.0
    if miss0 is not None:
        ok &= _mr_miss(store) == miss0
    store.unregister_recv("x", pool)
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)


WIDE = 3001  # float32 columns: 12004-byte rows


@pytest.mark.parametrize("method", METHODS)
def test_wide_rows(comm, monkeypatch, method):
    """Rows of 12004 bytes. Run with DDSTORE_MAX_READ_BYTES=4096 (method 1)
    to check that rows longer than one read are split, remainder included."""
    size = comm.Get_size()
    if method != 0:
        monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    rank = comm.Get_rank()
    store = dds.PyDDStore(comm, method=method)
    first = rank * 4
    rows = np.arange(first, first + 4)[:, None] * 10000.0 + np.arange(WIDE)
    store.add("w", rows.astype(np.float32))
    store.epoch_begin()
    idx = np.array([(rank + 1) % size * 4 + 3, 0, size * 4 - 1])
    out = np.zeros((len(idx), WIDE), dtype=np.float32)
    store.get_batch("w", out, idx)
    want = (idx[:, None] * 10000.0 + np.arange(WIDE)).astype(np.float32)
    ok = np.array_equal(out, want)
    one = np.zeros((1, WIDE), dtype=np.float32)
    store.get("w", one, int(idx[0]))
    ok &= np.array_equal(one, want[:1])
    comm.Barrier()
    finish(store)
    assert all_passed(comm, ok)
