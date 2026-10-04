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
METHODS = [0, pytest.param(1, marks=pytest.mark.skipif(not HAVE_CXI, reason="no CXI device"))]

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
    data = (np.arange(first, first + NROWS)[:, None] * 100 + np.arange(NCOLS)).astype(dtype)
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
@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.float32, np.int64, np.float64])
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


@pytest.mark.skipif(not (HAVE_CXI and HAVE_GPU and FABRIC == "cxi"),
                    reason="requires the cxi provider and a GPU")
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


@pytest.mark.parametrize("method", [pytest.param(1, marks=pytest.mark.skipif(not HAVE_CXI, reason="no CXI device"))])
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

    threads = [threading.Thread(target=worker, args=(1000 * comm.Get_rank() + t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    comm.Barrier()
    finish(store)
    assert not errors, f"worker thread(s) raised: {errors}"
    assert all_passed(comm, not bad)
