"""
Multi-rank tests — run with: mpirun -n 4 pytest test/test_multirank.py -v

Each rank stores a distinct value; tests verify cross-rank remote reads.
Requires at least 2 ranks; some tests require exactly 4.
"""

import numpy as np
import pytest
from mpi4py import MPI

import pyddstore as dds


def all_passed(comm, local_ok):
    return comm.allreduce(int(local_ok), op=MPI.LAND)


# ---------------------------------------------------------------------------
# remote get: each rank reads from every other rank
# ---------------------------------------------------------------------------


def test_remote_get_all_ranks(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    nrows, ncols = 8, 4

    store = dds.PyDDStore(comm)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    out = np.zeros((1, ncols), dtype=np.float32)
    local_ok = True
    for target_rank in range(size):
        global_idx = target_rank * nrows  # first row of each rank
        store.get("x", out, start=global_idx)
        expected = float(target_rank + 1)
        if not np.all(out == expected):
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


# ---------------------------------------------------------------------------
# boundary: last row of each rank's shard
# ---------------------------------------------------------------------------


def test_remote_get_last_row(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    nrows, ncols = 8, 4

    store = dds.PyDDStore(comm)
    data = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols)
    data += rank * 1000
    store.add("x", data)

    store.epoch_begin()
    out = np.zeros((1, ncols), dtype=np.float32)
    local_ok = True
    for target_rank in range(size):
        last_global_idx = target_rank * nrows + (nrows - 1)
        store.get("x", out, start=last_global_idx)
        expected = data[nrows - 1] + (target_rank - rank) * 1000
        # recompute expected from target rank's perspective
        expected = (
            np.arange((nrows - 1) * ncols, nrows * ncols, dtype=np.float32)
            + target_rank * 1000
        )
        if not np.allclose(out[0], expected):
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


# ---------------------------------------------------------------------------
# init + update on each rank, then remote get
# ---------------------------------------------------------------------------


def test_init_update_remote_get(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    nrows, ncols = 4, 8

    store = dds.PyDDStore(comm)
    store.init("y", nrows, ncols, itemsize=4)

    local_data = np.full((nrows, ncols), float(rank + 10), dtype=np.float32)
    store.update("y", local_data, offset=0)

    store.epoch_begin()
    out = np.zeros((1, ncols), dtype=np.float32)
    local_ok = True
    for target_rank in range(size):
        store.get("y", out, start=target_rank * nrows)
        expected = float(target_rank + 10)
        if not np.all(out == expected):
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


# ---------------------------------------------------------------------------
# multiple variables — independent correctness
# ---------------------------------------------------------------------------


def test_multiple_variables_remote(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    nrows, ncols = 4, 4

    store = dds.PyDDStore(comm)
    a = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    b = np.full((nrows, ncols), float(rank + 100), dtype=np.float32)
    store.add("a", a)
    store.add("b", b)

    store.epoch_begin()
    out = np.zeros((1, ncols), dtype=np.float32)
    local_ok = True
    for target_rank in range(size):
        idx = target_rank * nrows
        store.get("a", out, start=idx)
        if not np.all(out == float(target_rank + 1)):
            local_ok = False
        store.get("b", out, start=idx)
        if not np.all(out == float(target_rank + 100)):
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()


# ---------------------------------------------------------------------------
# ddstore_width: sub-communicator grouping (requires size >= 4)
# ---------------------------------------------------------------------------


def test_ddstore_width(comm):
    size = comm.Get_size()
    if size < 4:
        pytest.skip("requires at least 4 ranks")

    rank = comm.Get_rank()
    ddstore_width = 2
    nrows, ncols = 4, 4

    sub_comm = comm.Split(rank // ddstore_width, rank)
    sub_rank = sub_comm.Get_rank()
    sub_size = sub_comm.Get_size()

    store = dds.PyDDStore(sub_comm)
    data = np.full((nrows, ncols), float(rank + 1), dtype=np.float32)
    store.add("x", data)

    store.epoch_begin()
    out = np.zeros((1, ncols), dtype=np.float32)
    local_ok = True
    for target_sub_rank in range(sub_size):
        store.get("x", out, start=target_sub_rank * nrows)
        group_base = (rank // ddstore_width) * ddstore_width
        expected = float(group_base + target_sub_rank + 1)
        if not np.all(out == expected):
            local_ok = False
    store.epoch_end()

    assert all_passed(comm, local_ok)
    store.free()
