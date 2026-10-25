"""
Stores without an MPI communicator (methods 1 and 2 with rank=/size=): the
ranks exchange their handshake records through files in handshake_dir
instead of MPI_Allgather. Run inside a Slurm step with a CXI device, e.g.:
  srun -n 4 python -m pytest test/test_file_exchange.py -v

MPI is used here only to check results and share the test directory; the
stores themselves never see a communicator. The handshake directories are
created under DDSTORE_TEST_DIR (default: the current directory), which must
be on a filesystem every rank sees.
"""

import glob
import os
import shutil
import tempfile
import time

import numpy as np
import pytest
from mpi4py import MPI

import pyddstore as dds

HAVE_CXI = bool(glob.glob("/dev/cxi*")) and "SLURM_STEP_ID" in os.environ
FABRIC = os.environ.get("DDSTORE_FABRIC", "cxi")
pytestmark = pytest.mark.skipif(not HAVE_CXI, reason="no CXI device")

try:
    import torch

    HAVE_GPU = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    HAVE_GPU = False

NROWS, NCOLS = 16, 5


@pytest.fixture
def hs(comm, monkeypatch):
    """A fresh shared directory per test (removed afterwards)."""
    monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    path = None
    if comm.Get_rank() == 0:
        base = os.environ.get("DDSTORE_TEST_DIR", os.getcwd())
        path = tempfile.mkdtemp(prefix=".test_fx.", dir=base)
    path = comm.bcast(path)
    yield path
    comm.Barrier()
    if comm.Get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)


def all_passed(comm, local_ok):
    return comm.allreduce(int(local_ok), op=MPI.LAND)


def shard(rank, nrows=NROWS, dtype=np.float32):
    # global row g holds g*100 + column, so every element is identifiable
    first = rank * nrows
    return (np.arange(first, first + nrows)[:, None] * 100 + np.arange(NCOLS)).astype(
        dtype
    )


def expected_rows(idx, dtype=np.float32):
    idx = np.asarray(idx)
    return (idx[:, None] * 100 + np.arange(NCOLS)).astype(dtype)


def file_store(comm, hs, method, **kw):
    return dds.PyDDStore(
        None,
        method=method,
        handshake_dir=hs,
        rank=comm.Get_rank(),
        size=comm.Get_size(),
        **kw,
    )


@pytest.mark.parametrize("method", [1, 2])
def test_get_and_get_batch(comm, hs, method):
    rank, size = comm.Get_rank(), comm.Get_size()
    store = file_store(comm, hs, method)
    store.add("x", shard(rank))
    ok = store.info("x")[0] == NROWS * size

    rng = np.random.default_rng(rank)
    idx = rng.integers(0, NROWS * size, size=100)  # spans every rank, repeats
    out = np.zeros((len(idx), NCOLS), dtype=np.float32)
    store.get_batch("x", out, idx)
    ok = ok and np.array_equal(out, expected_rows(idx))

    one = np.zeros((1, NCOLS), dtype=np.float32)
    g = (rank + 1) % size * NROWS + 3  # a row of the next rank
    store.get("x", one, g)
    ok = ok and np.array_equal(one, expected_rows([g]))

    comm.Barrier()
    store.free()
    assert all_passed(comm, ok)


def test_uneven_shards_and_init_update(comm, hs):
    """Row counts differ per rank; init()+update() also exchanges records."""
    rank, size = comm.Get_rank(), comm.Get_size()
    n = rank + 2
    first = sum(r + 2 for r in range(rank))
    store = file_store(comm, hs, 1)
    store.init("y", n, NCOLS, 4)
    data = (np.arange(first, first + n)[:, None] * 100 + np.arange(NCOLS)).astype(
        np.float32
    )
    store.update("y", data, 0)
    total = sum(r + 2 for r in range(size))
    ok = store.info("y")[0] == total
    comm.Barrier()  # every rank has written its rows

    idx = np.arange(total)
    out = np.zeros((total, NCOLS), dtype=np.float32)
    store.get_batch("y", out, idx)
    ok = ok and np.array_equal(out, expected_rows(idx))
    comm.Barrier()
    store.free()
    assert all_passed(comm, ok)


def test_several_stores_and_variables(comm, hs):
    """Default exchange names keep stores created in the same order apart."""
    rank, size = comm.Get_rank(), comm.Get_size()
    a = file_store(comm, hs, 1)
    b = file_store(comm, hs, 1)
    a.add("x", shard(rank))
    b.add("x", shard(rank) + 0.5)
    a.add("z", shard(rank, dtype=np.float64))
    idx = np.arange(NROWS * size)
    out = np.zeros((len(idx), NCOLS), dtype=np.float32)
    a.get_batch("x", out, idx)
    ok = np.array_equal(out, expected_rows(idx))
    b.get_batch("x", out, idx)
    ok = ok and np.array_equal(out, expected_rows(idx) + 0.5)
    out64 = np.zeros((len(idx), NCOLS), dtype=np.float64)
    a.get_batch("z", out64, idx)
    ok = ok and np.array_equal(out64, expected_rows(idx, np.float64))
    comm.Barrier()
    a.free()
    b.free()
    assert all_passed(comm, ok)


def test_stale_exchange_files_ignored(comm, hs):
    """A second store reusing an exchange name ignores the first one's
    leftover combined file (its per-rank nonces don't match)."""
    rank, size = comm.Get_rank(), comm.Get_size()
    first = file_store(comm, hs, 1, exchange_ns="same")
    first.add("x", shard(rank))
    comm.Barrier()
    first.free()
    leftovers = glob.glob(os.path.join(hs, ".x.*.same.0.all"))
    ok = len(leftovers) == 1

    second = file_store(comm, hs, 1, exchange_ns="same")
    second.add("x", shard(rank) + 7)
    idx = np.arange(NROWS * size)
    out = np.zeros((len(idx), NCOLS), dtype=np.float32)
    second.get_batch("x", out, idx)
    ok = ok and np.array_equal(out, expected_rows(idx) + 7)
    comm.Barrier()
    second.free()
    assert all_passed(comm, ok)


@pytest.mark.parametrize("late", ["others", "rank0"])
def test_stale_rank_files_ignored(comm, hs, monkeypatch, late):
    """Per-rank and go files left by an interrupted run with the same session
    and ns: rank 0 ignores per-rank files of another generation, and a rank
    that read the old go file resends its own. late="others": rank 0 finds
    the stale files before the others write theirs; late="rank0": the others
    read the stale go file before rank 0 replaces it."""
    rank, size = comm.Get_rank(), comm.Get_size()
    monkeypatch.setenv("DDSTORE_SESSION", f"stale{late}")
    monkeypatch.setenv("DDSTORE_HANDSHAKE_TIMEOUT_S", "20")
    pre = os.path.join(hs, f".x.stale{late}.st.0.")
    rng = np.random.default_rng()
    old_gen = rng.integers(1, 2**63, dtype=np.uint64)
    if rank == 0:
        old_gen.tofile(pre + "go")
    else:  # call 0 of FileComm.allgather gathers one int64: 8 + 8 + 8 bytes
        np.array([old_gen, rng.integers(1, 2**63), 0], dtype=np.uint64).tofile(
            pre + str(rank)
        )
    comm.Barrier()
    if (rank == 0) == (late == "rank0"):
        time.sleep(1)
    fc = dds.FileComm(hs, rank, size, "st")
    ok = fc.allgather(("a", rank)) == [("a", r) for r in range(size)]
    ok = ok and fc.allgather(rank * 10) == [r * 10 for r in range(size)]
    fc.barrier()
    ok = ok and not os.path.exists(pre + "go")  # removed on call 1
    assert all_passed(comm, ok)


def test_method2_extra_joins_file_core(comm, hs):
    """Core ranks without MPI publish {name}.bin; an extra member joins."""
    rank, size = comm.Get_rank(), comm.Get_size()
    core = dds.PyDDStore(None, method=2, handshake_dir=hs, rank=rank, n_core=size)
    core.add("x", shard(rank))
    comm.Barrier()  # rank 0 publishes x.bin after the exchange returns
    ok = os.path.exists(os.path.join(hs, "x.bin"))

    extra = dds.PyDDStore(None, method=2, handshake_dir=hs, n_core=size)
    extra.join("x")
    idx = np.arange(NROWS * size)[::-1].copy()
    out = np.zeros((len(idx), NCOLS), dtype=np.float32)
    extra.get_batch("x", out, idx)
    ok = ok and np.array_equal(out, expected_rows(idx))
    comm.Barrier()
    extra.free()
    core.free()
    assert all_passed(comm, ok)


def test_same_layout_as_mpi_exchange(comm, hs):
    """The file exchange and MPI_Allgather give the same global layout."""
    if not dds.HAVE_MPI:
        pytest.skip("needs an MPI build (DDSTORE_MPI=1)")
    rank = comm.Get_rank()
    n = 3 + rank
    data = np.full((n, NCOLS), rank, dtype=np.float32)
    viafile = file_store(comm, hs, 1)
    viafile.add("x", data)
    viampi = dds.PyDDStore(comm, method=1)
    viampi.add("x", data)
    total = comm.allreduce(n)
    idx = np.arange(total)
    a = np.zeros((total, NCOLS), dtype=np.float32)
    b = np.zeros_like(a)
    viafile.get_batch("x", a, idx)
    viampi.get_batch("x", b, idx)
    ok = viafile.info("x")[0] == viampi.info("x")[0] == total and np.array_equal(a, b)
    comm.Barrier()
    viafile.free()
    viampi.free()
    assert all_passed(comm, ok)


def test_shape_mismatch_raises_everywhere(comm, hs):
    if comm.Get_size() < 2:
        pytest.skip("requires at least 2 ranks")
    rank = comm.Get_rank()
    store = file_store(comm, hs, 1)
    cols = NCOLS + (1 if rank == 1 else 0)
    with pytest.raises(ValueError, match="Invalid disp"):
        store.add("x", np.zeros((4, cols), dtype=np.float32))
    comm.Barrier()


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(method=0, rank=0, size=1), "method=1 or 2"),
        (dict(method=1, rank=0), "size="),
        (dict(method=1, rank=3, size=2), "rank"),
    ],
)
def test_bad_arguments(hs, kw, match):
    with pytest.raises((ValueError, OverflowError), match=match):
        dds.PyDDStore(None, handshake_dir=hs, **kw)


def test_needs_handshake_dir():
    with pytest.raises(ValueError, match="handshake_dir"):
        dds.PyDDStore(None, method=1, rank=0, size=1)


@pytest.mark.skipif(not HAVE_GPU or FABRIC != "cxi", reason="needs a GPU and cxi")
def test_gpu_source_and_destination(comm, hs):
    rank, size = comm.Get_rank(), comm.Get_size()
    dev = torch.device(
        "cuda", int(os.environ.get("SLURM_LOCALID", "0")) % torch.cuda.device_count()
    )
    src = torch.from_numpy(shard(rank)).to(dev)
    store = file_store(comm, hs, 1)
    store.add("x", src)
    idx = np.arange(NROWS * size)
    out = torch.zeros((len(idx), NCOLS), dtype=torch.float32, device=dev)
    store.get_batch("x", out, idx)
    ok = np.array_equal(out.cpu().numpy(), expected_rows(idx))
    comm.Barrier()
    store.free()
    assert all_passed(comm, ok)
