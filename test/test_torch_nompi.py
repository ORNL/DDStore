"""
pyddstore.torch without MPI: DistDataset over a FileComm (rank and size from
the launcher) or a torch.distributed group, with stores that exchange their
records through files (methods 1 and 2). Never imports mpi4py, so it runs
with either build (DDSTORE_MPI=0 or not). Inside a Slurm step with a CXI
device, e.g.:
  srun -n 4 python -m pytest test/test_torch_nompi.py -v

DDSTORE_TEST_DIR (default: the current directory) must be on a filesystem
every rank sees. Run it on its own: test_no_mpi4py_imported fails if
another test file in the same session imports mpi4py. Sets
DDSTORE_FORK_WORKERS=1 (forked DataLoader workers) for the whole session.
"""

import glob
import os
import shutil
import sys
import time

import numpy as np
import pytest

# Before pyddstore (and libfabric) start: forked DataLoader workers.
os.environ.setdefault("DDSTORE_FORK_WORKERS", "1")

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

import pyddstore.torch as pt  # noqa: E402
from pyddstore import FileComm  # noqa: E402
from pyddstore.torch import (
    DistDataset,
    DistDatasetReader,
    ThreadDataLoader,
)  # noqa: E402

HAVE_CXI = bool(glob.glob("/dev/cxi*")) and "SLURM_STEP_ID" in os.environ
pytestmark = pytest.mark.skipif(not HAVE_CXI, reason="no CXI device")
FABRIC = os.environ.get("DDSTORE_FABRIC", "cxi")
N = 37  # not a multiple of the rank count


class Source(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return torch.full((2, 3), float(i)), i


class Bad(Source):
    """Sample 5 has a different shape (lands on one rank only)."""

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return (torch.zeros(4) if i == 5 else x), y


def _step():
    return (
        f"{os.environ.get('SLURM_JOB_ID', '0')}.{os.environ.get('SLURM_STEP_ID', '0')}"
    )


@pytest.fixture(scope="module")
def hs():
    """One shared directory per step (all ranks derive the same name)."""
    base = os.environ.get("DDSTORE_TEST_DIR", os.getcwd())
    path = os.path.join(base, f".test_tnompi.{_step()}")
    os.makedirs(path, exist_ok=True)
    yield path
    comm = pt._file_comm(path)
    comm.allgather(None)  # every rank done with the tests
    # Rank 0 returns from that allgather as soon as it has written the
    # combined file, before the others have read it: wait for each rank to
    # say it is past the allgather before removing the directory.
    open(os.path.join(path, f".done.{comm.rank}"), "w").close()
    if comm.rank == 0:
        while len(glob.glob(os.path.join(path, ".done.*"))) < comm.size:
            time.sleep(0.01)
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def file_comm(hs, monkeypatch):
    monkeypatch.setenv("DDSTORE_FABRIC", FABRIC)
    monkeypatch.setenv("DDSTORE_COMM", "file")
    return pt._file_comm(hs)


def read_all(ds, num_workers=0):
    """Every sample through the batched path; True if all are right."""
    loader = ThreadDataLoader(ds, batch_size=8, num_workers=num_workers)
    seen = 0
    ok = True
    for x, y in loader:
        for k in range(len(y)):
            i = int(y[k])
            ok &= bool(torch.equal(x[k], torch.full((2, 3), float(i))))
            ok &= i == seen
            seen += 1
    loader.close()
    return ok and seen == N


def finish(ds):
    ds._comm.allgather(None)  # every rank done reading before teardown
    ds.ddstore.free()


@pytest.mark.parametrize("method", [1, 2])
def test_file_comm_dataset(file_comm, hs, method):
    ds = DistDataset(Source(), f"f{method}", handshake_dir=hs, method=method)
    ok = not ds._comm.is_mpi and ds.method == method and len(ds) == N
    ok &= read_all(ds, num_workers=2)
    finish(ds)
    assert all(file_comm.allgather(ok))


def test_default_method_is_1_without_mpi(file_comm, hs, monkeypatch):
    monkeypatch.delenv("DDSTORE_METHOD", raising=False)
    ds = DistDataset(Source(), "dm", handshake_dir=hs)
    ok = ds.method == 1 and read_all(ds)
    finish(ds)
    assert all(file_comm.allgather(ok))


def test_new_handshake_dir_after_removal(file_comm, hs):
    """Each handshake_dir gets its own file communicator: a dataset in a new
    directory still works after an earlier directory is removed."""
    a, b = os.path.join(hs, "dir_a"), os.path.join(hs, "dir_b")
    ds = DistDataset(Source(), "da", handshake_dir=a, method=1)
    ok = read_all(ds)
    finish(ds)
    file_comm.allgather(None)  # every rank done with dir_a
    if file_comm.rank == 0:
        shutil.rmtree(a)
    file_comm.allgather(None)
    ds = DistDataset(Source(), "db", handshake_dir=b, method=1)
    ok &= ds._comm.handshake_dir == b and read_all(ds)
    finish(ds)
    assert all(file_comm.allgather(ok))


def test_ddstore_width_groups(file_comm, hs):
    if file_comm.size < 2 or file_comm.size % 2:
        pytest.skip("needs an even number of ranks")
    width = file_comm.size // 2
    ds = DistDataset(Source(), "w", handshake_dir=hs, method=1, ddstore_width=width)
    ok = ds._group.size == width and ds._group.rank == file_comm.rank % width
    ok &= read_all(ds)
    finish(ds)
    assert all(file_comm.allgather(ok))


def test_method2_reader(file_comm, hs):
    core = DistDataset(Source(), "rd", handshake_dir=hs, method=2)
    reader = DistDatasetReader("rd", handshake_dir=hs, n_core=file_comm.size)
    ok = len(reader) == N and read_all(reader)
    file_comm.allgather(None)
    reader.ddstore.free()
    finish(core)
    assert all(file_comm.allgather(ok))


def read_forked(ds, num_workers):
    """Every sample through PyTorch's DataLoader with forked workers."""
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, num_workers=num_workers, timeout=60
    )
    seen = 0
    ok = True
    for x, y in loader:
        ok &= x.device.type == "cpu"
        for k in range(len(y)):
            i = int(y[k])
            ok &= bool(torch.equal(x[k], torch.full((2, 3), float(i))))
            ok &= i == seen
            seen += 1
    return ok and seen == N


@pytest.mark.parametrize("method", [1, 2])
def test_forked_dataloader_workers(file_comm, hs, method):
    ds = DistDataset(Source(), f"fk{method}", handshake_dir=hs, method=method)
    # The parent reads before forking (this hung workers under kdreg2).
    ok = int(ds[3][1]) == 3
    for _ in range(2):  # new workers every epoch
        ok &= read_forked(ds, num_workers=2)
    ok &= int(ds[N - 1][1]) == N - 1  # the parent's store still works
    finish(ds)
    assert all(file_comm.allgather(ok))


def test_forked_workers_method2_reader(file_comm, hs):
    core = DistDataset(Source(), "fkr", handshake_dir=hs, method=2)
    reader = DistDatasetReader("fkr", handshake_dir=hs, n_core=file_comm.size)
    ok = read_forked(reader, num_workers=2)
    file_comm.allgather(None)
    reader.ddstore.free()
    finish(core)
    assert all(file_comm.allgather(ok))


def test_fork_reader_refuses_kdreg2(file_comm, hs, monkeypatch):
    ds = DistDataset(Source(), "fkk", handshake_dir=hs, method=1)
    # In the parent kdreg2 is fine: a second reader of its own works.
    # (DDSTORE_FORK_WORKERS=0: otherwise opening it switches the
    # environment to memhooks, which the worker below would inherit.)
    monkeypatch.setenv("DDSTORE_FORK_WORKERS", "0")
    monkeypatch.setenv("FI_MR_CACHE_MONITOR", "kdreg2")
    r = ds.ddstore.fork_reader()
    out = np.zeros((1, 6), dtype=np.float32)
    r.get_batch(ds._var[0], out, [4])
    ok = bool((out == 4.0).all())
    r.free()
    # A forked worker sees kdreg2 (environment inherited) and refuses.
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=1, timeout=60)
    with pytest.raises(RuntimeError, match="DDSTORE_FORK_WORKERS"):
        next(iter(loader))
    finish(ds)
    assert all(file_comm.allgather(ok))


def test_error_raised_on_every_rank(file_comm, hs):
    # "differs from" (one rank sees it) or "differ ... across ranks" (a
    # rank's first sample is the bad one, with more ranks).
    with pytest.raises(ValueError, match="differ"):
        DistDataset(Bad(), "bad", handshake_dir=hs, method=1)


def test_method0_needs_mpi(file_comm, hs):
    with pytest.raises(ValueError, match="method=0"):
        DistDataset(Source(), "m0", handshake_dir=hs, method=0)


def test_file_comm_allgather(file_comm, hs):
    fc = FileComm(hs, file_comm.rank, file_comm.size, "direct")
    got = fc.allgather({"r": fc.rank, "pad": "x" * (fc.rank * 1000)})
    ok = [g["r"] for g in got] == list(range(fc.size))
    ok &= all(len(g["pad"]) == 1000 * g["r"] for g in got)
    fc.barrier()
    assert all(file_comm.allgather(ok))


def test_torch_distributed_group(file_comm, hs, monkeypatch):
    """A gloo group passed as comm=, and found for comm=None."""
    init = os.path.join(hs, "torch.init")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init}",
        rank=file_comm.rank,
        world_size=file_comm.size,
    )
    try:
        a = DistDataset(
            Source(), "tg", comm=dist.group.WORLD, handshake_dir=hs, method=1
        )
        ok = not a._comm.is_mpi and read_all(a)
        monkeypatch.setenv("DDSTORE_COMM", "torch")
        b = DistDataset(Source(), "tn", handshake_dir=hs, method=1)
        ok &= read_all(b)
        finish(a)
        finish(b)
        ok = all(file_comm.allgather(ok))
    finally:
        dist.destroy_process_group()
    if file_comm.rank == 0 and os.path.exists(init):
        os.remove(init)
    assert ok


def test_disjoint_torch_groups(file_comm, hs):
    """Two disjoint torch.distributed groups build datasets in one
    handshake_dir at the same time: their exchange files must not mix."""
    if file_comm.size < 2:
        pytest.skip("needs 2 ranks")
    init = os.path.join(hs, "torch2.init")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init}",
        rank=file_comm.rank,
        world_size=file_comm.size,
    )
    try:
        half = file_comm.size // 2
        groups = [
            dist.new_group(list(range(half))),
            dist.new_group(list(range(half, file_comm.size))),
        ]
        mine = groups[file_comm.rank >= half]
        ds = DistDataset(Source(), "dg", comm=mine, handshake_dir=hs, method=1)
        ok = ds._comm.size == dist.get_world_size(mine) and read_all(ds)
        finish(ds)
        ok = all(file_comm.allgather(ok))
    finally:
        dist.destroy_process_group()
    if file_comm.rank == 0 and os.path.exists(init):
        os.remove(init)
    assert ok


def test_no_mpi4py_imported():
    assert "mpi4py" not in sys.modules
